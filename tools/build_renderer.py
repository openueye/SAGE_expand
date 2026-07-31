from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

import torch

from sage.engine.rendering import (
    RENDERER_BUILD_MANIFEST_NAME,
    RENDERER_BUILD_MANIFEST_SCHEMA,
    RENDERER_BUILD_ENVIRONMENT_VARIABLES,
    RENDERER_BUILD_SOURCE,
    RENDERER_BUILD_WORKSPACE,
    RENDERER_COMPILE_FLAGS,
    RENDERER_CUDA_ARCH_LIST,
    RENDERER_PROVENANCE_COMMIT,
    RENDERER_SOURCE_TREE_SHA256,
    _tracked_renderer_source_paths,
    _validate_renderer_toolchain_package,
    _torch_build_identity,
    _expected_renderer_build_command,
    _expected_renderer_build_environment,
    _installed_conda_package_identity,
    renderer_source_tree_sha256,
)
from sage.foundation.hashing import sha256_file


def _command_version(command: list[str], pattern: str, description: str) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot capture {description}") from exc
    output = completed.stdout + completed.stderr
    match = re.search(pattern, output, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Cannot parse {description}")
    return match.group(1)


def _export_tracked_sources(destination: Path) -> list[dict[str, object]]:
    if destination.exists():
        raise RuntimeError("Renderer source export destination must be new")
    destination.mkdir(parents=True)
    inputs: list[dict[str, object]] = []
    source_root, paths = _tracked_renderer_source_paths()
    for path in paths:
        relative = path.relative_to(source_root)
        exported = destination / relative
        exported.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, exported, follow_symlinks=False)
        inputs.append({
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        })
    return inputs


@contextmanager
def _build_workspace(temporary_root: Path) -> Iterator[Path]:
    workspace = temporary_root / "sage-renderer-build"
    try:
        workspace.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Renderer build workspace already exists: {workspace}") from exc
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace)


def _build_environment(
    inherited: dict[str, str], *, cc: Path, cxx: Path, nvcc: Path, cuda_home: Path,
) -> dict[str, str]:
    environment = inherited.copy()
    for name in RENDERER_BUILD_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    environment.update({
        "CC": str(cc),
        "CXX": str(cxx),
        "NVCC": str(nvcc),
        "PATH": f"{cc.parent}:/usr/bin:/bin",
        "CUDA_HOME": str(cuda_home),
        "CUDAHOSTCXX": str(cxx),
        "CUDACXX": str(nvcc),
        "LDFLAGS": "-Wl,--strip-all",
        "MAX_JOBS": "1",
        "PYTHONNOUSERSITE": "1",
        "TORCH_CUDA_ARCH_LIST": RENDERER_CUDA_ARCH_LIST,
        "SOURCE_DATE_EPOCH": "0",
    })
    return environment


def _resolve_owned_tool(
    environment: dict[str, str], *, variable: str, fallback: str, environment_root: Path,
) -> Path:
    requested = environment.get(variable, fallback)
    located = shutil.which(requested)
    if located is None:
        raise RuntimeError(f"Renderer build requires {variable} executable {requested!r}")
    resolved = Path(located).resolve()
    if not resolved.is_relative_to(environment_root.resolve()):
        raise RuntimeError(f"Renderer build requires {variable} from the sage environment")
    return resolved


def _conda_package_identity(environment_root: Path, tool_path: Path) -> dict[str, str]:
    return _installed_conda_package_identity(environment_root, tool_path)


def _tool_identity(
    path: Path, *, version_pattern: str, description: str, environment_root: Path,
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "version": _command_version([str(path), "--version"], version_pattern, description),
        "conda_package": _conda_package_identity(environment_root, path),
    }


def _build_manifest(
    *,
    source_tree_sha256: str,
    source_inputs: list[dict[str, object]],
    command: list[str],
    cwd: Path,
    build_environment: dict[str, str],
    toolchain: dict[str, dict[str, object]],
    torch_build_identity: dict[str, object],
    extension_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": RENDERER_BUILD_MANIFEST_SCHEMA,
        "source_tree_sha256": source_tree_sha256,
        "source_files": source_inputs,
        "provenance_commit": RENDERER_PROVENANCE_COMMIT,
        "build_command": command,
        "build_cwd": str(cwd),
        "build_environment": {
            name: build_environment.get(name) for name in RENDERER_BUILD_ENVIRONMENT_VARIABLES
        },
        "toolchain": toolchain,
        "torch_build_identity": torch_build_identity,
        "extension_sha256": sha256_file(extension_path),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "nvcc_version": toolchain["nvcc"]["version"],
        "compiler": toolchain["cxx"]["version"],
        "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "cuda_arch_list": RENDERER_CUDA_ARCH_LIST,
        "compile_flags": list(RENDERER_COMPILE_FLAGS),
    }


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    actual_source_hash = renderer_source_tree_sha256()
    if actual_source_hash != RENDERER_SOURCE_TREE_SHA256:
        raise RuntimeError(
            "Vendored renderer source tree SHA-256 mismatch: "
            f"expected {RENDERER_SOURCE_TREE_SHA256}, got {actual_source_hash}"
        )
    environment_root = Path(sys.prefix).resolve()
    nvcc_path = _resolve_owned_tool(
        os.environ, variable="NVCC", fallback="nvcc", environment_root=environment_root,
    )
    cc_path = _resolve_owned_tool(
        os.environ, variable="CC", fallback="cc", environment_root=environment_root,
    )
    compiler_path = _resolve_owned_tool(
        os.environ, variable="CXX", fallback="c++", environment_root=environment_root,
    )
    cuda_home = nvcc_path.parent.parent
    toolchain = {
        "cc": _tool_identity(
            cc_path, version_pattern=r"^([^\n]+)", description="C compiler identity",
            environment_root=environment_root,
        ),
        "cxx": _tool_identity(
            compiler_path, version_pattern=r"^([^\n]+)", description="C++ compiler identity",
            environment_root=environment_root,
        ),
        "nvcc": _tool_identity(
            nvcc_path, version_pattern=r"V([0-9.]+)", description="CUDA nvcc identity",
            environment_root=environment_root,
        ),
    }
    for role, identity in toolchain.items():
        _validate_renderer_toolchain_package(role, identity["conda_package"])
    torch_build_identity = _torch_build_identity()
    build_environment = _build_environment(
        os.environ, cc=cc_path, cxx=compiler_path, nvcc=nvcc_path, cuda_home=cuda_home,
    )
    with _build_workspace(RENDERER_BUILD_WORKSPACE.parent) as workspace:
        if workspace != RENDERER_BUILD_WORKSPACE:
            raise RuntimeError("Renderer build workspace identity is invalid")
        source_export = workspace / "source"
        source_inputs = _export_tracked_sources(source_export)
        build_command = _expected_renderer_build_command()
        build_environment_identity = {
            name: build_environment.get(name) for name in RENDERER_BUILD_ENVIRONMENT_VARIABLES
        }
        if build_environment_identity != _expected_renderer_build_environment(toolchain):
            raise RuntimeError("Renderer build environment does not match the locked identity")
        subprocess.run(
            build_command,
            cwd=repository,
            env=build_environment,
            check=True,
        )
    importlib.invalidate_caches()
    package = importlib.import_module("diff_gaussian_rasterization")
    package_path = Path(package.__file__).resolve()
    extension_path = Path(package._C.__file__).resolve()
    if not package_path.is_relative_to(environment_root) or not extension_path.is_relative_to(environment_root):
        raise RuntimeError("Renderer build installed outside the sage environment")
    manifest = _build_manifest(
        source_tree_sha256=actual_source_hash,
        source_inputs=source_inputs,
        command=build_command,
        cwd=repository,
        build_environment=build_environment,
        toolchain=toolchain,
        torch_build_identity=torch_build_identity,
        extension_path=extension_path,
    )
    manifest_path = package_path.with_name(RENDERER_BUILD_MANIFEST_NAME)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"extension": str(extension_path), "build_manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
