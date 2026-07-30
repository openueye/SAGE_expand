from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
from typing import Protocol

import torch
from torch.nn import functional as F

from .contracts import FrameInputs
from .geometry import rotation_wc, world_to_camera
from .hashing import sha256_file
from .model import TrainableGaussians


RENDERER_PROVENANCE_COMMIT = "cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110"
RENDERER_SOURCE_TREE_SHA256 = "6a2380890c9d110e4f597fec76fabfe781a375da94f5b569a7e1c9bd9c32b4ec"
RENDERER_ADAPTER_SCHEMA = "sage-renderer-adapter-v1"
RENDERER_BUILD_MANIFEST_SCHEMA = "sage-renderer-build-v1"
RENDERER_BUILD_MANIFEST_NAME = "sage_build_identity.json"
RENDERER_CUDA_ARCH_LIST = "8.9"
RENDERER_BUILD_WORKSPACE = Path("/tmp/sage-renderer-build")
RENDERER_BUILD_SOURCE = RENDERER_BUILD_WORKSPACE / "source"
RENDERER_BUILD_ENVIRONMENT_VARIABLES = (
    "PATH", "CC", "CXX", "NVCC", "CPP", "CFLAGS", "CPPFLAGS", "CXXFLAGS",
    "LDFLAGS", "CUDAFLAGS", "NVCC_FLAGS", "CUDA_HOME", "CUDA_PATH",
    "CUDAHOSTCXX", "CUDACXX", "PYTORCH_NVCC", "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "PYTHONPATH",
    "PYTHONNOUSERSITE", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
    "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "SOURCE_DATE_EPOCH",
)
RENDERER_TOOLCHAIN_PACKAGE_CONTRACT = {
    "cc": {"name": "gcc_impl_linux-64", "version": "11.4.0"},
    "cxx": {"name": "gxx_impl_linux-64", "version": "11.4.0"},
    "nvcc": {"name": "cuda-nvcc", "version": "12.1.105"},
}
RENDERER_COMPILE_FLAGS = (
    f"TORCH_CUDA_ARCH_LIST={RENDERER_CUDA_ARCH_LIST}",
    "_GLIBCXX_USE_CXX11_ABI=0",
)
RENDERER_REBUILD_COMMAND = (
    "env -u PYTHONPATH PYTHONNOUSERSITE=1 conda run -n sage "
    "python tools/build_renderer.py"
)


@dataclass(frozen=True)
class RenderOutput:
    """Raw renderer output: RGB, accumulated depth moment, and alpha."""

    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor

    @property
    def accumulated_depth(self) -> torch.Tensor:
        """Return the unnormalized accumulated depth for new callers."""
        return self.depth


class Renderer(Protocol):
    def __call__(self, model: object, frame: FrameInputs) -> RenderOutput: ...


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _renderer_source_root() -> Path:
    return _repository_root() / "third_party" / "diff-gaussian-rasterization-w-depth"


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise RuntimeError(f"Build input tree is empty: {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _torch_build_identity() -> dict[str, object]:
    package_root = Path(torch.__file__).resolve().parent
    relative_files = [
        Path(torch._C.__file__).resolve().relative_to(package_root),
        *(Path("lib") / name for name in (
            "libc10.so", "libc10_cuda.so", "libtorch_cpu.so", "libtorch_cuda.so",
            "libtorch_python.so",
        )),
    ]
    runtime_files = []
    for relative in relative_files:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"PyTorch build input is missing: {path}")
        runtime_files.append({
            "path": relative.as_posix(), "sha256": sha256_file(path),
            "size": path.stat().st_size,
        })
    return {
        "package_root": str(package_root),
        "include_tree_sha256": _sha256_tree(package_root / "include"),
        "runtime_files": runtime_files,
    }


def _expected_renderer_build_command() -> list[str]:
    return [
        sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
        "--no-build-isolation", "--no-deps", str(RENDERER_BUILD_SOURCE),
    ]


def _equivalent_renderer_build_command(
    recorded: list[str],
    expected: list[str],
) -> bool:
    """Compare a locked build command without distinguishing interpreter aliases."""
    if not recorded or not expected or recorded[1:] != expected[1:]:
        return False
    try:
        return Path(recorded[0]).resolve(strict=True) == Path(expected[0]).resolve(
            strict=True
        )
    except OSError:
        return False


def _expected_renderer_build_environment(
    toolchain: dict[str, dict[str, object]],
) -> dict[str, str | None]:
    cc = str(toolchain["cc"]["path"])
    cxx = str(toolchain["cxx"]["path"])
    nvcc = str(toolchain["nvcc"]["path"])
    environment_root = Path(cc).parent.parent
    fixed = {
        "PATH": f"{environment_root / 'bin'}:/usr/bin:/bin",
        "CC": cc,
        "CXX": cxx,
        "NVCC": nvcc,
        "LDFLAGS": "-Wl,--strip-all",
        "CUDA_HOME": str(Path(nvcc).parent.parent),
        "CUDAHOSTCXX": cxx,
        "CUDACXX": nvcc,
        "TORCH_CUDA_ARCH_LIST": RENDERER_CUDA_ARCH_LIST,
        "MAX_JOBS": "1",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": "0",
    }
    return {name: fixed.get(name) for name in RENDERER_BUILD_ENVIRONMENT_VARIABLES}


def _installed_conda_package_identity(
    environment_root: Path, tool_path: Path,
) -> dict[str, str]:
    relative = tool_path.resolve().relative_to(environment_root.resolve()).as_posix()
    for record_path in sorted((environment_root / "conda-meta").glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read Conda package record {record_path}") from exc
        files = record.get("files") if isinstance(record, dict) else None
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise RuntimeError(f"Conda package record has invalid files for {tool_path}")
        if relative in files:
            identity = {name: record.get(name) for name in ("name", "version", "build", "sha256")}
            if not all(isinstance(value, str) and value for value in identity.values()):
                raise RuntimeError(f"Conda package record is incomplete for {tool_path}")
            if len(identity["sha256"]) != 64 or any(
                character not in "0123456789abcdef" for character in identity["sha256"]
            ):
                raise RuntimeError(f"Conda package record has an invalid SHA-256 for {tool_path}")
            return identity
    raise RuntimeError(f"Renderer tool is not owned by a locked Conda package: {tool_path}")


def _validate_renderer_toolchain_package(role: str, package: object) -> None:
    expected = RENDERER_TOOLCHAIN_PACKAGE_CONTRACT.get(role)
    if expected is None or not isinstance(package, dict) or set(package) != {
        "name", "version", "build", "sha256",
    }:
        raise RuntimeError("Renderer build manifest Conda tool identity is invalid")
    if any(not isinstance(package.get(name), str) or not package[name] for name in package):
        raise RuntimeError("Renderer build manifest Conda tool identity is invalid")
    if len(package["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in package["sha256"]
    ):
        raise RuntimeError("Renderer build manifest Conda tool identity is invalid")
    if package["name"] != expected["name"] or package["version"] != expected["version"]:
        raise RuntimeError(f"Renderer build manifest {role} package is not locked")


def renderer_source_tree_sha256() -> str:
    source_root, paths = _tracked_renderer_source_paths()
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tracked_renderer_source_paths() -> tuple[Path, list[Path]]:
    repository = _repository_root()
    source_root = _renderer_source_root()
    try:
        tracked = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z", "--", str(source_root.relative_to(repository))],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot enumerate tracked vendored renderer sources") from exc
    paths = [repository / raw.decode("utf-8") for raw in tracked if raw]
    if not paths:
        raise RuntimeError("Vendored renderer has no tracked source files")
    return source_root, paths


def _renderer_source_file_identities() -> list[dict[str, object]]:
    source_root, paths = _tracked_renderer_source_paths()
    return [{
        "path": path.relative_to(source_root).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    } for path in paths]


def _load_renderer_extension():
    try:
        package = import_module("diff_gaussian_rasterization")
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Vendored CUDA renderer is unavailable. Rebuild with: {RENDERER_REBUILD_COMMAND}") from exc
    package_path = Path(package.__file__).resolve()
    extension_path = Path(package._C.__file__).resolve()
    environment_root = Path(sys.prefix).resolve()
    if not package_path.is_relative_to(environment_root) or not extension_path.is_relative_to(environment_root):
        raise RuntimeError("CUDA renderer must load from the sage environment")
    forbidden = {"splaHybrid", ".local"}
    if any(part in forbidden for part in extension_path.parts):
        raise RuntimeError(f"CUDA renderer loaded from a forbidden external path: {extension_path}")
    return package


def _load_renderer_build_manifest(package: object, extension_path: Path) -> dict[str, object]:
    package_path = Path(package.__file__).resolve()
    manifest_path = package_path.with_name(RENDERER_BUILD_MANIFEST_NAME)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Renderer build manifest is missing or invalid. Rebuild with: {RENDERER_REBUILD_COMMAND}") from exc
    expected_fields = {
        "schema_version", "source_tree_sha256", "provenance_commit", "extension_sha256",
        "python_soabi", "torch_version", "cuda_version", "nvcc_version", "compiler",
        "abi", "cuda_arch_list", "compile_flags", "source_files", "build_command",
        "build_cwd", "build_environment", "toolchain", "torch_build_identity",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise RuntimeError("Renderer build manifest fields are invalid")
    if manifest.get("schema_version") != RENDERER_BUILD_MANIFEST_SCHEMA:
        raise RuntimeError("Renderer build manifest schema is unsupported")
    if manifest.get("source_tree_sha256") != RENDERER_SOURCE_TREE_SHA256:
        raise RuntimeError("Renderer build manifest source identity is invalid")
    if manifest.get("provenance_commit") != RENDERER_PROVENANCE_COMMIT:
        raise RuntimeError("Renderer build manifest provenance is invalid")
    if manifest.get("source_files") != _renderer_source_file_identities():
        raise RuntimeError("Renderer build manifest source inputs are invalid")
    if manifest.get("extension_sha256") != sha256_file(extension_path):
        raise RuntimeError("Renderer extension does not match its build manifest")
    command = manifest.get("build_command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise RuntimeError("Renderer build manifest command is invalid")
    if not _equivalent_renderer_build_command(
        command,
        _expected_renderer_build_command(),
    ):
        raise RuntimeError("Renderer build manifest command is invalid")
    if manifest.get("build_cwd") != str(_repository_root()):
        raise RuntimeError("Renderer build manifest working directory is invalid")
    build_environment = manifest.get("build_environment")
    if not isinstance(build_environment, dict):
        raise RuntimeError("Renderer build manifest environment is invalid")
    environment_root = Path(sys.prefix).resolve()
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != {"cc", "cxx", "nvcc"}:
        raise RuntimeError("Renderer build manifest toolchain is invalid")
    for role, identity in toolchain.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256", "version", "conda_package"}:
            raise RuntimeError("Renderer build manifest tool identity is invalid")
        tool_path = Path(identity["path"]).resolve() if isinstance(identity.get("path"), str) else None
        if tool_path is None or not tool_path.is_relative_to(environment_root) or not tool_path.is_file():
            raise RuntimeError("Renderer build manifest tool path is invalid")
        if not isinstance(identity.get("sha256"), str) or len(identity["sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in identity["sha256"]
        ):
            raise RuntimeError("Renderer build manifest tool identity is invalid")
        if identity.get("sha256") != sha256_file(tool_path):
            raise RuntimeError("Renderer build manifest tool binary has changed")
        package = identity.get("conda_package")
        _validate_renderer_toolchain_package(role, package)
        if package != _installed_conda_package_identity(environment_root, tool_path):
            raise RuntimeError("Renderer build manifest Conda tool identity has changed")
    if build_environment != _expected_renderer_build_environment(toolchain):
        raise RuntimeError("Renderer build manifest environment is invalid")
    if manifest.get("torch_build_identity") != _torch_build_identity():
        raise RuntimeError("Renderer build manifest PyTorch inputs do not match the active runtime")
    runtime_identity = {
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
    }
    if any(manifest.get(name) != value for name, value in runtime_identity.items()):
        raise RuntimeError("Renderer build manifest does not match the active runtime")
    if manifest.get("cuda_arch_list") != RENDERER_CUDA_ARCH_LIST:
        raise RuntimeError("Renderer build manifest CUDA architecture is invalid")
    if manifest.get("compile_flags") != list(RENDERER_COMPILE_FLAGS):
        raise RuntimeError("Renderer build manifest compile flags are invalid")
    if not all(isinstance(manifest.get(name), str) and manifest[name] for name in ("compiler", "nvcc_version")):
        raise RuntimeError("Renderer build manifest toolchain identity is invalid")
    if manifest["compiler"] != toolchain["cxx"]["version"]:
        raise RuntimeError("Renderer build manifest compiler identities disagree")
    if manifest["nvcc_version"] != toolchain["nvcc"]["version"]:
        raise RuntimeError("Renderer build manifest NVCC identities disagree")
    return manifest


def capture_renderer_identity() -> dict[str, object]:
    actual_tree_hash = renderer_source_tree_sha256()
    if actual_tree_hash != RENDERER_SOURCE_TREE_SHA256:
        raise RuntimeError(
            "Vendored renderer source tree SHA-256 mismatch: "
            f"expected {RENDERER_SOURCE_TREE_SHA256}, got {actual_tree_hash}"
        )
    package = _load_renderer_extension()
    extension_path = Path(package._C.__file__).resolve()
    build = _load_renderer_build_manifest(package, extension_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Renderer identity requires an available CUDA device")
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"{major}.{minor}"
    if compute_capability != build["cuda_arch_list"]:
        raise RuntimeError("Renderer build does not target the active CUDA device")
    return {
        "kind": "repository-cuda",
        "source_tree_sha256": build["source_tree_sha256"],
        "provenance_commit": build["provenance_commit"],
        "extension_sha256": build["extension_sha256"],
        "adapter_schema": RENDERER_ADAPTER_SCHEMA,
        "python_soabi": build["python_soabi"],
        "torch_version": build["torch_version"],
        "cuda_version": build["cuda_version"],
        "nvcc_version": build["nvcc_version"],
        "compiler": build["compiler"],
        "abi": build["abi"],
        "compute_capability": compute_capability,
        "compile_flags": build["compile_flags"],
    }


def alpha_normalize_render_output_depth(output: RenderOutput, *, min_alpha: float) -> RenderOutput:
    """Compatibility re-export; production rendering never normalizes depth."""
    from .losses import alpha_normalize_render_output_depth as normalize

    return normalize(output, min_alpha=min_alpha)


def _camera_matrices(
    frame: FrameInputs, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation_cw = rotation_wc(frame.pose, device=device).T
    center = torch.tensor(frame.pose.translation, dtype=torch.float32, device=device)
    world_to_view = torch.eye(4, dtype=torch.float32, device=device)
    world_to_view[:3, :3] = rotation_cw
    world_to_view[:3, 3] = -(rotation_cw @ center)
    world_to_view = world_to_view.T.contiguous()
    near, far = 0.01, 100.0
    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    projection[0, 0] = 2 * frame.intrinsics.fx / frame.intrinsics.width
    projection[1, 1] = 2 * frame.intrinsics.fy / frame.intrinsics.height
    projection[2, 0] = 2 * frame.intrinsics.cx / frame.intrinsics.width - 1
    projection[2, 1] = 2 * frame.intrinsics.cy / frame.intrinsics.height - 1
    projection[2, 2] = far / (far - near)
    projection[2, 3] = 1
    projection[3, 2] = -(far * near) / (far - near)
    return world_to_view, (world_to_view @ projection).contiguous(), center


def render(model: TrainableGaussians, frame: FrameInputs) -> RenderOutput:
    if model.means3d.device.type != "cuda":
        raise RuntimeError("Repository renderer requires CUDA; CPU rendering is not supported")
    backend = _load_renderer_extension()
    device = model.means3d.device
    view, projection, center = _camera_matrices(frame, device)
    settings_values = {
        "image_height": frame.intrinsics.height, "image_width": frame.intrinsics.width,
        "tanfovx": frame.intrinsics.width / (2 * frame.intrinsics.fx),
        "tanfovy": frame.intrinsics.height / (2 * frame.intrinsics.fy),
        "bg": torch.zeros(3, dtype=torch.float32, device=device), "scale_modifier": 1.0,
        "viewmatrix": view, "projmatrix": projection, "sh_degree": 0, "campos": center,
        "prefiltered": False,
    }
    fields = getattr(backend.GaussianRasterizationSettings, "_fields", ())
    if "debug" in fields:
        settings_values["debug"] = False
    if "antialiasing" in fields:
        settings_values["antialiasing"] = False
    settings = backend.GaussianRasterizationSettings(**settings_values)
    rasterizer = backend.GaussianRasterizer(settings)
    means2d = torch.zeros_like(model.means3d, requires_grad=True)
    rotations = F.normalize(model.rotations, dim=1).contiguous()
    rgb, _, _ = rasterizer(
        means3D=model.means3d.contiguous(), means2D=means2d,
        colors_precomp=model.colors.contiguous(), opacities=model.opacities.contiguous(),
        scales=model.scales.contiguous(), rotations=rotations,
    )
    camera_z = world_to_camera(model.means3d, frame.pose)[:, 2]
    depth_silhouette_colors = torch.stack(
        (camera_z, torch.ones_like(camera_z), camera_z.square()), dim=1
    )
    depth_silhouette, _, _ = backend.GaussianRasterizer(settings)(
        means3D=model.means3d.contiguous(), means2D=torch.zeros_like(model.means3d),
        colors_precomp=depth_silhouette_colors.contiguous(), opacities=model.opacities.contiguous(),
        scales=model.scales.contiguous(), rotations=rotations,
    )
    return RenderOutput(rgb.permute(1, 2, 0), depth_silhouette[0], depth_silhouette[1].clamp(0, 1))
