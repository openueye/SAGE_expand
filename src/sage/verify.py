from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import site
import subprocess
import sys

import numpy as np
import sage
import torch

from .foundation.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _module_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Required runtime package is unavailable: {name}")
    return Path(spec.origin).resolve()


def _verify_conda_prefix() -> None:
    prefix_value = os.environ.get("CONDA_PREFIX")
    if not prefix_value:
        raise RuntimeError("sage verify must run in a Conda environment")
    if Path(prefix_value).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError("Active Conda prefix does not own the running Python interpreter")


def execution_preflight(config: object, *, device: str = "cuda") -> dict[str, object]:
    from .data.providers.spnet import OnlineSPNetProvider
    from .engine.model import TrainableGaussians
    from .engine.rendering import render
    from .foundation.config import SageConfig
    from .foundation.contracts import (
        CameraIntrinsics,
        DepthEvidence,
        FrameInputs,
        GrowthInputs,
        InputSourceFamily,
        INVALID_SOURCE_TYPE,
        MappingObservation,
        Pose,
        SourceType,
    )

    if not isinstance(config, SageConfig):
        raise ValueError("Execution preflight requires the frozen SAGE configuration")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Execution preflight requires an available CUDA device")
    height = width = 32
    depth = np.zeros((height, width), dtype=np.float32)
    valid = depth > 0
    source_types = np.full((height, width), INVALID_SOURCE_TYPE, dtype=np.uint8)
    confidence = valid.astype(np.float32)
    frame = FrameInputs(
        index=0,
        stem="sage-preflight",
        timestamp_ns=0,
        intrinsics=CameraIntrinsics(width, height, 32.0, 32.0, width / 2, height / 2),
        pose=Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        rgb=np.zeros((height, width, 3), dtype=np.float32),
        mapping=MappingObservation(
            depth, source_types, confidence, InputSourceFamily.LIDAR_WORLD,
        ),
        growth=GrowthInputs((DepthEvidence(
            SourceType.LIDAR_CENTER,
            depth,
            valid,
            confidence,
            InputSourceFamily.LIDAR_WORLD,
        ),)),
    )
    target = torch.device(device)
    model = TrainableGaussians(
        torch.tensor([[0.0, 0.0, 1.0]], device=target),
        torch.full((1, 3), 0.5, device=target),
        torch.zeros((1, 1), device=target),
        torch.full((1, 1), -4.0, device=target),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=target),
        torch.zeros(1, dtype=torch.int64, device=target),
        torch.zeros(1, dtype=torch.int64, device=target),
        torch.full((1,), int(SourceType.LIDAR_CENTER), dtype=torch.uint8, device=target),
        torch.ones(1, device=target),
    ).eval()
    rendered = render(model, frame)
    if (
        rendered.rgb.shape != (height, width, 3)
        or rendered.depth.shape != (height, width)
        or rendered.alpha.shape != (height, width)
        or not all(bool(torch.isfinite(value).all()) for value in (
            rendered.rgb, rendered.depth, rendered.alpha,
        ))
    ):
        raise RuntimeError("CUDA renderer preflight returned an invalid output")
    spnet = OnlineSPNetProvider(
        config.growth_sources.spnet,
        device=device,
        model_root=config.model_root,
    )
    evidence = spnet.evidence_for(frame)
    if evidence.depth_m.shape != (height, width):
        raise RuntimeError("SPNet preflight returned an invalid output")
    return {
        "renderer_output_shape": [height, width],
        "spnet_output_shape": list(evidence.depth_m.shape),
        "spnet_model_id": spnet.identity.model_id,
    }


def verify(
    *,
    require_models: bool = False,
    model_root: Path | None = None,
    require_clean_worktree: bool = False,
) -> dict[str, object]:
    package_path = Path(sage.__file__).resolve().parent
    expected_package = (ROOT / "src" / "sage").resolve()
    if package_path != expected_package:
        raise RuntimeError(f"sage resolved outside the repository src tree: {package_path}")

    user_site = Path(site.getusersitepackages()).resolve()
    resolved_sys_path = {
        Path(entry or os.getcwd()).resolve()
        for entry in sys.path
        if entry or os.getcwd()
    }
    if site.ENABLE_USER_SITE or user_site in resolved_sys_path:
        raise RuntimeError("Python user site must be disabled and absent from sys.path")
    _verify_conda_prefix()

    dirty = bool(_git("status", "--porcelain"))
    if require_clean_worktree and dirty:
        raise RuntimeError("Clean worktree required")

    from .data.providers.spnet import (
        SPNET_SOURCE_ID,
        verify_spnet_source_identity,
    )
    from .engine.metrics import _calibration_path
    from .engine.model_registry import (
        ALEXNET_MODEL_ID,
        SPNET_MODEL_ID,
        ModelRegistry,
        default_registry_path,
    )
    from .engine.rendering import (
        _gsplat_extension_path,
        _gsplat_package_path,
        capture_renderer_identity,
    )
    from .foundation.code_identity import repository_code_identity

    renderer_identity = capture_renderer_identity()
    renderer_package_path = _gsplat_package_path()
    renderer_extension_path = _gsplat_extension_path()
    resolved_model_root = (
        Path(model_root).expanduser().resolve()
        if model_root is not None
        else (
            Path(os.environ["SAGE_MODEL_ROOT"]).expanduser().resolve()
            if os.environ.get("SAGE_MODEL_ROOT")
            else None
        )
    )
    model_paths: dict[str, str | None] = {SPNET_MODEL_ID: None, ALEXNET_MODEL_ID: None}
    if require_models and resolved_model_root is None:
        raise RuntimeError(
            "A model root is required when --require-models is set; "
            "configure runtime.model_root or SAGE_MODEL_ROOT"
        )
    registry = ModelRegistry.load(default_registry_path())
    model_identities = {
        model_id: {"path": None, "sha256": registry.require(model_id).sha256}
        for model_id in model_paths
    }
    if resolved_model_root is not None:
        model_paths = {
            model_id: str(registry.resolve(model_id, model_root=resolved_model_root))
            for model_id in model_paths
        }
        model_identities = {
            model_id: {"path": model_paths[model_id], "sha256": registry.require(model_id).sha256}
            for model_id in model_paths
        }
    calibration_path, calibration_version, calibration_package = _calibration_path()
    del calibration_version, calibration_package
    spnet_source_root, spnet_source_commit, spnet_source_tree_sha256 = verify_spnet_source_identity()

    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "package_path": str(package_path),
        "sys_path": [str(entry) for entry in sys.path],
        "user_site_path": str(user_site),
        "git_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": str(dirty),
        "code_identity": repository_code_identity(),
        "user_site_enabled": str(bool(site.ENABLE_USER_SITE)),
        "torchvision_path": str(_module_path("torchvision")),
        "torchmetrics_path": str(_module_path("torchmetrics")),
        "opencv_path": str(_module_path("cv2")),
        "opencv": __import__("cv2").__version__,
        "renderer_package_path": str(renderer_package_path),
        "renderer_extension_path": str(renderer_extension_path),
        "renderer_identity": renderer_identity,
        "spnet_source_path": str((spnet_source_root / "src" / "networks.py").resolve()),
        "spnet_identity": {
            "source_id": SPNET_SOURCE_ID,
            "source_commit": spnet_source_commit,
            "source_tree_sha256": spnet_source_tree_sha256,
            "model_id": SPNET_MODEL_ID,
        },
        "metric_calibration_path": str(calibration_path),
        "model_paths": model_paths,
        "model_identities": model_identities,
        "conda_lock_path": str((ROOT / "conda-lock.yml").resolve()),
        "conda_lock_sha256": sha256_file(ROOT / "conda-lock.yml"),
    }
    from .foundation.runtime_identity import runtime_identity_from_report
    report["runtime_identity"] = runtime_identity_from_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the standalone SAGE v1 install identity.")
    parser.add_argument("--require-models", action="store_true", help="resolve both declared SPNet and AlexNet models")
    arguments = parser.parse_args()
    for key, value in verify(require_models=arguments.require_models).items():
        encoded = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        print(f"{key}={encoded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
