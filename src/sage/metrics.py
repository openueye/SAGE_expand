from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
from threading import Lock
import warnings

import torch
from torch.nn import functional as F

from .hashing import sha256_file
from .losses import photometric_loss
from .model_registry import ALEXNET_MODEL_ID, ModelRegistry, default_registry_path, MODEL_ROOT_ENVIRONMENT_VARIABLE


EVALUATOR_SCHEMA_VERSION = "sage-image-metrics-v1"
TORCHMETRICS_VERSION = "1.9.0"
TORCHVISION_VERSION_PREFIX = "0.20.1"
LPIPS_CALIBRATION_RELATIVE_PATH = "functional/image/lpips_models/alex.pth"
LPIPS_CALIBRATION_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
_LPIPS_BUILD_LOCK = Lock()
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*weights_only=False.*",
)


@contextmanager
def _force_torch_load_weights_only() -> object:
    original_torch_load = torch.load

    def load_with_weights_only(*args, **kwargs):
        kwargs.setdefault("weights_only", True)
        return original_torch_load(*args, **kwargs)

    torch.load = load_with_weights_only
    try:
        yield
    finally:
        torch.load = original_torch_load


@dataclass(frozen=True)
class ImageQualityMetrics:
    psnr: float
    ssim: float
    lpips: float


@dataclass(frozen=True)
class ImageMetricIdentity:
    model_id: str
    backbone_sha256: str
    torchmetrics_version: str
    torchvision_version: str
    calibration_sha256: str
    calibration_package: str
    evaluator_schema: str
    device: str
    dtype: str

    def dependency_payload(self) -> dict[str, object]:
        return {
            "kind": "repository-offline",
            "model_id": self.model_id,
            "weights_sha256": self.backbone_sha256,
            "evaluator_schema": self.evaluator_schema,
            "torchmetrics_version": self.torchmetrics_version,
            "torchvision_version": self.torchvision_version,
            "calibration_sha256": self.calibration_sha256,
            "calibration_package": self.calibration_package,
            "device": self.device,
            "dtype": self.dtype,
        }


def psnr_ssim(rendered: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if rendered.shape != target.shape or rendered.ndim != 3 or rendered.shape[-1] != 3:
        raise ValueError("RGB tensors must share HxWx3 shape")
    rendered = rendered.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(rendered, target)
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    _, _, ssim = photometric_loss(rendered, target)
    return psnr, ssim


def _calibration_path() -> tuple[Path, str, str]:
    torchmetrics = importlib.import_module("torchmetrics")
    version = str(torchmetrics.__version__)
    if version != TORCHMETRICS_VERSION:
        raise ValueError(f"Image metrics require torchmetrics=={TORCHMETRICS_VERSION}, got {version}")
    package_path = Path(torchmetrics.__file__).resolve().parent
    calibration = package_path / "functional" / "image" / "lpips_models" / "alex.pth"
    if not calibration.is_file():
        raise ValueError(f"TorchMetrics LPIPS Alex calibration is missing: {calibration}")
    actual = sha256_file(calibration)
    if actual != LPIPS_CALIBRATION_SHA256:
        raise ValueError(
            "TorchMetrics LPIPS Alex calibration SHA-256 mismatch: "
            f"expected {LPIPS_CALIBRATION_SHA256}, got {actual}"
        )
    return calibration, version, f"torchmetrics=={version}:{LPIPS_CALIBRATION_RELATIVE_PATH}"


def _load_alexnet_features(path: Path) -> torch.nn.Sequential:
    try:
        import torchvision

        model = torchvision.models.alexnet(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("AlexNet backbone checkpoint must contain a state dictionary")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"AlexNet backbone is structurally incompatible: {path}") from exc
    return model.features


def _build_lpips(features: torch.nn.Sequential) -> torch.nn.Module:
    lpips_module = importlib.import_module("torchmetrics.functional.image.lpips")
    metric_type = getattr(importlib.import_module("torchmetrics.image.lpip"), "LearnedPerceptualImagePatchSimilarity")
    original_resolver = lpips_module._get_tv_model_features

    def explicit_features(net: str, pretrained: bool = False) -> torch.nn.Sequential:
        if net != "alexnet" or not pretrained:
            raise ValueError(f"Identity-bound LPIPS only supports pretrained local AlexNet, got {net}")
        return features

    with _LPIPS_BUILD_LOCK:
        lpips_module._get_tv_model_features = explicit_features
        try:
            with _force_torch_load_weights_only():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message=".*weights_only=False.*",
                    )
                    return metric_type(net_type="alex", normalize=True).eval().requires_grad_(False)
        finally:
            lpips_module._get_tv_model_features = original_resolver


class ImageMetricEvaluator:
    """Compute render/target diagnostics with explicit offline model identity."""

    def __init__(
        self,
        device: str | torch.device,
        *,
        model_root: Path | None = None,
        registry_path: Path | None = None,
        model_id: str = ALEXNET_MODEL_ID,
    ) -> None:
        self.device = torch.device(device)
        if model_root is None:
            configured_root = os.environ.get(MODEL_ROOT_ENVIRONMENT_VARIABLE)
            if not configured_root:
                raise ValueError(
                    "Image metrics require an explicit model root; set "
                    f"{MODEL_ROOT_ENVIRONMENT_VARIABLE} or pass model_root"
                )
            model_root = Path(configured_root)
        registry = ModelRegistry.load(registry_path or default_registry_path())
        record = registry.require(model_id)
        backbone = registry.resolve(model_id, model_root=Path(model_root))
        calibration, torchmetrics_version, calibration_package = _calibration_path()
        try:
            import torchvision

            torchvision_version = str(torchvision.__version__)
        except ImportError as exc:
            raise ValueError("Image metrics require torchvision") from exc
        if not torchvision_version.startswith(TORCHVISION_VERSION_PREFIX):
            raise ValueError(
                f"Image metrics require torchvision {TORCHVISION_VERSION_PREFIX}.*, got {torchvision_version}"
            )
        self.identity = ImageMetricIdentity(
            model_id=record.model_id,
            backbone_sha256=record.sha256,
            torchmetrics_version=torchmetrics_version,
            torchvision_version=torchvision_version,
            calibration_sha256=LPIPS_CALIBRATION_SHA256,
            calibration_package=calibration_package,
            evaluator_schema=EVALUATOR_SCHEMA_VERSION,
            device=str(self.device),
            dtype="float32",
        )
        self.lpips = _build_lpips(_load_alexnet_features(backbone)).to(self.device)

    @torch.inference_mode()
    def __call__(self, rendered: torch.Tensor, target: torch.Tensor) -> ImageQualityMetrics:
        rendered = rendered.to(device=self.device, dtype=torch.float32).clamp(0, 1)
        target = target.to(device=self.device, dtype=torch.float32).clamp(0, 1)
        psnr, ssim = psnr_ssim(rendered, target)
        lpips = self.lpips(
            rendered.permute(2, 0, 1).unsqueeze(0),
            target.permute(2, 0, 1).unsqueeze(0),
        )
        return ImageQualityMetrics(float(psnr), float(ssim), float(lpips))
