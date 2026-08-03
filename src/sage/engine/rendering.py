from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import sysconfig
from typing import Protocol

import gsplat
import torch
from torch.nn import functional as F

from ..core_input import MappingFrame
from .geometry import rotation_wc
from ..foundation.hashing import sha256_file
from .model import TrainableGaussians


RENDERER_PACKAGE = "gsplat"
RENDERER_PACKAGE_VERSION = "1.5.3"
RENDERER_ADAPTER_SCHEMA = "sage-renderer-adapter-v2"
RENDERER_NEAR_PLANE = 0.01
RENDERER_FAR_PLANE = 100.0


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


@dataclass(frozen=True)
class RenderStaticFields:
    """Geometry-only renderer inputs reused during appearance optimization."""

    backend: object
    means3d: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    background: torch.Tensor


@dataclass(frozen=True)
class RenderCameraFields:
    """Per-frame camera tensors reused across optimization renders."""

    view: torch.Tensor
    intrinsics: torch.Tensor


class Renderer(Protocol):
    def __call__(self, model: object, frame: MappingFrame) -> RenderOutput: ...


def _gsplat_extension_path() -> Path:
    from gsplat.cuda._backend import _C as extension

    return Path(extension.__file__).resolve()


def _gsplat_package_path() -> Path:
    package_file = getattr(gsplat, "__file__", None)
    if not isinstance(package_file, str):
        raise RuntimeError("gsplat package path is unavailable")
    package_path = Path(package_file).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("gsplat must load from the active Conda environment")
    if str(gsplat.__version__) != RENDERER_PACKAGE_VERSION:
        raise RuntimeError(
            "gsplat version does not match the locked renderer version: "
            f"expected {RENDERER_PACKAGE_VERSION}, got {gsplat.__version__}"
        )
    return package_path


def capture_renderer_identity() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Renderer identity requires an available CUDA device")
    _gsplat_package_path()
    major, minor = torch.cuda.get_device_capability()
    return {
        "kind": "pip-package",
        "package": RENDERER_PACKAGE,
        "package_version": RENDERER_PACKAGE_VERSION,
        "extension_sha256": sha256_file(_gsplat_extension_path()),
        "adapter_schema": RENDERER_ADAPTER_SCHEMA,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "compute_capability": f"{major}.{minor}",
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }


def _camera_matrices(
    frame: MappingFrame, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the world-to-camera transform and pinhole intrinsics gsplat expects."""
    rotation_cw = rotation_wc(frame.pose, device=device).T
    center = torch.tensor(frame.pose.translation, dtype=torch.float32, device=device)
    view = torch.eye(4, dtype=torch.float32, device=device)
    view[:3, :3] = rotation_cw
    view[:3, 3] = -(rotation_cw @ center)
    intrinsics = torch.tensor([
        [frame.intrinsics.fx, 0.0, frame.intrinsics.cx],
        [0.0, frame.intrinsics.fy, frame.intrinsics.cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    return view.contiguous(), intrinsics


def prepare_render_camera(
    frame: MappingFrame,
    device: torch.device | str,
) -> RenderCameraFields:
    view, intrinsics = _camera_matrices(frame, torch.device(device))
    return RenderCameraFields(view, intrinsics)


def prepare_render_static(model: TrainableGaussians) -> RenderStaticFields:
    """Prepare fields that stay frozen throughout appearance refinement."""
    if model.means3d.device.type != "cuda":
        raise RuntimeError(
            "gsplat renderer requires CUDA; CPU rendering is not supported"
        )
    with torch.no_grad():
        means3d = model.means3d.detach().contiguous()
        scales = model.scales.detach().contiguous()
        rotations = F.normalize(
            model.rotations.detach(),
            dim=1,
        ).contiguous()
    return RenderStaticFields(
        backend=gsplat,
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        background=torch.zeros(3, dtype=torch.float32, device=model.means3d.device),
    )


def render(
    model: TrainableGaussians,
    frame: MappingFrame,
    *,
    static: RenderStaticFields | None = None,
    camera: RenderCameraFields | None = None,
) -> RenderOutput:
    if model.means3d.device.type != "cuda":
        raise RuntimeError("gsplat renderer requires CUDA; CPU rendering is not supported")
    device = model.means3d.device
    if static is None:
        backend = gsplat
        means3d = model.means3d.contiguous()
        scales = model.scales.contiguous()
        rotations = F.normalize(model.rotations, dim=1).contiguous()
        background = torch.zeros(3, dtype=torch.float32, device=device)
    else:
        if static.means3d.device != device:
            raise ValueError(
                "Static renderer fields must share the model device"
            )
        backend = static.backend
        means3d = static.means3d
        scales = static.scales
        rotations = static.rotations
        background = static.background
    if camera is None:
        camera = prepare_render_camera(frame, device)
    elif camera.view.device != device or camera.intrinsics.device != device:
        raise ValueError("Cached renderer camera must share the model device")
    view, intrinsics = camera.view, camera.intrinsics
    colors, alphas, _meta = backend.rasterization(
        means=means3d, quats=rotations, scales=scales,
        opacities=model.opacities.squeeze(-1).contiguous(), colors=model.colors.contiguous(),
        viewmats=view[None], Ks=intrinsics[None],
        width=frame.intrinsics.width, height=frame.intrinsics.height,
        near_plane=RENDERER_NEAR_PLANE, far_plane=RENDERER_FAR_PLANE,
        sh_degree=None, render_mode="RGB+D",
    )
    expected_colors = (1, frame.intrinsics.height, frame.intrinsics.width, 4)
    expected_alphas = (1, frame.intrinsics.height, frame.intrinsics.width, 1)
    if colors.shape != expected_colors or alphas.shape != expected_alphas:
        raise RuntimeError(
            "gsplat renderer returned invalid RGB+D/alpha shapes: "
            f"colors={tuple(colors.shape)}, alphas={tuple(alphas.shape)}"
        )
    alpha = alphas[0, ..., 0].clamp(0, 1)
    rgb = colors[0, ..., :3] + (1 - alpha[..., None]) * background
    return RenderOutput(rgb, colors[0, ..., 3], alpha)


class CachedRenderer:
    """Renderer adapter that owns immutable camera tensors for one scene run."""

    def __init__(self) -> None:
        self._cameras: dict[tuple[object, ...], RenderCameraFields] = {}

    def __call__(
        self,
        model: TrainableGaussians,
        frame: MappingFrame,
        *,
        static: RenderStaticFields | None = None,
    ) -> RenderOutput:
        device = model.means3d.device
        key = (
            str(device),
            frame.index,
            frame.timestamp_ns,
            frame.intrinsics,
            frame.pose,
        )
        camera = self._cameras.get(key)
        if camera is None:
            camera = prepare_render_camera(frame, device)
            self._cameras[key] = camera
        return render(model, frame, static=static, camera=camera)
