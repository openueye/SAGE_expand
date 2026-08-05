"""Mandatory per-frame visualization export for evaluation runs.

Every evaluated frame gets four PNGs under
``<root>/render/{GT,RenderRGB,depth,GT-RenderRGB_diff}/{keyframe,non_keyframe}/``,
built from the RGB/depth tensors evaluation already rendered - no extra
render pass, and no effect on the evaluation metrics themselves.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..core_input import MappingFrame
from .geometry import is_mapping_frame
from .rendering import RenderOutput


RENDER_KINDS = ("GT", "RenderRGB", "depth", "GT-RenderRGB_diff")
_KEYFRAME_BUCKETS = ("keyframe", "non_keyframe")


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    clamped = np.clip(image, 0.0, 1.0)
    return np.rint(clamped * 255.0).astype(np.uint8)


def _colorize(scalar: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    """Turbo-colormap a scalar HxW map into an HxWx3 RGB uint8 image."""
    span = vmax - vmin
    normalized = (scalar - vmin) / span if span > 0 else np.zeros_like(scalar)
    gray = np.clip(np.rint(np.clip(normalized, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return bgr[..., ::-1]


class EvaluationRenderExporter:
    """Writes the mandatory render/ tree; one instance per evaluation output."""

    def __init__(self, root: Path, *, keyframe_every: int) -> None:
        if type(keyframe_every) is not int or keyframe_every < 1:
            raise ValueError("EvaluationRenderExporter requires a positive keyframe_every")
        self._root = Path(root) / "render"
        self._keyframe_every = keyframe_every
        for kind in RENDER_KINDS:
            for bucket in _KEYFRAME_BUCKETS:
                (self._root / kind / bucket).mkdir(parents=True, exist_ok=True)

    def export(self, frame: MappingFrame, output: RenderOutput) -> None:
        frame_id = frame.stem
        bucket = (
            "keyframe"
            if is_mapping_frame(frame.index, map_every=self._keyframe_every)
            else "non_keyframe"
        )
        gt_rgb = np.asarray(frame.rgb, dtype=np.float32)
        rendered_rgb = output.rgb.detach().to("cpu").numpy().astype(np.float32)
        depth = output.depth.detach().to("cpu").numpy().astype(np.float32)
        diff = np.abs(gt_rgb - rendered_rgb).mean(axis=-1)
        finite_depth = depth[np.isfinite(depth) & (depth > 0)]
        depth_vmin, depth_vmax = (
            (float(finite_depth.min()), float(finite_depth.max()))
            if finite_depth.size
            else (0.0, 1.0)
        )
        payloads = {
            "GT": _to_uint8_rgb(gt_rgb),
            "RenderRGB": _to_uint8_rgb(rendered_rgb),
            "depth": _colorize(depth, vmin=depth_vmin, vmax=depth_vmax),
            "GT-RenderRGB_diff": _colorize(diff, vmin=0.0, vmax=1.0),
        }
        for kind, array in payloads.items():
            path = self._root / kind / bucket / f"{frame_id}.png"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array, mode="RGB").save(path, format="PNG")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to save {kind} visualization for frame {frame_id!r} to {path}"
                ) from exc


__all__ = ["RENDER_KINDS", "EvaluationRenderExporter"]
