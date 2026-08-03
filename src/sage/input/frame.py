"""Canonical, host-side frame contract shared by every SAGE input adapter.

`FrameInputs` is the single seam between an input adapter and SAGE Core. It is
device-independent: adapters produce NumPy arrays only, and Core converts to
torch near the model call. Arrays are copied and marked read-only on
construction so an adapter, a writer and a Core consumer can hold the same
frame without one of them mutating it under the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _frozen(array: np.ndarray, dtype: np.dtype | type, name: str) -> np.ndarray:
    value = np.array(array, dtype=dtype, copy=True)
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    value.setflags(write=False)
    return value


def _frozen_bool(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array")
    value = value.copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class DepthObservation:
    """One depth source's observation of a canonical frame, in camera z-metres."""

    depth_m: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray | None = None
    support_count: np.ndarray | None = None
    temporal_age_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        depth = _frozen(self.depth_m, np.float32, "DepthObservation depth_m")
        if depth.ndim != 2:
            raise ValueError("DepthObservation depth_m must be a 2-D array")
        valid = _frozen_bool(self.valid_mask, "DepthObservation valid_mask")
        if valid.shape != depth.shape:
            raise ValueError("DepthObservation valid_mask must match depth_m")
        if np.any(valid & (depth <= 0)):
            raise ValueError("DepthObservation valid pixels must have positive depth")
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "valid_mask", valid)
        for name, dtype, lower, upper in (
            ("confidence", np.float32, 0.0, 1.0),
            ("support_count", np.uint16, None, None),
            ("temporal_age_s", np.float32, 0.0, None),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            frozen = _frozen(value, dtype, f"DepthObservation {name}")
            if frozen.shape != depth.shape:
                raise ValueError(f"DepthObservation {name} must match depth_m")
            if lower is not None and (frozen < lower).any():
                raise ValueError(f"DepthObservation {name} must be at least {lower}")
            if upper is not None and (frozen > upper).any():
                raise ValueError(f"DepthObservation {name} must be at most {upper}")
            if np.any(~valid & (frozen != 0)):
                raise ValueError(f"DepthObservation {name} must be zero outside valid_mask")
            object.__setattr__(self, name, frozen)

    @property
    def valid_pixels(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def coverage(self) -> float:
        return float(self.valid_mask.mean())


@dataclass(frozen=True)
class FrameMetadata:
    """Provenance that Core may log but must never branch on."""

    source_timestamp_ns: Mapping[str, int]
    frame_names: Mapping[str, str]
    diagnostics: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        for name, value_types in (
            ("source_timestamp_ns", (int,)),
            ("frame_names", (str,)),
            ("diagnostics", (float, int, str)),
        ):
            payload = getattr(self, name)
            if not isinstance(payload, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, value_types)
                or isinstance(value, bool)
                for key, value in payload.items()
            ):
                raise ValueError(f"FrameMetadata {name} must map names to {value_types}")
            object.__setattr__(self, name, MappingProxyType(dict(payload)))


@dataclass(frozen=True)
class FrameInputs:
    """One accepted canonical frame; the only thing SAGE Core consumes."""

    frame_index: int
    timestamp_ns: int
    image_rgb: np.ndarray
    intrinsics: np.ndarray
    reference_from_camera: np.ndarray
    lidar_center: DepthObservation | None
    lidar_fused: DepthObservation | None
    metadata: FrameMetadata

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("FrameInputs frame_index must be a non-negative integer")
        if type(self.timestamp_ns) is not int:
            raise ValueError("FrameInputs timestamp_ns must be an integer")
        image = _frozen(self.image_rgb, np.float32, "FrameInputs image_rgb")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("FrameInputs image_rgb must be HxWx3")
        if ((image < 0) | (image > 1)).any():
            raise ValueError("FrameInputs image_rgb must be within [0, 1]")
        intrinsics = _frozen(self.intrinsics, np.float64, "FrameInputs intrinsics")
        if intrinsics.shape != (3, 3):
            raise ValueError("FrameInputs intrinsics must be a 3x3 matrix")
        if (
            intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or not np.array_equal(intrinsics[1:, 0], [0.0, 0.0])
            or intrinsics[0, 1] != 0.0
            or intrinsics[2, 1] != 0.0
            or intrinsics[2, 2] != 1.0
        ):
            raise ValueError("FrameInputs intrinsics must be a skew-free pinhole matrix")
        pose = _frozen(self.reference_from_camera, np.float64, "FrameInputs reference_from_camera")
        if pose.shape != (4, 4) or not np.array_equal(pose[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("FrameInputs reference_from_camera must be a 4x4 homogeneous matrix")
        rotation = pose[:3, :3]
        # Same tolerance as the calibration itself: the pose inherits whatever
        # precision the published extrinsics were quoted at.
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-4,
        ):
            raise ValueError("FrameInputs reference_from_camera rotation must be in SO(3)")
        object.__setattr__(self, "image_rgb", image)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "reference_from_camera", pose)
        if not isinstance(self.metadata, FrameMetadata):
            raise ValueError("FrameInputs metadata must be a FrameMetadata")
        shape = image.shape[:2]
        for name in ("lidar_center", "lidar_fused"):
            observation = getattr(self, name)
            if observation is None:
                continue
            if not isinstance(observation, DepthObservation):
                raise ValueError(f"FrameInputs {name} must be a DepthObservation")
            if observation.depth_m.shape != shape:
                raise ValueError(f"FrameInputs {name} must match the image grid")
        if self.lidar_center is not None and self.lidar_fused is not None and np.any(
            self.lidar_center.valid_mask & self.lidar_fused.valid_mask
        ):
            raise ValueError("Fused observation must be invalid wherever center is valid")

    @property
    def stem(self) -> str:
        """Stable per-frame asset name; also the Prepared Scene file stem."""
        return f"{self.frame_index:06d}"

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height) of the canonical image grid."""
        return int(self.image_rgb.shape[1]), int(self.image_rgb.shape[0])

    def observation(self, source_name: str) -> DepthObservation | None:
        if source_name == "LIDAR_CENTER":
            return self.lidar_center
        if source_name == "LIDAR_FUSED":
            return self.lidar_fused
        raise ValueError(f"FrameInputs has no observation named {source_name!r}")


__all__ = ["DepthObservation", "FrameInputs", "FrameMetadata"]
