"""Small builders for the Core-facing frame Core actually optimizes.

Tests exercise mapping, growth and rendering, which consume `MappingFrame`.
Building one directly keeps a test free to inject evidence combinations no
adapter would emit (SPNet evidence, empty growth), while still going through
the same canonical frame the real pipeline delivers.
"""

from __future__ import annotations

import numpy as np

from sage.core_input import MappingFrame, camera_intrinsics, camera_pose
from sage.foundation.contracts import (
    GrowthInputs,
    INVALID_SOURCE_TYPE,
    MappingObservation,
)
from sage.input.frame import DepthObservation, FrameInputs, FrameMetadata


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def pose_matrix(translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = translation
    return matrix


def canonical_frame(
    *,
    frame_index: int = 0,
    timestamp_ns: int = 0,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    reference_from_camera: np.ndarray | None = None,
    center: DepthObservation | None = None,
    fused: DepthObservation | None = None,
) -> FrameInputs:
    height, width = rgb.shape[:2]
    if center is None:
        depth = np.zeros((height, width), dtype=np.float32)
        center = DepthObservation(depth, depth > 0)
    return FrameInputs(
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        image_rgb=rgb,
        intrinsics=intrinsics,
        reference_from_camera=(
            pose_matrix() if reference_from_camera is None else reference_from_camera
        ),
        lidar_center=center,
        lidar_fused=fused,
        metadata=FrameMetadata({}, {}, {}),
    )


def mapping_frame(
    *,
    frame_index: int = 0,
    timestamp_ns: int = 0,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    reference_from_camera: np.ndarray | None = None,
    mapping: MappingObservation | None = None,
    growth: GrowthInputs | None = None,
) -> MappingFrame:
    canonical = canonical_frame(
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        rgb=rgb,
        intrinsics=intrinsics,
        reference_from_camera=reference_from_camera,
    )
    height, width = rgb.shape[:2]
    if mapping is None:
        mapping = MappingObservation(
            np.zeros((height, width), dtype=np.float32),
            np.full((height, width), INVALID_SOURCE_TYPE, dtype=np.uint8),
            np.zeros((height, width), dtype=np.float32),
        )
    return MappingFrame(
        canonical=canonical,
        mapping=mapping,
        growth=GrowthInputs(()) if growth is None else growth,
        intrinsics=camera_intrinsics(canonical),
        pose=camera_pose(canonical),
        rgb=np.array(canonical.image_rgb, dtype=np.float32),
    )


__all__ = ["canonical_frame", "intrinsics_matrix", "mapping_frame", "pose_matrix"]
