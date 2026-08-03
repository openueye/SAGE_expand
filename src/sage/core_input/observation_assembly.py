"""The one place in SAGE Core that merges canonical observations.

Adapters deliver `lidar_center` and `lidar_fused` separately so source-aware
growth keeps its provenance. Mapping still needs a single depth target, so the
priority rule lives here — once — and mapping, refinement and evaluation all
call it instead of each re-deriving "center wins, fused fills holes".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..foundation.contracts import (
    CameraIntrinsics,
    DepthEvidence,
    GrowthInputs,
    INVALID_SOURCE_TYPE,
    MappingObservation,
    Pose,
    SourceType,
)
from ..foundation.source_policy import descriptor_for_type
from ..input.frame import DepthObservation, FrameInputs


_SOURCE_BY_NAME = {
    "LIDAR_CENTER": SourceType.LIDAR_CENTER,
    "LIDAR_FUSED": SourceType.LIDAR_FUSED,
}


def camera_intrinsics(frame: FrameInputs) -> CameraIntrinsics:
    """Core's view descriptor for a canonical frame's pinhole camera."""
    width, height = frame.image_size
    matrix = frame.intrinsics
    return CameraIntrinsics(
        width, height,
        float(matrix[0, 0]), float(matrix[1, 1]),
        float(matrix[0, 2]), float(matrix[1, 2]),
    )


def camera_pose(frame: FrameInputs) -> Pose:
    """`reference_from_camera` as the translation+quaternion Core renders with."""
    from ..input.rosbag.transforms import matrix_to_quaternion_xyzw

    matrix = frame.reference_from_camera
    quaternion = matrix_to_quaternion_xyzw(matrix[:3, :3])
    return Pose(
        float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]),
        float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]),
    )


def _confidence(observation: DepthObservation, source_type: SourceType) -> np.ndarray:
    if observation.confidence is not None:
        return np.asarray(observation.confidence, dtype=np.float32)
    default = descriptor_for_type(source_type).default_confidence
    return np.where(observation.valid_mask, default, 0.0).astype(np.float32)


@dataclass(frozen=True)
class AssembledObservations:
    mapping: MappingObservation
    growth: GrowthInputs


class CoreObservationAssembler:
    """Assemble canonical observations into Core's mapping target and growth evidence."""

    def __init__(self, enabled_sources: tuple[str, ...]) -> None:
        names = tuple(enabled_sources)
        if not names or names[0] != "LIDAR_CENTER" or any(
            name not in _SOURCE_BY_NAME for name in names
        ) or len(set(names)) != len(names):
            raise ValueError(
                "CoreObservationAssembler requires LIDAR_CENTER first and only canonical LiDAR sources"
            )
        self._enabled = names

    @property
    def enabled_sources(self) -> tuple[str, ...]:
        return self._enabled

    def assemble(self, frame: FrameInputs) -> AssembledObservations:
        height, width = frame.image_rgb.shape[:2]
        depth = np.zeros((height, width), dtype=np.float32)
        sources = np.full((height, width), INVALID_SOURCE_TYPE, dtype=np.uint8)
        confidences = np.zeros((height, width), dtype=np.float32)
        claimed = np.zeros((height, width), dtype=bool)
        evidences: list[DepthEvidence] = []
        for name in self._enabled:
            observation = frame.observation(name)
            source_type = _SOURCE_BY_NAME[name]
            if observation is None:
                raise ValueError(f"Canonical frame {frame.stem} is missing enabled source {name}")
            confidence = _confidence(observation, source_type)
            evidences.append(DepthEvidence(
                source_type,
                np.asarray(observation.depth_m, dtype=np.float32),
                np.asarray(observation.valid_mask, dtype=bool),
                confidence,
            ))
            # Higher-priority sources come first and keep the pixel; the canonical
            # frame already guarantees fused is invalid wherever center is valid,
            # so this only guards a source the adapter left overlapping.
            take = observation.valid_mask & ~claimed
            depth[take] = observation.depth_m[take]
            sources[take] = int(source_type)
            confidences[take] = confidence[take]
            claimed |= take
        return AssembledObservations(
            mapping=MappingObservation(depth, sources, confidences),
            growth=GrowthInputs(tuple(evidences)),
        )


__all__ = [
    "AssembledObservations",
    "CoreObservationAssembler",
    "camera_intrinsics",
    "camera_pose",
]
