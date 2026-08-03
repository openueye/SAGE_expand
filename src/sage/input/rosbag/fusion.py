"""Build the two canonical LiDAR observations for one accepted frame.

`LIDAR_CENTER` is the frame's own associated scan projected into its own
camera. `LIDAR_FUSED` is the rest of the window filling what center could not
see: it is deliberately invalid wherever center is valid, so the two
observations never disagree about the same pixel and growth keeps a clean
provenance split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frame import DepthObservation
from .calibration import Calibration, OutputCamera
from .decoder import decode_points_xyz, parse_pointcloud2
from .projection import project_to_depth
from .reader import RosbagReader
from .spec import RosbagInputSpec
from .synchronizer import AssociatedFrame
from .transforms import transform_points


@dataclass(frozen=True)
class FrameObservations:
    center: DepthObservation
    fused: DepthObservation | None
    statistics: dict[str, int]


def _reference_points(
    reader: RosbagReader, frame: AssociatedFrame, spec: RosbagInputSpec,
) -> np.ndarray:
    """Points of a frame's scan, expressed in the reference frame."""
    message = parse_pointcloud2(reader.payload(frame.lidar))
    points = decode_points_xyz(message)
    if spec.points_frame == "reference_frame":
        # The producer already registered this scan into the reference frame;
        # applying the pose chain again would double-transform it.
        return points.astype(np.float64, copy=False)
    assert frame.reference_from_lidar is not None
    return transform_points(frame.reference_from_lidar, points)


def build_observations(
    reader: RosbagReader,
    spec: RosbagInputSpec,
    calibration: Calibration,
    frame: AssociatedFrame,
    window: tuple[AssociatedFrame, ...],
    *,
    camera: OutputCamera,
) -> FrameObservations:
    camera_from_reference = np.linalg.inv(frame.reference_from_camera)
    depth_range = (spec.depth.min_depth_m, spec.depth.max_depth_m)
    projected = []
    ages_s = []
    statistics = {"points": 0, "positive_depth_points": 0, "in_image_points": 0}
    members = window if spec.enable_fused else (frame,)
    for member in members:
        depth, member_statistics = project_to_depth(
            _reference_points(reader, member, spec),
            camera_from_reference=camera_from_reference,
            camera=camera,
            min_depth_m=depth_range[0],
            max_depth_m=depth_range[1],
        )
        projected.append(depth)
        ages_s.append(abs(member.lidar.effective_timestamp_ns - frame.timestamp_ns) / 1e9)
        if member.candidate_index == frame.candidate_index:
            statistics = {key: int(value) for key, value in member_statistics.items()}
    stack = np.stack(projected)
    center_position = next(
        position for position, member in enumerate(members)
        if member.candidate_index == frame.candidate_index
    )
    center_depth = stack[center_position].copy()
    center_valid = center_depth > 0
    center = DepthObservation(center_depth, center_valid)
    statistics["center_valid_pixels"] = int(center_valid.sum())
    if not spec.enable_fused:
        statistics["fused_valid_pixels"] = 0
        return FrameObservations(center, None, statistics)

    valid = stack > 0
    support = valid.sum(axis=0).astype(np.uint16)
    masked = np.where(valid, stack, np.inf)
    nearest = np.min(masked, axis=0)
    farthest = np.max(np.where(valid, stack, -np.inf), axis=0)
    conflicts = (support > 1) & ((farthest - nearest) > spec.fusion.conflict_threshold_m)
    fused_depth = np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)
    fused_depth[conflicts] = 0.0
    fused_valid = (fused_depth > 0) & ~center_valid
    fused_depth = np.where(fused_valid, fused_depth, 0.0).astype(np.float32)
    winner = np.argmin(masked, axis=0)
    age = np.asarray(ages_s, dtype=np.float32)[winner]
    fused = DepthObservation(
        fused_depth,
        fused_valid,
        support_count=np.where(fused_valid, support, 0).astype(np.uint16),
        temporal_age_s=np.where(fused_valid, age, 0.0).astype(np.float32),
    )
    statistics["fused_valid_pixels"] = int(fused_valid.sum())
    statistics["conflict_pixels"] = int(conflicts.sum())
    return FrameObservations(center, fused, statistics)


__all__ = ["FrameObservations", "build_observations"]
