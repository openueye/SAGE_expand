from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .poses import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix


def _rigid_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} must have a homogeneous last row")
    return matrix


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    left = q0 / np.linalg.norm(q0)
    right = q1 / np.linalg.norm(q1)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (np.sin((1.0 - alpha) * angle) / scale) * left + (np.sin(alpha * angle) / scale) * right


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    world_from_base: np.ndarray

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int:
            raise ValueError("Pose timestamp_ns must be an integer")
        object.__setattr__(self, "world_from_base", _rigid_matrix(self.world_from_base, "world_from_base"))


class PoseTrack:
    """Strict SE(3) interpolation with no extrapolation or endpoint clamping."""

    def __init__(self, samples: tuple[PoseSample, ...]) -> None:
        if len(samples) < 2:
            raise ValueError("PoseTrack requires at least two samples")
        timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
        if np.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("PoseTrack timestamps must be strictly increasing")
        self._samples = samples
        self._timestamps = timestamps

    @property
    def coverage_ns(self) -> tuple[int, int]:
        return int(self._timestamps[0]), int(self._timestamps[-1])

    def bracket(self, timestamp_ns: int, *, max_distance_ns: int | None = None) -> tuple[int, int, float]:
        timestamp = int(timestamp_ns)
        if timestamp < self._timestamps[0] or timestamp > self._timestamps[-1]:
            raise ValueError(f"Timestamp {timestamp} is outside odometry coverage {self.coverage_ns}")
        right = int(np.searchsorted(self._timestamps, timestamp, side="left"))
        if right < len(self._timestamps) and int(self._timestamps[right]) == timestamp:
            left = right
            alpha = 0.0
        else:
            left = right - 1
            span = int(self._timestamps[right] - self._timestamps[left])
            alpha = (timestamp - int(self._timestamps[left])) / span
        if max_distance_ns is not None:
            distances = (
                abs(timestamp - int(self._timestamps[left])),
                abs(int(self._timestamps[right]) - timestamp),
            )
            span = int(self._timestamps[right] - self._timestamps[left])
            if min(distances) > int(max_distance_ns) or span > 2 * int(max_distance_ns):
                raise ValueError(f"Timestamp {timestamp} has no odometry bracket within {max_distance_ns} ns")
        return left, right, float(alpha)

    def interpolate(self, timestamp_ns: int, *, max_distance_ns: int | None = None) -> np.ndarray:
        left, right, alpha = self.bracket(timestamp_ns, max_distance_ns=max_distance_ns)
        if left == right:
            return self._samples[left].world_from_base.copy()
        first = self._samples[left].world_from_base
        second = self._samples[right].world_from_base
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
        q0 = matrix_to_quaternion_xyzw(first[:3, :3])
        q1 = matrix_to_quaternion_xyzw(second[:3, :3])
        matrix[:3, :3] = quaternion_xyzw_to_matrix(_slerp(q0, q1, alpha))
        return matrix

    def bracket_receipt(self, timestamp_ns: int, *, max_distance_ns: int | None = None) -> dict[str, object]:
        left, right, alpha = self.bracket(timestamp_ns, max_distance_ns=max_distance_ns)
        return {
            "query_timestamp_ns": int(timestamp_ns),
            "previous_timestamp_ns": int(self._timestamps[left]),
            "next_timestamp_ns": int(self._timestamps[right]),
            "alpha": alpha,
        }


def _xyz(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape Nx3, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud must contain only finite points")
    return points.astype(np.float64, copy=False)


class LidarWorldNormalizer:
    def normalize(self, points_world: np.ndarray, *, frame_id: str) -> np.ndarray:
        if frame_id != "odom":
            raise ValueError(f"LIDAR_WORLD requires frame_id=odom, got {frame_id!r}")
        return _xyz(points_world).astype(np.float32, copy=True)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 1:
            raise ValueError("Camera dimensions must be positive integers")
        if not np.isfinite(values).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("Camera intrinsics must be finite with positive focal lengths")


@dataclass(frozen=True)
class WorldCloudEvidence:
    image_index: int
    image_timestamp_ns: int
    cloud_id: str
    cloud_timestamp_ns: int
    world_xyz: np.ndarray
    center_source_type: str
    fused_source_type: str
    normalization_receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if (self.center_source_type, self.fused_source_type) != ("LIDAR_WORLD_CENTER", "LIDAR_WORLD_FUSED5"):
            raise ValueError("World cloud source types do not form a supported source family")
        if not isinstance(self.cloud_id, str) or not self.cloud_id:
            raise ValueError("World cloud id must be non-empty")
        if not isinstance(self.normalization_receipt, Mapping):
            raise ValueError("World cloud normalization receipt must be an object")
        object.__setattr__(self, "world_xyz", _xyz(self.world_xyz).astype(np.float32, copy=True))


@dataclass(frozen=True)
class FinalizedSensorFrame:
    image_index: int
    stem: str
    image_timestamp_ns: int
    rgb: np.ndarray
    intrinsics: CameraIntrinsics
    world_from_camera: np.ndarray
    cloud: WorldCloudEvidence

    def __post_init__(self) -> None:
        color = np.asarray(self.rgb)
        expected = (self.intrinsics.height, self.intrinsics.width, 3)
        if color.dtype != np.uint8 or color.shape != expected:
            raise ValueError(f"Rectified RGB must be uint8 with shape {expected}")
        if self.cloud.image_index != self.image_index or self.cloud.image_timestamp_ns != self.image_timestamp_ns:
            raise ValueError("World cloud evidence must belong to its image slot")
        object.__setattr__(self, "rgb", color.copy())
        object.__setattr__(self, "world_from_camera", _rigid_matrix(self.world_from_camera, "world_from_camera"))


@dataclass(frozen=True)
class RejectedSensorFrame:
    image_index: int
    stem: str
    image_timestamp_ns: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Rejected sensor frame reason must be non-empty")


@dataclass(frozen=True)
class ProjectedCenter:
    target: FinalizedSensorFrame
    source_frames: tuple[FinalizedSensorFrame, ...]
    center_depth: np.ndarray
    fused_depth: np.ndarray
    fused_mask: np.ndarray
    stats: Mapping[str, int]


@dataclass(frozen=True)
class RejectedCenter:
    target_image_index: int
    target_stem: str
    target_timestamp_ns: int
    dependency_indices: tuple[int, ...]
    root_reasons: tuple[str, ...]


class CenteredFiveProjector:
    def __init__(self, *, min_depth_m: float = 0.1, max_depth_m: float = 200.0, conflict_threshold_m: float = 1.0) -> None:
        values = np.asarray([min_depth_m, max_depth_m, conflict_threshold_m], dtype=np.float64)
        if not np.isfinite(values).all() or min_depth_m <= 0 or max_depth_m <= min_depth_m or conflict_threshold_m < 0:
            raise ValueError("Centered-five depth limits are invalid")
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
        self._conflict_threshold_m = float(conflict_threshold_m)

    @staticmethod
    def _project_depth(
        points_world: np.ndarray,
        *,
        camera_from_world: np.ndarray,
        intrinsics: CameraIntrinsics,
        min_depth_m: float,
        max_depth_m: float,
    ) -> np.ndarray:
        depth = np.zeros((intrinsics.height, intrinsics.width), dtype=np.dtype("<f4"))
        if len(points_world) == 0:
            return depth
        points_homogeneous = np.concatenate(
            [points_world.astype(np.float64, copy=False), np.ones((len(points_world), 1), dtype=np.float64)],
            axis=1,
        )
        points_camera = (camera_from_world @ points_homogeneous.T).T[:, :3]
        z = points_camera[:, 2]
        valid = np.isfinite(points_camera).all(axis=1) & (z >= min_depth_m) & (z <= max_depth_m)
        points_camera = points_camera[valid]
        z = z[valid]
        if not len(z):
            return depth
        u = np.rint(intrinsics.fx * points_camera[:, 0] / z + intrinsics.cx).astype(np.int64)
        v = np.rint(intrinsics.fy * points_camera[:, 1] / z + intrinsics.cy).astype(np.int64)
        inside = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
        u, v, z = u[inside], v[inside], z[inside].astype(np.float32)
        zbuffer = np.full(depth.shape, np.inf, dtype=np.float32)
        np.minimum.at(zbuffer, (v, u), z)
        finite = np.isfinite(zbuffer)
        depth[finite] = zbuffer[finite]
        return depth

    def project(self, frames: tuple[FinalizedSensorFrame, ...]) -> ProjectedCenter:
        if len(frames) != 5:
            raise ValueError("Centered-five projection requires exactly five finalized slots")
        indices = tuple(frame.image_index for frame in frames)
        if indices != tuple(range(indices[0], indices[0] + 5)):
            raise ValueError("Centered-five projection requires five consecutive image slots")
        families = {(frame.cloud.center_source_type, frame.cloud.fused_source_type) for frame in frames}
        if len(families) != 1:
            raise ValueError("Centered-five projection cannot mix source families")
        target = frames[2]
        if any(frame.intrinsics != target.intrinsics for frame in frames):
            raise ValueError("Centered-five source slots must share camera intrinsics")
        camera_from_world = np.linalg.inv(target.world_from_camera)
        source_depths = np.stack([
            self._project_depth(
                frame.cloud.world_xyz,
                camera_from_world=camera_from_world,
                intrinsics=target.intrinsics,
                min_depth_m=self._min_depth_m,
                max_depth_m=self._max_depth_m,
            )
            for frame in frames
        ])
        center = source_depths[2].astype(np.dtype("<f4"), copy=True)
        valid = source_depths > 0
        valid_count = valid.sum(axis=0)
        minimum = np.min(np.where(valid, source_depths, np.inf), axis=0)
        maximum = np.max(np.where(valid, source_depths, -np.inf), axis=0)
        conflicts = (valid_count > 1) & ((maximum - minimum) > self._conflict_threshold_m)
        fused = np.where(np.isfinite(minimum), minimum, 0.0).astype(np.dtype("<f4"))
        fused[conflicts] = 0.0
        center_valid = center > 0
        fused[center_valid] = center[center_valid]
        fused_mask = (fused > 0).astype(np.uint8)
        stats = {
            "center_valid_pixels": int(center_valid.sum()),
            "fused_valid_pixels": int(fused_mask.sum()),
            "conflict_pixels": int(conflicts.sum()),
            "anchor_restored_pixels": int((center_valid & conflicts).sum()),
        }
        return ProjectedCenter(target, frames, center, fused, fused_mask, stats)


class CenteredFiveWindow:
    """Five-slot state machine; rejected slots remain in sequence and propagate."""

    def __init__(self, projector: CenteredFiveProjector | None = None) -> None:
        self._projector = projector or CenteredFiveProjector()
        self._slots: deque[FinalizedSensorFrame | RejectedSensorFrame] = deque(maxlen=5)
        self._last_index: int | None = None
        self._closed = False

    def push(
        self, slot: FinalizedSensorFrame | RejectedSensorFrame,
    ) -> tuple[ProjectedCenter | RejectedCenter, ...]:
        if self._closed:
            raise RuntimeError("CenteredFiveWindow is already closed")
        if self._last_index is not None and slot.image_index != self._last_index + 1:
            raise ValueError("CenteredFiveWindow requires consecutive image slots")
        self._last_index = slot.image_index
        self._slots.append(slot)
        if len(self._slots) < 5:
            return ()
        slots = tuple(self._slots)
        rejected = tuple(item for item in slots if isinstance(item, RejectedSensorFrame))
        if rejected:
            target = slots[2]
            reasons = tuple(dict.fromkeys(item.reason for item in rejected))
            return (RejectedCenter(
                target_image_index=target.image_index,
                target_stem=target.stem,
                target_timestamp_ns=target.image_timestamp_ns,
                dependency_indices=tuple(item.image_index for item in slots),
                root_reasons=reasons,
            ),)
        return (self._projector.project(slots),)  # type: ignore[arg-type]

    def finish(self) -> tuple[ProjectedCenter | RejectedCenter, ...]:
        self._closed = True
        return ()
