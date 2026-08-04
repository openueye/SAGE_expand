"""The complete, explicit description of one ROSBAG input.

Nothing here is inferred from a dataset name. Topics are named by the user,
message types are detected but never guessed at semantically, and every
association and fusion rule is a declared value that ends up in the adapter
provenance identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping


ADAPTER_TYPE = "rosbag2"
EXECUTION_BATCH_V1 = "batch-v1"
EXECUTION_ONLINE_WINDOW_V2 = "online-window-v2"
EXECUTION_MODES = (EXECUTION_BATCH_V1, EXECUTION_ONLINE_WINDOW_V2)

TIMESTAMP_POLICIES = ("header_stamp",)
LIDAR_ASSOCIATIONS = ("nearest_to_anchor", "latest_not_after_anchor")
POSE_INTERPOLATIONS = ("linear_slerp",)
POINTS_FRAMES = ("lidar_frame", "reference_frame")

_CONFIGURABLE_SYNCHRONIZATION_FIELDS = frozenset({
    "lidar_association",
    "max_lidar_skew_ms",
    "max_pose_gap_ms",
})
_CONFIGURABLE_FUSION_FIELDS = frozenset({
    "window",
    "max_age_ms",
    "conflict_threshold_m",
})

# Only one fusion policy is implemented. It is symmetric around the anchor and
# therefore NOT causal: frame k uses scans from k-2..k+2. The name says so, and
# `causal_trailing` is deliberately left unimplemented rather than aliased onto
# this behaviour.
# ponytail: single policy; add causal_trailing as a second builder when a
# causal run is actually needed.
FUSION_POLICIES = ("centered_window",)


def _positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


@dataclass(frozen=True)
class SynchronizationSpec:
    anchor: str = "image"
    timestamp_policy: str = "header_stamp"
    lidar_association: str = "nearest_to_anchor"
    max_lidar_skew_ms: float = 20.0
    pose_interpolation: str = "linear_slerp"
    max_pose_gap_ms: float = 100.0
    allow_pose_extrapolation: bool = False

    def __post_init__(self) -> None:
        if self.anchor != "image":
            raise ValueError("synchronization.anchor must be image")
        if self.timestamp_policy not in TIMESTAMP_POLICIES:
            raise ValueError(
                f"synchronization.timestamp_policy must be one of {TIMESTAMP_POLICIES}"
            )
        if self.lidar_association not in LIDAR_ASSOCIATIONS:
            raise ValueError(
                f"synchronization.lidar_association must be one of {LIDAR_ASSOCIATIONS}"
            )
        if self.pose_interpolation not in POSE_INTERPOLATIONS:
            raise ValueError(
                f"synchronization.pose_interpolation must be one of {POSE_INTERPOLATIONS}"
            )
        if self.allow_pose_extrapolation is not False:
            raise ValueError("Pose extrapolation is not supported; it must stay false")
        object.__setattr__(self, "max_lidar_skew_ms", _positive_number(
            self.max_lidar_skew_ms, "synchronization.max_lidar_skew_ms",
        ))
        object.__setattr__(self, "max_pose_gap_ms", _positive_number(
            self.max_pose_gap_ms, "synchronization.max_pose_gap_ms",
        ))

    @property
    def max_lidar_skew_ns(self) -> int:
        return int(round(self.max_lidar_skew_ms * 1_000_000))

    @property
    def max_pose_gap_ns(self) -> int:
        return int(round(self.max_pose_gap_ms * 1_000_000))

    def payload(self) -> dict[str, object]:
        return {
            "anchor": self.anchor,
            "timestamp_policy": self.timestamp_policy,
            "lidar_association": self.lidar_association,
            "max_lidar_skew_ms": self.max_lidar_skew_ms,
            "pose_interpolation": self.pose_interpolation,
            "max_pose_gap_ms": self.max_pose_gap_ms,
            "allow_pose_extrapolation": self.allow_pose_extrapolation,
        }


@dataclass(frozen=True)
class FusionSpec:
    policy: str = "centered_window"
    window: int = 5
    max_age_ms: float = 500.0
    z_buffer: str = "nearest"
    conflict_threshold_m: float = 1.0
    current_scan_priority: bool = True
    warmup_policy: str = "drop_incomplete_window"

    def __post_init__(self) -> None:
        if self.policy not in FUSION_POLICIES:
            raise ValueError(f"fusion.policy must be one of {FUSION_POLICIES}")
        if type(self.window) is not int or self.window < 1 or self.window % 2 == 0:
            raise ValueError("fusion.window must be a positive odd number of scans")
        if self.z_buffer != "nearest":
            raise ValueError("fusion.z_buffer must be nearest")
        if self.warmup_policy != "drop_incomplete_window":
            raise ValueError("fusion.warmup_policy must be drop_incomplete_window")
        if self.current_scan_priority is not True:
            raise ValueError("fusion.current_scan_priority must stay true")
        object.__setattr__(self, "max_age_ms", _positive_number(self.max_age_ms, "fusion.max_age_ms"))
        conflict = float(self.conflict_threshold_m)
        if not conflict >= 0:
            raise ValueError("fusion.conflict_threshold_m must be non-negative")
        object.__setattr__(self, "conflict_threshold_m", conflict)

    @property
    def causal(self) -> bool:
        """A centered window reads future scans; say so everywhere it is recorded."""
        return False

    @property
    def half_window(self) -> int:
        return self.window // 2

    @property
    def max_age_ns(self) -> int:
        return int(round(self.max_age_ms * 1_000_000))

    def payload(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "causal": self.causal,
            "max_scans": self.window,
            "max_age_ms": self.max_age_ms,
            "z_buffer": self.z_buffer,
            "conflict_threshold_m": self.conflict_threshold_m,
            "current_scan_priority": self.current_scan_priority,
            "warmup_policy": self.warmup_policy,
        }


@dataclass(frozen=True)
class DepthSpec:
    min_depth_m: float = 0.1
    max_depth_m: float = 200.0

    def __post_init__(self) -> None:
        minimum = _positive_number(self.min_depth_m, "depth.min_depth_m")
        maximum = _positive_number(self.max_depth_m, "depth.max_depth_m")
        if maximum <= minimum:
            raise ValueError("depth.max_depth_m must exceed depth.min_depth_m")
        object.__setattr__(self, "min_depth_m", minimum)
        object.__setattr__(self, "max_depth_m", maximum)

    def payload(self) -> dict[str, object]:
        return {"min_depth_m": self.min_depth_m, "max_depth_m": self.max_depth_m}


@dataclass(frozen=True)
class RosbagInputSpec:
    rosbag_path: Path
    calibration_path: Path
    lidar_topic: str
    image_topic: str
    odometry_topic: str
    points_frame: str = "lidar_frame"
    enable_fused: bool = True
    synchronization: SynchronizationSpec = SynchronizationSpec()
    fusion: FusionSpec = FusionSpec()
    depth: DepthSpec = DepthSpec()
    frame_limit: int | None = None
    execution: str = EXECUTION_BATCH_V1
    allow_empty_image_frame_id: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rosbag_path", Path(self.rosbag_path).expanduser().resolve())
        object.__setattr__(self, "calibration_path", Path(self.calibration_path).expanduser().resolve())
        for name in ("lidar_topic", "image_topic", "odometry_topic"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("/"):
                raise ValueError(f"input.topics {name} must be an absolute topic name")
        if len({self.lidar_topic, self.image_topic, self.odometry_topic}) != 3:
            raise ValueError("input.topics must name three distinct topics")
        if self.points_frame not in POINTS_FRAMES:
            raise ValueError(f"input.lidar.points_frame must be one of {POINTS_FRAMES}")
        if type(self.enable_fused) is not bool:
            raise ValueError("input.lidar.enable_fused must be a boolean")
        if self.frame_limit is not None and (type(self.frame_limit) is not int or self.frame_limit < 1):
            raise ValueError("input.frame_limit must be a positive integer when set")
        if self.execution not in EXECUTION_MODES:
            raise ValueError(f"input.execution must be one of {EXECUTION_MODES}")
        if type(self.allow_empty_image_frame_id) is not bool:
            raise ValueError("input.allow_empty_image_frame_id must be a boolean")
        if self.allow_empty_image_frame_id and self.execution != EXECUTION_ONLINE_WINDOW_V2:
            raise ValueError(
                "input.allow_empty_image_frame_id requires execution=online-window-v2"
            )

    @property
    def sources(self) -> tuple[str, ...]:
        return ("LIDAR_CENTER", "LIDAR_FUSED") if self.enable_fused else ("LIDAR_CENTER",)

    def topic_roles(self) -> dict[str, str]:
        return {
            "lidar": self.lidar_topic,
            "image": self.image_topic,
            "odometry": self.odometry_topic,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, base_dir: Path) -> "RosbagInputSpec":
        """Build from the config `input:` section; unknown keys are rejected."""
        expected = {"type", "rosbag", "calibration", "topics", "lidar", "synchronization", "fusion", "depth"}
        optional = {"frame_limit", "execution", "allow_empty_image_frame_id"}
        if not isinstance(payload, Mapping) or not expected <= set(payload) <= expected | optional:
            raise ValueError(
                "input (rosbag2) must define: " + ", ".join(sorted(expected))
                + "; optional: " + ", ".join(sorted(optional))
            )
        if payload["type"] != ADAPTER_TYPE:
            raise ValueError(f"input.type must be {ADAPTER_TYPE}")
        topics = payload["topics"]
        if not isinstance(topics, Mapping) or set(topics) != {"lidar", "image", "odometry"}:
            raise ValueError("input.topics must define exactly lidar, image and odometry")
        lidar = payload["lidar"]
        if not isinstance(lidar, Mapping) or set(lidar) != {"points_frame", "enable_fused"}:
            raise ValueError("input.lidar must define points_frame and enable_fused")
        synchronization = payload["synchronization"]
        if (
            not isinstance(synchronization, Mapping)
            or frozenset(synchronization) != _CONFIGURABLE_SYNCHRONIZATION_FIELDS
        ):
            raise ValueError(
                "input.synchronization must define exactly: "
                + ", ".join(sorted(_CONFIGURABLE_SYNCHRONIZATION_FIELDS))
            )
        fusion = payload["fusion"]
        if (
            not isinstance(fusion, Mapping)
            or frozenset(fusion) != _CONFIGURABLE_FUSION_FIELDS
        ):
            raise ValueError(
                "input.fusion must define exactly: "
                + ", ".join(sorted(_CONFIGURABLE_FUSION_FIELDS))
            )
        return cls(
            rosbag_path=_resolve(payload["rosbag"], base_dir),
            calibration_path=_resolve(payload["calibration"], base_dir),
            lidar_topic=str(topics["lidar"]),
            image_topic=str(topics["image"]),
            odometry_topic=str(topics["odometry"]),
            points_frame=str(lidar["points_frame"]),
            enable_fused=bool(lidar["enable_fused"]),
            synchronization=SynchronizationSpec(**dict(synchronization)),
            fusion=FusionSpec(**dict(fusion)),
            depth=DepthSpec(**dict(payload["depth"])),
            frame_limit=payload.get("frame_limit"),
            execution=str(payload.get("execution", EXECUTION_BATCH_V1)),
            allow_empty_image_frame_id=payload.get("allow_empty_image_frame_id", False),
        )


def _resolve(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input paths must be non-empty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


__all__ = [
    "ADAPTER_TYPE",
    "DepthSpec",
    "EXECUTION_BATCH_V1",
    "EXECUTION_ONLINE_WINDOW_V2",
    "FusionSpec",
    "RosbagInputSpec",
    "SynchronizationSpec",
]
