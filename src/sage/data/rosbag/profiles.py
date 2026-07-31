from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping

from ...foundation.prepared_scene_contract import (
    PREPARED_SCENE_SCHEMA,
    PROFILE_CONTRACTS,
    validate_source_contract as validate_prepared_scene_contract,
)

SCHEMA_VERSION = PREPARED_SCENE_SCHEMA


@dataclass(frozen=True)
class SourceContract:
    mode: str
    topic: str
    frame_id: str
    center_source_type: str
    fused_source_type: str
    fallback_policy: str
    producer_processing: str


@dataclass(frozen=True)
class CameraContract:
    source_model: str = "FishPoly"
    output_model: str = "PINHOLE"
    output_grid: tuple[int, int] = (1296, 1600)


@dataclass(frozen=True)
class TimingContract:
    clock: str
    pose_policy: str
    point_time_policy: str
    point_offset_range_s: tuple[float, float] | None


@dataclass(frozen=True)
class DepthContract:
    definition: str
    fusion_policy: str
    fusion_window: int = 5
    z_buffer: str = "nearest"
    conflict_threshold_m: float = 1.0
    min_depth_m: float = 0.1
    max_depth_m: float = 200.0


@dataclass(frozen=True)
class PreparationProfile:
    name: str
    source: SourceContract
    camera: CameraContract
    timing: TimingContract
    depth: DepthContract
    cloud_max_dt_ms: float = 20.0
    odometry_max_dt_ms: float = 50.0

    def manifest_contract(self) -> dict[str, object]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "preparation_profile": self.name,
            "source": asdict(self.source),
            "camera": asdict(self.camera),
            "timing": asdict(self.timing),
            "depth": asdict(self.depth),
        }
        payload["camera"]["output_grid"] = list(self.camera.output_grid)
        if self.timing.point_offset_range_s is not None:
            payload["timing"]["point_offset_range_s"] = list(self.timing.point_offset_range_s)
        return payload


def _profile_from_contract(name: str) -> PreparationProfile:
    contract = PROFILE_CONTRACTS[name]
    camera = dict(contract["camera"])
    timing = dict(contract["timing"])
    depth = dict(contract["depth"])
    camera["output_grid"] = tuple(camera["output_grid"])
    if timing["point_offset_range_s"] is not None:
        timing["point_offset_range_s"] = tuple(timing["point_offset_range_s"])
    return PreparationProfile(
        name=name,
        source=SourceContract(**contract["source"]),
        camera=CameraContract(**camera),
        timing=TimingContract(**timing),
        depth=DepthContract(**depth),
    )


_PROFILES: Mapping[str, PreparationProfile] = MappingProxyType({
    name: _profile_from_contract(name) for name in PROFILE_CONTRACTS
})


def profile_for_name(name: str) -> PreparationProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unsupported preparation_profile={name!r}; expected one of: {supported}") from exc


def validate_source_contract(payload: Mapping[str, object]) -> PreparationProfile:
    if not isinstance(payload, Mapping):
        raise ValueError("Prepared Scene source contract must be an object")
    validate_prepared_scene_contract(payload)
    return profile_for_name(str(payload["preparation_profile"]))
