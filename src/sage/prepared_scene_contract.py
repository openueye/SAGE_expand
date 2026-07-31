from __future__ import annotations

from typing import Mapping


PREPARED_SCENE_SCHEMA = "sage-prepared-scene-v1"

_COMMON_CAMERA = {"source_model": "FishPoly", "output_model": "PINHOLE", "output_grid": [1296, 1600]}
_COMMON_DEPTH = {
    "definition": "target-camera-z-meters", "fusion_window": 5, "z_buffer": "nearest",
    "conflict_threshold_m": 1.0, "min_depth_m": 0.1, "max_depth_m": 200.0,
}
PROFILE_CONTRACTS = {
    "odin1-calibrated-native-v1": {
        "source": {
            "mode": "LIDAR_RAW", "topic": "/odin1/cloud_raw", "frame_id": "odin1_base_link",
            "center_source_type": "LIDAR_RAW", "fused_source_type": "LIDAR_FUSED5",
            "fallback_policy": "none", "producer_processing": "consumer-per-point-deskew",
        },
        "camera": _COMMON_CAMERA,
        "timing": {
            "clock": "message_header_stamp", "pose_policy": "se3-interpolation-no-extrapolation-v1",
            "point_time_policy": "header-plus-offset-time-seconds-v1",
            "point_offset_range_s": [0.0, 0.1],
        },
        "depth": {**_COMMON_DEPTH, "fusion_policy": "raw-centered-5-v1"},
    },
    "odin1-slam-world-native-v1": {
        "source": {
            "mode": "SLAM_WORLD", "topic": "/odin1/cloud_slam", "frame_id": "odom",
            "center_source_type": "LIDAR_SLAM_CENTER", "fused_source_type": "LIDAR_SLAM_FUSED5",
            "fallback_policy": "none", "producer_processing": "opaque-recorded-output",
        },
        "camera": _COMMON_CAMERA,
        "timing": {
            "clock": "message_header_stamp", "pose_policy": "se3-interpolation-no-extrapolation-v1",
            "point_time_policy": "unavailable-in-topic",
            "point_offset_range_s": None,
        },
        "depth": {**_COMMON_DEPTH, "fusion_policy": "slam-world-centered-5-v1"},
    },
}


def validate_source_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    if manifest.get("schema_version") != PREPARED_SCENE_SCHEMA:
        raise ValueError(f"Prepared Scene must use {PREPARED_SCENE_SCHEMA}")
    profile_name = manifest.get("preparation_profile")
    if profile_name not in PROFILE_CONTRACTS:
        raise ValueError("Prepared Scene preparation_profile is unsupported")
    expected = PROFILE_CONTRACTS[str(profile_name)]
    for name in ("source", "camera", "timing", "depth"):
        if manifest.get(name) != expected[name]:
            raise ValueError(f"Prepared Scene {name} does not match preparation_profile={profile_name}")
    return expected
