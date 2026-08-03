from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Iterable


REPLAY_RECEIPT_SCHEMA = "prepared-scene-replay-v1"
MATERIALIZED_RECEIPT_SCHEMA = "materialized-prepared-scene-v1"
STREAM_RECEIPT_SCHEMA = "fixed-lag-stream-v1"

_REQUIRED_STREAM_TOPIC_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "nav_msgs/msg/Odometry",
    "sensor_msgs/msg/PointCloud2",
}


def validate_completion_receipt(
    receipt: object,
    *,
    allowed_schemas: Iterable[str],
    expected_adapter: str | None = None,
    expected_source_mode: str | None = None,
    expected_emitted: int | None = None,
    require_bag_exhausted: bool = False,
    require_write_through: bool = False,
    require_peak_cuda: bool = True,
) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise ValueError("FrameSource completion receipt must be an object")
    required = {
        "receipt_schema", "complete", "aborted", "error", "fallback_used", "source_mode",
        "emitted_centers", "rejected_centers_count", "source_families",
        "frame_hash_chain_sha256", "write_through_enabled",
    }
    missing = sorted(name for name in required if name not in receipt)
    if missing:
        raise ValueError(f"FrameSource completion receipt is missing fields: {', '.join(missing)}")
    schema = receipt["receipt_schema"]
    expected_schema = {
        "prepared-scene": REPLAY_RECEIPT_SCHEMA,
        "rosbag-fixed-lag-v1": STREAM_RECEIPT_SCHEMA,
    }.get(expected_adapter)
    if schema not in set(allowed_schemas) or (
        expected_schema is not None and schema != expected_schema
    ):
        raise ValueError("FrameSource completion receipt schema is invalid")
    if schema in {MATERIALIZED_RECEIPT_SCHEMA, STREAM_RECEIPT_SCHEMA}:
        if (
            type(receipt.get("bag_exhausted")) is not bool
            or type(receipt.get("truncated")) is not bool
            or receipt["truncated"] == receipt["bag_exhausted"]
        ):
            raise ValueError("Prepared Scene production exhaustion state is invalid")
        if (require_bag_exhausted or require_write_through) and not receipt["bag_exhausted"]:
            raise ValueError("Formal ROSBAG processing must exhaust the ROSBAG")
    if (
        receipt["complete"] is not True
        or receipt["aborted"] is not False
        or receipt["error"] is not None
        or receipt["fallback_used"] is not False
        or not isinstance(receipt["source_mode"], str)
        or type(receipt["emitted_centers"]) is not int
        or type(receipt["rejected_centers_count"]) is not int
        or not isinstance(receipt["source_families"], list)
        or not receipt["source_families"]
        or not all(isinstance(value, str) for value in receipt["source_families"])
        or set(receipt["source_families"]) != {receipt["source_mode"]}
        or type(receipt["write_through_enabled"]) is not bool
    ):
        raise ValueError("FrameSource completion receipt lifecycle fields are invalid")
    if expected_emitted is not None and receipt["emitted_centers"] != expected_emitted:
        raise ValueError("FrameSource completion receipt emitted count does not match mapping")
    if expected_source_mode is not None and receipt["source_mode"] != expected_source_mode:
        raise ValueError("FrameSource completion receipt source mode contradicts the run identity")
    runtime = receipt.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, dict):
            raise ValueError("FrameSource runtime evidence must be an object")
        allowed_runtime = {
            "first_frame_wait_seconds", "mapping_wall_seconds", "mapping_fps",
        }
        if set(runtime) - allowed_runtime:
            raise ValueError("FrameSource runtime evidence contains unknown fields")
        for name in allowed_runtime & set(runtime):
            value = runtime[name]
            if value == "unmeasured":
                continue
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"FrameSource runtime evidence {name} is invalid")
    if require_write_through and receipt["write_through_enabled"] is not True:
        raise ValueError("Formal mapping requires write-through")
    chain_value = receipt["frame_hash_chain_sha256"]
    if (
        not isinstance(chain_value, str) or len(chain_value) != 64
        or any(character not in "0123456789abcdef" for character in chain_value)
    ):
        raise ValueError("FrameSource completion receipt hash chain is incomplete")
    if schema == STREAM_RECEIPT_SCHEMA:
        _validate_stream_receipt(
            receipt,
            require_write_through=require_write_through,
            require_peak_cuda=require_peak_cuda,
        )
    return deepcopy(receipt)


def validate_sage_completion_receipt(
    receipt: object,
    *,
    expected_adapter: str | None = None,
    expected_source_mode: str | None = None,
    expected_emitted: int | None = None,
    require_bag_exhausted: bool = False,
    require_write_through: bool = False,
    require_peak_cuda: bool = True,
) -> dict[str, object]:
    """validate_completion_receipt() pinned to the two receipt schemas SAGE publishes."""
    return validate_completion_receipt(
        receipt,
        allowed_schemas={REPLAY_RECEIPT_SCHEMA, STREAM_RECEIPT_SCHEMA},
        expected_adapter=expected_adapter,
        expected_source_mode=expected_source_mode,
        expected_emitted=expected_emitted,
        require_bag_exhausted=require_bag_exhausted,
        require_write_through=require_write_through,
        require_peak_cuda=require_peak_cuda,
    )


def _validate_stream_receipt(
    receipt: dict[str, object],
    *,
    require_write_through: bool,
    require_peak_cuda: bool,
) -> None:
    required_stream = {
        "selected_image_slots", "finalized_slots", "rejected_slots", "edge_slots_not_emitted",
        "emitted_frame_records", "queue_capacity", "max_queue_depth",
        "producer_stall_seconds", "consumer_wait_seconds", "peak_cpu_memory_bytes",
        "write_through_manifest", "write_through_artifacts_sha256", "rejected_centers",
    }
    if require_peak_cuda:
        required_stream.add("peak_cuda_memory_bytes")
    missing = sorted(name for name in required_stream if name not in receipt)
    if missing:
        raise ValueError(f"Fixed-lag completion receipt is missing fields: {', '.join(missing)}")
    records = receipt["emitted_frame_records"]
    if not isinstance(records, list) or len(records) != receipt["emitted_centers"]:
        raise ValueError("Fixed-lag completion receipt frame provenance is incomplete")
    rejected = receipt["rejected_centers"]
    if not isinstance(rejected, list) or len(rejected) != receipt["rejected_centers_count"]:
        raise ValueError("Fixed-lag completion receipt rejection provenance is incomplete")
    if require_write_through and (
        not isinstance(receipt["write_through_manifest"], str)
        or not isinstance(receipt["write_through_artifacts_sha256"], dict)
        or not receipt["write_through_artifacts_sha256"]
    ):
        raise ValueError("Formal fixed-lag mapping requires write-through")
    if not require_write_through:
        if receipt["write_through_enabled"] is True and (
            not isinstance(receipt["write_through_manifest"], str)
            or not isinstance(receipt["write_through_artifacts_sha256"], dict)
            or not receipt["write_through_artifacts_sha256"]
        ):
            raise ValueError("Enabled streaming write-through evidence is incomplete")
        if receipt["write_through_enabled"] is False and (
            receipt["write_through_manifest"] is not None
            or receipt["write_through_artifacts_sha256"] is not None
        ):
            raise ValueError("Disabled streaming write-through evidence must be null")
    producer_identity = receipt.get("producer_identity")
    if receipt.get("source_mode") != "LIDAR_WORLD":
        raise ValueError("Fixed-lag completion receipt source semantics are invalid")
    input_contract = producer_identity.get("input_contract") if isinstance(producer_identity, dict) else None
    topics = input_contract.get("topics") if isinstance(input_contract, dict) else None
    if not isinstance(topics, dict) or len(topics) != 3:
        raise ValueError("Fixed-lag completion receipt lacks observed topic identities")
    for topic, observed in topics.items():
        if (
            not isinstance(observed, dict)
            or not isinstance(observed.get("type"), str)
            or observed.get("serialization") != "cdr"
            or type(observed.get("message_count")) is not int
            or observed["message_count"] < 1
            or not isinstance(observed.get("header_timestamp_range_ns"), list)
            or len(observed["header_timestamp_range_ns"]) != 2
            or any(type(value) is not int for value in observed["header_timestamp_range_ns"])
            or observed["header_timestamp_range_ns"][0] > observed["header_timestamp_range_ns"][1]
        ):
            raise ValueError(f"Fixed-lag topic observation is invalid: {topic}")
    if {observed["type"] for observed in topics.values()} != _REQUIRED_STREAM_TOPIC_TYPES:
        raise ValueError("Fixed-lag completion receipt topic source semantics are invalid")
    source_layout = input_contract.get("source_layout")
    odometry_frames = input_contract.get("odometry_frames")
    if (
        not isinstance(source_layout, dict)
        or source_layout.get("frame_ids") != ["odom"]
        or not isinstance(odometry_frames, dict)
        or odometry_frames.get("frame_ids") != ["odom"]
        or not isinstance(odometry_frames.get("child_frame_ids"), list)
        or len(odometry_frames["child_frame_ids"]) != 1
        or not isinstance(odometry_frames["child_frame_ids"][0], str)
        or not odometry_frames["child_frame_ids"][0]
    ):
        raise ValueError("Fixed-lag completion receipt frame source semantics are invalid")
    integer_metrics = {
        "selected_image_slots", "finalized_slots", "rejected_slots", "edge_slots_not_emitted",
        "queue_capacity", "max_queue_depth", "peak_cpu_memory_bytes",
    }
    if require_peak_cuda:
        integer_metrics.add("peak_cuda_memory_bytes")
    for name in integer_metrics:
        if type(receipt[name]) is not int or receipt[name] < 0:
            raise ValueError(f"Fixed-lag completion metric {name} is invalid")
    for name in ("producer_stall_seconds", "consumer_wait_seconds"):
        if not isinstance(receipt[name], (int, float)) or receipt[name] < 0:
            raise ValueError(f"Fixed-lag completion metric {name} is invalid")
    chain = bytes(32)
    for record in records:
        if not isinstance(record, dict) or not {
            "frame_id", "image_index", "image_timestamp_ns", "cloud_id", "cloud_timestamp_ns",
            "window_image_indices", "window_cloud_ids", "window_normalization_receipts",
            "center_source_type", "fused_source_type", "normalization_receipt", "frame_hashes",
        }.issubset(record):
            raise ValueError("Fixed-lag completion receipt frame provenance is incomplete")
        hashes = record["frame_hashes"]
        if not isinstance(hashes, dict) or set(hashes) != {
            "image", "center_depth", "fused5_depth", "fused5_mask",
        }:
            raise ValueError("Fixed-lag completion receipt frame hashes are incomplete")
        if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
            raise ValueError("Fixed-lag completion receipt frame hash is invalid")
        if (
            not isinstance(record["frame_id"], str) or not record["frame_id"]
            or type(record["image_index"]) is not int
            or type(record["image_timestamp_ns"]) is not int
            or not isinstance(record["cloud_id"], str) or not record["cloud_id"]
            or type(record["cloud_timestamp_ns"]) is not int
            or not isinstance(record["window_image_indices"], list)
            or len(record["window_image_indices"]) != 5
            or not isinstance(record["window_cloud_ids"], list)
            or len(record["window_cloud_ids"]) != 5
            or not isinstance(record["window_normalization_receipts"], list)
            or len(record["window_normalization_receipts"]) != 5
            or not isinstance(record["normalization_receipt"], dict)
        ):
            raise ValueError("Fixed-lag completion receipt frame association is invalid")
        if (
            record["center_source_type"] != "LIDAR_WORLD_CENTER"
            or record["fused_source_type"] != "LIDAR_WORLD_FUSED5"
        ):
            raise ValueError("Fixed-lag completion receipt source semantics are invalid")
        _validate_normalization_receipt(
            record["normalization_receipt"],
            source_mode=str(receipt["source_mode"]),
        )
        for window_receipt in record["window_normalization_receipts"]:
            if (
                not isinstance(window_receipt, dict)
                or not isinstance(window_receipt.get("cloud_id"), str)
                or type(window_receipt.get("image_index")) is not int
            ):
                raise ValueError(
                    "Fixed-lag completion receipt window source semantics are invalid"
                )
            _validate_normalization_receipt(
                window_receipt.get("receipt"),
                source_mode=str(receipt["source_mode"]),
            )
        chain = hashlib.sha256(
            chain + record["frame_id"].encode("utf-8")
            + json.dumps(hashes, sort_keys=True).encode("utf-8")
        ).digest()
    if chain.hex() != receipt["frame_hash_chain_sha256"]:
        raise ValueError("Fixed-lag completion receipt hash chain does not match frame provenance")
    for item in rejected:
        if not isinstance(item, dict) or not {
            "target_image_index", "target_stem", "target_timestamp_ns",
            "dependency_indices", "root_reasons",
        }.issubset(item):
            raise ValueError("Fixed-lag completion receipt rejection provenance is invalid")


def _validate_normalization_receipt(
    value: object,
    *,
    source_mode: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("Fixed-lag completion receipt normalization source semantics are invalid")
    if source_mode != "LIDAR_WORLD" or (
        value.get("mode") != "direct-world"
        or value.get("frame_id") != "odom"
        or value.get("point_time_policy") != "unavailable-in-topic"
        or value.get("producer_processing") != "opaque-recorded-output"
        or not isinstance(value.get("image_pose_bracket"), dict)
    ):
        raise ValueError(
            "Fixed-lag completion receipt normalization source semantics are invalid"
        )
