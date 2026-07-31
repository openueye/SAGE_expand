from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from time import perf_counter
from typing import Protocol

import numpy as np

from .config import SceneConfig
from .contracts import (
    CameraIntrinsics,
    DepthEvidence,
    FrameInputs,
    GrowthInputs,
    INVALID_SOURCE_TYPE,
    MappingObservation,
    Pose,
    SourceType,
)
from .scene import (
    PreparedScene,
    _resize_nearest,
    _resize_rgb,
    scene_content_sha256,
    sha256_file,
    transform_contract_sha256,
)
from .receipt_contract import REPLAY_RECEIPT_SCHEMA
from .source_policy import SOURCE_POLICY_VERSION, descriptor_for_type


class FrameSource(Protocol):
    """Small mapper-facing seam for ordered, source-identified frame inputs."""

    def start_identity(self) -> dict[str, object]: ...

    def frames(self) -> Iterator[FrameInputs]: ...

    def set_runtime_metrics(self, metrics: Mapping[str, object]) -> None: ...

    def prepare_close(self) -> dict[str, object]: ...

    def commit(self) -> None: ...

    def rollback_commit(self, error: BaseException | str) -> None: ...

    def abort(self, error: BaseException | str) -> dict[str, object]: ...


def _source_mode_for_frame(frame: FrameInputs) -> str | None:
    family = frame.mapping.source_family
    return family


class PreparedSceneFrameSource:
    """Lifecycle adapter over the existing, validated PreparedScene loader."""

    def __init__(self, config: SceneConfig, *, frame_limit: int = -1) -> None:
        if type(frame_limit) is not int or frame_limit == 0 or frame_limit < -1:
            raise ValueError("FrameSource frame_limit must be -1 or a positive integer")
        self._scene = PreparedScene(config)
        self._frame_limit = frame_limit
        self._manifest: dict[str, object] | None = None
        self._started = False
        self._exhausted = False
        self._aborted = False
        self._prepared_receipt: dict[str, object] | None = None
        self._emitted = 0
        self._source_modes: set[str] = set()
        self._runtime_metrics: dict[str, object] = {}

    def start_identity(self) -> dict[str, object]:
        if self._aborted:
            raise RuntimeError("PreparedSceneFrameSource has already been aborted")
        if self._manifest is None:
            self._manifest = self._scene._validate_contract()
        manifest = self._manifest
        source_mode = str(manifest.get("source", {}).get("mode", manifest.get("source_mode", "LIDAR_RAW")))
        profile = str(manifest.get("preparation_profile", "sage-prepared-scene-v1"))
        return {
            "adapter": "prepared-scene",
            "receipt_schema": REPLAY_RECEIPT_SCHEMA,
            "preparation_profile": profile,
            "source_mode": source_mode,
            "source_policy_version": str(manifest["source_policy_version"]),
            "fallback_policy": "none",
            "prepared_manifest_sha256": sha256_file(self._scene.prepared_manifest_path),
            "transform_contract_sha256": transform_contract_sha256(manifest),
            "content_sha256": scene_content_sha256(self._scene.input_files(limit=self._frame_limit)),
        }

    def frames(self) -> Iterator[FrameInputs]:
        if self._started:
            raise RuntimeError("PreparedSceneFrameSource.frames() is single-use")
        if self._aborted:
            raise RuntimeError("PreparedSceneFrameSource has already been aborted")
        identity = self.start_identity()
        declared_mode = str(identity["source_mode"])
        first_frame_started = perf_counter()
        first_frame_recorded = False
        self._started = True
        try:
            for frame in self._scene.frames(limit=self._frame_limit):
                mode = _source_mode_for_frame(frame)
                if mode is not None and mode != declared_mode:
                    raise ValueError("Prepared Scene frame source family contradicts its manifest")
                self._source_modes.add(declared_mode)
                if len(self._source_modes) > 1:
                    raise ValueError("FrameSource cannot mix raw and SLAM LiDAR source families")
                self._emitted += 1
                if not first_frame_recorded:
                    self._runtime_metrics["first_frame_wait_seconds"] = perf_counter() - first_frame_started
                    first_frame_recorded = True
                yield frame
            self._exhausted = True
        except BaseException:
            raise

    def prepare_close(self) -> dict[str, object]:
        if not self._started or not self._exhausted:
            raise RuntimeError("PreparedSceneFrameSource.prepare_close() requires a fully consumed iterator")
        if self._prepared_receipt is not None:
            raise RuntimeError("PreparedSceneFrameSource is already prepared")
        counts = self._manifest.get("counts", {})
        rejected_count = int(counts.get("rejected_centers", 0)) if isinstance(counts, dict) else 0
        self._prepared_receipt = {
            **self.start_identity(),
            "receipt_schema": REPLAY_RECEIPT_SCHEMA,
            "complete": True,
            "aborted": False,
            "emitted_centers": self._emitted,
            "rejected_centers": rejected_count,
            "rejected_centers_count": rejected_count,
            "write_through_enabled": False,
            "frame_hash_chain_sha256": self._manifest.get("frame_hash_chain_sha256"),
            "source_families": sorted(
                self._source_modes or {str(self.start_identity()["source_mode"])}
            ),
            "fallback_used": False,
            "error": None,
            "runtime": {
                "first_frame_wait_seconds": self._runtime_metrics.get(
                    "first_frame_wait_seconds", "unmeasured"
                ),
                **{
                    key: value for key, value in self._runtime_metrics.items()
                    if key != "first_frame_wait_seconds"
                },
            },
        }
        return deepcopy(self._prepared_receipt)

    def set_runtime_metrics(self, metrics: Mapping[str, object]) -> None:
        if not isinstance(metrics, Mapping):
            raise ValueError("FrameSource runtime metrics must be an object")
        self._runtime_metrics.update(deepcopy({
            key: value for key, value in metrics.items()
            if key in {"first_frame_wait_seconds", "mapping_wall_seconds", "mapping_fps"}
        }))

    def commit(self) -> None:
        if self._prepared_receipt is None:
            raise RuntimeError("PreparedSceneFrameSource.commit() requires prepare_close()")

    def rollback_commit(self, error: BaseException | str) -> None:
        if self._prepared_receipt is None:
            raise RuntimeError("PreparedSceneFrameSource.rollback_commit() requires prepare_close()")

    def close(self) -> dict[str, object]:
        receipt = self.prepare_close()
        self.commit()
        return receipt

    def abort(self, error: BaseException | str) -> dict[str, object]:
        if isinstance(error, BaseException):
            message = f"{type(error).__name__}: {error}"
        else:
            message = str(error)
        if not message:
            raise ValueError("FrameSource abort error must be non-empty")
        identity = self.start_identity()
        self._aborted = True
        return {
            **identity,
            "receipt_schema": REPLAY_RECEIPT_SCHEMA,
            "complete": False,
            "aborted": True,
            "emitted_centers": self._emitted,
            "rejected_centers": 0,
            "source_families": sorted(self._source_modes),
            "fallback_used": False,
            "error": message,
        }


def _matrix_to_pose(matrix: np.ndarray) -> Pose:
    from .data.rosbag.poses import matrix_to_quaternion_xyzw

    quaternion = matrix_to_quaternion_xyzw(np.asarray(matrix, dtype=np.float64)[:3, :3])
    return Pose(
        float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]),
        float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]),
    )


class OdinBagFixedLagFrameSource:
    """Adapter for SAGE's finite source-aware Odin1 fixed-lag stream."""

    def __init__(
        self,
        config: SceneConfig,
        *,
        frame_limit: int = -1,
        non_formal: bool = False,
    ) -> None:
        if type(frame_limit) is not int or frame_limit == 0 or frame_limit < -1:
            raise ValueError("FrameSource frame_limit must be -1 or a positive integer")
        if config.input_adapter != "rosbag-fixed-lag-v1":
            raise ValueError("OdinBagFixedLagFrameSource requires input_adapter=rosbag-fixed-lag-v1")
        from .data.rosbag.streaming import OdinBagFixedLagResultStream
        if config.preparation_profile is None:
            raise ValueError("Streaming scene config requires preparation_profile")
        self._stream = OdinBagFixedLagResultStream(
            config.rosbag_dir,
            preparation_profile=config.preparation_profile,
            calibration_path=config.calibration_path,
            queue_capacity=config.stream_queue_size,
            emitted_limit=frame_limit if frame_limit > 0 else None,
            non_formal=non_formal,
            confirm_raw_offset_time_seconds_from_scan_start=(
                config.confirm_raw_offset_time_seconds_from_scan_start
            ),
            confirm_base_from_lidar_identity=config.confirm_base_from_lidar_identity,
            require_clean_worktree=config.require_clean_worktree,
            write_through_dir=config.write_through_dir,
        )
        self._config = config
        self._started = False
        self._exhausted = False
        self._prepared_receipt: dict[str, object] | None = None
        self._emitted = 0
        self._source_modes: set[str] = set()
        self._runtime_metrics: dict[str, object] = {}

    def start_identity(self) -> dict[str, object]:
        identity = self._stream.start_identity()
        return {
            **deepcopy(identity),
            "source_policy_version": SOURCE_POLICY_VERSION,
            "frame_index_policy": "emitted-sequence-index-original-image-index-in-receipt-v1",
        }

    def set_runtime_metrics(self, metrics: Mapping[str, object]) -> None:
        setter = getattr(self._stream, "set_runtime_metrics", None)
        if setter is None:
            raise RuntimeError("Streaming producer does not expose runtime metrics")
        setter(metrics)
        self._runtime_metrics.update(deepcopy({
            key: value for key, value in metrics.items()
            if key in {"first_frame_wait_seconds", "mapping_wall_seconds", "mapping_fps"}
        }))

    def frames(self) -> Iterator[FrameInputs]:
        if self._started:
            raise RuntimeError("OdinBagFixedLagFrameSource.frames() is single-use")
        self._started = True
        first_frame_started = perf_counter()
        first_frame_recorded = False
        for result in self._stream.frames():
            source_mode = result.target.cloud.center_source_type
            self._source_modes.add("SLAM_WORLD" if source_mode.startswith("LIDAR_SLAM") else "LIDAR_RAW")
            if len(self._source_modes) > 1:
                raise ValueError("FrameSource cannot mix raw and SLAM LiDAR source families")
            target = result.target
            center_type = SourceType[target.cloud.center_source_type]
            fused_type = SourceType[target.cloud.fused_source_type]
            center_depth = np.asarray(result.center_depth, dtype=np.float32)
            fused_depth = np.asarray(result.fused_depth, dtype=np.float32)
            frame_intrinsics = target.intrinsics
            rgb = target.rgb.astype(np.float32) / 255.0
            if self._config.resize_width is not None:
                output_size = (self._config.resize_width, self._config.resize_height)
                scale_x = self._config.resize_width / target.intrinsics.width
                scale_y = self._config.resize_height / target.intrinsics.height
                frame_intrinsics = CameraIntrinsics(
                    self._config.resize_width, self._config.resize_height,
                    target.intrinsics.fx * scale_x, target.intrinsics.fy * scale_y,
                    target.intrinsics.cx * scale_x, target.intrinsics.cy * scale_y,
                )
                rgb = _resize_rgb(rgb, output_size)
                center_depth = _resize_nearest(center_depth, output_size, dtype=np.float32)
                fused_depth = _resize_nearest(fused_depth, output_size, dtype=np.float32)
                result_mask = _resize_nearest(
                    np.asarray(result.fused_mask, dtype=np.float32), output_size, dtype=np.float32,
                )
            else:
                result_mask = np.asarray(result.fused_mask, dtype=np.float32)
            center_valid = center_depth > 0
            fused_valid = result_mask.astype(bool) & (fused_depth > 0)
            fused_enabled = self._config.enabled_depth_sources == (
                target.cloud.center_source_type, target.cloud.fused_source_type,
            )
            if not fused_enabled:
                fused_valid = np.zeros_like(center_valid)
            mapping_depth = np.where(center_valid, center_depth, np.where(fused_valid, fused_depth, 0.0)).astype(np.float32)
            mapping_sources = np.full(mapping_depth.shape, INVALID_SOURCE_TYPE, dtype=np.uint8)
            mapping_sources[center_valid] = int(center_type)
            mapping_sources[~center_valid & fused_valid] = int(fused_type)
            mapping_confidence = np.zeros(mapping_depth.shape, dtype=np.float32)
            mapping_confidence[center_valid] = descriptor_for_type(center_type).default_confidence
            mapping_confidence[~center_valid & fused_valid] = descriptor_for_type(fused_type).default_confidence
            evidences = [DepthEvidence(
                center_type, center_depth, center_valid,
                np.where(center_valid, descriptor_for_type(center_type).default_confidence, 0.0).astype(np.float32),
            )]
            if fused_enabled:
                evidences.append(DepthEvidence(
                    fused_type, fused_depth, fused_valid,
                    np.where(fused_valid, descriptor_for_type(fused_type).default_confidence, 0.0).astype(np.float32),
                ))
            self._emitted += 1
            if not first_frame_recorded:
                self.set_runtime_metrics({
                    "first_frame_wait_seconds": perf_counter() - first_frame_started,
                })
                first_frame_recorded = True
            yield FrameInputs(
                index=self._emitted - 1,
                stem=target.stem,
                timestamp_ns=target.image_timestamp_ns,
                intrinsics=frame_intrinsics,
                pose=_matrix_to_pose(target.world_from_camera),
                rgb=rgb,
                mapping=MappingObservation(mapping_depth, mapping_sources, mapping_confidence),
                growth=GrowthInputs(tuple(evidences)),
            )
        self._exhausted = True

    def prepare_close(self) -> dict[str, object]:
        if not self._started or not self._exhausted:
            raise RuntimeError("OdinBagFixedLagFrameSource.prepare_close() requires a fully consumed iterator")
        if self._prepared_receipt is not None:
            raise RuntimeError("OdinBagFixedLagFrameSource is already prepared")
        receipt = self._stream.prepare_close()
        if receipt.get("fallback_used") is not False:
            raise ValueError("Streaming source receipt reports fallback")
        artifacts = receipt.get("write_through_artifacts_sha256")
        if self._config.write_through_dir is not None and (
            not isinstance(artifacts, dict) or not artifacts
        ):
            raise ValueError("Streaming write-through receipt lacks artifact hashes")
        self._prepared_receipt = {
            **self.start_identity(),
            **receipt,
            "complete": True,
            "emitted_centers": self._emitted,
            "source_families": sorted(self._source_modes),
            "runtime": {
                **(
                    receipt.get("runtime", {})
                    if isinstance(receipt.get("runtime"), dict) else {}
                ),
                **deepcopy(self._runtime_metrics),
                "first_frame_wait_seconds": self._runtime_metrics.get(
                    "first_frame_wait_seconds",
                    receipt.get("runtime", {}).get("first_frame_wait_seconds", "unmeasured")
                    if isinstance(receipt.get("runtime"), dict) else "unmeasured",
                ),
            },
        }
        return deepcopy(self._prepared_receipt)

    def commit(self) -> None:
        if self._prepared_receipt is None:
            raise RuntimeError("OdinBagFixedLagFrameSource.commit() requires prepare_close()")
        self._stream.commit()

    def rollback_commit(self, error: BaseException | str) -> None:
        if self._prepared_receipt is None:
            raise RuntimeError("OdinBagFixedLagFrameSource.rollback_commit() requires prepare_close()")
        self._stream.rollback_commit(error)

    def close(self) -> dict[str, object]:
        receipt = self.prepare_close()
        self.commit()
        return receipt

    def abort(self, error: BaseException | str) -> dict[str, object]:
        identity = self.start_identity()
        receipt = self._stream.abort(error)
        return {
            **identity,
            **receipt,
            "complete": False,
            "emitted_centers": self._emitted,
            "source_families": sorted(self._source_modes),
        }


def frame_source_for_config(
    config: SceneConfig,
    *,
    frame_limit: int = -1,
    non_formal: bool = False,
) -> FrameSource:
    if config.input_adapter == "prepared-scene":
        return PreparedSceneFrameSource(config, frame_limit=frame_limit)
    if config.input_adapter == "rosbag-fixed-lag-v1":
        return OdinBagFixedLagFrameSource(
            config,
            frame_limit=frame_limit,
            non_formal=non_formal,
        )
    raise ValueError(f"Unsupported scene input_adapter: {config.input_adapter}")
