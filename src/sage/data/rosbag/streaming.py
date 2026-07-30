from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import resource
import time
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread
from typing import Callable, Generic, Iterable, Mapping, TypeVar

from .writer import sha256_file


T = TypeVar("T")


@dataclass(frozen=True)
class _StreamError:
    error: BaseException


class BoundedResultStream(Generic[T]):
    """Expose one ordered producer through a bounded, lossless iterator.

    The producer is deliberately kept behind this small interface.  It may do sensor
    decoding and projection, while the consumer controls how quickly the producer can
    advance by filling the bounded queue.  A normal close is only valid after the
    iterator has consumed the producer's completion sentinel; early termination must
    use :meth:`abort` so an incomplete receipt cannot be mistaken for a complete run.
    """

    def __init__(
        self,
        producer: Callable[[], Iterable[T]],
        *,
        identity: Mapping[str, object],
        queue_capacity: int = 1,
    ) -> None:
        if not callable(producer):
            raise ValueError("BoundedResultStream producer must be callable")
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("BoundedResultStream queue_capacity must be a positive integer")
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("BoundedResultStream identity must be a non-empty object")
        self._producer = producer
        self._identity = deepcopy(dict(identity))
        self._queue_capacity = queue_capacity
        self._queue: Queue[object] = Queue(maxsize=queue_capacity)
        self._stop = Event()
        self._thread: Thread | None = None
        self._started = False
        self._exhausted = False
        self._aborted = False
        self._error: BaseException | None = None
        self._items_emitted = 0
        self._max_queue_depth = 0
        self._producer_stall_seconds = 0.0
        self._consumer_wait_seconds = 0.0

    def start_identity(self) -> dict[str, object]:
        if self._aborted:
            raise RuntimeError("BoundedResultStream has already been aborted")
        return deepcopy(self._identity)

    @property
    def items_emitted(self) -> int:
        return self._items_emitted

    def _offer(self, item: object) -> bool:
        stalled_at = None
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                if stalled_at is not None:
                    self._producer_stall_seconds += time.monotonic() - stalled_at
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
                return True
            except Full:
                if stalled_at is None:
                    stalled_at = time.monotonic()
                continue
        return False

    def _run_producer(self) -> None:
        try:
            for item in self._producer():
                if not self._offer(item):
                    return
            self._offer(None)
        except BaseException as exc:  # surfaced on the consumer thread
            self._error = exc
            self._offer(_StreamError(exc))

    def frames(self):
        if self._started:
            raise RuntimeError("BoundedResultStream.frames() is single-use")
        if self._aborted:
            raise RuntimeError("BoundedResultStream has already been aborted")
        self._started = True
        self._thread = Thread(target=self._run_producer, name="bounded-result-producer", daemon=True)
        self._thread.start()
        while True:
            waiting_at = time.monotonic()
            item = self._queue.get()
            self._consumer_wait_seconds += time.monotonic() - waiting_at
            if item is None:
                self._exhausted = True
                break
            if isinstance(item, _StreamError):
                self._error = item.error
                raise item.error
            self._items_emitted += 1
            yield item

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("BoundedResultStream producer did not stop")

    def close(self) -> dict[str, object]:
        if not self._started:
            raise RuntimeError("BoundedResultStream must be iterated before close()")
        if not self._exhausted:
            raise RuntimeError("BoundedResultStream.close() requires a fully consumed iterator")
        self._join()
        if self._error is not None:
            raise RuntimeError("BoundedResultStream producer failed") from self._error
        return {
            **deepcopy(self._identity),
            "complete": True,
            "aborted": False,
            "items_emitted": self._items_emitted,
            "queue_capacity": self._queue_capacity,
            "max_queue_depth": self._max_queue_depth,
            "producer_stall_seconds": self._producer_stall_seconds,
            "consumer_wait_seconds": self._consumer_wait_seconds,
            "peak_cpu_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "error": None,
        }

    def abort(self, error: BaseException | str) -> dict[str, object]:
        if isinstance(error, BaseException):
            message = f"{type(error).__name__}: {error}"
        else:
            message = str(error)
        if not message:
            raise ValueError("BoundedResultStream abort error must be non-empty")
        self._aborted = True
        self._stop.set()
        self._join()
        return {
            **deepcopy(self._identity),
            "complete": False,
            "aborted": True,
            "items_emitted": self._items_emitted,
            "queue_capacity": self._queue_capacity,
            "max_queue_depth": self._max_queue_depth,
            "producer_stall_seconds": self._producer_stall_seconds,
            "consumer_wait_seconds": self._consumer_wait_seconds,
            "peak_cpu_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "error": message,
        }


class OdinBagFixedLagResultStream:
    """Source-aware fixed-lag centered-five results for a finite Odin1 bag."""

    def __init__(
        self,
        rosbag_dir,
        *,
        preparation_profile: str = "odin1-slam-world-native-v1",
        calibration_path=None,
        queue_capacity: int = 1,
        non_formal: bool = False,
        image_start: int = 0,
        image_limit: int | None = None,
        emitted_limit: int | None = None,
        confirm_raw_offset_time_seconds_from_scan_start: bool = False,
        confirm_base_from_lidar_identity: bool = False,
        write_through_dir=None,
    ) -> None:
        from .producer import _bag_shards, iter_projected_centers, prepare_streaming_bag_runtime

        if image_start < 0 or (image_limit is not None and image_limit < 5):
            raise ValueError("Diagnostic image_start must be non-negative and image_limit must be at least five")
        if emitted_limit is not None and emitted_limit < 1:
            raise ValueError("emitted_limit must be positive when configured")
        calibration = Path(calibration_path).resolve() if calibration_path is not None else Path(rosbag_dir).resolve() / "cam_in_ex.txt"
        bag_root = Path(rosbag_dir).resolve()
        metadata = bag_root / "metadata.yaml"
        shards = _bag_shards(bag_root, metadata)
        pre_input_files = {
            "metadata.yaml": metadata,
            **{
                f"rosbag/{path.name}": path
                for path in shards
            },
            "cam_in_ex.txt": calibration,
        }
        pre_input_hashes = {
            label: sha256_file(path) for label, path in sorted(pre_input_files.items())
        }
        effective_image_limit = image_limit
        planned_image_limit = image_limit
        if emitted_limit is not None and image_start == 0 and image_limit is None:
            planned_image_limit = emitted_limit + 4
        self._runtime = prepare_streaming_bag_runtime(
            rosbag_dir,
            preparation_profile=preparation_profile,
            calibration_path=calibration_path,
            non_formal=bool(non_formal or image_start != 0 or image_limit is not None),
            confirm_raw_offset_time_seconds_from_scan_start=confirm_raw_offset_time_seconds_from_scan_start,
            confirm_base_from_lidar_identity=confirm_base_from_lidar_identity,
        )
        self._iter_projected_centers = iter_projected_centers
        self._receipt = {
            "receipt_schema": "fixed-lag-stream-v1",
            "selected_image_slots": 0,
            "finalized_slots": 0,
            "rejected_slots": 0,
            "edge_slots_not_emitted": 0,
            "source_mode": self._runtime.profile.source.mode,
            "fallback_used": False,
            "raw_offset_time_confirmed": bool(confirm_raw_offset_time_seconds_from_scan_start),
            "base_from_lidar_identity_confirmed": bool(confirm_base_from_lidar_identity),
        }
        self._rejected_centers = []
        self._frame_chain = bytes(32)
        self._emitted_frame_records: list[dict[str, object]] = []
        self._runtime_metrics: dict[str, object] = {}
        identity = {
            "adapter": "rosbag-fixed-lag-v1",
            "receipt_schema": "fixed-lag-stream-v1",
            "preparation_profile": self._runtime.profile.name,
            "source": self._runtime.profile.manifest_contract()["source"],
            "input_files_sha256": {
                label: sha256_file(path)
                for label, path in sorted({**self._runtime.input_files, "cam_in_ex.txt": self._runtime.calibration}.items())
            },
            "producer_identity": deepcopy(self._runtime.producer_identity),
            "queue_capacity": queue_capacity,
            "emitted_limit": emitted_limit if emitted_limit is not None else -1,
            "image_selection": {
                "start": image_start,
                "limit": planned_image_limit if planned_image_limit is not None else -1,
            },
        }
        post_input_files = {**self._runtime.input_files, "cam_in_ex.txt": self._runtime.calibration}
        try:
            post_input_hashes = {
                label: sha256_file(path) for label, path in sorted(post_input_files.items())
            }
        except BaseException:
            self._runtime.close()
            raise
        if pre_input_hashes != post_input_hashes:
            self._runtime.close()
            raise ValueError("ROSBAG inputs changed while the streaming index was built")
        identity["input_files_sha256"] = post_input_hashes
        self._writer = None
        self._write_through_dir = Path(write_through_dir).resolve() if write_through_dir is not None else None
        self._write_through_non_formal = bool(
            non_formal or image_start != 0 or image_limit is not None
            or emitted_limit is not None
        )
        self._prepared_receipt: dict[str, object] | None = None
        self._runtime_closed = False

        def produce():
            for result in self._iter_projected_centers(
                self._runtime,
                image_start=image_start,
                image_limit=effective_image_limit,
                emitted_limit=emitted_limit,
                receipt=self._receipt,
                rejected_centers=self._rejected_centers,
            ):
                target = result.target
                if self._writer is not None:
                    self._writer.write(result)
                    assert self._writer.last_frame_hashes is not None
                    frame_hashes = deepcopy(self._writer.last_frame_hashes)
                else:
                    frame_hashes = {
                        "image": hashlib.sha256(target.rgb.tobytes(order="C")).hexdigest(),
                        "center_depth": hashlib.sha256(result.center_depth.tobytes(order="C")).hexdigest(),
                        "fused5_depth": hashlib.sha256(result.fused_depth.tobytes(order="C")).hexdigest(),
                        "fused5_mask": hashlib.sha256(result.fused_mask.tobytes(order="C")).hexdigest(),
                    }
                self._frame_chain = hashlib.sha256(
                    self._frame_chain
                    + target.stem.encode("utf-8")
                    + json.dumps(frame_hashes, sort_keys=True).encode("utf-8")
                ).digest()
                self._emitted_frame_records.append({
                    "frame_id": target.stem,
                    "image_index": target.image_index,
                    "image_timestamp_ns": target.image_timestamp_ns,
                    "cloud_id": target.cloud.cloud_id,
                    "cloud_timestamp_ns": target.cloud.cloud_timestamp_ns,
                    "window_image_indices": [frame.image_index for frame in result.source_frames],
                    "window_cloud_ids": [frame.cloud.cloud_id for frame in result.source_frames],
                    "window_normalization_receipts": [
                        {
                            "image_index": frame.image_index,
                            "cloud_id": frame.cloud.cloud_id,
                            "receipt": deepcopy(frame.cloud.normalization_receipt),
                        }
                        for frame in result.source_frames
                    ],
                    "center_source_type": target.cloud.center_source_type,
                    "fused_source_type": target.cloud.fused_source_type,
                    "normalization_receipt": deepcopy(target.cloud.normalization_receipt),
                    "frame_hashes": frame_hashes,
                })
                yield result

        self._stream = BoundedResultStream(
            produce,
            identity=identity,
            queue_capacity=queue_capacity,
        )

    @property
    def profile(self):
        return self._runtime.profile

    def start_identity(self) -> dict[str, object]:
        return self._stream.start_identity()

    def set_runtime_metrics(self, metrics: Mapping[str, object]) -> None:
        if not isinstance(metrics, Mapping):
            raise ValueError("Streaming runtime metrics must be an object")
        self._runtime_metrics.update(deepcopy(dict(metrics)))

    def frames(self):
        if self._write_through_dir is not None and self._writer is None:
            from .writer import PreparedSceneWriter

            self._writer = PreparedSceneWriter(
                self._write_through_dir,
                profile=self._runtime.profile,
                input_files={**self._runtime.input_files, "cam_in_ex.txt": self._runtime.calibration},
                producer_identity=self._runtime.producer_identity,
                non_formal=self._write_through_non_formal,
            )
        yield from self._stream.frames()

    def _close_runtime(self) -> None:
        if not self._runtime_closed:
            self._runtime.close()
            self._runtime_closed = True

    def prepare_close(self) -> dict[str, object]:
        if self._prepared_receipt is not None:
            raise RuntimeError("OdinBagFixedLagResultStream is already prepared")
        try:
            receipt = self._stream.close()
            complete_receipt = {
                **receipt,
                **deepcopy(self._receipt),
                "emitted_centers": receipt["items_emitted"],
                "rejected_centers_count": len(self._rejected_centers),
                "source_families": [self._runtime.profile.source.mode],
                "write_through_enabled": self._writer is not None,
                "emitted_frame_records": deepcopy(self._emitted_frame_records),
                "frame_hash_chain_sha256": (
                    self._writer.chain.hex() if self._writer is not None else self._frame_chain.hex()
                ),
                **deepcopy(self._runtime_metrics),
            }
            complete_receipt["image_selection"] = {
                "start": self._stream.start_identity()["image_selection"]["start"],
                "limit": int(self._receipt["selected_image_slots"]),
            }
            complete_receipt["rejected_centers"] = [
                {
                    "target_image_index": item.target_image_index,
                    "target_stem": item.target_stem,
                    "target_timestamp_ns": item.target_timestamp_ns,
                    "dependency_indices": list(item.dependency_indices),
                    "root_reasons": list(item.root_reasons),
                }
                for item in self._rejected_centers
            ]
            write_through_manifest = None
            complete_receipt["write_through_artifacts_sha256"] = None
            if self._writer is not None:
                complete_receipt["write_through_manifest"] = str(
                    self._write_through_dir / "sage_prepared_scene_manifest.json"
                )
                try:
                    write_through_manifest = str(self._writer.prepare(
                        self._rejected_centers, production_receipt=complete_receipt,
                    ))
                    manifest = json.loads(
                        (self._writer.staging / "sage_prepared_scene_manifest.json").read_text(
                            encoding="utf-8",
                        )
                    )
                    complete_receipt = deepcopy(manifest["production_receipt"])
                except BaseException as exc:
                    self._writer.abort(exc)
                    raise
            complete_receipt["write_through_manifest"] = write_through_manifest
            self._prepared_receipt = deepcopy(complete_receipt)
            return deepcopy(complete_receipt)
        finally:
            self._close_runtime()

    def commit(self) -> None:
        if self._prepared_receipt is None:
            raise RuntimeError("OdinBagFixedLagResultStream.commit() requires prepare_close()")
        if self._writer is not None:
            self._writer.commit()

    def rollback_commit(self, error: BaseException | str) -> None:
        if self._prepared_receipt is None:
            raise RuntimeError("OdinBagFixedLagResultStream.rollback_commit() requires prepare_close()")
        if self._writer is not None:
            self._writer.rollback_commit(error)

    def close(self) -> dict[str, object]:
        receipt = self.prepare_close()
        self.commit()
        return receipt

    def abort(self, error: BaseException | str) -> dict[str, object]:
        try:
            receipt = self._stream.abort(error)
            if self._writer is not None:
                self._writer.abort(error)
            return {
                **receipt,
                **deepcopy(self._receipt),
                "complete": False,
                "aborted": True,
                "error": receipt["error"],
                "emitted_centers": self._stream.items_emitted,
                "rejected_centers_count": len(self._rejected_centers),
                "source_families": [self._runtime.profile.source.mode],
                "emitted_frame_records": deepcopy(self._emitted_frame_records),
                "frame_hash_chain_sha256": self._frame_chain.hex(),
                "write_through_manifest": None,
                "write_through_artifacts_sha256": None,
                "write_through_enabled": self._writer is not None,
                "rejected_centers": [
                    {
                        "target_image_index": item.target_image_index,
                        "target_stem": item.target_stem,
                        "target_timestamp_ns": item.target_timestamp_ns,
                        "dependency_indices": list(item.dependency_indices),
                        "root_reasons": list(item.root_reasons),
                    }
                    for item in self._rejected_centers
                ],
            }
        finally:
            self._close_runtime()
