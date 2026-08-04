"""Transient, bounded-RAM input cache between mapping and refinement.

The cache contains only frames that Stage 1 actually maps.  It is a retryable
handoff, not a published artifact: the mapping manifest records its canonical
identity, Stage 2 validates that identity before use, and the pipeline removes
the cache only after successful final publication.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from ..core_input import MappingFrame
from ..foundation.contracts import (
    CameraIntrinsics,
    DenseGeometryPrior,
    DensePriorDiagnostics,
    GrowthInputs,
    MappingObservation,
    Pose,
)
from ..foundation.identity_schema import validate_dataset_identity
from ..input.frame import FrameInputs, FrameMetadata
from ..input.identity import InputIdentities
from ..input.rosbag.transforms import quaternion_xyzw_to_matrix


STAGE2_CACHE_SCHEMA = "sage-stage2-input-cache-v1"
_MANIFEST_NAME = "manifest.json"
_FRAMES_DIR = "frames"
_PRIORS_DIR = "dense-priors"
_PRIORS_MANIFEST_NAME = "manifest.json"
_PRIORS_SCHEMA = "sage-stage2-dense-prior-cache-v1"


def default_stage2_cache_path(structure_output: Path) -> Path:
    """The cache lives beside, never inside, the published Stage 1 artifact."""
    structure = Path(structure_output).resolve()
    return structure.parent / ".stage2-input-cache"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantized_rgb(frame: MappingFrame) -> np.ndarray:
    rgb = np.asarray(frame.rgb, dtype=np.float32)
    quantized = np.rint(rgb * 255.0)
    if not np.allclose(quantized / 255.0, rgb, atol=1e-6):
        raise ValueError("Stage 2 cache requires 8-bit canonical RGB")
    return quantized.astype(np.uint8)


class Stage2InputCacheWriter:
    """Write mapping frames once, atomically publish only after Stage 1 EOF."""

    def __init__(self, destination: Path) -> None:
        self.destination = Path(destination).resolve()
        if self.destination.exists():
            raise ValueError(f"Stage 2 input cache already exists: {self.destination}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._staging = Path(tempfile.mkdtemp(
            prefix=f".{self.destination.name}.staging-", dir=self.destination.parent,
        ))
        (self._staging / _FRAMES_DIR).mkdir()
        self._records: list[dict[str, object]] = []
        self._written_indices: set[int] = set()
        self._published = False

    def write(self, frame: MappingFrame) -> None:
        if self._published:
            raise RuntimeError("Stage 2 input cache is already published")
        if frame.index in self._written_indices:
            raise ValueError(f"Stage 2 cache frame index is duplicated: {frame.index}")
        ordinal = len(self._records)
        filename = f"{ordinal:06d}-{frame.index:09d}.npz"
        path = self._staging / _FRAMES_DIR / filename
        pose = np.asarray((
            frame.pose.tx, frame.pose.ty, frame.pose.tz,
            frame.pose.qx, frame.pose.qy, frame.pose.qz, frame.pose.qw,
        ), dtype=np.float64)
        intrinsics = np.asarray((
            frame.intrinsics.width, frame.intrinsics.height,
            frame.intrinsics.fx, frame.intrinsics.fy,
            frame.intrinsics.cx, frame.intrinsics.cy,
        ), dtype=np.float64)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                frame_index=np.asarray(frame.index, dtype=np.int64),
                timestamp_ns=np.asarray(frame.timestamp_ns, dtype=np.int64),
                rgb=_quantized_rgb(frame),
                depth_m=np.asarray(frame.mapping.depth_m, dtype=np.float32),
                source_types=np.asarray(frame.mapping.source_types, dtype=np.uint8),
                confidences=np.asarray(frame.mapping.confidences, dtype=np.float32),
                intrinsics=intrinsics,
                pose=pose,
            )
        self._written_indices.add(frame.index)
        self._records.append({
            "frame_index": frame.index,
            "timestamp_ns": frame.timestamp_ns,
            "stem": frame.stem,
            "path": f"{_FRAMES_DIR}/{filename}",
            "sha256": _sha256_file(path),
        })

    @property
    def mapping_frame_count(self) -> int:
        return len(self._records)

    def finalize(self, identities: InputIdentities) -> Path:
        if self._published:
            raise RuntimeError("Stage 2 input cache is already published")
        if not self._records:
            raise ValueError("Stage 2 input cache received no mapping frames")
        manifest = {
            "schema_version": STAGE2_CACHE_SCHEMA,
            "input_identities": identities.payload(),
            "frames": self._records,
        }
        (self._staging / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        self._staging.rename(self.destination)
        self._published = True
        return self.destination

    def abort(self) -> None:
        if self._published:
            shutil.rmtree(self.destination, ignore_errors=True)
        else:
            shutil.rmtree(self._staging, ignore_errors=True)


class Stage2InputCache:
    """Verified on-disk mapping frames with a bounded CPU LRU."""

    def __init__(self, root: Path, *, cpu_lru_frames: int = 4) -> None:
        self.root = Path(root).resolve()
        if type(cpu_lru_frames) is not int or cpu_lru_frames < 1:
            raise ValueError("Stage 2 cache CPU LRU capacity must be positive")
        self._capacity = cpu_lru_frames
        self._manifest = self._load_manifest()
        self._records = {
            int(record["frame_index"]): dict(record)
            for record in self._manifest["frames"]
        }
        self._lru: OrderedDict[int, MappingFrame] = OrderedDict()

    def _load_manifest(self) -> dict[str, object]:
        path = self.root / _MANIFEST_NAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read Stage 2 input cache manifest: {path}") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "input_identities", "frames"}
            or payload["schema_version"] != STAGE2_CACHE_SCHEMA
            or not isinstance(payload["frames"], list)
            or not payload["frames"]
        ):
            raise ValueError("Stage 2 input cache manifest is invalid")
        InputIdentities.from_payload(payload["input_identities"])
        expected = {"frame_index", "timestamp_ns", "stem", "path", "sha256"}
        previous = -1
        for record in payload["frames"]:
            if (
                not isinstance(record, dict)
                or set(record) != expected
                or type(record["frame_index"]) is not int
                or record["frame_index"] <= previous
                or type(record["timestamp_ns"]) is not int
                or not isinstance(record["stem"], str)
                or not isinstance(record["path"], str)
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64
            ):
                raise ValueError("Stage 2 input cache frame manifest is invalid")
            previous = record["frame_index"]
        return payload

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(self._records)

    @property
    def input_identities(self) -> InputIdentities:
        return InputIdentities.from_payload(self._manifest["input_identities"])

    def validate_checkpoint_identity(self, checkpoint_identity: object) -> None:
        validate_dataset_identity(checkpoint_identity, self.input_identities)

    def load(self, frame_index: int) -> MappingFrame:
        cached = self._lru.get(frame_index)
        if cached is not None:
            self._lru.move_to_end(frame_index)
            return cached
        try:
            record = self._records[frame_index]
        except KeyError as exc:
            raise ValueError(f"Stage 2 cache has no mapping frame {frame_index}") from exc
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Stage 2 cache frame path is invalid")
        path = self.root / relative
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Stage 2 cache frame hash does not match: {relative}")
        frame = self._decode(path, record)
        self._lru[frame_index] = frame
        while len(self._lru) > self._capacity:
            self._lru.popitem(last=False)
        return frame

    @staticmethod
    def _decode(path: Path, record: dict[str, object]) -> MappingFrame:
        try:
            with np.load(path, allow_pickle=False) as payload:
                expected = {
                    "frame_index", "timestamp_ns", "rgb", "depth_m", "source_types",
                    "confidences", "intrinsics", "pose",
                }
                if set(payload.files) != expected:
                    raise ValueError("Stage 2 cache frame payload fields are invalid")
                index = int(payload["frame_index"])
                timestamp = int(payload["timestamp_ns"])
                rgb_u8 = np.array(payload["rgb"], dtype=np.uint8, copy=True)
                depth = np.array(payload["depth_m"], dtype=np.float32, copy=True)
                sources = np.array(payload["source_types"], dtype=np.uint8, copy=True)
                confidences = np.array(payload["confidences"], dtype=np.float32, copy=True)
                intrinsics = np.array(payload["intrinsics"], dtype=np.float64, copy=True)
                pose_values = np.array(payload["pose"], dtype=np.float64, copy=True)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode Stage 2 cache frame: {path}") from exc
        if index != record["frame_index"] or timestamp != record["timestamp_ns"]:
            raise ValueError("Stage 2 cache frame does not match its manifest")
        if (
            rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3 or depth.shape != rgb_u8.shape[:2]
            or sources.shape != depth.shape or confidences.shape != depth.shape
            or intrinsics.shape != (6,) or pose_values.shape != (7,)
        ):
            raise ValueError("Stage 2 cache frame dimensions are invalid")
        camera = CameraIntrinsics(
            int(intrinsics[0]), int(intrinsics[1]),
            float(intrinsics[2]), float(intrinsics[3]),
            float(intrinsics[4]), float(intrinsics[5]),
        )
        if (camera.height, camera.width) != depth.shape:
            raise ValueError("Stage 2 cache intrinsics do not match frame dimensions")
        pose = Pose(*[float(value) for value in pose_values])
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = quaternion_xyzw_to_matrix(pose_values[3:])
        matrix[:3, 3] = pose_values[:3]
        K = np.asarray([
            [camera.fx, 0.0, camera.cx],
            [0.0, camera.fy, camera.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        canonical = FrameInputs(
            frame_index=index,
            timestamp_ns=timestamp,
            image_rgb=rgb_u8.astype(np.float32) / 255.0,
            intrinsics=K,
            reference_from_camera=matrix,
            lidar_center=None,
            lidar_fused=None,
            metadata=FrameMetadata({}, {}, {}),
        )
        return MappingFrame(
            canonical=canonical,
            mapping=MappingObservation(depth, sources, confidences),
            growth=GrowthInputs(()),
            intrinsics=camera,
            pose=pose,
            rgb=canonical.image_rgb.copy(),
        )


class Stage2DensePriorCacheWriter:
    """Persist Stage 2 dense targets without retaining every image in RAM."""

    def __init__(self, input_cache: Stage2InputCache) -> None:
        self._input_cache = input_cache
        self.destination = input_cache.root / _PRIORS_DIR
        if self.destination.exists():
            raise ValueError(f"Stage 2 dense-prior cache already exists: {self.destination}")
        self._staging = Path(tempfile.mkdtemp(
            prefix=f".{_PRIORS_DIR}.staging-", dir=input_cache.root,
        ))
        self._records: list[dict[str, object]] = []
        self._written_indices: set[int] = set()
        self._published = False

    def write(self, frame_index: int, prior: DenseGeometryPrior) -> None:
        if self._published:
            raise RuntimeError("Stage 2 dense-prior cache is already published")
        if frame_index in self._written_indices:
            raise ValueError(f"Stage 2 dense-prior frame is duplicated: {frame_index}")
        if not isinstance(prior, DenseGeometryPrior):
            raise ValueError("Stage 2 dense-prior cache requires DenseGeometryPrior")
        filename = f"{len(self._records):06d}-{frame_index:09d}.npz"
        path = self._staging / filename
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                raw_depth_m=np.asarray(prior.raw_depth_m, dtype=np.float32),
                aligned_depth_m=np.asarray(prior.aligned_depth_m, dtype=np.float32),
                valid_mask=np.asarray(prior.valid_mask, dtype=np.bool_),
                confidence=np.asarray(prior.confidence, dtype=np.float32),
                scale_grid=np.asarray(prior.scale_grid, dtype=np.float32),
            )
        self._written_indices.add(frame_index)
        self._records.append({
            "frame_index": frame_index,
            "path": filename,
            "sha256": _sha256_file(path),
            "diagnostics": asdict(prior.diagnostics),
        })

    def finalize(self) -> Path:
        if self._published:
            raise RuntimeError("Stage 2 dense-prior cache is already published")
        if tuple(record["frame_index"] for record in self._records) != self._input_cache.frame_indices:
            raise ValueError("Stage 2 dense-prior cache frame set does not match input cache")
        (self._staging / _PRIORS_MANIFEST_NAME).write_text(
            json.dumps({
                "schema_version": _PRIORS_SCHEMA,
                "input_identities": self._input_cache.input_identities.payload(),
                "frames": self._records,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._staging.rename(self.destination)
        self._published = True
        return self.destination

    def abort(self) -> None:
        if self._published:
            shutil.rmtree(self.destination, ignore_errors=True)
        else:
            shutil.rmtree(self._staging, ignore_errors=True)


class Stage2DensePriorCache:
    """Verified dense priors with a bounded CPU LRU."""

    def __init__(self, input_cache: Stage2InputCache, *, cpu_lru_frames: int = 4) -> None:
        if type(cpu_lru_frames) is not int or cpu_lru_frames < 1:
            raise ValueError("Stage 2 dense-prior CPU LRU capacity must be positive")
        self._input_cache = input_cache
        self.root = input_cache.root / _PRIORS_DIR
        self._capacity = cpu_lru_frames
        self._records = self._load_manifest()
        self._lru: OrderedDict[int, DenseGeometryPrior] = OrderedDict()

    def _load_manifest(self) -> dict[int, dict[str, object]]:
        path = self.root / _PRIORS_MANIFEST_NAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read Stage 2 dense-prior manifest: {path}") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "input_identities", "frames"}
            or payload["schema_version"] != _PRIORS_SCHEMA
            or InputIdentities.from_payload(payload["input_identities"])
            != self._input_cache.input_identities
            or not isinstance(payload["frames"], list)
        ):
            raise ValueError("Stage 2 dense-prior manifest is invalid")
        records: dict[int, dict[str, object]] = {}
        expected = {"frame_index", "path", "sha256", "diagnostics"}
        for record in payload["frames"]:
            if (
                not isinstance(record, dict)
                or set(record) != expected
                or type(record["frame_index"]) is not int
                or record["frame_index"] in records
                or not isinstance(record["path"], str)
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64
                or not isinstance(record["diagnostics"], dict)
            ):
                raise ValueError("Stage 2 dense-prior frame manifest is invalid")
            records[record["frame_index"]] = dict(record)
        if tuple(records) != self._input_cache.frame_indices:
            raise ValueError("Stage 2 dense-prior manifest frame set is invalid")
        return records

    def load(self, frame_index: int) -> DenseGeometryPrior:
        cached = self._lru.get(frame_index)
        if cached is not None:
            self._lru.move_to_end(frame_index)
            return cached
        try:
            record = self._records[frame_index]
        except KeyError as exc:
            raise ValueError(f"Stage 2 dense-prior cache has no frame {frame_index}") from exc
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Stage 2 dense-prior frame path is invalid")
        path = self.root / relative
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"Stage 2 dense-prior hash does not match: {relative}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                expected = {
                    "raw_depth_m", "aligned_depth_m", "valid_mask", "confidence", "scale_grid",
                }
                if set(payload.files) != expected:
                    raise ValueError("Stage 2 dense-prior payload fields are invalid")
                prior = DenseGeometryPrior(
                    np.array(payload["raw_depth_m"], dtype=np.float32, copy=True),
                    np.array(payload["aligned_depth_m"], dtype=np.float32, copy=True),
                    np.array(payload["valid_mask"], dtype=np.bool_, copy=True),
                    np.array(payload["confidence"], dtype=np.float32, copy=True),
                    np.array(payload["scale_grid"], dtype=np.float32, copy=True),
                    DensePriorDiagnostics(**record["diagnostics"]),
                )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Cannot decode Stage 2 dense-prior frame: {path}") from exc
        self._lru[frame_index] = prior
        while len(self._lru) > self._capacity:
            self._lru.popitem(last=False)
        return prior


def remove_stage2_cache(root: Path) -> None:
    """Delete only the exact transient cache after complete-pipeline success."""
    cache = Path(root).resolve()
    if cache.name != ".stage2-input-cache":
        raise ValueError("Refusing to delete a non-transient Stage 2 cache path")
    if cache.exists():
        if cache.is_symlink() or not cache.is_dir():
            raise ValueError("Stage 2 cache path is not a directory")
        shutil.rmtree(cache)


def remove_stage2_dense_prior_cache(input_cache: Stage2InputCache) -> None:
    """Remove an uncommitted dense-prior retry residue from one input cache."""
    cache = input_cache.root / _PRIORS_DIR
    if cache.exists():
        if cache.is_symlink() or not cache.is_dir():
            raise ValueError("Stage 2 dense-prior cache path is not a directory")
        shutil.rmtree(cache)


__all__ = [
    "STAGE2_CACHE_SCHEMA",
    "Stage2DensePriorCache",
    "Stage2DensePriorCacheWriter",
    "Stage2InputCache",
    "Stage2InputCacheWriter",
    "default_stage2_cache_path",
    "remove_stage2_dense_prior_cache",
    "remove_stage2_cache",
]
