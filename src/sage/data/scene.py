from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ..foundation.config import SceneConfig
from ..foundation.contracts import CameraIntrinsics, DepthEvidence, FrameInputs, GrowthInputs, INVALID_SOURCE_TYPE, MappingObservation, Pose, SourceType
from ..foundation.hashing import sha256_file as _sha256_file
from ..foundation.prepared_scene_contract import PREPARED_SCENE_SCHEMA, validate_source_contract
from ..foundation.receipt_contract import (
    MATERIALIZED_RECEIPT_SCHEMA,
    STREAM_RECEIPT_SCHEMA,
    validate_completion_receipt,
)
from ..foundation.source_policy import descriptor_for_type


PREPARED_SCENE_MANIFEST_NAME = "sage_prepared_scene_manifest.json"


def _read_intrinsics(path: Path) -> CameraIntrinsics:
    if not path.is_file():
        raise ValueError(f"Missing camera intrinsics: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Unsupported camera line: {line}")
        model, width, height = parts[1], int(parts[2]), int(parts[3])
        values = [float(item) for item in parts[4:]]
        if model == "PINHOLE" and len(values) >= 4:
            fx, fy, cx, cy = values[:4]
        elif model == "SIMPLE_PINHOLE" and len(values) >= 3:
            fx = fy = values[0]
            cx, cy = values[1:3]
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        return CameraIntrinsics(width, height, fx, fy, cx, cy)
    raise ValueError(f"Missing camera record: {path}")


def _rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing Prepared Scene table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        frame_id = row.get("frame_id")
        if not frame_id or frame_id in result:
            raise ValueError(f"Frame table has an invalid or duplicate frame_id: {frame_id}")
        result[frame_id] = row
    return result


def _resize_nearest(array: np.ndarray, size: tuple[int, int], *, dtype: np.dtype) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype == np.bool_:
        source = source.astype(np.uint8)
    image = Image.fromarray(source.astype(np.float32, copy=False), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=dtype).copy()


def _resize_rgb(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _validate_frame_hash_chain(
    prepared_scene_dir: Path,
    declared_artifacts: dict[str, object],
    associations: dict[str, dict[str, str]],
    expected_final: object,
) -> None:
    chain_path = prepared_scene_dir / "audit" / "frame_hash_chain.json"
    try:
        records = json.loads(chain_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Prepared Scene v2 frame hash chain is not valid JSON") from exc
    if not isinstance(records, list) or len(records) != len(associations):
        raise ValueError("Prepared Scene v2 frame hash chain count is invalid")
    chain = bytes(32)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"frame_id", "chain_sha256"}:
            raise ValueError("Prepared Scene v2 frame hash chain record is invalid")
        stem = record["frame_id"]
        if not isinstance(stem, str) or not stem or Path(stem).name != stem or stem in seen or stem not in associations:
            raise ValueError("Prepared Scene v2 frame hash chain frame identity is invalid")
        chain_digest = record["chain_sha256"]
        if not isinstance(chain_digest, str) or len(chain_digest) != 64 or any(
            character not in "0123456789abcdef" for character in chain_digest
        ):
            raise ValueError("Prepared Scene v2 frame hash chain digest is invalid")
        seen.add(stem)
        frame_hashes = {
            "image": declared_artifacts.get(f"scene/images/{stem}.png"),
            "center_depth": declared_artifacts.get(f"center_depth/{stem}.npy"),
            "fused5_depth": declared_artifacts.get(f"fused5_depth/{stem}.npy"),
            "fused5_mask": declared_artifacts.get(f"fused5_mask/{stem}.npy"),
        }
        if any(not isinstance(value, str) for value in frame_hashes.values()):
            raise ValueError("Prepared Scene v2 frame hash chain is missing frame artifacts")
        chain = hashlib.sha256(
            chain + stem.encode("utf-8") + json.dumps(frame_hashes, sort_keys=True).encode("utf-8")
        ).digest()
        if chain.hex() != chain_digest:
            raise ValueError("Prepared Scene v2 frame hash chain record does not match artifacts")
    if not isinstance(expected_final, str) or expected_final != chain.hex():
        raise ValueError("Prepared Scene v2 frame hash chain final digest is invalid")


class PreparedScene:
    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        candidates = (
            self.config.scene_dir / PREPARED_SCENE_MANIFEST_NAME,
            self.config.prepared_scene_dir / PREPARED_SCENE_MANIFEST_NAME,
        )
        existing = tuple(dict.fromkeys(path for path in candidates if path.is_file()))
        if len(existing) > 1:
            raise ValueError("Prepared Scene manifest is ambiguous across configured roots")
        self.prepared_manifest_path = existing[0] if existing else candidates[0]
        self.transform_contract_path = self.config.prepared_scene_dir / "calib" / "tf_chain.json"

    @classmethod
    def from_root(cls, root: Path) -> PreparedScene:
        """Construct the canonical full-contract validator for one Prepared Scene root."""
        prepared_root = Path(root).resolve()
        manifest_path = prepared_root / PREPARED_SCENE_MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Prepared Scene manifest is unreadable: {manifest_path}") from exc
        source = manifest.get("source") if isinstance(manifest, dict) else None
        if not isinstance(source, dict):
            raise ValueError("Prepared Scene manifest source contract is invalid")
        center_source = source.get("center_source_type")
        fused_source = source.get("fused_source_type")
        if not isinstance(center_source, str) or not isinstance(fused_source, str):
            raise ValueError("Prepared Scene manifest source types are invalid")
        return cls(SceneConfig(
            scene_dir=(prepared_root / "scene").resolve(),
            prepared_scene_dir=prepared_root,
            center_depth_dir=(prepared_root / "center_depth").resolve(),
            fused5_depth_dir=(prepared_root / "fused5_depth").resolve(),
            fused5_mask_dir=(prepared_root / "fused5_mask").resolve(),
            enabled_depth_sources=(center_source, fused_source),
        ))

    def _ordered_associations(self, limit: int) -> tuple[dict[str, dict[str, str]], list[tuple[str, dict[str, str]]]]:
        associations = _rows_by_id(self.config.prepared_scene_dir / "associations" / "frame_associations.csv")
        try:
            ordered = sorted(associations.items(), key=lambda item: int(item[1]["image_timestamp_ns"]))
        except (KeyError, TypeError, ValueError) as exc:
            path = self.config.prepared_scene_dir / "associations" / "frame_associations.csv"
            raise ValueError(f"Prepared Scene associations contain an invalid timestamp: {path}") from exc
        if limit > 0:
            ordered = ordered[:limit]
        return associations, ordered

    def _frame_input_paths(self, stem: str) -> tuple[tuple[str, Path], ...]:
        return (
            (f"rgb/{stem}", self.config.scene_dir / "images" / f"{stem}.png"),
            (f"center_depth/{stem}", self.config.center_depth_path / f"{stem}.npy"),
            (f"fused5_depth/{stem}", self.config.fused5_depth_dir / f"{stem}.npy"),
            (f"fused5_mask/{stem}", self.config.fused5_mask_dir / f"{stem}.npy"),
        )

    def input_files(self, *, limit: int = -1) -> tuple[tuple[str, Path], ...]:
        _, ordered = self._ordered_associations(limit)
        files = [
            ("intrinsics", self.config.scene_dir / "sparse" / "0" / "cameras.txt"),
            ("associations", self.config.prepared_scene_dir / "associations" / "frame_associations.csv"),
            ("poses", self.config.prepared_scene_dir / "poses" / "camera_poses.csv"),
        ]
        for stem, _ in ordered:
            files.extend(self._frame_input_paths(stem))
        return tuple(files)

    def frames(self, *, limit: int = -1) -> Iterator[FrameInputs]:
        manifest = self._validate_contract()
        source = manifest["source"]
        assert isinstance(source, dict)
        center_source_type = SourceType[source["center_source_type"]]
        fused_source_type = SourceType[source["fused_source_type"]]
        fused5_enabled = descriptor_for_type(fused_source_type).name in self.config.enabled_depth_sources
        intrinsics = _read_intrinsics(self.config.scene_dir / "sparse" / "0" / "cameras.txt")
        _, ordered = self._ordered_associations(limit)
        poses = _rows_by_id(self.config.prepared_scene_dir / "poses" / "camera_poses.csv")
        for index, (stem, association) in enumerate(ordered):
            pose_row = poses.get(stem)
            if pose_row is None:
                raise ValueError(f"Missing finalized pose for frame: {stem}")
            if int(pose_row["image_timestamp_ns"]) != int(association["image_timestamp_ns"]):
                raise ValueError(f"Pose and association timestamps disagree for frame: {stem}")
            frame_paths = dict(self._frame_input_paths(stem))
            image_path = frame_paths[f"rgb/{stem}"]
            center_path = frame_paths[f"center_depth/{stem}"]
            fused_path = frame_paths[f"fused5_depth/{stem}"]
            fused_mask_path = frame_paths[f"fused5_mask/{stem}"]
            for label, path in (
                ("RGB image", image_path),
                ("center depth", center_path),
                ("fused5 depth", fused_path),
                ("fused5 mask", fused_mask_path),
            ):
                if not path.is_file():
                    raise ValueError(f"Missing {label} for frame {stem}: {path}")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            center_depth_loaded = np.load(center_path)
            fused_depth_loaded = np.load(fused_path)
            if center_depth_loaded.dtype != np.dtype("<f4") or fused_depth_loaded.dtype != np.dtype("<f4"):
                raise ValueError(f"Prepared Scene depth must be little-endian float32: {stem}")
            center_depth = np.asarray(center_depth_loaded, dtype=np.float32)
            fused_depth = np.asarray(fused_depth_loaded, dtype=np.float32)
            fused_mask_raw = np.asarray(np.load(fused_mask_path))
            native_shape = (intrinsics.height, intrinsics.width)
            if rgb.shape != (*native_shape, 3) or center_depth.shape != native_shape or fused_depth.shape != native_shape or fused_mask_raw.shape != native_shape:
                raise ValueError(f"Frame dimensions do not match intrinsics: {stem}")
            if fused_mask_raw.dtype.kind not in "biu":
                raise ValueError(f"Fused5 mask must be boolean-like for frame {stem}")
            if not np.isfinite(center_depth).all() or not np.isfinite(fused_depth).all() or (
                center_depth < 0
            ).any() or (fused_depth < 0).any():
                raise ValueError(f"Prepared Scene depth must be finite and non-negative: {stem}")
            center_valid = np.isfinite(center_depth) & (center_depth > 0)
            fused_valid = fused_mask_raw.astype(bool) & np.isfinite(fused_depth) & (fused_depth > 0)
            if not np.array_equal(fused_mask_raw.astype(bool), fused_depth > 0):
                raise ValueError(f"Prepared Scene fused mask does not match fused depth: {stem}")
            if center_depth[center_valid].tobytes() != fused_depth[center_valid].tobytes():
                raise ValueError(f"Prepared Scene center anchor is not byte-exact in fused depth: {stem}")
            if not fused5_enabled:
                fused_valid = np.zeros(native_shape, dtype=bool)
            frame_intrinsics = intrinsics
            if self.config.resize_width is not None:
                output_size = (self.config.resize_width, self.config.resize_height)
                scale_x = self.config.resize_width / intrinsics.width
                scale_y = self.config.resize_height / intrinsics.height
                frame_intrinsics = CameraIntrinsics(
                    self.config.resize_width, self.config.resize_height,
                    intrinsics.fx * scale_x, intrinsics.fy * scale_y,
                    intrinsics.cx * scale_x, intrinsics.cy * scale_y,
                )
                rgb = _resize_rgb(rgb, output_size)
                center_depth = _resize_nearest(center_depth, output_size, dtype=np.float32)
                center_valid = _resize_nearest(center_valid, output_size, dtype=np.float32).astype(bool)
                fused_depth = _resize_nearest(fused_depth, output_size, dtype=np.float32)
                fused_valid = _resize_nearest(fused_valid, output_size, dtype=np.float32).astype(bool)
            mapping_depth = np.where(center_valid, center_depth, np.where(fused_valid, fused_depth, 0.0)).astype(np.float32)
            mapping_sources = np.full(mapping_depth.shape, INVALID_SOURCE_TYPE, dtype=np.uint8)
            mapping_sources[center_valid] = int(center_source_type)
            mapping_sources[~center_valid & fused_valid] = int(fused_source_type)
            mapping_confidence = np.zeros(mapping_depth.shape, dtype=np.float32)
            raw_confidence = descriptor_for_type(center_source_type).default_confidence
            fused_confidence = descriptor_for_type(fused_source_type).default_confidence
            mapping_confidence[center_valid] = raw_confidence
            mapping_confidence[~center_valid & fused_valid] = fused_confidence
            evidences = [DepthEvidence(
                center_source_type, center_depth, center_valid,
                np.where(center_valid, raw_confidence, 0.0).astype(np.float32),
            )]
            if fused5_enabled:
                evidences.append(DepthEvidence(
                    fused_source_type, fused_depth, fused_valid,
                    np.where(fused_valid, fused_confidence, 0.0).astype(np.float32),
                ))
            observation = MappingObservation(mapping_depth, mapping_sources, mapping_confidence)
            pose = Pose(*(float(pose_row[name]) for name in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")))
            yield FrameInputs(
                index=index,
                stem=stem,
                timestamp_ns=int(association["image_timestamp_ns"]),
                intrinsics=frame_intrinsics,
                pose=pose,
                rgb=rgb,
                mapping=observation,
                growth=GrowthInputs(tuple(evidences)),
            )
        if not ordered:
            raise ValueError("Prepared Scene contains no frames")

    def _validate_contract(self) -> dict[str, object]:
        if not self.prepared_manifest_path.is_file():
            raise ValueError(f"Missing Prepared Scene manifest: {self.prepared_manifest_path}")
        try:
            manifest = json.loads(self.prepared_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Prepared Scene manifest must be valid JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError("Prepared Scene manifest must be a JSON object")
        if manifest.get("schema_version") == PREPARED_SCENE_SCHEMA:
            return self._validate_manifest_contract(manifest)
        raise ValueError(f"Prepared Scene must use {PREPARED_SCENE_SCHEMA}")

    def _validate_manifest_contract(self, manifest: dict[str, object]) -> dict[str, object]:
        expected_fields = {
            "schema_version", "preparation_profile", "source", "camera", "timing", "depth",
            "complete", "non_formal", "source_policy_version", "rejection_policy",
            "producer_identity", "production_receipt", "input_files_sha256", "artifacts_sha256",
            "counts", "frame_hash_chain_sha256",
        }
        if set(manifest) != expected_fields:
            raise ValueError("Prepared Scene manifest fields do not match sage-prepared-scene-v1")
        contract = validate_source_contract(manifest)
        source = contract["source"]
        assert isinstance(source, dict)
        expected_sources = (
            (source["center_source_type"],),
            (source["center_source_type"], source["fused_source_type"]),
        )
        if self.config.enabled_depth_sources not in expected_sources:
            raise ValueError("Configured depth sources do not match Prepared Scene source family")
        configured_mode = "SLAM_WORLD" if self.config.enabled_depth_sources[0] == "LIDAR_SLAM_CENTER" else "LIDAR_RAW"
        if source["mode"] != configured_mode:
            raise ValueError("Prepared Scene source mode does not match configured source family")
        if self.config.scene_dir != (self.config.prepared_scene_dir / "scene").resolve():
            raise ValueError("Prepared Scene v2 scene_dir must name the manifest scene artifact root")
        expected_roots = (
            (self.config.center_depth_path, self.config.prepared_scene_dir / "center_depth", "center_depth"),
            (self.config.fused5_depth_dir, self.config.prepared_scene_dir / "fused5_depth", "fused5_depth"),
            (self.config.fused5_mask_dir, self.config.prepared_scene_dir / "fused5_mask", "fused5_mask"),
        )
        for configured, expected, name in expected_roots:
            if configured != expected.resolve():
                raise ValueError(f"Configured {name} does not match Prepared Scene v2 artifact root")
        receipt = manifest.get("production_receipt")
        if (
            manifest.get("complete") is not True
            or manifest.get("source_policy_version") != "sage-source-policy-v2"
            or manifest.get("rejection_policy") != "complete-centered-5-no-slot-compression-v1"
        ):
            raise ValueError("Prepared Scene v2 completion and no-fallback receipt is invalid")
        receipt = validate_completion_receipt(
            receipt,
            allowed_schemas={MATERIALIZED_RECEIPT_SCHEMA, STREAM_RECEIPT_SCHEMA},
            expected_emitted=int(manifest["counts"]["emitted_centers"]),
            require_write_through=True,
            require_peak_cuda=False,
        )
        if receipt.get("source_mode") != source["mode"]:
            raise ValueError("Prepared Scene v2 receipt source mode is inconsistent")
        for name in ("input_files_sha256", "artifacts_sha256"):
            hashes = manifest.get(name)
            if not isinstance(hashes, dict) or not hashes or any(
                not isinstance(label, str) or not label
                or not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for label, value in hashes.items()
            ):
                raise ValueError(f"Prepared Scene v2 {name} is invalid")
        declared = manifest["artifacts_sha256"]
        assert isinstance(declared, dict)
        if receipt["receipt_schema"] == "fixed-lag-stream-v1" and receipt["write_through_artifacts_sha256"] != declared:
            raise ValueError("Prepared Scene v2 write-through artifact hashes are inconsistent")
        current_paths: dict[str, Path] = {}
        for path in sorted(self.config.prepared_scene_dir.rglob("*")):
            if not path.is_file() or path == self.prepared_manifest_path:
                continue
            if path.is_symlink():
                raise ValueError(f"Prepared Scene v2 artifact must not be symbolic: {path}")
            current_paths[path.relative_to(self.config.prepared_scene_dir).as_posix()] = path
        if set(declared) != set(current_paths) or any(
            declared[label] != sha256_file(path) for label, path in current_paths.items()
        ):
            raise ValueError("Prepared Scene v2 artifact hashes do not match the manifest")
        associations = _rows_by_id(self.config.prepared_scene_dir / "associations" / "frame_associations.csv")
        for stem, row in associations.items():
            if (
                row.get("center_source_type") != source["center_source_type"]
                or row.get("fused_source_type") != source["fused_source_type"]
            ):
                raise ValueError(f"Prepared Scene v2 association source type mismatch: {stem}")
        counts = manifest.get("counts")
        if not isinstance(counts, dict) or counts.get("emitted_centers") != len(associations):
            raise ValueError("Prepared Scene v2 emitted center count is invalid")
        if receipt.get("emitted_centers") != counts["emitted_centers"]:
            raise ValueError("Prepared Scene v2 receipt emitted center count is inconsistent")
        if receipt.get("rejected_centers_count") != counts.get("rejected_centers"):
            raise ValueError("Prepared Scene v2 receipt rejected center count is inconsistent")
        _validate_frame_hash_chain(
            self.config.prepared_scene_dir,
            declared,
            associations,
            manifest["frame_hash_chain_sha256"],
        )
        return manifest

def sha256_file(path: Path) -> str:
    try:
        return _sha256_file(path)
    except OSError as exc:
        raise ValueError(f"Cannot hash Prepared Scene input: {path}") from exc


def transform_contract_sha256(manifest: dict[str, object]) -> str:
    if manifest.get("schema_version") != PREPARED_SCENE_SCHEMA:
        raise ValueError(f"Prepared Scene must use {PREPARED_SCENE_SCHEMA}")
    embedded_transform = {
        name: manifest[name]
        for name in ("preparation_profile", "source", "camera", "timing", "depth")
    }
    return hashlib.sha256(
        json.dumps(embedded_transform, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scene_content_sha256(files: tuple[tuple[str, Path], ...]) -> str:
    digest = hashlib.sha256()
    for label, path in files:
        digest.update(f"{label}\0{sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()
