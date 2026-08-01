from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from ...foundation.hashing import sha256_file
from ...foundation.receipt_contract import (
    MATERIALIZED_RECEIPT_SCHEMA,
    STREAM_RECEIPT_SCHEMA,
    validate_completion_receipt,
)
from .poses import matrix_to_quaternion_xyzw

from .geometry import ProjectedCenter, RejectedCenter
from .profiles import PreparationProfile, validate_source_contract


MANIFEST_NAME = "sage_prepared_scene_manifest.json"


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate_center(result: ProjectedCenter, profile: PreparationProfile) -> None:
    expected_shape = tuple(profile.camera.output_grid)
    target = result.target
    if (target.intrinsics.height, target.intrinsics.width) != expected_shape:
        raise ValueError(f"Prepared Scene output grid must be {expected_shape}")
    if (target.cloud.center_source_type, target.cloud.fused_source_type) != (
        profile.source.center_source_type,
        profile.source.fused_source_type,
    ):
        raise ValueError("Projected center source family does not match preparation profile")
    for name, value in (("center_depth", result.center_depth), ("fused_depth", result.fused_depth)):
        array = np.asarray(value)
        if array.shape != expected_shape or array.dtype != np.dtype("<f4"):
            raise ValueError(f"{name} must be little-endian float32 on the profile output grid")
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"{name} must contain finite non-negative target-camera Z")
    mask = np.asarray(result.fused_mask)
    if mask.shape != expected_shape or mask.dtype.kind not in "biu":
        raise ValueError("fused5_mask must be boolean-like on the profile output grid")
    center_valid = result.center_depth > 0
    if result.center_depth[center_valid].tobytes() != result.fused_depth[center_valid].tobytes():
        raise ValueError("Fused depth must restore the center anchor byte-exact")


def _artifact_hashes(
    root: Path,
    *,
    known_hashes: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Hash publication artifacts, reusing hashes captured after immutable writes."""
    known = dict(known_hashes or {})
    artifacts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        artifact_hash = known.get(relative)
        artifacts[relative] = artifact_hash if artifact_hash is not None else sha256_file(path)
    return artifacts


class PreparedSceneWriter:
    """Incremental Prepared Scene v2 writer with a separate atomic publish step."""

    def __init__(
        self,
        output_dir: Path,
        *,
        profile: PreparationProfile,
        input_files: Mapping[str, Path],
        producer_identity: Mapping[str, object],
        non_formal: bool,
        input_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self.output = Path(output_dir).resolve()
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite Prepared Scene: {self.output}")
        if not input_files:
            raise ValueError("Prepared Scene publication requires hashed input files")
        if not isinstance(producer_identity, Mapping) or not producer_identity:
            raise ValueError("Prepared Scene publication requires producer identity")
        self.profile = profile
        self.input_files = {label: Path(path) for label, path in sorted(input_files.items())}
        if input_hashes is None:
            self.input_hashes = {
                label: sha256_file(path) for label, path in self.input_files.items()
            }
        else:
            self.input_hashes = dict(input_hashes)
            if set(self.input_hashes) != set(self.input_files) or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.input_hashes.values()
            ):
                raise ValueError("Prepared Scene input hashes must cover every input with SHA-256 values")
        self.producer_identity = dict(producer_identity)
        self.non_formal = bool(non_formal)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=f".{self.output.name}.staging-", dir=self.output.parent))
        for relative in (
            "scene/images", "scene/sparse/0", "center_depth", "fused5_depth",
            "fused5_mask", "associations", "poses", "audit",
        ):
            (self.staging / relative).mkdir(parents=True, exist_ok=True)
        self.association_rows: list[dict[str, object]] = []
        self.pose_rows: list[dict[str, object]] = []
        self.audit_rows: list[dict[str, object]] = []
        self.chain = bytes(32)
        self.chain_records: list[dict[str, str]] = []
        self._written_artifact_hashes: dict[str, str] = {}
        self.last_frame_hashes: dict[str, str] | None = None
        self.emitted_count = 0
        self.prepared = False
        self.closed = False

    def write(self, result: ProjectedCenter) -> None:
        if self.closed or self.prepared:
            raise RuntimeError("PreparedSceneWriter is already closed")
        _validate_center(result, self.profile)
        target = result.target
        if self.emitted_count == 0:
            first = target.intrinsics
            cameras = f"1 PINHOLE {first.width} {first.height} {first.fx:.17g} {first.fy:.17g} {first.cx:.17g} {first.cy:.17g}\n"
            (self.staging / "scene/sparse/0/cameras.txt").write_text(cameras, encoding="utf-8")
        stem = target.stem
        image_path = self.staging / "scene/images" / f"{stem}.png"
        center_path = self.staging / "center_depth" / f"{stem}.npy"
        fused_path = self.staging / "fused5_depth" / f"{stem}.npy"
        mask_path = self.staging / "fused5_mask" / f"{stem}.npy"
        Image.fromarray(target.rgb, mode="RGB").save(image_path, format="PNG")
        np.save(center_path, np.asarray(result.center_depth, dtype=np.dtype("<f4")), allow_pickle=False)
        np.save(fused_path, np.asarray(result.fused_depth, dtype=np.dtype("<f4")), allow_pickle=False)
        np.save(mask_path, np.asarray(result.fused_mask, dtype=np.uint8), allow_pickle=False)
        frame_hashes = {
            "image": sha256_file(image_path),
            "center_depth": sha256_file(center_path),
            "fused5_depth": sha256_file(fused_path),
            "fused5_mask": sha256_file(mask_path),
        }
        self._written_artifact_hashes.update({
            image_path.relative_to(self.staging).as_posix(): frame_hashes["image"],
            center_path.relative_to(self.staging).as_posix(): frame_hashes["center_depth"],
            fused_path.relative_to(self.staging).as_posix(): frame_hashes["fused5_depth"],
            mask_path.relative_to(self.staging).as_posix(): frame_hashes["fused5_mask"],
        })
        self.last_frame_hashes = dict(frame_hashes)
        self.chain = hashlib.sha256(
            self.chain + stem.encode("utf-8") + json.dumps(frame_hashes, sort_keys=True).encode("utf-8")
        ).digest()
        self.chain_records.append({"frame_id": stem, "chain_sha256": self.chain.hex()})
        self.association_rows.append({
            "frame_id": stem,
            "image_index": target.image_index,
            "image_timestamp_ns": target.image_timestamp_ns,
            "cloud_id": target.cloud.cloud_id,
            "cloud_timestamp_ns": target.cloud.cloud_timestamp_ns,
            "image_to_cloud_dt_ns": abs(target.image_timestamp_ns - target.cloud.cloud_timestamp_ns),
            "window_image_indices": json.dumps([frame.image_index for frame in result.source_frames], separators=(",", ":")),
            "window_cloud_ids": json.dumps([frame.cloud.cloud_id for frame in result.source_frames], separators=(",", ":")),
            "window_normalization_receipts": json.dumps([
                {
                    "image_index": frame.image_index,
                    "cloud_id": frame.cloud.cloud_id,
                    "receipt": frame.cloud.normalization_receipt,
                }
                for frame in result.source_frames
            ], sort_keys=True, separators=(",", ":")),
            "center_source_type": target.cloud.center_source_type,
            "fused_source_type": target.cloud.fused_source_type,
            "normalization_receipt": json.dumps(target.cloud.normalization_receipt, sort_keys=True, separators=(",", ":")),
        })
        pose = target.world_from_camera
        quaternion = matrix_to_quaternion_xyzw(pose[:3, :3])
        self.pose_rows.append({
            "frame_id": stem,
            "image_timestamp_ns": target.image_timestamp_ns,
            "tx": float(pose[0, 3]), "ty": float(pose[1, 3]), "tz": float(pose[2, 3]),
            "qx": float(quaternion[0]), "qy": float(quaternion[1]),
            "qz": float(quaternion[2]), "qw": float(quaternion[3]),
        })
        self.audit_rows.append({"frame_id": stem, **{name: int(value) for name, value in result.stats.items()}})
        self.emitted_count += 1

    def _write_tables(self, rejected_centers: tuple[RejectedCenter, ...]) -> None:
        _write_csv(
            self.staging / "associations/frame_associations.csv",
            (
                "frame_id", "image_index", "image_timestamp_ns", "cloud_id", "cloud_timestamp_ns",
                "image_to_cloud_dt_ns", "window_image_indices", "window_cloud_ids",
                "window_normalization_receipts", "center_source_type", "fused_source_type",
                "normalization_receipt",
            ),
            self.association_rows,
        )
        _write_csv(
            self.staging / "associations/rejected_centers.csv",
            ("target_image_index", "target_stem", "target_timestamp_ns", "dependency_indices", "root_reasons"),
            ({
                "target_image_index": item.target_image_index,
                "target_stem": item.target_stem,
                "target_timestamp_ns": item.target_timestamp_ns,
                "dependency_indices": json.dumps(item.dependency_indices, separators=(",", ":")),
                "root_reasons": json.dumps(item.root_reasons, separators=(",", ":")),
            } for item in rejected_centers),
        )
        _write_csv(
            self.staging / "poses/camera_poses.csv",
            ("frame_id", "image_timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"),
            self.pose_rows,
        )
        audit_fields = ("frame_id", *sorted({key for row in self.audit_rows for key in row if key != "frame_id"}))
        _write_csv(self.staging / "audit/projection_stats.csv", audit_fields, self.audit_rows)
        (self.staging / "audit/frame_hash_chain.json").write_text(
            json.dumps(self.chain_records, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    def prepare(
        self,
        rejected_centers: Iterable[RejectedCenter],
        *,
        production_receipt: Mapping[str, object] | None = None,
    ) -> Path:
        if self.closed or self.prepared:
            raise RuntimeError("PreparedSceneWriter is already closed")
        if self.emitted_count == 0:
            raise ValueError("Prepared Scene publication requires at least one emitted center")
        if production_receipt is None:
            raise ValueError("Prepared Scene publication requires an explicit production receipt")
        rejection_records = tuple(rejected_centers)
        receipt = dict(production_receipt)
        receipt.update({
            "emitted_centers": self.emitted_count,
            "rejected_centers_count": len(rejection_records),
            "frame_hash_chain_sha256": self.chain.hex(),
        })
        if receipt["source_mode"] != self.profile.source.mode:
            raise ValueError("Prepared Scene receipt source mode does not match profile")
        self._write_tables(rejection_records)
        if {label: sha256_file(path) for label, path in self.input_files.items()} != self.input_hashes:
            raise ValueError("Prepared Scene inputs changed during production")
        artifacts = _artifact_hashes(
            self.staging,
            known_hashes=self._written_artifact_hashes,
        )
        if receipt["receipt_schema"] == STREAM_RECEIPT_SCHEMA:
            receipt["write_through_artifacts_sha256"] = dict(artifacts)
        validate_completion_receipt(
            receipt,
            allowed_schemas={MATERIALIZED_RECEIPT_SCHEMA, STREAM_RECEIPT_SCHEMA},
            expected_source_mode=self.profile.source.mode,
            expected_emitted=self.emitted_count,
            require_write_through=receipt["receipt_schema"] == STREAM_RECEIPT_SCHEMA,
            require_peak_cuda=False,
        )
        manifest = {
            **self.profile.manifest_contract(),
            "complete": True,
            "non_formal": self.non_formal,
            "source_policy_version": "sage-source-policy-v2",
            "rejection_policy": "complete-centered-5-no-slot-compression-v1",
            "producer_identity": self.producer_identity,
            "production_receipt": receipt,
            "input_files_sha256": self.input_hashes,
            "artifacts_sha256": artifacts,
            "counts": {
                "emitted_centers": self.emitted_count,
                "rejected_centers": len(rejection_records),
            },
            "frame_hash_chain_sha256": self.chain.hex(),
        }
        validate_source_contract(manifest)
        temporary_manifest = self.staging / f".{MANIFEST_NAME}.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, self.staging / MANIFEST_NAME)
        self.prepared = True
        return self.output / MANIFEST_NAME

    def commit(self) -> Path:
        if self.closed:
            raise RuntimeError("PreparedSceneWriter is already closed")
        if not self.prepared:
            raise RuntimeError("PreparedSceneWriter.commit() requires prepare()")
        os.replace(self.staging, self.output)
        self.closed = True
        return self.output / MANIFEST_NAME

    def finish(
        self,
        rejected_centers: Iterable[RejectedCenter],
        *,
        production_receipt: Mapping[str, object] | None = None,
    ) -> Path:
        self.prepare(rejected_centers, production_receipt=production_receipt)
        return self.commit()

    def abort(self, error: BaseException | str) -> Path | None:
        if self.closed or not self.staging.exists():
            return None
        message = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        incomplete = self.staging / "state.json"
        incomplete.write_text(json.dumps({
            "state": "incomplete",
            "error": message,
            "emitted_centers": self.emitted_count,
            "source_mode": self.profile.source.mode,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        target = self.output.with_name(f".{self.output.name}.incomplete-")
        diagnostic = Path(tempfile.mkdtemp(prefix=target.name, dir=self.output.parent))
        diagnostic.rmdir()
        self.staging.rename(diagnostic)
        self.closed = True
        return diagnostic

    def rollback_commit(self, error: BaseException | str) -> Path | None:
        if not self.closed or not self.output.is_dir():
            return None
        message = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        (self.output / "state.json").write_text(json.dumps({
            "state": "incomplete",
            "error": message,
            "emitted_centers": self.emitted_count,
            "source_mode": self.profile.source.mode,
            "rolled_back_after_commit": True,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        target = self.output.with_name(f".{self.output.name}.incomplete-")
        diagnostic = Path(tempfile.mkdtemp(prefix=target.name, dir=self.output.parent))
        diagnostic.rmdir()
        self.output.rename(diagnostic)
        return diagnostic
