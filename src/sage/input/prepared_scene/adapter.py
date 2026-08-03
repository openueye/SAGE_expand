"""PreparedSceneAdapter: canonical frames read back from disk.

It needs no ROS, no topics and no calibration: the persisted frames already are
the canonical contract. Every identity is recomputed from the assets rather
than trusted from the manifest, so a swapped depth map is caught even when the
manifest still claims the old digest.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..adapter import ResolvedInput
from ..contract import ResolvedInputContract
from ..frame import DepthObservation, FrameInputs, FrameMetadata
from ..identity import CanonicalSequenceDigest, canonical_json_sha256
from ..preflight import PreflightReport
from .manifest import (
    FRAMES_NAME,
    MANIFEST_NAME,
    SOURCE_ASSETS,
    SOURCE_DIRS,
    SOURCE_FIELDS,
    SceneManifest,
    frame_chain_digest,
    safe_relative_path,
)


ADAPTER_TYPE = "prepared_scene"

_OBSERVATION_DTYPES = {
    "depth": np.float32,
    "valid": np.bool_,
    "support": np.uint16,
    "age": np.float32,
}


def _load_array(root: Path, relative: str, asset: str, digest_by_asset: dict[str, str], key: str) -> np.ndarray:
    path = safe_relative_path(root, relative)
    if not path.is_file():
        raise ValueError(f"Prepared Scene asset is missing: {relative}")
    payload = path.read_bytes()
    _verify_digest(digest_by_asset, key, payload, relative)
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    expected = _OBSERVATION_DTYPES[asset]
    if array.dtype != np.dtype(expected):
        raise ValueError(
            f"Prepared Scene asset {relative} must be {np.dtype(expected)}, got {array.dtype}"
        )
    return array


def _verify_digest(digest_by_asset: dict[str, str], key: str, payload: bytes, relative: str) -> None:
    declared = digest_by_asset.get(key)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"Prepared Scene frame record has no valid hash for {relative}")
    if hashlib.sha256(payload).hexdigest() != declared:
        raise ValueError(f"Prepared Scene asset does not match its recorded hash: {relative}")


class PreparedSceneAdapter:
    """Read one published Prepared Scene as canonical frames."""

    def __init__(self, scene_path: Path) -> None:
        self._root = Path(scene_path).expanduser().resolve()

    def preflight(self) -> ResolvedInput:
        report = PreflightReport(ADAPTER_TYPE)
        root = self._root
        if not root.is_dir():
            raise ValueError(f"Prepared Scene directory does not exist: {root}")
        manifest = SceneManifest.load(root / MANIFEST_NAME)
        records = self._records(root, manifest)
        contract = ResolvedInputContract(
            adapter_type=ADAPTER_TYPE,
            canonical=manifest.canonical,
            adapter_details=manifest.adapter_details(),
        )
        digest = CanonicalSequenceDigest(contract.canonical_contract_identity)
        chain = bytes(32)
        for record in records:
            frame = self._frame(root, manifest, record)
            digest.update(frame)
            chain = frame_chain_digest(
                chain, frame.stem, record["assets_sha256"],
                {key: value for key, value in record.items() if key != "assets_sha256"},
            )
        if chain.hex() != manifest.frame_assets_hash_chain:
            report.add_error("Prepared Scene frame asset hash chain does not match its manifest")
        sequence_identity = digest.hexdigest()
        if sequence_identity != manifest.canonical_sequence_identity:
            report.add_error(
                "Prepared Scene canonical sequence identity does not match its assets"
            )
        report.record_counts(candidate_frames=len(records), accepted_frames=len(records))
        report.record_statistics("assets", {
            "frames": len(records),
            "sources": list(manifest.canonical.sources),
        })
        report.raise_for_errors()

        def frames() -> Iterator[FrameInputs]:
            for record in records:
                yield self._frame(root, manifest, record)

        return ResolvedInput(
            contract=contract,
            report=report,
            source_identity=canonical_json_sha256({
                "scene": root.name,
                "frame_assets_hash_chain": manifest.frame_assets_hash_chain,
                "provenance": dict(manifest.provenance),
            }),
            _frames=frames,
            _sequence_identity=sequence_identity,
        )

    def _records(self, root: Path, manifest: SceneManifest) -> list[dict[str, object]]:
        path = root / FRAMES_NAME
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"Prepared Scene frame table is unreadable: {path}") from exc
        records = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Prepared Scene frames.jsonl line {index} is not JSON") from exc
            expected = {
                "frame_index", "timestamp_ns", "image", "intrinsics",
                "reference_from_camera", "assets_sha256", *SOURCE_FIELDS.values(),
            }
            if not isinstance(record, dict) or set(record) != expected:
                raise ValueError(f"Prepared Scene frame record {index} fields are invalid")
            if record["frame_index"] != index:
                raise ValueError(
                    f"Prepared Scene frame indices must be contiguous from zero; got "
                    f"{record['frame_index']} at line {index}"
                )
            records.append(record)
        if len(records) != manifest.canonical.frame_count:
            raise ValueError(
                f"Prepared Scene declares {manifest.canonical.frame_count} frames but frames.jsonl "
                f"has {len(records)}"
            )
        return records

    def _frame(self, root: Path, manifest: SceneManifest, record: dict[str, object]) -> FrameInputs:
        digest_by_asset = record["assets_sha256"]
        if not isinstance(digest_by_asset, dict):
            raise ValueError("Prepared Scene frame record assets_sha256 must be an object")
        image_path = safe_relative_path(root, str(record["image"]))
        if not image_path.is_file():
            raise ValueError(f"Prepared Scene image is missing: {record['image']}")
        payload = image_path.read_bytes()
        _verify_digest(digest_by_asset, "image", payload, str(record["image"]))
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        observations: dict[str, DepthObservation | None] = {}
        for source_name, field in SOURCE_FIELDS.items():
            entry = record[field]
            if source_name not in manifest.canonical.sources:
                if entry is not None:
                    raise ValueError(
                        f"Prepared Scene frame declares {source_name} but the contract disables it"
                    )
                observations[field] = None
                continue
            if not isinstance(entry, dict) or set(entry) != {
                asset for asset, _ in SOURCE_ASSETS[source_name]
            }:
                raise ValueError(
                    f"Prepared Scene frame {record['frame_index']} has an invalid {source_name} record"
                )
            arrays = {
                attribute: _load_array(
                    root, str(entry[asset]), asset, digest_by_asset,
                    f"{SOURCE_DIRS[source_name]}.{asset}",
                )
                for asset, attribute in SOURCE_ASSETS[source_name]
            }
            observations[field] = DepthObservation(
                depth_m=arrays["depth_m"],
                valid_mask=arrays["valid_mask"],
                support_count=arrays.get("support_count"),
                temporal_age_s=arrays.get("temporal_age_s"),
            )
        return FrameInputs(
            frame_index=int(record["frame_index"]),
            timestamp_ns=int(record["timestamp_ns"]),
            image_rgb=rgb.astype(np.float32) / 255.0,
            intrinsics=np.asarray(record["intrinsics"], dtype=np.float64),
            reference_from_camera=np.asarray(record["reference_from_camera"], dtype=np.float64),
            lidar_center=observations["lidar_center"],
            lidar_fused=observations["lidar_fused"],
            metadata=FrameMetadata(
                source_timestamp_ns={},
                frame_names={
                    "reference": manifest.canonical.reference_frame,
                    "camera": manifest.canonical.camera_frame,
                },
                diagnostics={},
            ),
        )


__all__ = ["PreparedSceneAdapter"]
