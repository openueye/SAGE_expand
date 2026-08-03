"""Prepared Scene schema: manifest, frame records and asset addressing.

A Prepared Scene is the persisted form of a canonical frame sequence — not a
second input semantics. The manifest therefore carries the same canonical
contract a ROSBAG run would freeze, plus the hashes needed to prove that no
individual asset was swapped after publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ..contract import CanonicalInputContract


PREPARED_SCENE_SCHEMA = "sage_prepared_scene"
PREPARED_SCENE_REVISION = 1

MANIFEST_NAME = "manifest.json"
FRAMES_NAME = "frames.jsonl"

IMAGE_DIR = "images"
STORAGE_LAYOUT = "frame_assets"
IMAGE_STORAGE = "png"
DEPTH_STORAGE = "float32_npy"

# Per source: the asset key each frame record must carry, and the observation
# attribute it round-trips.
SOURCE_ASSETS = {
    "LIDAR_CENTER": (("depth", "depth_m"), ("valid", "valid_mask")),
    "LIDAR_FUSED": (
        ("depth", "depth_m"),
        ("valid", "valid_mask"),
        ("support", "support_count"),
        ("age", "temporal_age_s"),
    ),
}
SOURCE_DIRS = {"LIDAR_CENTER": "lidar_center", "LIDAR_FUSED": "lidar_fused"}
SOURCE_FIELDS = {"LIDAR_CENTER": "lidar_center", "LIDAR_FUSED": "lidar_fused"}

_ASSET_SUFFIX = {"depth": "depth.npy", "valid": "valid.npy", "support": "support.npy", "age": "age.npy"}


def asset_path(source_name: str, stem: str, asset: str) -> str:
    return f"{SOURCE_DIRS[source_name]}/{stem}.{_ASSET_SUFFIX[asset]}"


def image_path(stem: str) -> str:
    return f"{IMAGE_DIR}/{stem}.png"


def safe_relative_path(root: Path, relative: str) -> Path:
    """Resolve a manifest-declared asset path, refusing anything that escapes."""
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        raise ValueError(f"Prepared Scene asset path must be relative: {relative!r}")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Prepared Scene asset path escapes the scene: {relative!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Prepared Scene asset path escapes the scene: {relative!r}")
    if resolved.is_symlink() or (root / candidate).is_symlink():
        raise ValueError(f"Prepared Scene asset must not be a symbolic link: {relative!r}")
    return resolved


def frame_chain_digest(chain: bytes, stem: str, asset_hashes: Mapping[str, str], record: Mapping[str, object]) -> bytes:
    """Fold one frame's assets and its canonical JSON line into the chain."""
    return hashlib.sha256(
        chain
        + stem.encode("utf-8")
        + json.dumps(dict(asset_hashes), sort_keys=True, separators=(",", ":")).encode("utf-8")
        + json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


@dataclass(frozen=True)
class SceneManifest:
    canonical: CanonicalInputContract
    canonical_contract_identity: str
    canonical_sequence_identity: str
    frame_assets_hash_chain: str
    provenance: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "schema_name": PREPARED_SCENE_SCHEMA,
            "schema_revision": PREPARED_SCENE_REVISION,
            "storage_layout": STORAGE_LAYOUT,
            "image_storage": IMAGE_STORAGE,
            "depth_storage": DEPTH_STORAGE,
            "canonical": self.canonical.payload(),
            "canonical_contract_identity": self.canonical_contract_identity,
            "canonical_sequence_identity": self.canonical_sequence_identity,
            "frame_assets_hash_chain": self.frame_assets_hash_chain,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def load(cls, path: Path) -> "SceneManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Prepared Scene manifest is unreadable: {path}") from exc
        expected = {
            "schema_name", "schema_revision", "storage_layout", "image_storage",
            "depth_storage", "canonical", "canonical_contract_identity",
            "canonical_sequence_identity", "frame_assets_hash_chain", "provenance",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(
                "Unsupported prepared scene schema.\n\n"
                "This branch accepts only the canonical SAGE prepared scene format.\n"
                "Regenerate the prepared scene using the current input pipeline."
            )
        if (
            payload["schema_name"] != PREPARED_SCENE_SCHEMA
            or payload["schema_revision"] != PREPARED_SCENE_REVISION
            or payload["storage_layout"] != STORAGE_LAYOUT
            or payload["image_storage"] != IMAGE_STORAGE
            or payload["depth_storage"] != DEPTH_STORAGE
        ):
            raise ValueError(
                "Unsupported prepared scene schema.\n\n"
                "This branch accepts only the canonical SAGE prepared scene format.\n"
                "Regenerate the prepared scene using the current input pipeline."
            )
        canonical = CanonicalInputContract.from_payload(payload["canonical"])
        if canonical.identity != payload["canonical_contract_identity"]:
            raise ValueError("Prepared Scene canonical contract identity does not match its contract")
        return cls(
            canonical=canonical,
            canonical_contract_identity=str(payload["canonical_contract_identity"]),
            canonical_sequence_identity=str(payload["canonical_sequence_identity"]),
            frame_assets_hash_chain=str(payload["frame_assets_hash_chain"]),
            provenance=dict(payload["provenance"]),
        )

    def adapter_details(self) -> dict[str, object]:
        return {
            "adapter_type": "prepared_scene",
            "schema_name": PREPARED_SCENE_SCHEMA,
            "schema_revision": PREPARED_SCENE_REVISION,
            "storage_layout": STORAGE_LAYOUT,
            "image_storage": IMAGE_STORAGE,
            "depth_storage": DEPTH_STORAGE,
        }


def frame_record(
    *,
    frame_index: int,
    timestamp_ns: int,
    intrinsics: list[list[float]],
    reference_from_camera: list[list[float]],
    sources: Mapping[str, Mapping[str, str]],
    stem: str,
) -> dict[str, object]:
    """One `frames.jsonl` line; the order of keys is irrelevant, the content is not."""
    record: dict[str, object] = {
        "frame_index": frame_index,
        "timestamp_ns": timestamp_ns,
        "image": image_path(stem),
        "intrinsics": intrinsics,
        "reference_from_camera": reference_from_camera,
    }
    for name, field in SOURCE_FIELDS.items():
        record[field] = dict(sources[name]) if name in sources else None
    return record


__all__ = [
    "FRAMES_NAME",
    "MANIFEST_NAME",
    "PREPARED_SCENE_REVISION",
    "PREPARED_SCENE_SCHEMA",
    "SOURCE_ASSETS",
    "SOURCE_DIRS",
    "SOURCE_FIELDS",
    "SceneManifest",
    "asset_path",
    "frame_chain_digest",
    "frame_record",
    "image_path",
    "safe_relative_path",
]
