"""Persist a canonical frame sequence as a Prepared Scene.

`frames.jsonl` and every asset form one indivisible scene, so the writer stages
the whole thing and publishes it atomically. A reader only ever sees a
directory that already carries a complete, hash-covered manifest.

This is used by `sage prepare` only. Input adapters never write.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image

from ..contract import CanonicalInputContract
from ..frame import FrameInputs
from ..identity import CanonicalSequenceDigest
from .manifest import (
    FRAMES_NAME,
    IMAGE_DIR,
    MANIFEST_NAME,
    SOURCE_ASSETS,
    SOURCE_DIRS,
    SceneManifest,
    asset_path,
    frame_chain_digest,
    frame_record,
    image_path,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _publish(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _write_npy(path: Path, array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return _publish(path, buffer.getvalue())


def _write_png(path: Path, image_rgb: np.ndarray) -> str:
    """Store canonical RGB on the 8-bit raster it came from, losslessly."""
    quantized = np.rint(np.asarray(image_rgb, dtype=np.float64) * 255.0)
    if not np.allclose(quantized / 255.0, image_rgb, atol=1e-6):
        raise ValueError(
            "Canonical RGB is not on the 8-bit raster; PNG storage would not round-trip"
        )
    buffer = io.BytesIO()
    Image.fromarray(quantized.astype(np.uint8), mode="RGB").save(buffer, format="PNG", optimize=False)
    return _publish(path, buffer.getvalue())


class PreparedSceneWriter:
    """Stage, hash and atomically publish one Prepared Scene."""

    def __init__(self, destination: Path, *, canonical: CanonicalInputContract, provenance: Mapping[str, object]) -> None:
        self._destination = Path(destination).resolve()
        if self._destination.exists():
            raise ValueError(f"Prepared Scene destination already exists: {self._destination}")
        self._canonical = canonical
        self._provenance = dict(provenance)
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        self._staging = Path(tempfile.mkdtemp(
            prefix=f".{self._destination.name}.staging-", dir=self._destination.parent,
        ))

    def write(self, frames: Iterable[FrameInputs]) -> Path:
        staging = self._staging
        try:
            (staging / IMAGE_DIR).mkdir()
            for source_name in self._canonical.sources:
                (staging / SOURCE_DIRS[source_name]).mkdir()
            digest = CanonicalSequenceDigest(self._canonical.identity)
            chain = bytes(32)
            records: list[str] = []
            for frame in frames:
                digest.update(frame)
                record, chain = self._write_frame(staging, frame, chain)
                records.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
            if digest.frame_count != self._canonical.frame_count:
                raise ValueError(
                    f"Prepared Scene received {digest.frame_count} frames, contract declares "
                    f"{self._canonical.frame_count}"
                )
            (staging / FRAMES_NAME).write_text("\n".join(records) + "\n", encoding="utf-8")
            manifest = SceneManifest(
                canonical=self._canonical,
                canonical_contract_identity=self._canonical.identity,
                canonical_sequence_identity=digest.hexdigest(),
                frame_assets_hash_chain=chain.hex(),
                provenance=self._provenance,
            )
            (staging / MANIFEST_NAME).write_text(
                json.dumps(manifest.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            staging.rename(self._destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return self._destination

    def _write_frame(
        self, staging: Path, frame: FrameInputs, chain: bytes,
    ) -> tuple[dict[str, object], bytes]:
        stem = frame.stem
        asset_hashes = {"image": _write_png(staging / image_path(stem), frame.image_rgb)}
        sources: dict[str, dict[str, str]] = {}
        for source_name in self._canonical.sources:
            observation = frame.observation(source_name)
            if observation is None:
                raise ValueError(
                    f"Frame {stem} is missing observation {source_name} declared by the contract"
                )
            entry: dict[str, str] = {}
            for asset, attribute in SOURCE_ASSETS[source_name]:
                values = getattr(observation, attribute)
                if values is None:
                    raise ValueError(
                        f"Frame {stem} observation {source_name} has no {attribute} to persist"
                    )
                relative = asset_path(source_name, stem, asset)
                asset_hashes[f"{SOURCE_DIRS[source_name]}.{asset}"] = _write_npy(
                    staging / relative, values,
                )
                entry[asset] = relative
            sources[source_name] = entry
        record = frame_record(
            frame_index=frame.frame_index,
            timestamp_ns=frame.timestamp_ns,
            intrinsics=np.asarray(frame.intrinsics, dtype=np.float64).tolist(),
            reference_from_camera=np.asarray(frame.reference_from_camera, dtype=np.float64).tolist(),
            sources=sources,
            stem=stem,
        )
        record["assets_sha256"] = asset_hashes
        return record, frame_chain_digest(
            chain, stem, asset_hashes, {key: value for key, value in record.items() if key != "assets_sha256"},
        )

    def abort(self) -> None:
        shutil.rmtree(self._staging, ignore_errors=True)


__all__ = ["PreparedSceneWriter"]
