"""The three input identities every SAGE adapter must be able to recompute.

They are deliberately separate. `canonical_sequence_identity` and
`canonical_contract_identity` describe what SAGE Core actually consumed, so a
ROSBAG run and a Prepared Scene exported from that same ROSBAG share them and
can exchange checkpoints. `source_identity` and `adapter_provenance_identity`
describe where the bytes came from and how they were parsed; they are recorded
for audit and must never gate that exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .frame import DepthObservation, FrameInputs


def canonical_json_sha256(payload: object) -> str:
    """Digest of a JSON-serializable payload under one canonical encoding."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(digest: "hashlib._Hash", label: str, array: np.ndarray | None) -> None:
    digest.update(label.encode("utf-8"))
    if array is None:
        digest.update(b"\0none")
        return
    contiguous = np.ascontiguousarray(array)
    digest.update(f"\0{contiguous.dtype.str}\0{contiguous.shape}\0".encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))


def _observation_digest(digest: "hashlib._Hash", label: str, observation: DepthObservation | None) -> None:
    if observation is None:
        digest.update(f"{label}\0absent".encode("utf-8"))
        return
    for name in ("depth_m", "valid_mask", "confidence", "support_count", "temporal_age_s"):
        _array_digest(digest, f"{label}.{name}", getattr(observation, name))


class CanonicalSequenceDigest:
    """Incremental digest over the ordered semantic frame sequence.

    Only Core-visible semantics enter it: index, timestamp, image, K, pose and
    every observation array. Asset paths, topics and adapter details do not, so
    two adapters that deliver the same frames agree bit for bit.
    """

    def __init__(self, contract_identity: str) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(f"sage-canonical-sequence-v1\0{contract_identity}\n".encode("utf-8"))
        self._frames = 0

    def update(self, frame: FrameInputs) -> None:
        if frame.frame_index != self._frames:
            raise ValueError(
                f"Canonical sequence expects frame_index={self._frames}, got {frame.frame_index}"
            )
        self._frames += 1
        self._digest.update(f"\nframe\0{frame.frame_index}\0{frame.timestamp_ns}\0".encode("utf-8"))
        _array_digest(self._digest, "image_rgb", frame.image_rgb)
        _array_digest(self._digest, "intrinsics", frame.intrinsics)
        _array_digest(self._digest, "reference_from_camera", frame.reference_from_camera)
        _observation_digest(self._digest, "lidar_center", frame.lidar_center)
        _observation_digest(self._digest, "lidar_fused", frame.lidar_fused)

    @property
    def frame_count(self) -> int:
        return self._frames

    def hexdigest(self) -> str:
        return hashlib.sha256(
            self._digest.digest() + f"\0frames\0{self._frames}".encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class InputIdentities:
    source_identity: str
    adapter_provenance_identity: str
    canonical_contract_identity: str
    canonical_sequence_identity: str

    def __post_init__(self) -> None:
        for name in (
            "source_identity",
            "adapter_provenance_identity",
            "canonical_contract_identity",
            "canonical_sequence_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Input identity {name} must be a sha256 hex digest")

    def payload(self) -> dict[str, str]:
        return {
            "source_identity": self.source_identity,
            "adapter_provenance_identity": self.adapter_provenance_identity,
            "canonical_contract_identity": self.canonical_contract_identity,
            "canonical_sequence_identity": self.canonical_sequence_identity,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "InputIdentities":
        if not isinstance(payload, Mapping) or set(payload) != {
            "source_identity",
            "adapter_provenance_identity",
            "canonical_contract_identity",
            "canonical_sequence_identity",
        }:
            raise ValueError("Input identity payload fields are invalid")
        return cls(**{name: str(value) for name, value in payload.items()})


def validate_core_identity_match(recorded: object, current: InputIdentities) -> None:
    """Reject reuse across a different canonical sequence or contract.

    Provenance identities are informational: the same canonical sequence read
    from a ROSBAG or from its exported Prepared Scene must stay interchangeable.
    """
    identities = InputIdentities.from_payload(recorded)
    mismatched = [
        name
        for name in ("canonical_sequence_identity", "canonical_contract_identity")
        if getattr(identities, name) != getattr(current, name)
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint input identity is incompatible with the current runtime contract "
            f"({', '.join(mismatched)}). Old checkpoints are not supported in this branch."
        )


__all__ = [
    "CanonicalSequenceDigest",
    "InputIdentities",
    "canonical_json_sha256",
    "file_sha256",
    "validate_core_identity_match",
]
