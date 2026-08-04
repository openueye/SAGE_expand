"""Frozen semantics of one resolved input.

The canonical half is what SAGE Core is allowed to see and what checkpoint
compatibility is bound to. The adapter half records how the bytes were parsed;
it is written to run artifacts for audit and never consulted by Core.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from .identity import canonical_json_sha256


CANONICAL_CONTRACT_SCHEMA = "sage_canonical_input"
CANONICAL_CONTRACT_REVISION = 1
ONLINE_CANONICAL_CONTRACT_REVISION = 2

DEPTH_DEFINITION = "camera_optical_z"
LENGTH_UNIT = "meter"
POSE_CONVENTION = "reference_from_camera"
CAMERA_MODEL = "pinhole"

CANONICAL_SOURCES = ("LIDAR_CENTER", "LIDAR_FUSED")


@dataclass(frozen=True)
class CanonicalInputContract:
    """Adapter-independent semantics of an accepted canonical frame sequence."""

    frame_count: int | None
    reference_frame: str
    camera_frame: str
    image_size: tuple[int, int]
    sources: tuple[str, ...]
    fusion: Mapping[str, object]
    camera_model: str = CAMERA_MODEL
    depth_definition: str = DEPTH_DEFINITION
    length_unit: str = LENGTH_UNIT
    pose_convention: str = POSE_CONVENTION
    schema_revision: int = CANONICAL_CONTRACT_REVISION

    def __post_init__(self) -> None:
        if self.schema_revision not in {
            CANONICAL_CONTRACT_REVISION,
            ONLINE_CANONICAL_CONTRACT_REVISION,
        }:
            raise ValueError("Canonical contract schema_revision is unsupported")
        if self.frame_count is None:
            if self.schema_revision != ONLINE_CANONICAL_CONTRACT_REVISION:
                raise ValueError(
                    "Only the online canonical contract may defer frame_count until EOF"
                )
        elif type(self.frame_count) is not int or self.frame_count < 1:
            raise ValueError("Canonical contract frame_count must be a positive integer")
        for name in ("reference_frame", "camera_frame"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Canonical contract {name} must be a non-empty name")
        size = tuple(int(value) for value in self.image_size)
        if len(size) != 2 or min(size) < 1:
            raise ValueError("Canonical contract image_size must be positive (width, height)")
        object.__setattr__(self, "image_size", size)
        sources = tuple(self.sources)
        if not sources or len(set(sources)) != len(sources) or any(
            name not in CANONICAL_SOURCES for name in sources
        ):
            raise ValueError(f"Canonical contract sources must be a subset of {CANONICAL_SOURCES}")
        if sources[0] != "LIDAR_CENTER":
            raise ValueError("Canonical contract must always enable LIDAR_CENTER first")
        object.__setattr__(self, "sources", sources)
        if (
            self.camera_model != CAMERA_MODEL
            or self.depth_definition != DEPTH_DEFINITION
            or self.length_unit != LENGTH_UNIT
            or self.pose_convention != POSE_CONVENTION
        ):
            raise ValueError("Canonical contract may not redefine the frozen frame semantics")
        fusion = dict(self.fusion)
        if not isinstance(fusion.get("policy"), str) or not fusion["policy"]:
            raise ValueError("Canonical contract fusion must declare an explicit policy name")
        if not isinstance(fusion.get("causal"), bool):
            raise ValueError("Canonical contract fusion must declare whether it is causal")
        json.dumps(fusion, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "fusion", fusion)

    def payload(self) -> dict[str, object]:
        return {
            "schema_name": CANONICAL_CONTRACT_SCHEMA,
            "schema_revision": self.schema_revision,
            "frame_count": self.frame_count,
            "reference_frame": self.reference_frame,
            "camera_frame": self.camera_frame,
            "camera_model": self.camera_model,
            "image_size": list(self.image_size),
            "depth_definition": self.depth_definition,
            "length_unit": self.length_unit,
            "pose_convention": self.pose_convention,
            "sources": list(self.sources),
            "fusion": dict(self.fusion),
        }

    @classmethod
    def from_payload(cls, payload: object) -> "CanonicalInputContract":
        if not isinstance(payload, Mapping):
            raise ValueError("Canonical contract payload must be an object")
        expected = {
            "schema_name", "schema_revision", "frame_count", "reference_frame",
            "camera_frame", "camera_model", "image_size", "depth_definition",
            "length_unit", "pose_convention", "sources", "fusion",
        }
        if set(payload) != expected:
            raise ValueError("Canonical contract payload fields are invalid")
        if payload["schema_name"] != CANONICAL_CONTRACT_SCHEMA or payload["schema_revision"] not in {
            CANONICAL_CONTRACT_REVISION,
            ONLINE_CANONICAL_CONTRACT_REVISION,
        }:
            raise ValueError(
                "Unsupported canonical input contract schema. This branch accepts only "
                f"{CANONICAL_CONTRACT_SCHEMA} revisions "
                f"{CANONICAL_CONTRACT_REVISION} and {ONLINE_CANONICAL_CONTRACT_REVISION}."
            )
        return cls(
            frame_count=(
                None if payload["frame_count"] is None else int(payload["frame_count"])
            ),
            reference_frame=str(payload["reference_frame"]),
            camera_frame=str(payload["camera_frame"]),
            image_size=tuple(payload["image_size"]),
            sources=tuple(str(name) for name in payload["sources"]),
            fusion=dict(payload["fusion"]),
            camera_model=str(payload["camera_model"]),
            depth_definition=str(payload["depth_definition"]),
            length_unit=str(payload["length_unit"]),
            pose_convention=str(payload["pose_convention"]),
            schema_revision=int(payload["schema_revision"]),
        )

    @property
    def identity(self) -> str:
        payload = self.payload()
        if self.schema_revision == ONLINE_CANONICAL_CONTRACT_REVISION:
            # The online reader must start its incremental frame digest before
            # it can know how many frames will be accepted. The final count
            # remains in the persisted contract and in the sequence digest,
            # but it cannot perturb the contract seed at EOF; otherwise the
            # resulting sequence digest would be bound to a provisional,
            # non-persisted contract identity.
            payload["frame_count"] = None
        return canonical_json_sha256(payload)


@dataclass(frozen=True)
class ResolvedInputContract:
    adapter_type: str
    canonical: CanonicalInputContract
    adapter_details: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_type, str) or not self.adapter_type:
            raise ValueError("Resolved contract adapter_type must be a non-empty name")
        if not isinstance(self.canonical, CanonicalInputContract):
            raise ValueError("Resolved contract canonical must be a CanonicalInputContract")
        details = dict(self.adapter_details)
        if details.get("adapter_type") != self.adapter_type:
            raise ValueError("Resolved contract adapter_details must declare its adapter_type")
        json.dumps(details, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "adapter_details", details)

    @property
    def canonical_contract_identity(self) -> str:
        return self.canonical.identity

    @property
    def adapter_provenance_identity(self) -> str:
        return canonical_json_sha256(dict(self.adapter_details))

    def payload(self) -> dict[str, object]:
        return {
            "adapter_type": self.adapter_type,
            "canonical": self.canonical.payload(),
            "adapter_details": dict(self.adapter_details),
            "canonical_contract_identity": self.canonical_contract_identity,
            "adapter_provenance_identity": self.adapter_provenance_identity,
        }


__all__ = [
    "CANONICAL_SOURCES",
    "CanonicalInputContract",
    "ONLINE_CANONICAL_CONTRACT_REVISION",
    "ResolvedInputContract",
]
