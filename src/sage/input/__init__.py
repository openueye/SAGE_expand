"""Canonical SAGE input layer: one contract, two adapters."""

from .adapter import InputAdapter, ResolvedInput
from .contract import CanonicalInputContract, ResolvedInputContract
from .factory import create_input_adapter
from .frame import DepthObservation, FrameInputs, FrameMetadata, source_frame_label
from .identity import InputIdentities, validate_core_identity_match
from .preflight import PreflightReport

__all__ = [
    "CanonicalInputContract",
    "DepthObservation",
    "FrameInputs",
    "FrameMetadata",
    "InputAdapter",
    "InputIdentities",
    "PreflightReport",
    "ResolvedInput",
    "ResolvedInputContract",
    "source_frame_label",
    "create_input_adapter",
    "validate_core_identity_match",
]
