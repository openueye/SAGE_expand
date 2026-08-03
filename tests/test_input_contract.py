"""Unit coverage for the canonical input contract and Core assembly."""

from __future__ import annotations

import numpy as np
import pytest

from sage.core_input import CoreObservationAssembler
from sage.foundation.contracts import INVALID_SOURCE_TYPE, SourceType
from sage.foundation.source_policy import descriptor_for_type
from sage.input.contract import CanonicalInputContract, ResolvedInputContract
from sage.input.frame import DepthObservation, FrameInputs, FrameMetadata
from sage.input.identity import (
    CanonicalSequenceDigest,
    InputIdentities,
    validate_core_identity_match,
)

from helpers import canonical_frame, intrinsics_matrix


def _observation(depth_value: float, mask: np.ndarray) -> DepthObservation:
    depth = np.where(mask, depth_value, 0.0).astype(np.float32)
    return DepthObservation(depth, mask)


def _frame(*, center: np.ndarray, fused: np.ndarray | None = None) -> FrameInputs:
    height, width = center.shape
    return canonical_frame(
        rgb=np.full((height, width, 3), 0.25, dtype=np.float32),
        intrinsics=intrinsics_matrix(10.0, 10.0, width / 2, height / 2),
        center=_observation(5.0, center),
        fused=None if fused is None else _observation(7.0, fused),
    )


def test_canonical_arrays_are_immutable() -> None:
    frame = _frame(center=np.ones((2, 2), dtype=bool))
    for array in (frame.image_rgb, frame.intrinsics, frame.reference_from_camera):
        assert not array.flags.writeable
    assert not frame.lidar_center.depth_m.flags.writeable
    with pytest.raises(ValueError):
        frame.image_rgb[0, 0, 0] = 1.0


def test_canonical_frame_copies_its_inputs() -> None:
    source = np.full((2, 2, 3), 0.5, dtype=np.float32)
    frame = canonical_frame(
        rgb=source, intrinsics=intrinsics_matrix(1.0, 1.0, 1.0, 1.0),
    )
    source[0, 0, 0] = 0.9
    assert frame.image_rgb[0, 0, 0] == pytest.approx(0.5)


def test_fused_must_not_overlap_center() -> None:
    overlap = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="Fused observation must be invalid"):
        _frame(center=overlap, fused=overlap)


def test_depth_observation_rejects_invalid_pixel_with_depth() -> None:
    depth = np.full((2, 2), 3.0, dtype=np.float32)
    with pytest.raises(ValueError, match="support_count must be zero outside"):
        DepthObservation(
            depth,
            np.zeros((2, 2), dtype=bool),
            support_count=np.ones((2, 2), dtype=np.uint16),
        )


def test_assembler_lets_center_win_and_fused_fill_holes() -> None:
    center = np.array([[True, False], [False, False]])
    fused = np.array([[False, True], [False, False]])
    assembled = CoreObservationAssembler(("LIDAR_CENTER", "LIDAR_FUSED")).assemble(
        _frame(center=center, fused=fused)
    )
    mapping = assembled.mapping
    assert mapping.depth_m.tolist() == [[5.0, 7.0], [0.0, 0.0]]
    assert mapping.source_types.tolist() == [
        [int(SourceType.LIDAR_CENTER), int(SourceType.LIDAR_FUSED)],
        [int(INVALID_SOURCE_TYPE), int(INVALID_SOURCE_TYPE)],
    ]
    assert mapping.confidences[0, 0] == pytest.approx(
        descriptor_for_type(SourceType.LIDAR_CENTER).default_confidence
    )
    assert mapping.confidences[0, 1] == pytest.approx(
        descriptor_for_type(SourceType.LIDAR_FUSED).default_confidence
    )


def test_assembler_keeps_both_sources_separate_for_growth() -> None:
    center = np.array([[True, False], [False, False]])
    fused = np.array([[False, True], [True, False]])
    assembled = CoreObservationAssembler(("LIDAR_CENTER", "LIDAR_FUSED")).assemble(
        _frame(center=center, fused=fused)
    )
    by_source = {evidence.source_type: evidence for evidence in assembled.growth.evidences}
    assert set(by_source) == {SourceType.LIDAR_CENTER, SourceType.LIDAR_FUSED}
    assert by_source[SourceType.LIDAR_CENTER].valid_mask.tolist() == center.tolist()
    assert by_source[SourceType.LIDAR_FUSED].valid_mask.tolist() == fused.tolist()


def test_assembler_requires_every_enabled_source() -> None:
    frame = _frame(center=np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="missing enabled source LIDAR_FUSED"):
        CoreObservationAssembler(("LIDAR_CENTER", "LIDAR_FUSED")).assemble(frame)


def _contract(frame_count: int = 2) -> CanonicalInputContract:
    return CanonicalInputContract(
        frame_count=frame_count,
        reference_frame="odom",
        camera_frame="camera",
        image_size=(2, 2),
        sources=("LIDAR_CENTER",),
        fusion={"policy": "centered_window", "causal": False, "max_scans": 5},
    )


def test_canonical_contract_round_trips_through_its_payload() -> None:
    contract = _contract()
    restored = CanonicalInputContract.from_payload(contract.payload())
    assert restored.payload() == contract.payload()
    assert restored.identity == contract.identity


def test_canonical_contract_rejects_a_foreign_schema() -> None:
    payload = {**_contract().payload(), "schema_name": "legacy_scene"}
    with pytest.raises(ValueError, match="Unsupported canonical input contract schema"):
        CanonicalInputContract.from_payload(payload)


def test_contract_identity_separates_canonical_from_provenance() -> None:
    canonical = _contract()
    first = ResolvedInputContract(
        adapter_type="rosbag2",
        canonical=canonical,
        adapter_details={"adapter_type": "rosbag2", "topics": {"lidar": "/a"}},
    )
    second = ResolvedInputContract(
        adapter_type="prepared_scene",
        canonical=canonical,
        adapter_details={"adapter_type": "prepared_scene"},
    )
    assert first.canonical_contract_identity == second.canonical_contract_identity
    assert first.adapter_provenance_identity != second.adapter_provenance_identity


def test_sequence_digest_tracks_frame_content_and_order() -> None:
    frames = [
        canonical_frame(
            frame_index=index,
            timestamp_ns=index,
            rgb=np.full((2, 2, 3), 0.5, dtype=np.float32),
            intrinsics=intrinsics_matrix(1.0, 1.0, 1.0, 1.0),
        )
        for index in range(2)
    ]

    def digest(sequence) -> str:
        value = CanonicalSequenceDigest("contract")
        for frame in sequence:
            value.update(frame)
        return value.hexdigest()

    assert digest(frames) == digest(frames)
    changed = canonical_frame(
        frame_index=1,
        timestamp_ns=1,
        rgb=np.full((2, 2, 3), 0.6, dtype=np.float32),
        intrinsics=intrinsics_matrix(1.0, 1.0, 1.0, 1.0),
    )
    assert digest(frames) != digest([frames[0], changed])
    with pytest.raises(ValueError, match="expects frame_index"):
        digest([frames[1]])


def _identities(sequence: str, contract: str) -> InputIdentities:
    return InputIdentities(
        source_identity="0" * 64,
        adapter_provenance_identity="1" * 64,
        canonical_contract_identity=contract,
        canonical_sequence_identity=sequence,
    )


def test_core_identity_match_ignores_provenance_but_not_content() -> None:
    current = _identities("a" * 64, "b" * 64)
    other_provenance = InputIdentities(
        source_identity="2" * 64,
        adapter_provenance_identity="3" * 64,
        canonical_contract_identity="b" * 64,
        canonical_sequence_identity="a" * 64,
    )
    validate_core_identity_match(other_provenance.payload(), current)
    with pytest.raises(ValueError, match="incompatible"):
        validate_core_identity_match(_identities("c" * 64, "b" * 64).payload(), current)
