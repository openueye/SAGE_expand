"""Transient Stage 2 cache coverage."""

from __future__ import annotations

import numpy as np
import pytest

from sage.core_input import CoreObservationAssembler
from sage.foundation.contracts import DenseGeometryPrior, DensePriorDiagnostics
from sage.input.frame import DepthObservation
from sage.input.identity import InputIdentities
from sage.refinement.stage2_cache import (
    Stage2DensePriorCache,
    Stage2DensePriorCacheWriter,
    Stage2InputCache,
    Stage2InputCacheWriter,
    remove_stage2_cache,
    remove_stage2_dense_prior_cache,
)

from helpers import canonical_frame, intrinsics_matrix


def _mapping_frame(index: int):
    valid = np.array([[True, False], [False, True]], dtype=bool)
    depth = np.where(valid, 3.0, 0.0).astype(np.float32)
    canonical = canonical_frame(
        frame_index=index,
        timestamp_ns=100 + index,
        rgb=np.full((2, 2, 3), 128 / 255, dtype=np.float32),
        intrinsics=intrinsics_matrix(2.0, 2.0, 1.0, 1.0),
        center=DepthObservation(depth, valid),
    )
    return CoreObservationAssembler(("LIDAR_CENTER",)).assemble(canonical)


def _identities() -> InputIdentities:
    return InputIdentities(
        source_identity="0" * 64,
        adapter_provenance_identity="1" * 64,
        canonical_contract_identity="2" * 64,
        canonical_sequence_identity="3" * 64,
    )


def test_stage2_cache_round_trip_is_identity_bound_and_cpu_bounded(tmp_path) -> None:
    destination = tmp_path / ".stage2-input-cache"
    writer = Stage2InputCacheWriter(destination)
    writer.write(_mapping_frame(0))
    writer.write(_mapping_frame(4))
    writer.finalize(_identities())

    cache = Stage2InputCache(destination, cpu_lru_frames=1)
    cache.validate_checkpoint_identity({"input": _identities().payload()})
    first = cache.load(0)
    second = cache.load(4)

    assert cache.frame_indices == (0, 4)
    np.testing.assert_allclose(first.rgb, 128 / 255)
    np.testing.assert_allclose(second.mapping.depth_m, [[3, 0], [0, 3]])
    assert len(cache._lru) == 1
    with pytest.raises(ValueError, match="incompatible"):
        cache.validate_checkpoint_identity({"input": {
            **_identities().payload(),
            "canonical_sequence_identity": "4" * 64,
        }})


def test_stage2_cache_is_removed_only_from_its_explicit_path(tmp_path) -> None:
    destination = tmp_path / ".stage2-input-cache"
    writer = Stage2InputCacheWriter(destination)
    writer.write(_mapping_frame(0))
    writer.finalize(_identities())

    remove_stage2_cache(destination)

    assert not destination.exists()
    with pytest.raises(ValueError, match="non-transient"):
        remove_stage2_cache(tmp_path / "unrelated")


def test_stage2_dense_priors_are_disk_backed_with_a_bounded_lru(tmp_path) -> None:
    destination = tmp_path / ".stage2-input-cache"
    writer = Stage2InputCacheWriter(destination)
    writer.write(_mapping_frame(0))
    writer.write(_mapping_frame(4))
    writer.finalize(_identities())
    frames = Stage2InputCache(destination)
    prior_writer = Stage2DensePriorCacheWriter(frames)
    for index in frames.frame_indices:
        valid = np.array([[True, False], [False, True]], dtype=bool)
        prior_writer.write(index, DenseGeometryPrior(
            raw_depth_m=np.where(valid, 4.0, 0.0).astype(np.float32),
            aligned_depth_m=np.where(valid, 3.0, 0.0).astype(np.float32),
            valid_mask=valid,
            confidence=np.where(valid, 0.5, 0.0).astype(np.float32),
            scale_grid=np.ones((1, 1), dtype=np.float32),
            diagnostics=DensePriorDiagnostics(2, 1, 1.0, 0.0, False),
        ))
    prior_writer.finalize()

    priors = Stage2DensePriorCache(frames, cpu_lru_frames=1)
    assert priors.load(0).diagnostics.lidar_support == 2
    np.testing.assert_allclose(priors.load(4).aligned_depth_m, [[3, 0], [0, 3]])
    assert len(priors._lru) == 1

    remove_stage2_dense_prior_cache(frames)

    assert not (destination / "dense-priors").exists()
