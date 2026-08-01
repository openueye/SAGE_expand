import json
import numpy as np
import pytest
import torch

from sage.data.providers.spnet import (
    SPNetFrameGrid,
    SPNetIdentity,
    derive_spnet_grid,
    predict_spnet_evidence,
    predict_spnet_evidence_bundle,
)
from sage.data.providers.spnet_cache import (
    DenseSPNetFrame,
    ReusingDenseSPNetProvider,
    load_dense_spnet_cache,
    write_dense_spnet_cache,
)
from sage.foundation.contracts import (
    CameraIntrinsics,
    DepthEvidence,
    FrameInputs,
    GrowthInputs,
    MappingObservation,
    Pose,
    SourceType,
)
from sage.foundation.hashing import sha256_file
from sage.mapping.mapper import MappingRun, mapping_spnet_evidence
from sage.mapping.mapping_artifacts import _validate_run
from sage.refinement.refinement_run import _load_mapping_dense_cache


def _frame() -> FrameInputs:
    height = width = 8
    center_depth = np.zeros((height, width), dtype=np.float32)
    center_depth[::4, ::4] = 5.0
    center_valid = center_depth > 0
    center = DepthEvidence(
        SourceType.LIDAR_SLAM_CENTER,
        center_depth,
        center_valid,
        center_valid.astype(np.float32),
    )
    return FrameInputs(
        4,
        "000004",
        4,
        CameraIntrinsics(width, height, 10.0, 10.0, 4.0, 4.0),
        Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        np.full((height, width, 3), 0.5, dtype=np.float32),
        MappingObservation(
            center_depth.copy(),
            np.where(
                center_valid,
                int(SourceType.LIDAR_SLAM_CENTER),
                255,
            ).astype(np.uint8),
            center_valid.astype(np.float32),
        ),
        GrowthInputs((center,)),
    )


class _Predictor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, rgb, depth, mask) -> torch.Tensor:
        self.calls += 1
        assert rgb.shape == (1, 3, 32, 32)
        assert depth.shape == mask.shape == (1, 1, 32, 32)
        return torch.full_like(depth, 0.025)


def test_spnet_bundle_reuses_one_prediction_for_dense_and_growth_evidence() -> None:
    frame = _frame()
    predictor = _Predictor()

    bundle = predict_spnet_evidence_bundle(
        frame,
        predictor,
        depth_scale_m=200.0,
        confidence=0.4,
        sample_stride=2,
    )
    expected_growth = predict_spnet_evidence(
        frame,
        _Predictor(),
        depth_scale_m=200.0,
        confidence=0.4,
        sample_stride=2,
    )

    assert predictor.calls == 1
    np.testing.assert_array_equal(
        bundle.growth_evidence.depth_m, expected_growth.depth_m,
    )
    np.testing.assert_array_equal(
        bundle.growth_evidence.valid_mask, expected_growth.valid_mask,
    )
    np.testing.assert_array_equal(
        bundle.growth_evidence.confidence, expected_growth.confidence,
    )
    assert bundle.dense_evidence.valid_mask.all()
    np.testing.assert_allclose(bundle.dense_evidence.depth_m, 5.0)
    expected_valid = np.zeros((8, 8), dtype=np.bool_)
    expected_valid[::2, ::2] = True
    expected_valid[::4, ::4] = False
    np.testing.assert_array_equal(bundle.growth_evidence.valid_mask, expected_valid)


def _identity(*indices: int) -> SPNetIdentity:
    return SPNetIdentity(
        mode="online",
        source_id="test-source",
        model_id="test-model",
        weights_sha256="a" * 64,
        source_content_sha256="b" * 64,
        source_commit="c" * 40,
        depth_scale_m=200.0,
        confidence=0.4,
        sample_stride=2,
        frame_grids=tuple(
            SPNetFrameGrid(index, f"{index:06d}", derive_spnet_grid((8, 8)))
            for index in indices
        ),
    )


class _FallbackProvider:
    def __init__(self) -> None:
        self.identity = _identity()
        self.calls = 0

    def dense_evidence_for(self, frame: FrameInputs) -> DepthEvidence:
        self.calls += 1
        depth = np.full(frame.rgb.shape[:2], 6.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        return DepthEvidence(
            SourceType.SPNET_BLIND,
            depth,
            valid,
            np.full_like(depth, 0.4),
        )


class _LegacyMappingProvider:
    def __init__(self) -> None:
        self.identity = _identity()
        self.calls = 0

    def evidence_for(self, frame: FrameInputs) -> DepthEvidence:
        self.calls += 1
        depth = np.full(frame.rgb.shape[:2], 7.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        return DepthEvidence(
            SourceType.SPNET_BLIND,
            depth,
            valid,
            np.full_like(depth, 0.4),
        )


def test_mapping_preserves_legacy_spnet_provider_contract() -> None:
    provider = _LegacyMappingProvider()

    evidence, dense_frame = mapping_spnet_evidence(provider, _frame())

    assert provider.calls == 1
    assert evidence.source_type == SourceType.SPNET_BLIND
    assert dense_frame is None


def test_mapping_artifacts_allow_legacy_provider_without_dense_cache() -> None:
    _validate_run(MappingRun(
        model=object(),
        commits=(),
        processed_frames=1,
        duration_seconds=1.0,
        fps=1.0,
        spnet_actual_invocations=1,
    ))


def test_dense_spnet_cache_round_trip_reuses_cached_frames(tmp_path) -> None:
    frame = _frame()
    cached_depth = np.full(frame.rgb.shape[:2], 5.0, dtype=np.float32)
    cache_path = tmp_path / "spnet_dense.pt"
    write_dense_spnet_cache(
        cache_path,
        (DenseSPNetFrame(frame.index, frame.stem, cached_depth),),
        _identity(frame.index),
    )
    cache = load_dense_spnet_cache(cache_path)
    fallback = _FallbackProvider()
    provider = ReusingDenseSPNetProvider(cache, fallback)

    first_frame = FrameInputs(
        0,
        "000000",
        frame.timestamp_ns,
        frame.intrinsics,
        frame.pose,
        frame.rgb,
        frame.mapping,
        frame.growth,
    )
    live = provider.dense_evidence_for(first_frame)
    reused = provider.dense_evidence_for(frame)

    assert fallback.calls == 1
    np.testing.assert_allclose(live.depth_m, 6.0)
    np.testing.assert_allclose(reused.depth_m, 5.0)
    assert provider.reused_frames == 1
    assert provider.live_frames == 1
    assert [item.frame_index for item in provider.identity.frame_grids] == [0, 4]


def test_refinement_loads_dense_cache_only_through_structure_manifest(tmp_path) -> None:
    frame = _frame()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    cache_path = tmp_path / "spnet_dense.pt"
    write_dense_spnet_cache(
        cache_path,
        (DenseSPNetFrame(
            frame.index,
            frame.stem,
            np.full(frame.rgb.shape[:2], 5.0, dtype=np.float32),
        ),),
        _identity(frame.index),
    )
    manifest = {
        "artifacts": {
            "checkpoint": {
                "path": checkpoint.name,
                "sha256": sha256_file(checkpoint),
            },
            "dense_spnet_cache": {
                "path": cache_path.name,
                "sha256": sha256_file(cache_path),
                "frame_count": 1,
            },
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    loaded = _load_mapping_dense_cache(
        checkpoint,
        source_checkpoint_sha256=sha256_file(checkpoint),
    )
    assert loaded is not None
    assert [item.frame_index for item in loaded[0].frames] == [frame.index]

    cache_path.write_bytes(cache_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="cache hash"):
        _load_mapping_dense_cache(
            checkpoint,
            source_checkpoint_sha256=sha256_file(checkpoint),
        )


def test_refinement_without_structure_manifest_falls_back_online(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    assert _load_mapping_dense_cache(
        checkpoint,
        source_checkpoint_sha256=sha256_file(checkpoint),
    ) is None
