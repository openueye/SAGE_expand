import numpy as np
import torch
from dataclasses import replace
from types import SimpleNamespace

from sage.engine.growth import GrowthBuilder
from sage.foundation.config import (
    ALPHA_NORMALIZED_DEPTH_POLICY,
    GaussianInitializationConfig,
    GrowthConfig,
    ResidualThreshold,
)
from sage.foundation.contracts import (
    CameraIntrinsics,
    DepthEvidence,
    FrameInputs,
    GrowthInputs,
    InputSourceFamily,
    INVALID_SOURCE_TYPE,
    MappingObservation,
    Pose,
    SourceType,
)


def _frame(candidate_depth_m: float) -> FrameInputs:
    return FrameInputs(
        index=0,
        stem="growth-depth-policy-test",
        timestamp_ns=0,
        intrinsics=CameraIntrinsics(1, 1, 1.0, 1.0, 0.0, 0.0),
        pose=Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        rgb=np.zeros((1, 1, 3), dtype=np.float32),
        mapping=MappingObservation(
            np.zeros((1, 1), dtype=np.float32),
            np.full((1, 1), INVALID_SOURCE_TYPE, dtype=np.uint8),
            np.zeros((1, 1), dtype=np.float32),
        ),
        growth=GrowthInputs((
            DepthEvidence(
                SourceType.LIDAR_CENTER,
                np.full((1, 1), candidate_depth_m, dtype=np.float32),
                np.ones((1, 1), dtype=np.bool_),
                np.ones((1, 1), dtype=np.float32),
                InputSourceFamily.LIDAR_RAW,
            ),
        )),
    )


def _builder() -> GrowthBuilder:
    return GrowthBuilder(
        GrowthConfig(),
        device="cpu",
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
    )


def _covered_render() -> SimpleNamespace:
    """已覆盖的渲染 (alpha 高于 coverage 阈值)，alpha 归一化深度为 9.0/0.9 = 10.0。

    alpha 必须高于 coverage_alpha_threshold，否则 coverage 通道会先于几何通道
    决定结果，这两个用例就测不到深度残差门。
    """
    return SimpleNamespace(
        rgb=torch.zeros((1, 1, 3)),
        accumulated_depth=torch.tensor([[9.0]]),
        alpha=torch.tensor([[0.9]]),
    )


def test_growth_depth_policy_is_alpha_normalized() -> None:
    assert GrowthConfig().depth_policy == ALPHA_NORMALIZED_DEPTH_POLICY


def test_growth_does_not_duplicate_matching_alpha_normalized_depth() -> None:
    result = _builder().build(_frame(10.0), _covered_render())

    assert result.batch.means3d.shape[0] == 0
    assert result.stats.by_source["LIDAR_CENTER"]["rejection_reasons"] == {
        "gate_blocked": 1,
    }


def test_growth_keeps_real_alpha_normalized_depth_discrepancy() -> None:
    result = _builder().build(_frame(8.0), _covered_render())

    assert result.batch.means3d.shape[0] == 1
    assert result.stats.by_source["LIDAR_CENTER"]["accepted"] == 1


_HEIGHT = 8
_WIDTH = 8


def _multi_pixel_frame() -> FrameInputs:
    """两个源各占一半画面，互不重叠，避免 2D 仲裁干扰预算测试。"""
    left = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    left[:, : _WIDTH // 2] = True
    right = ~left
    return FrameInputs(
        index=1,
        stem="growth-budget-test",
        timestamp_ns=0,
        intrinsics=CameraIntrinsics(_WIDTH, _HEIGHT, 20.0, 20.0, _WIDTH / 2, _HEIGHT / 2),
        pose=Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        rgb=np.full((_HEIGHT, _WIDTH, 3), 0.5, dtype=np.float32),
        mapping=MappingObservation(
            np.zeros((_HEIGHT, _WIDTH), dtype=np.float32),
            np.full((_HEIGHT, _WIDTH), INVALID_SOURCE_TYPE, dtype=np.uint8),
            np.zeros((_HEIGHT, _WIDTH), dtype=np.float32),
        ),
        growth=GrowthInputs((
            DepthEvidence(
                SourceType.LIDAR_CENTER,
                np.full((_HEIGHT, _WIDTH), 5.0, dtype=np.float32),
                left,
                left.astype(np.float32),
                InputSourceFamily.LIDAR_RAW,
            ),
            DepthEvidence(
                SourceType.SPNET_COMPLETED,
                np.full((_HEIGHT, _WIDTH), 7.0, dtype=np.float32),
                right,
                right.astype(np.float32),
                InputSourceFamily.LIDAR_RAW,
            ),
        )),
    )


def _uncovered_render(rendered_rgb: float) -> SimpleNamespace:
    """完全未覆盖：alpha=0 使渲染深度无效，几何通道恒不成立。"""
    return SimpleNamespace(
        rgb=torch.full((_HEIGHT, _WIDTH, 3), rendered_rgb),
        accumulated_depth=torch.zeros((_HEIGHT, _WIDTH)),
        alpha=torch.zeros((_HEIGHT, _WIDTH)),
    )


def _quota_builder(quotas: dict[str, int] | None) -> GrowthBuilder:
    return GrowthBuilder(
        GrowthConfig(max_new_per_commit=quotas),
        device="cpu",
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
    )


def _quotas(lidar: int, spnet: int) -> dict[str, int]:
    return {"LIDAR_CENTER": lidar, "LIDAR_FUSED": lidar, "SPNET_COMPLETED": spnet}


def test_growth_quota_caps_each_source_independently() -> None:
    frame = _multi_pixel_frame()
    render = _uncovered_render(0.0)
    per_source = _HEIGHT * _WIDTH // 2

    unlimited = _quota_builder(None).build(frame, render)
    assert unlimited.batch.means3d.shape[0] == _HEIGHT * _WIDTH

    for lidar, spnet in ((per_source, per_source), (10, per_source), (per_source, 10), (1, 1)):
        result = _quota_builder(_quotas(lidar, spnet)).build(frame, render)

        assert int(result.stats.by_source["LIDAR_CENTER"]["accepted"]) == min(lidar, per_source)
        # 关键：LiDAR 收紧不得吃掉 SPNet 的额度，两者互不影响
        assert int(result.stats.by_source["SPNET_COMPLETED"]["accepted"]) == min(spnet, per_source)
        assert result.batch.means3d.shape[0] == min(lidar, per_source) + min(spnet, per_source)


def test_growth_quota_does_not_starve_low_priority_source() -> None:
    """全局总量 + 词典序会让 SPNet 归零；每源配额必须避免这一点。"""
    frame = _multi_pixel_frame()
    result = _quota_builder(_quotas(_HEIGHT * _WIDTH, 4)).build(frame, _uncovered_render(0.0))

    assert int(result.stats.by_source["SPNET_COMPLETED"]["accepted"]) == 4
    spnet = result.stats.by_source["SPNET_COMPLETED"]
    # 统计恒等式由 _finish_stats 强制；这里额外确认超额被显式归因
    assert spnet["rejection_reasons"]["quota_exhausted"] == _HEIGHT * _WIDTH // 2 - 4


def test_arbitration_follows_declared_priority_not_iteration_order() -> None:
    """2D 仲裁必须按 descriptor.priority 裁决，而不是按证据占位顺序。"""
    overlap = np.ones((_HEIGHT, _WIDTH), dtype=np.bool_)
    frame = replace(
        _multi_pixel_frame(),
        growth=GrowthInputs((
            DepthEvidence(
                SourceType.LIDAR_CENTER,
                np.full((_HEIGHT, _WIDTH), 5.0, dtype=np.float32),
                overlap,
                overlap.astype(np.float32),
                InputSourceFamily.LIDAR_RAW,
            ),
            DepthEvidence(
                SourceType.SPNET_COMPLETED,
                np.full((_HEIGHT, _WIDTH), 5.0, dtype=np.float32),
                overlap,
                overlap.astype(np.float32),
                InputSourceFamily.LIDAR_RAW,
            ),
        )),
    )
    result = _quota_builder(None).build(frame, _uncovered_render(0.0))

    assert result.batch.means3d.shape[0] == _HEIGHT * _WIDTH
    assert torch.all(result.batch.source_types == int(SourceType.LIDAR_CENTER))
    assert result.stats.by_source["SPNET_COMPLETED"]["rejection_reasons"] == {
        "duplicate_2d": _HEIGHT * _WIDTH,
    }


def test_growth_quota_samples_uniformly_instead_of_truncating() -> None:
    frame = _multi_pixel_frame()
    quota = 4
    result = _quota_builder(_quotas(quota, quota)).build(frame, _uncovered_render(0.0))
    repeat = _quota_builder(_quotas(quota, quota)).build(frame, _uncovered_render(0.0))

    # 光栅序截断只会留下最上面几行；均匀抽样必须跨越整个纵向范围
    heights = result.batch.means3d[:, 1]
    full = _quota_builder(None).build(frame, _uncovered_render(0.0)).batch.means3d[:, 1]
    span = float(heights.max() - heights.min())
    assert span > 0.5 * float(full.max() - full.min())
    assert torch.equal(result.batch.means3d, repeat.batch.means3d)


def test_growth_coverage_is_alpha_only() -> None:
    """Low alpha remains a growth need regardless of rendered RGB residual."""
    frame = _multi_pixel_frame()
    result = _builder().build(frame, _uncovered_render(0.48))

    assert result.batch.means3d.shape[0] == _HEIGHT * _WIDTH


def test_growth_rejects_candidates_already_present_in_persistent_map() -> None:
    first = _builder().build(_frame(8.0), _covered_render())
    assert first.batch.means3d.shape[0] == 1

    repeated = _builder().build(
        _frame(8.0),
        _covered_render(),
        existing_points=first.batch.means3d,
    )

    assert repeated.batch.means3d.shape[0] == 0
    assert repeated.stats.by_source["LIDAR_CENTER"]["rejection_reasons"] == {
        "duplicate_3d_existing": 1,
    }


def test_persistent_dedup_uses_distance_not_voxel_identity() -> None:
    frame = _frame(1.0)
    builder = GrowthBuilder(
        GrowthConfig(candidate_duplicate_3d_threshold_m=0.05),
        device="cpu",
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
    )

    # These points are in adjacent voxels but only 2 mm apart.
    near_boundary = builder.build(
        frame,
        _covered_render(),
        existing_points=torch.tensor([[0.051, 0.0, 1.0]]),
    )
    assert near_boundary.batch.means3d.shape[0] == 1

    # The candidate is at (0, 0, 1); move it across the voxel boundary.
    shifted_frame = replace(
        frame, pose=Pose(0.049, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    near_boundary = builder.build(
        shifted_frame,
        _covered_render(),
        existing_points=torch.tensor([[0.051, 0.0, 1.0]]),
    )
    assert near_boundary.batch.means3d.shape[0] == 0
    assert near_boundary.stats.by_source["LIDAR_CENTER"]["rejection_reasons"] == {
        "duplicate_3d_existing": 1,
    }

    # Same voxel does not imply duplicate when the Euclidean distance is large.
    same_voxel_far = builder.build(
        frame,
        _covered_render(),
        existing_points=torch.tensor([[0.049, 0.049, 1.049]]),
    )
    assert same_voxel_far.batch.means3d.shape[0] == 1


def test_growth_keeps_canonical_roles_for_slam_input() -> None:
    frame = replace(
        _multi_pixel_frame(),
        growth=GrowthInputs((
            DepthEvidence(
                SourceType.LIDAR_CENTER,
                np.full((_HEIGHT, _WIDTH), 5.0, dtype=np.float32),
                np.ones((_HEIGHT, _WIDTH), dtype=np.bool_),
                np.ones((_HEIGHT, _WIDTH), dtype=np.float32),
                InputSourceFamily.SLAM_WORLD,
            ),
        )),
    )
    result = GrowthBuilder(
        GrowthConfig(max_new_per_commit={
            "LIDAR_CENTER": 3,
            "LIDAR_FUSED": 3,
            "SPNET_COMPLETED": 3,
        }),
        device="cpu",
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
    ).build(frame, _uncovered_render(0.0))

    assert result.batch.means3d.shape[0] == 3
    assert result.stats.by_source["LIDAR_CENTER"]["accepted"] == 3


def test_growth_zero_quota_disables_only_that_source() -> None:
    result = _quota_builder(_quotas(0, 4)).build(
        _multi_pixel_frame(), _uncovered_render(0.0),
    )

    assert int(result.stats.by_source["LIDAR_CENTER"]["accepted"]) == 0
    assert int(result.stats.by_source["SPNET_COMPLETED"]["accepted"]) == 4
    assert result.stats.by_source["LIDAR_CENTER"]["rejection_reasons"] == {
        "quota_exhausted": _HEIGHT * _WIDTH // 2,
    }


def test_growth_conflict_threshold_is_strict() -> None:
    residuals = {
        "LIDAR_CENTER": ResidualThreshold(0.2, 0.0),
        "LIDAR_FUSED": ResidualThreshold(0.2, 0.0),
        "SPNET_COMPLETED": ResidualThreshold(0.2, 0.0),
    }
    builder = GrowthBuilder(
        GrowthConfig(residual_thresholds=residuals),
        device="cpu",
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
    )

    at_threshold = builder.build(_frame(10.2), _covered_render())
    beyond_threshold = builder.build(_frame(10.2001), _covered_render())

    assert at_threshold.batch.means3d.shape[0] == 0
    assert beyond_threshold.batch.means3d.shape[0] == 1


def test_growth_stats_partition_mixed_rejection_reasons() -> None:
    result = _quota_builder(_quotas(2, 2)).build(
        _multi_pixel_frame(), _uncovered_render(0.0),
    )

    for source_stats in result.stats.by_source.values():
        reasons = source_stats["rejection_reasons"]
        assert source_stats["rejected"] == sum(reasons.values())
        assert source_stats["proposed"] == source_stats["accepted"] + source_stats["rejected"]
