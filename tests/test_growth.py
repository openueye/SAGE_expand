import numpy as np
import torch
from types import SimpleNamespace

from sage.engine.growth import GrowthBuilder
from sage.foundation.config import (
    ALPHA_NORMALIZED_DEPTH_POLICY,
    GaussianInitializationConfig,
    GrowthConfig,
)
from sage.foundation.source_policy import SOURCE_DESCRIPTORS
from sage.foundation.contracts import (
    CameraIntrinsics,
    DepthEvidence,
    FrameInputs,
    GrowthInputs,
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
                SourceType.LIDAR_RAW,
                np.full((1, 1), candidate_depth_m, dtype=np.float32),
                np.ones((1, 1), dtype=np.bool_),
                np.ones((1, 1), dtype=np.float32),
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
    assert result.stats.by_source["LIDAR_RAW"]["rejection_reasons"] == {
        "gate_blocked": 1,
    }


def test_growth_keeps_real_alpha_normalized_depth_discrepancy() -> None:
    result = _builder().build(_frame(8.0), _covered_render())

    assert result.batch.means3d.shape[0] == 1
    assert result.stats.by_source["LIDAR_RAW"]["accepted"] == 1


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
                SourceType.LIDAR_RAW,
                np.full((_HEIGHT, _WIDTH), 5.0, dtype=np.float32),
                left,
                left.astype(np.float32),
            ),
            DepthEvidence(
                SourceType.SPNET_BLIND,
                np.full((_HEIGHT, _WIDTH), 7.0, dtype=np.float32),
                right,
                right.astype(np.float32),
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
    return {"LIDAR_RAW": lidar, "LIDAR_FUSED5": lidar, "SPNET_BLIND": spnet}


def test_growth_quota_caps_each_source_independently() -> None:
    frame = _multi_pixel_frame()
    render = _uncovered_render(0.0)
    per_source = _HEIGHT * _WIDTH // 2

    unlimited = _quota_builder(None).build(frame, render)
    assert unlimited.batch.means3d.shape[0] == _HEIGHT * _WIDTH

    for lidar, spnet in ((per_source, per_source), (10, per_source), (per_source, 10), (1, 1)):
        result = _quota_builder(_quotas(lidar, spnet)).build(frame, render)

        assert int(result.stats.by_source["LIDAR_RAW"]["accepted"]) == min(lidar, per_source)
        # 关键：LiDAR 收紧不得吃掉 SPNet 的额度，两者互不影响
        assert int(result.stats.by_source["SPNET_BLIND"]["accepted"]) == min(spnet, per_source)
        assert result.batch.means3d.shape[0] == min(lidar, per_source) + min(spnet, per_source)


def test_growth_quota_does_not_starve_low_priority_source() -> None:
    """全局总量 + 词典序会让 SPNet 归零；每源配额必须避免这一点。"""
    frame = _multi_pixel_frame()
    result = _quota_builder(_quotas(_HEIGHT * _WIDTH, 4)).build(frame, _uncovered_render(0.0))

    assert int(result.stats.by_source["SPNET_BLIND"]["accepted"]) == 4
    spnet = result.stats.by_source["SPNET_BLIND"]
    # 统计恒等式由 _finish_stats 强制；这里额外确认超额被显式归因
    assert spnet["rejection_reasons"]["quota_exhausted"] == _HEIGHT * _WIDTH // 2 - 4


def test_arbitration_follows_declared_priority_not_iteration_order() -> None:
    """2D 仲裁必须按 descriptor.priority 裁决，而不是按传入顺序。

    `GrowthInputs` 已在契约层强制证据按优先级排序，所以这里直接打仲裁接缝：
    即使字典以相反顺序传入，重叠像素也必须判给 LiDAR。
    """
    overlap = torch.ones((_HEIGHT, _WIDTH), dtype=torch.bool)
    unused = torch.zeros((_HEIGHT, _WIDTH))
    reversed_order = {
        int(SourceType.SPNET_BLIND): (overlap, unused, unused),
        int(SourceType.LIDAR_RAW): (overlap, unused, unused),
    }
    stats = GrowthBuilder._empty_stats(SOURCE_DESCRIPTORS)

    keeps = GrowthBuilder._arbitrate_2d(reversed_order, SOURCE_DESCRIPTORS, stats)

    assert int(keeps[int(SourceType.LIDAR_RAW)].sum()) == _HEIGHT * _WIDTH
    assert int(keeps[int(SourceType.SPNET_BLIND)].sum()) == 0
    assert stats["SPNET_BLIND"]["rejection_reasons"]["duplicate_2d"] == _HEIGHT * _WIDTH
    # 返回顺序也必须是优先级序，下游 3D 去重的"先到先得"依赖它
    assert list(keeps) == [int(SourceType.LIDAR_RAW), int(SourceType.SPNET_BLIND)]


def test_growth_quota_samples_uniformly_instead_of_truncating() -> None:
    frame = _multi_pixel_frame()
    quota = 4
    result = _quota_builder(_quotas(quota, quota)).build(frame, _uncovered_render(0.0))

    # 光栅序截断只会留下最上面几行；均匀抽样必须跨越整个纵向范围
    heights = result.batch.means3d[:, 1]
    full = _quota_builder(None).build(frame, _uncovered_render(0.0)).batch.means3d[:, 1]
    span = float(heights.max() - heights.min())
    assert span > 0.5 * float(full.max() - full.min())


def test_growth_reports_rgb_residual_suppression_separately() -> None:
    """未覆盖但渲染颜色接近目标的低纹理像素，被 RGB 残差项挡下时要能量化。"""
    frame = _multi_pixel_frame()
    result = _builder().build(frame, _uncovered_render(0.48))

    assert result.batch.means3d.shape[0] == 0
    reasons = result.stats.by_source["LIDAR_RAW"]["rejection_reasons"]
    assert reasons == {"rgb_residual_suppressed": _HEIGHT * _WIDTH // 2}
