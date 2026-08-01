import numpy as np
import torch
from types import SimpleNamespace

from sage.engine.growth import GrowthBuilder
from sage.foundation.config import (
    ALPHA_NORMALIZED_DEPTH_POLICY,
    GaussianInitializationConfig,
    GrowthConfig,
)
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


def _half_alpha_render() -> SimpleNamespace:
    return SimpleNamespace(
        rgb=torch.zeros((1, 1, 3)),
        accumulated_depth=torch.tensor([[5.0]]),
        alpha=torch.tensor([[0.5]]),
    )


def test_growth_depth_policy_is_alpha_normalized() -> None:
    assert GrowthConfig().depth_policy == ALPHA_NORMALIZED_DEPTH_POLICY


def test_growth_does_not_duplicate_matching_alpha_normalized_depth() -> None:
    result = _builder().build(_frame(10.0), _half_alpha_render())

    assert result.batch.means3d.shape[0] == 0
    assert result.stats.by_source["LIDAR_RAW"]["rejection_reasons"] == {
        "gate_blocked": 1,
    }


def test_growth_keeps_real_alpha_normalized_depth_discrepancy() -> None:
    result = _builder().build(_frame(8.0), _half_alpha_render())

    assert result.batch.means3d.shape[0] == 1
    assert result.stats.by_source["LIDAR_RAW"]["accepted"] == 1
