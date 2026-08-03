import numpy as np
import torch

from sage.data.providers.spnet import SPNetIdentity
from sage.engine.rendering import RenderOutput
from sage.foundation.config import (
    GaussianInitializationConfig,
    GrowthConfig,
    MappingConfig,
    MappingLossConfig,
    PruningConfig,
)
from sage.foundation.contracts import (
    DepthEvidence,
    GrowthInputs,
    MappingObservation,
    SourceType,
)
from sage.core_input import MappingFrame

from helpers import intrinsics_matrix, mapping_frame, pose_matrix
from sage.mapping.mapper import MappingEngine


class _Provider:
    identity = SPNetIdentity(mode="online", source_id="mapping-integration-test")

    def __init__(self) -> None:
        self.calls = 0

    def evidence_for(self, frame: MappingFrame) -> DepthEvidence:
        self.calls += 1
        depth = np.zeros(frame.rgb.shape[:2], dtype=np.float32)
        depth[0, 1] = 1.0
        valid = depth > 0
        return DepthEvidence(
            SourceType.SPNET_COMPLETED,
            depth,
            valid,
            valid.astype(np.float32),
        )


class _GradientRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model, frame: MappingFrame) -> RenderOutput:
        self.calls += 1
        shape = (frame.intrinsics.height, frame.intrinsics.width)
        rgb = (model.colors.mean(dim=0) + 0.1).reshape(1, 1, 3).expand(*shape, 3)
        depth = model.means3d[:, 2].mean().expand(shape)
        alpha = (model.opacities.mean() * 0).expand(shape)
        return RenderOutput(rgb, depth, alpha)


def _frame(index: int, translation_x: float) -> MappingFrame:
    height, width = 1, 2
    depth = np.zeros((height, width), dtype=np.float32)
    depth[0, 0] = 1.0
    valid = depth > 0
    source_types = np.full((height, width), 255, dtype=np.uint8)
    source_types[valid] = int(SourceType.LIDAR_CENTER)
    confidence = valid.astype(np.float32)
    center = DepthEvidence(SourceType.LIDAR_CENTER, depth, valid, confidence)
    return mapping_frame(
        frame_index=index,
        timestamp_ns=index,
        rgb=np.full((height, width, 3), 0.2, dtype=np.float32),
        intrinsics=intrinsics_matrix(2.0, 2.0, 0.5, 0.0),
        reference_from_camera=pose_matrix((translation_x, 0.0, 0.0)),
        mapping=MappingObservation(depth, source_types, confidence),
        growth=GrowthInputs((center,)),
    )


def test_mapping_engine_runs_growth_append_and_spnet_with_canonical_roles() -> None:
    provider = _Provider()
    renderer = _GradientRenderer()
    engine = MappingEngine(
        MappingConfig(
            map_every=1,
            keyframe_every=1,
            iterations=1,
            prune_every=1,
            prune_stop_after=0,
        ),
        PruningConfig(),
        GrowthConfig(),
        device="cpu",
        renderer=renderer,
        spnet_provider=provider,
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
        loss_policy=MappingLossConfig(),
    )

    result = engine.run((_frame(0, 0.0), _frame(1, 0.2)))

    assert result.processed_frames == 2
    assert len(result.commits) == 2
    assert provider.calls == 1
    assert result.commits[1].spnet_invoked
    assert result.commits[1].added_by_source == {
        "LIDAR_CENTER": 1,
        "LIDAR_FUSED": 0,
        "SPNET_COMPLETED": 1,
    }
    assert result.model.source_types.tolist() == [
        int(SourceType.LIDAR_CENTER),
        int(SourceType.LIDAR_CENTER),
        int(SourceType.SPNET_COMPLETED),
    ]
    assert result.optimizer_lifecycle == "persistent"
    assert result.optimizer_append_migrations == 1
    assert result.spnet_anchor_source_types == ("LIDAR_CENTER",)
