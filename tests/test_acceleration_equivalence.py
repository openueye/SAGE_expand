from __future__ import annotations

import numpy as np
import torch

from sage.appearance_config import (
    APPEARANCE_REFINEMENT_SCHEMA,
    FIRST_MAPPING_FRAME_EXPOSURE_GAUGE,
    SEEDED_RANDOM_MAPPING_FRAME_VARIANT,
    AppearanceRefinementConfig,
)
from sage.appearance_refinement import (
    AppearanceExposureNuisance,
    AppearanceObjective,
    AppearanceRefiner,
)
from sage.config import MappingLossConfig
from sage.contracts import (
    CameraIntrinsics,
    DenseGeometryPrior,
    DensePriorDiagnostics,
    MappingObservation,
    SourceType,
)
from sage.dense_geometry_config import DenseGeometryPolicy, DensePriorPolicy
from sage.dense_geometry_objective import (
    dense_geometry_objective,
    prepare_dense_geometry_static,
)
from sage.losses import mapping_loss


def _mapping(shape: tuple[int, int]) -> MappingObservation:
    depth = np.full(shape, 2.0, dtype=np.float32)
    source_types = np.full(
        shape,
        int(SourceType.LIDAR_SLAM_CENTER),
        dtype=np.uint8,
    )
    confidence = np.ones(shape, dtype=np.float32)
    return MappingObservation(depth, source_types, confidence)


def test_refinement_mapping_fast_path_preserves_depth_term() -> None:
    torch.manual_seed(3)
    shape = (12, 10)
    mapping = _mapping(shape)
    rendered_rgb = torch.rand((*shape, 3), dtype=torch.float32)
    target_rgb = torch.rand((*shape, 3), dtype=torch.float32)
    alpha = torch.full(shape, 0.8, dtype=torch.float32)
    accumulated_depth = alpha * 2.1
    policy = MappingLossConfig(depth_weight=0.25, alpha_support_a0=0.5)

    _, full_terms = mapping_loss(
        rendered_rgb,
        target_rgb,
        accumulated_depth,
        mapping,
        rendered_alpha=alpha,
        policy=policy,
    )
    source_types = torch.as_tensor(mapping.source_types)
    fast_loss, fast_terms = mapping_loss(
        rendered_rgb,
        target_rgb,
        accumulated_depth,
        mapping,
        rendered_alpha=alpha,
        policy=policy,
        target_depth=torch.as_tensor(mapping.depth_m),
        source_masks={
            "center": source_types == int(SourceType.LIDAR_SLAM_CENTER),
            "fused5": source_types == int(SourceType.LIDAR_SLAM_FUSED5),
        },
        include_image=False,
        include_diagnostics=False,
        validate_rgb=False,
    )

    torch.testing.assert_close(fast_loss, full_terms["depth"])
    torch.testing.assert_close(fast_terms["depth"], full_terms["depth"])


def test_cached_dense_normal_objective_matches_uncached_gradient() -> None:
    height, width = 10, 12
    rows, columns = np.mgrid[:height, :width]
    target_depth = (2.0 + rows * 0.01 + columns * 0.02).astype(np.float32)
    valid = np.ones((height, width), dtype=np.bool_)
    confidence = np.ones((height, width), dtype=np.float32)
    prior = DenseGeometryPrior(
        target_depth,
        target_depth,
        valid,
        confidence,
        np.ones((2, 2), dtype=np.float32),
        DensePriorDiagnostics(120, 4, 0.0, 0.0, False),
    )
    intrinsics = CameraIntrinsics(width, height, 40.0, 40.0, 5.5, 4.5)
    prior_policy = DensePriorPolicy(2, 2, 1, 2.0, 0.01)
    geometry = DenseGeometryPolicy(
        dense_depth_weight=0.0,
        dense_normal_weight=1.0,
        normal_smoothness_weight=0.0,
        edge_weight_gamma=10.0,
        alpha_support_a0=0.5,
        max_relative_depth_jump=0.2,
    )
    rgb = torch.linspace(
        0,
        1,
        height * width * 3,
        dtype=torch.float32,
    ).reshape(height, width, 3)
    alpha = torch.full((height, width), 0.9, dtype=torch.float32)
    accumulated_a = (torch.from_numpy(target_depth) * alpha).requires_grad_()
    accumulated_b = accumulated_a.detach().clone().requires_grad_()

    uncached, _ = dense_geometry_objective(
        accumulated_a,
        alpha,
        prior,
        rgb,
        intrinsics,
        prior_policy,
        geometry,
    )
    static = prepare_dense_geometry_static(
        prior,
        rgb,
        intrinsics,
        prior_policy,
        geometry,
    )
    cached, _ = dense_geometry_objective(
        accumulated_b,
        alpha,
        prior,
        rgb,
        intrinsics,
        prior_policy,
        geometry,
        static=static,
    )
    uncached.backward()
    cached.backward()

    torch.testing.assert_close(cached, uncached)
    torch.testing.assert_close(accumulated_b.grad, accumulated_a.grad)


class _TinyAppearanceModel(torch.nn.Module):
    PARAMETER_NAMES = ("colors", "opacity_logits")

    def __init__(self) -> None:
        super().__init__()
        self.colors = torch.nn.Parameter(torch.full((2, 3), 0.25))
        self.opacity_logits = torch.nn.Parameter(torch.zeros((2, 1)))

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def count(self) -> int:
        return self.colors.shape[0]


def _refinement_config() -> AppearanceRefinementConfig:
    return AppearanceRefinementConfig(
        schema_version=APPEARANCE_REFINEMENT_SCHEMA,
        iterations=4,
        seed=0,
        sampling_variant=SEEDED_RANDOM_MAPPING_FRAME_VARIANT,
        color_learning_rate=0.01,
        opacity_learning_rate=0.01,
        opacity_anchor_weight=0.1,
        lidar_depth_weight=0.1,
        dense_normal_weight=0.1,
        dense_normal_edge_weight_gamma=10.0,
        dense_normal_alpha_support_a0=0.5,
        dense_normal_max_relative_depth_jump=0.1,
        dense_prior=DensePriorPolicy(1, 1, 1, 1.0, 0.01),
        exposure_learning_rate=0.01,
        exposure_log_gain_bound=0.5,
        exposure_bias_bound=0.25,
        exposure_anchor_weight=1.0,
        exposure_gauge_variant=FIRST_MAPPING_FRAME_EXPOSURE_GAUGE,
        ssim_weight=0.2,
        milestone_steps=(2, 4),
    )


def test_refiner_keeps_only_milestone_history_and_updates_model() -> None:
    model = _TinyAppearanceModel()
    config = _refinement_config()
    exposure = AppearanceExposureNuisance((0, 5), config, device="cpu")
    baseline = model.colors.detach().clone()

    def objective(
        item: _TinyAppearanceModel,
        frame_index: int,
        step: int,
    ) -> AppearanceObjective:
        photo = (item.colors - 0.8).square().mean()
        depth = item.opacity_logits.square().mean()
        return AppearanceObjective(
            total=photo + depth,
            terms={"photo": photo, "lidar_depth": depth},
            diagnostics={
                "dense_valid_normal_pixels": photo.new_tensor(4.0),
                "dense_normal_weight_mass": photo.new_tensor(2.0),
            },
        )

    result = AppearanceRefiner(config).optimize(
        model,
        (0, 5),
        objective,
        exposure_nuisance=exposure,
    )

    assert [entry["step"] for entry in result.steps] == [2, 4]
    assert [entry["step"] for entry in result.milestones] == [2, 4]
    assert len(result.selected_frame_indices) == config.iterations
    assert not torch.equal(model.colors, baseline)


def test_refiner_reports_intervals_and_milestones_without_history_growth() -> None:
    model = _TinyAppearanceModel()
    config = _refinement_config()
    exposure = AppearanceExposureNuisance((0, 5), config, device="cpu")
    reports: list[tuple[int, int, int, float]] = []

    def objective(
        item: _TinyAppearanceModel,
        _frame_index: int,
        _step: int,
    ) -> AppearanceObjective:
        photo = (item.colors - 0.8).square().mean()
        return AppearanceObjective(
            total=photo,
            terms={"photo": photo},
            diagnostics={
                "dense_valid_normal_pixels": photo.new_tensor(4.0),
                "dense_normal_weight_mass": photo.new_tensor(2.0),
            },
        )

    def report(step: int, total: int, frame: int, loss: torch.Tensor) -> None:
        reports.append((step, total, frame, float(loss)))

    result = AppearanceRefiner(config).optimize(
        model,
        (0, 5),
        objective,
        exposure_nuisance=exposure,
        progress_every=3,
        progress_callback=report,
    )

    assert [step for step, *_ in reports] == [2, 3, 4]
    assert all(total == config.iterations for _, total, *_ in reports)
    assert len(result.steps) == len(config.milestone_steps)
