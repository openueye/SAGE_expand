from __future__ import annotations

import numpy as np
import torch

from sage.data.rosbag import writer as rosbag_writer
from sage.data.rosbag.geometry import PoseSample, PoseTrack, RawWorldNormalizer
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
from sage.evaluation import EvaluationDepthPolicy, evaluate_render_output
from sage.losses import mapping_loss
from sage.rendering import RenderOutput


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


def _pose(timestamp_ns: int, *, x: float, yaw_radians: float) -> PoseSample:
    cosine = np.cos(yaw_radians)
    sine = np.sin(yaw_radians)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix[0, 3] = x
    return PoseSample(timestamp_ns, matrix)


def test_batch_pose_interpolation_and_raw_normalization_match_scalar_path() -> None:
    poses = PoseTrack((
        _pose(0, x=0.0, yaw_radians=0.0),
        _pose(10, x=2.0, yaw_radians=np.pi / 2),
        _pose(20, x=4.0, yaw_radians=np.pi),
    ))
    timestamps = np.asarray([0, 5, 10, 15, 20], dtype=np.int64)
    batched = poses.interpolate_many(timestamps, max_distance_ns=10)
    scalar = np.stack([
        poses.interpolate(int(timestamp), max_distance_ns=10)
        for timestamp in timestamps
    ])
    np.testing.assert_allclose(batched, scalar, atol=1e-12, rtol=1e-12)

    points = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    offsets = timestamps.astype(np.float64)
    normalizer = RawWorldNormalizer(
        poses,
        np.eye(4),
        offset_time_scale_ns=1.0,
        max_pose_distance_ns=10,
    )
    actual = normalizer.normalize(
        points,
        cloud_timestamp_ns=0,
        offset_time=offsets,
    )
    expected = np.empty_like(points)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    for index, timestamp in enumerate(timestamps):
        expected[index] = (
            poses.interpolate(int(timestamp), max_distance_ns=10)
            @ homogeneous[index]
        )[:3]
    np.testing.assert_allclose(actual, expected.astype(np.float32), atol=1e-6, rtol=1e-6)


def test_artifact_hashes_reuses_hashes_for_immutable_frame_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    known = tmp_path / "scene/images/000000.png"
    fresh = tmp_path / "audit/projection_stats.csv"
    known.parent.mkdir(parents=True)
    fresh.parent.mkdir(parents=True)
    known.write_bytes(b"already-hashed")
    fresh.write_text("frame_id\n000000\n", encoding="utf-8")
    calls: list[str] = []

    def fake_sha256(path) -> str:
        calls.append(path.name)
        return "f" * 64

    monkeypatch.setattr(rosbag_writer, "sha256_file", fake_sha256)
    artifacts = rosbag_writer._artifact_hashes(
        tmp_path,
        known_hashes={"scene/images/000000.png": "0" * 64},
    )

    assert artifacts == {
        "audit/projection_stats.csv": "f" * 64,
        "scene/images/000000.png": "0" * 64,
    }
    assert calls == ["projection_stats.csv"]


def test_evaluation_materializes_geometry_scalars_as_python_values() -> None:
    mapping = MappingObservation(
        np.asarray([[2.0, 4.0], [0.0, 0.0]], dtype=np.float32),
        np.asarray([
            [int(SourceType.LIDAR_SLAM_CENTER), int(SourceType.LIDAR_SLAM_FUSED5)],
            [255, 255],
        ], dtype=np.uint8),
        np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    )
    output = RenderOutput(
        rgb=torch.zeros((2, 2, 3), dtype=torch.float32),
        depth=torch.tensor([[1.8, 0.2], [0.0, 0.0]], dtype=torch.float32),
        alpha=torch.tensor([[0.9, 0.02], [0.0, 0.0]], dtype=torch.float32),
    )

    report = evaluate_render_output(
        output,
        mapping,
        policy=EvaluationDepthPolicy(depth_policy="raw-accumulated-v1"),
    )

    center = report["depth"]["center"]
    fused = report["depth"]["fused5"]
    assert center["observation_count"] == 1
    assert center["render_valid_count"] == 1
    assert isinstance(center["mae_m"], float)
    assert fused["render_valid_count"] == 1
    assert isinstance(report["alpha"]["total"]["finite_count"], int)
