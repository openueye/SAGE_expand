import numpy as np
import pytest
import torch

from sage.engine.losses import mapping_loss, mapping_training_loss
from sage.foundation.config import MappingLossConfig
from sage.foundation.contracts import InputSourceFamily, MappingObservation, SourceType


def _observation() -> MappingObservation:
    return MappingObservation(
        np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32),
        np.array([
            [int(SourceType.LIDAR_CENTER), int(SourceType.LIDAR_FUSED)],
            [255, int(SourceType.LIDAR_CENTER)],
        ], dtype=np.uint8),
        np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        InputSourceFamily.SLAM_WORLD,
    )


def test_mapping_training_loss_matches_full_loss_and_gradients() -> None:
    policy = MappingLossConfig()
    observation = _observation()
    target_rgb = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
         [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]],
        dtype=torch.float32,
    )
    rendered_rgb = (target_rgb + 0.05).requires_grad_()
    rendered_depth = torch.tensor(
        [[0.8, 1.7], [0.0, 3.2]], dtype=torch.float32, requires_grad=True,
    )
    rendered_alpha = torch.tensor(
        [[0.8, 0.9], [0.1, 0.8]], dtype=torch.float32, requires_grad=True,
    )

    full, _ = mapping_loss(
        rendered_rgb,
        target_rgb,
        rendered_depth,
        observation,
        rendered_alpha=rendered_alpha,
        policy=policy,
    )
    full_gradients = torch.autograd.grad(
        full, (rendered_rgb, rendered_depth, rendered_alpha), retain_graph=True,
    )

    fast = mapping_training_loss(
        rendered_rgb,
        target_rgb,
        rendered_depth,
        observation,
        rendered_alpha=rendered_alpha,
        policy=policy,
    )
    fast_gradients = torch.autograd.grad(
        fast, (rendered_rgb, rendered_depth, rendered_alpha),
    )

    torch.testing.assert_close(fast, full)
    for actual, expected in zip(fast_gradients, full_gradients):
        torch.testing.assert_close(actual, expected)


def test_depth_alignment_uses_support_mass_and_detaches_alpha_support() -> None:
    observation = MappingObservation(
        np.ones((1, 2), dtype=np.float32),
        np.full((1, 2), int(SourceType.LIDAR_CENTER), dtype=np.uint8),
        np.ones((1, 2), dtype=np.float32),
        InputSourceFamily.SLAM_WORLD,
    )
    target_rgb = torch.zeros((1, 2, 3), dtype=torch.float32)
    rendered_depth = torch.tensor([[1.0, 0.4]], requires_grad=True)
    rendered_alpha = torch.tensor([[0.5, 0.1]], requires_grad=True)
    policy = MappingLossConfig(
        image_weight=1e-12,
        depth_weight=1.0,
        depth_coverage_weight=0.0,
        alpha_support_a0=0.5,
    )

    total, terms = mapping_loss(
        target_rgb,
        target_rgb,
        rendered_depth,
        observation,
        rendered_alpha=rendered_alpha,
        policy=policy,
    )

    expected = (1.0 * 1.0 + 0.2 * 3.0) / 1.2
    assert terms["depth"].item() == pytest.approx(expected, rel=1e-5)
    alpha_gradient = torch.autograd.grad(total, rendered_alpha)[0]
    detached_support_depth = (
        (rendered_depth / rendered_alpha - 1.0).abs()
        * torch.tensor([[1.0, 0.2]])
    ).sum() / (1.2 + policy.epsilon)
    expected_alpha_gradient = torch.autograd.grad(
        detached_support_depth, rendered_alpha,
    )[0]
    torch.testing.assert_close(alpha_gradient, expected_alpha_gradient)


def test_depth_coverage_hinge_penalizes_only_lidar_valid_low_support() -> None:
    observation = MappingObservation(
        np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
        np.full((1, 3), int(SourceType.LIDAR_CENTER), dtype=np.uint8),
        np.ones((1, 3), dtype=np.float32),
        InputSourceFamily.SLAM_WORLD,
    )
    target_rgb = torch.zeros((1, 3, 3), dtype=torch.float32)
    rendered_depth = torch.ones((1, 3), requires_grad=True)
    rendered_alpha = torch.tensor([[0.5, 0.1, 0.9]], requires_grad=True)
    policy = MappingLossConfig(
        image_weight=1e-12,
        depth_weight=0.0,
        depth_coverage_weight=1.0,
        alpha_support_a0=0.85,
        depth_coverage_threshold=0.85,
    )

    total, terms = mapping_loss(
        target_rgb,
        target_rgb,
        rendered_depth,
        observation,
        rendered_alpha=rendered_alpha,
        policy=policy,
    )

    assert terms["depth_coverage"].item() == pytest.approx(0.06125, rel=1e-5)
    total.backward()
    torch.testing.assert_close(
        rendered_alpha.grad,
        torch.tensor([[-0.35, 0.0, 0.0]]),
        atol=1e-6,
        rtol=0,
    )
