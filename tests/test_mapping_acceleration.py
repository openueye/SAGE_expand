import numpy as np
import torch

from sage.engine.losses import mapping_loss, mapping_training_loss
from sage.foundation.config import MappingLossConfig
from sage.foundation.contracts import MappingObservation, SourceType


def _observation() -> MappingObservation:
    return MappingObservation(
        np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32),
        np.array([
            [int(SourceType.LIDAR_SLAM_CENTER), int(SourceType.LIDAR_SLAM_FUSED5)],
            [255, int(SourceType.LIDAR_SLAM_CENTER)],
        ], dtype=np.uint8),
        np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
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
