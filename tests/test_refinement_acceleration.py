import numpy as np
import torch

from sage.foundation.contracts import (
    CameraIntrinsics,
    DenseGeometryPrior,
    DensePriorDiagnostics,
)
from sage.refinement.dense_geometry_config import (
    DenseGeometryPolicy,
    DensePriorPolicy,
)
from sage.refinement.dense_geometry_objective import (
    dense_geometry_objective,
    dense_normal_objective,
    prepare_dense_geometry_static,
    prepare_dense_normal_static,
)


def _inputs():
    height, width = 12, 16
    intrinsics = CameraIntrinsics(width, height, 20.0, 20.0, 8.0, 6.0)
    rows, columns = np.indices((height, width), dtype=np.float32)
    depth = (4.0 + 0.01 * rows + 0.02 * columns).astype(np.float32)
    valid = np.ones((height, width), dtype=np.bool_)
    confidence = np.ones((height, width), dtype=np.float32)
    prior = DenseGeometryPrior(
        depth.copy(),
        depth.copy(),
        valid,
        confidence,
        np.ones((2, 2), dtype=np.float32),
        DensePriorDiagnostics(0, 4, 0.0, 0.0, False),
    )
    prior_policy = DensePriorPolicy(2, 2, 1, 2.0)
    geometry = DenseGeometryPolicy(0.0, 1.0, 0.0, 10.0, 0.2, 0.5)
    rgb = torch.linspace(0.0, 1.0, height * width * 3).reshape(height, width, 3)
    return intrinsics, prior, prior_policy, geometry, rgb


def test_dense_normal_fast_path_matches_general_objective_and_gradients() -> None:
    intrinsics, prior, prior_policy, geometry, rgb = _inputs()
    accumulated_depth = torch.from_numpy(prior.aligned_depth_m * 0.8).requires_grad_()
    alpha = torch.full_like(accumulated_depth, 0.8, requires_grad=True)

    general_static = prepare_dense_geometry_static(
        prior, rgb, intrinsics, prior_policy, geometry,
    )
    _, general_terms = dense_geometry_objective(
        accumulated_depth,
        alpha,
        prior,
        rgb,
        intrinsics,
        prior_policy,
        geometry,
        static=general_static,
    )
    general_gradients = torch.autograd.grad(
        general_terms["normal"], (accumulated_depth, alpha), retain_graph=True,
    )

    normal_static = prepare_dense_normal_static(
        prior, rgb, intrinsics, prior_policy, geometry,
    )
    result = dense_normal_objective(
        accumulated_depth,
        alpha,
        intrinsics,
        geometry,
        normal_static,
    )
    fast_gradients = torch.autograd.grad(
        result.loss, (accumulated_depth, alpha),
    )

    torch.testing.assert_close(result.loss, general_terms["normal"])
    torch.testing.assert_close(
        result.valid_pixels, general_terms["valid_normal_pixels"],
    )
    torch.testing.assert_close(
        result.weight_mass, general_terms["normal_weight_mass"],
    )
    for actual, expected in zip(fast_gradients, general_gradients):
        torch.testing.assert_close(actual, expected)
