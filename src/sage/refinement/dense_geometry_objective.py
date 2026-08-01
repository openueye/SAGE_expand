"""Dense LiDAR-anchored objective for SAGE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..engine.losses import alpha_normalized_depth
from ..foundation.contracts import CameraIntrinsics, DenseGeometryPrior
from .dense_geometry_config import DenseGeometryPolicy, DensePriorPolicy


_CAMERA_RAYS: dict[
    tuple[str, torch.dtype, int, int, float, float, float, float],
    tuple[torch.Tensor, torch.Tensor],
] = {}


@dataclass(frozen=True)
class DenseNormalStatic:
    """Target-only tensors retained by the refinement normal objective."""

    weights: torch.Tensor
    prior_valid: torch.Tensor
    target_normals: torch.Tensor
    target_valid: torch.Tensor
    normal_weights: torch.Tensor


@dataclass(frozen=True)
class DenseNormalObjective:
    loss: torch.Tensor
    valid_pixels: torch.Tensor
    weight_mass: torch.Tensor


def _camera_rays(
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    key = (
        str(depth.device),
        depth.dtype,
        height,
        width,
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.cx),
        float(intrinsics.cy),
    )
    rays = _CAMERA_RAYS.get(key)
    if rays is None:
        rows, columns = torch.meshgrid(
            torch.arange(height, dtype=depth.dtype, device=depth.device),
            torch.arange(width, dtype=depth.dtype, device=depth.device),
            indexing="ij",
        )
        rays = (
            (columns - intrinsics.cx) / intrinsics.fx,
            (rows - intrinsics.cy) / intrinsics.fy,
        )
        _CAMERA_RAYS[key] = rays
    return rays


def _camera_points(
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    x_ray, y_ray = _camera_rays(depth, intrinsics)
    return torch.stack(
        (
            x_ray * depth,
            y_ray * depth,
            depth,
        ),
        dim=2,
    )


def _zero(reference: torch.Tensor) -> torch.Tensor:
    safe = torch.where(
        torch.isfinite(reference),
        reference,
        torch.zeros_like(reference),
    )
    return safe.sum() * 0


def _depth_normals(
    depth: torch.Tensor,
    valid_depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
    *,
    max_relative_depth_jump: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    normals = torch.zeros(
        (height, width, 3),
        dtype=depth.dtype,
        device=depth.device,
    )
    normal_valid = torch.zeros(
        (height, width),
        dtype=torch.bool,
        device=depth.device,
    )
    if height < 3 or width < 3:
        return normals, normal_valid
    points = _camera_points(depth, intrinsics)
    tangent_x = points[1:-1, 2:] - points[1:-1, :-2]
    tangent_y = points[2:, 1:-1] - points[:-2, 1:-1]
    normal = torch.cross(tangent_x, tangent_y, dim=2)
    norm = torch.linalg.vector_norm(normal, dim=2)
    stencil_valid = (
        valid_depth[1:-1, 1:-1]
        & valid_depth[1:-1, :-2]
        & valid_depth[1:-1, 2:]
        & valid_depth[:-2, 1:-1]
        & valid_depth[2:, 1:-1]
        & torch.isfinite(norm)
        & (norm > torch.finfo(depth.dtype).eps)
    )
    if max_relative_depth_jump is not None:
        center = depth[1:-1, 1:-1]
        neighbors = torch.stack(
            (
                depth[1:-1, :-2],
                depth[1:-1, 2:],
                depth[:-2, 1:-1],
                depth[2:, 1:-1],
            )
        )
        relative_jump = (
            (neighbors - center).abs()
            / center.abs().clamp_min(torch.finfo(depth.dtype).eps)
        )
        stencil_valid &= (
            torch.isfinite(relative_jump).all(dim=0)
            & (relative_jump <= max_relative_depth_jump).all(dim=0)
        )
    normals[1:-1, 1:-1] = F.normalize(normal, dim=2)
    normal_valid[1:-1, 1:-1] = stencil_valid
    return normals, normal_valid


def _stencil_minimum(values: torch.Tensor) -> torch.Tensor:
    height, width = values.shape
    output = torch.zeros_like(values)
    if height < 3 or width < 3:
        return output
    output[1:-1, 1:-1] = torch.stack(
        (
            values[1:-1, 1:-1],
            values[1:-1, :-2],
            values[1:-1, 2:],
            values[:-2, 1:-1],
            values[2:, 1:-1],
        ),
        dim=0,
    ).amin(dim=0)
    return output


def edge_weights(
    target_rgb: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return detached center, horizontal and vertical edge attenuations."""
    gray = target_rgb.detach().mean(dim=2)
    horizontal_gradient = (gray[:, 1:] - gray[:, :-1]).abs()
    vertical_gradient = (gray[1:, :] - gray[:-1, :]).abs()
    horizontal_weight = torch.exp(-gamma * horizontal_gradient)
    vertical_weight = torch.exp(-gamma * vertical_gradient)
    center_gradient = torch.maximum(
        torch.maximum(
            F.pad(horizontal_gradient, (1, 0)),
            F.pad(horizontal_gradient, (0, 1)),
        ),
        torch.maximum(
            F.pad(vertical_gradient, (0, 0, 1, 0)),
            F.pad(vertical_gradient, (0, 0, 0, 1)),
        ),
    )
    return (
        torch.exp(-gamma * center_gradient),
        horizontal_weight,
        vertical_weight,
    )


def _center_edge_weights(target_rgb: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute only the center attenuation consumed by normal refinement."""
    gray = target_rgb.detach().mean(dim=2)
    horizontal_gradient = (gray[:, 1:] - gray[:, :-1]).abs()
    vertical_gradient = (gray[1:, :] - gray[:-1, :]).abs()
    center_gradient = torch.maximum(
        torch.maximum(
            F.pad(horizontal_gradient, (1, 0)),
            F.pad(horizontal_gradient, (0, 1)),
        ),
        torch.maximum(
            F.pad(vertical_gradient, (0, 0, 1, 0)),
            F.pad(vertical_gradient, (0, 0, 0, 1)),
        ),
    )
    return torch.exp(-gamma * center_gradient)


def dense_prior_support_counts(
    prior: DenseGeometryPrior,
    intrinsics: CameraIntrinsics,
    *,
    max_relative_depth_jump: float | None = None,
) -> tuple[int, int]:
    """Return valid dense-depth and target-normal center counts."""
    expected = (intrinsics.height, intrinsics.width)
    if prior.aligned_depth_m.shape != expected:
        raise ValueError("Dense geometry prior dimensions must match intrinsics")
    target_depth = torch.from_numpy(prior.aligned_depth_m)
    prior_valid = torch.from_numpy(prior.valid_mask)
    _, target_valid = _depth_normals(
        target_depth,
        prior_valid,
        intrinsics,
        max_relative_depth_jump=max_relative_depth_jump,
    )
    return int(prior_valid.sum()), int(target_valid.sum())


def prepare_dense_geometry_static(
    prior: DenseGeometryPrior,
    rgb: torch.Tensor,
    intrinsics: CameraIntrinsics,
    prior_policy: DensePriorPolicy,
    geometry: DenseGeometryPolicy,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Materialize target-only dense-normal terms once per input frame."""
    target_depth = torch.as_tensor(
        prior.aligned_depth_m,
        dtype=rgb.dtype,
        device=rgb.device,
    )
    weights = torch.as_tensor(
        prior.confidence,
        dtype=rgb.dtype,
        device=rgb.device,
    ).pow(prior_policy.confidence_exponent)
    prior_valid = torch.as_tensor(
        prior.valid_mask,
        dtype=torch.bool,
        device=rgb.device,
    )
    target_normals, target_valid = _depth_normals(
        target_depth,
        prior_valid,
        intrinsics,
        max_relative_depth_jump=geometry.max_relative_depth_jump,
    )
    edge = edge_weights(rgb, geometry.edge_weight_gamma)
    return {
        "target_depth": target_depth,
        "weights": weights,
        "prior_valid": prior_valid,
        "target_normals": target_normals,
        "target_valid": target_valid,
        "edge_weights": edge,
        "normal_weights": _stencil_minimum(weights) * edge[0],
    }


def prepare_dense_normal_static(
    prior: DenseGeometryPrior,
    rgb: torch.Tensor,
    intrinsics: CameraIntrinsics,
    prior_policy: DensePriorPolicy,
    geometry: DenseGeometryPolicy,
) -> DenseNormalStatic:
    """Materialize only target tensors consumed by refinement's normal loss."""
    target_depth = torch.as_tensor(
        prior.aligned_depth_m,
        dtype=rgb.dtype,
        device=rgb.device,
    )
    weights = torch.as_tensor(
        prior.confidence,
        dtype=rgb.dtype,
        device=rgb.device,
    ).pow(prior_policy.confidence_exponent)
    prior_valid = torch.as_tensor(
        prior.valid_mask,
        dtype=torch.bool,
        device=rgb.device,
    )
    target_normals, target_valid = _depth_normals(
        target_depth,
        prior_valid,
        intrinsics,
        max_relative_depth_jump=geometry.max_relative_depth_jump,
    )
    center_weight = _center_edge_weights(rgb, geometry.edge_weight_gamma)
    return DenseNormalStatic(
        weights=weights,
        prior_valid=prior_valid,
        target_normals=target_normals,
        target_valid=target_valid,
        normal_weights=_stencil_minimum(weights) * center_weight,
    )


def dense_normal_objective(
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
    intrinsics: CameraIntrinsics,
    geometry: DenseGeometryPolicy,
    static: DenseNormalStatic,
) -> DenseNormalObjective:
    """Return the published refinement normal loss without unused objectives."""
    expected = (intrinsics.height, intrinsics.width)
    if rendered_depth.shape != expected or rendered_alpha.shape != expected:
        raise ValueError("Dense normal render dimensions must match intrinsics")
    predicted = alpha_normalized_depth(rendered_depth, rendered_alpha)
    valid = (
        static.prior_valid
        & (static.weights > 0)
        & torch.isfinite(predicted)
        & (predicted > 0)
        & torch.isfinite(rendered_alpha)
        & (rendered_alpha.detach() >= geometry.alpha_support_a0)
    )
    predicted_normals, predicted_valid = _depth_normals(
        predicted,
        valid,
        intrinsics,
        max_relative_depth_jump=geometry.max_relative_depth_jump,
    )
    normal_valid = predicted_valid & static.target_valid
    normal_values = 1 - (
        predicted_normals * static.target_normals
    ).sum(dim=2).clamp(-1, 1)
    normal_mass = torch.where(
        normal_valid,
        static.normal_weights,
        torch.zeros_like(static.normal_weights),
    ).sum()
    normal = (
        torch.where(
            normal_valid,
            normal_values * static.normal_weights,
            torch.zeros_like(normal_values),
        ).sum()
        / normal_mass.clamp_min(1)
    )
    return DenseNormalObjective(
        loss=normal,
        valid_pixels=normal_valid.to(rendered_depth.dtype).sum(),
        weight_mass=normal_mass,
    )


def dense_geometry_objective(
    rendered_depth: torch.Tensor,
    rendered_alpha: torch.Tensor,
    prior: DenseGeometryPrior,
    rgb: torch.Tensor,
    intrinsics: CameraIntrinsics,
    prior_policy: DensePriorPolicy,
    geometry: DenseGeometryPolicy,
    static: (
        dict[
            str,
            torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
        | None
    ) = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return robust dense depth, normal alignment, and edge-aware normal smoothness."""
    expected = (intrinsics.height, intrinsics.width)
    if rendered_depth.shape != expected or rendered_alpha.shape != expected:
        raise ValueError("Dense geometry render dimensions must match intrinsics")
    if prior.aligned_depth_m.shape != expected or rgb.shape != (*expected, 3):
        raise ValueError("Dense geometry prior and RGB dimensions must match intrinsics")
    if static is None:
        target_depth = torch.as_tensor(
            prior.aligned_depth_m,
            dtype=rendered_depth.dtype,
            device=rendered_depth.device,
        )
        weights = torch.as_tensor(
            prior.confidence,
            dtype=rendered_depth.dtype,
            device=rendered_depth.device,
        ).pow(prior_policy.confidence_exponent)
        prior_valid = torch.as_tensor(
            prior.valid_mask,
            dtype=torch.bool,
            device=rendered_depth.device,
        )
    else:
        target_depth = static["target_depth"]
        weights = static["weights"]
        prior_valid = static["prior_valid"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (target_depth, weights, prior_valid)
        ):
            raise ValueError("Dense geometry static cache is invalid")
    predicted = alpha_normalized_depth(rendered_depth, rendered_alpha)
    valid = (
        prior_valid
        & (weights > 0)
        & torch.isfinite(predicted)
        & (predicted > 0)
        & torch.isfinite(rendered_alpha)
        & (rendered_alpha.detach() >= geometry.alpha_support_a0)
    )
    mass = torch.where(valid, weights, torch.zeros_like(weights)).sum()
    if geometry.dense_depth_weight:
        residual = (
            torch.sqrt(
                (predicted - target_depth).square()
                + prior_policy.robust_epsilon_m**2
            )
            - prior_policy.robust_epsilon_m
        )
        depth = (
            torch.where(
                valid,
                residual * weights,
                torch.zeros_like(residual),
            ).sum()
            / mass.clamp_min(1)
        )
    else:
        depth = _zero(predicted)
    predicted_normals, predicted_valid = _depth_normals(
        predicted,
        valid,
        intrinsics,
        max_relative_depth_jump=geometry.max_relative_depth_jump,
    )
    if static is None:
        target_normals, target_valid = _depth_normals(
            target_depth,
            prior_valid,
            intrinsics,
            max_relative_depth_jump=geometry.max_relative_depth_jump,
        )
        center, horizontal, vertical = edge_weights(
            rgb,
            geometry.edge_weight_gamma,
        )
        normal_weights = _stencil_minimum(weights) * center
    else:
        target_normals = static["target_normals"]
        target_valid = static["target_valid"]
        edge = static["edge_weights"]
        normal_weights = static["normal_weights"]
        if (
            not isinstance(target_normals, torch.Tensor)
            or not isinstance(target_valid, torch.Tensor)
            or not isinstance(normal_weights, torch.Tensor)
            or not isinstance(edge, tuple)
            or len(edge) != 3
        ):
            raise ValueError("Dense geometry static cache is invalid")
        center, horizontal, vertical = edge
    normal_valid = predicted_valid & target_valid
    normal_values = 1 - (predicted_normals * target_normals).sum(dim=2).clamp(-1, 1)
    normal_mass = torch.where(normal_valid, normal_weights, torch.zeros_like(normal_weights)).sum()
    normal = (
        torch.where(
            normal_valid,
            normal_values * normal_weights,
            torch.zeros_like(normal_values),
        ).sum()
        / normal_mass.clamp_min(1)
    )
    if geometry.normal_smoothness_weight:
        horizontal_valid = predicted_valid[:, 1:] & predicted_valid[:, :-1]
        vertical_valid = predicted_valid[1:, :] & predicted_valid[:-1, :]
        horizontal_values = 1 - (
            predicted_normals[:, 1:] * predicted_normals[:, :-1]
        ).sum(dim=2).clamp(-1, 1)
        vertical_values = 1 - (
            predicted_normals[1:] * predicted_normals[:-1]
        ).sum(dim=2).clamp(-1, 1)
        smooth = (
            torch.where(
                horizontal_valid,
                horizontal_values * horizontal,
                torch.zeros_like(horizontal_values),
            ).sum()
            + torch.where(
                vertical_valid,
                vertical_values * vertical,
                torch.zeros_like(vertical_values),
            ).sum()
        ) / (horizontal_valid.sum() + vertical_valid.sum()).clamp_min(1)
    else:
        smooth = _zero(predicted)
    total = geometry.dense_depth_weight * depth + geometry.dense_normal_weight * normal + geometry.normal_smoothness_weight * smooth
    return total, {
        "total": total,
        "depth": depth,
        "normal": normal,
        "smoothness": smooth,
        "weight_mass": mass,
        "normal_weight_mass": normal_mass,
        "valid_depth_pixels": valid.to(rendered_depth.dtype).sum(),
        "valid_normal_pixels": normal_valid.to(rendered_depth.dtype).sum(),
    }
