from __future__ import annotations

import torch
from torch.nn import functional as F

from ..foundation.config import MappingLossConfig
from ..foundation.contracts import MappingObservation, SourceType


_SSIM_WINDOWS: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def photometric_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    *,
    ssim_weight: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rendered.shape != target.shape or rendered.ndim != 3 or rendered.shape[-1] != 3:
        raise ValueError("RGB tensors must share HxWx3 shape")
    # RGB finiteness is checked once at the mapping-loss boundary. Keeping this
    # helper focused avoids a second per-step device synchronization.
    x = rendered.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    window_key = (str(x.device), x.dtype)
    window = _SSIM_WINDOWS.get(window_key)
    if window is None:
        coordinates = torch.arange(11, dtype=x.dtype, device=x.device) - 5
        gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
        gaussian = gaussian / gaussian.sum()
        window = (
            torch.outer(gaussian, gaussian)
            .view(1, 1, 11, 11)
            .expand(3, 1, 11, 11)
            .contiguous()
        )
        _SSIM_WINDOWS[window_key] = window
    mu_x = F.conv2d(x, window, padding=5, groups=3)
    mu_y = F.conv2d(y, window, padding=5, groups=3)
    sigma_x = F.conv2d(x * x, window, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(y * y, window, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(x * y, window, padding=5, groups=3) - mu_x * mu_y
    ssim = (((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) /
            ((mu_x.square() + mu_y.square() + 0.01**2) * (sigma_x + sigma_y + 0.03**2))).mean()
    l1 = F.l1_loss(rendered, target)
    image = (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim.clamp(-1, 1))
    return image, l1, ssim


def alpha_normalized_depth(
    accumulated_depth: torch.Tensor,
    alpha: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Apply the registered ``Z / max(A, epsilon)`` depth expression."""
    if accumulated_depth.shape != alpha.shape:
        raise ValueError("Accumulated depth and alpha must share HxW shape")
    finite_depth = torch.isfinite(accumulated_depth)
    finite_alpha = torch.isfinite(alpha) & (alpha >= 0)
    safe_depth = torch.where(finite_depth, accumulated_depth, torch.zeros_like(accumulated_depth))
    safe_alpha = torch.where(finite_alpha, alpha, torch.zeros_like(alpha))
    normalized = safe_depth / safe_alpha.clamp_min(epsilon)
    valid = finite_depth & finite_alpha & (alpha > 0) & torch.isfinite(normalized)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def _source_masks(target_mapping: MappingObservation, device: torch.device) -> dict[str, torch.Tensor]:
    source_types = torch.as_tensor(target_mapping.source_types, dtype=torch.uint8, device=device)
    return {
        "center": (source_types == int(SourceType.LIDAR_RAW))
        | (source_types == int(SourceType.LIDAR_SLAM_CENTER)),
        "fused5": (source_types == int(SourceType.LIDAR_FUSED5))
        | (source_types == int(SourceType.LIDAR_SLAM_FUSED5)),
    }


def _zero(reference: torch.Tensor) -> torch.Tensor:
    safe = torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
    return safe.sum() * 0


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean with a tensor denominator; do not synchronize the host in the hot path."""
    safe_values = torch.where(mask & torch.isfinite(values), values, torch.zeros_like(values))
    denominator = mask.to(dtype=values.dtype).sum()
    return safe_values.sum() / denominator.clamp_min(1)


def _alpha_statistics(
    alpha: torch.Tensor,
    mask: torch.Tensor,
    *,
    a0: float,
    reference: torch.Tensor,
    collect_quantiles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = mask & torch.isfinite(alpha)
    values = torch.where(finite, alpha, torch.zeros_like(alpha))
    count = finite.to(dtype=reference.dtype).sum()
    mean = values.sum() / count.clamp_min(1)
    below = ((values < a0) & finite).to(dtype=reference.dtype).sum() / count.clamp_min(1)
    if not collect_quantiles:
        zero = _zero(reference)
        return mean, zero, zero, below, count
    # Quantiles are intentionally restricted to the commit-final diagnostic pass.
    selected = alpha[finite]
    if selected.numel() == 0:
        zero = _zero(reference)
        return mean, zero, zero, below, count
    return mean, torch.quantile(selected, 0.10), torch.quantile(selected, 0.50), below, count


def mapping_loss(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    target_mapping: MappingObservation,
    *,
    rendered_alpha: torch.Tensor,
    policy: MappingLossConfig,
    collect_diagnostics: bool = False,
    target_depth: torch.Tensor | None = None,
    source_masks: dict[str, torch.Tensor] | None = None,
    include_image: bool = True,
    include_diagnostics: bool = True,
    validate_rgb: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute mapping loss while allowing refinement to skip unused work.

    Defaults preserve the mapping-stage behavior. Appearance refinement already
    owns its photometric loss and only consumes the LiDAR depth term, so it can
    pass cached targets and disable duplicate image/diagnostic reductions.
    """
    if not isinstance(target_mapping, MappingObservation):
        raise ValueError("mapping_loss requires a MappingObservation target")
    if target_mapping.depth_m.shape != rendered_depth.shape:
        raise ValueError("MappingObservation depth must match rendered depth")

    if target_depth is None:
        target_depth = torch.as_tensor(
            target_mapping.depth_m,
            dtype=rendered_depth.dtype,
            device=rendered_depth.device,
        )
    elif (
        target_depth.shape != rendered_depth.shape
        or target_depth.device != rendered_depth.device
        or target_depth.dtype != rendered_depth.dtype
    ):
        raise ValueError("Cached mapping depth must match the rendered depth")
    if validate_rgb and not bool(
        torch.isfinite(rendered_rgb).all()
        & torch.isfinite(target_rgb).all()
    ):
        raise FloatingPointError("Mapping loss received non-finite rendered or target RGB")
    if include_image:
        image, l1, ssim = photometric_loss(
            rendered_rgb, target_rgb, ssim_weight=policy.ssim_weight,
        )
    else:
        image = rendered_rgb.new_zeros(())
        l1 = image
        ssim = image
    if rendered_alpha.shape != rendered_depth.shape:
        raise ValueError("Policy mapping loss requires matching rendered alpha")

    target_valid = torch.isfinite(target_depth) & (target_depth > 0)
    raw_valid = torch.isfinite(rendered_depth)
    device = rendered_depth.device
    masks = (
        source_masks
        if source_masks is not None
        else _source_masks(target_mapping, device)
    )
    observation_valid = target_valid & raw_valid
    depth_for_loss = alpha_normalized_depth(rendered_depth, rendered_alpha)
    alpha_valid = torch.isfinite(rendered_alpha) & (rendered_alpha >= 0)
    observation_valid &= alpha_valid & torch.isfinite(depth_for_loss)
    support = torch.where(
        alpha_valid,
        (rendered_alpha / policy.alpha_support_a0).clamp(0, 1),
        torch.zeros_like(rendered_alpha),
    ).detach()
    support = torch.where(
        observation_valid & (depth_for_loss > 0), support, torch.zeros_like(support),
    )
    loss_valid = support > 0
    a0 = policy.alpha_support_a0
    # |M_s| is defined by the target/source observation, not renderer support.
    fixed_depth_denominator = target_valid

    safe_depth = torch.where(torch.isfinite(depth_for_loss), depth_for_loss, torch.zeros_like(depth_for_loss))
    safe_target = torch.where(target_valid, target_depth, torch.zeros_like(target_depth))
    residual = (safe_depth - safe_target).abs()
    numerator = residual * support
    depth = _masked_mean(numerator, fixed_depth_denominator)
    total = (
        policy.image_weight * image + policy.depth_weight * depth
        if include_image
        else depth
    )
    if not include_diagnostics:
        return total, {"depth": depth}

    zero = _zero(rendered_depth) + _zero(rendered_alpha)
    geo_center = _masked_mean(numerator, fixed_depth_denominator & masks["center"])
    geo_fused5 = _masked_mean(numerator, fixed_depth_denominator & masks["fused5"])
    center_mae = _masked_mean(residual * observation_valid.to(residual.dtype), fixed_depth_denominator & masks["center"])
    fused5_mae = _masked_mean(residual * observation_valid.to(residual.dtype), fixed_depth_denominator & masks["fused5"])

    hit_center = zero
    hit_fused5 = zero
    hit = hit_center + hit_fused5
    alpha_mean, alpha_p10, alpha_p50, alpha_below_a0, _ = _alpha_statistics(
        rendered_alpha,
        target_valid,
        a0=a0,
        reference=rendered_depth,
        collect_quantiles=collect_diagnostics,
    )
    center_alpha = _alpha_statistics(
        rendered_alpha, target_valid & masks["center"], a0=a0, reference=rendered_depth,
        collect_quantiles=collect_diagnostics,
    )
    fused5_alpha = _alpha_statistics(
        rendered_alpha, target_valid & masks["fused5"], a0=a0, reference=rendered_depth,
        collect_quantiles=collect_diagnostics,
    )
    return total, {
        "image": image,
        "total": total,
        "photo": image,
        "l1": l1,
        "ssim": ssim,
        "depth": depth,
        "geo_center": geo_center,
        "geo_fused5": geo_fused5,
        "hit": hit,
        "hit_center": hit_center,
        "hit_fused5": hit_fused5,
        "depth_valid_pixels": (loss_valid & target_valid).to(dtype=rendered_depth.dtype).sum(),
        "depth_valid_center_pixels": (loss_valid & target_valid & masks["center"]).to(dtype=rendered_depth.dtype).sum(),
        "depth_valid_fused5_pixels": (loss_valid & target_valid & masks["fused5"]).to(dtype=rendered_depth.dtype).sum(),
        "depth_center_mae": center_mae,
        "depth_fused5_mae": fused5_mae,
        "depth_mean_alpha": _masked_mean(
            rendered_alpha,
            loss_valid,
        ),
        "alpha_mean": alpha_mean,
        "alpha_p10": alpha_p10,
        "alpha_p50": alpha_p50,
        "alpha_below_a0_fraction": alpha_below_a0,
        "alpha_center_mean": center_alpha[0],
        "alpha_center_p10": center_alpha[1],
        "alpha_center_p50": center_alpha[2],
        "alpha_fused5_mean": fused5_alpha[0],
        "alpha_fused5_p10": fused5_alpha[1],
        "alpha_fused5_p50": fused5_alpha[2],
    }
