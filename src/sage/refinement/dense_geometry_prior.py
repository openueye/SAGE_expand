"""LiDAR-anchored dense depth targets for SAGE."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as F

from ..data.providers.spnet import DenseSPNetProvider
from ..foundation.contracts import (
    DenseGeometryPrior,
    DensePriorDiagnostics,
    DepthEvidence,
    FrameInputs,
    MappingObservation,
    SourceType,
)
from .dense_geometry_config import (
    DENSE_ALIGNMENT_NONWORSENING_V2,
    DensePriorPolicy,
)


def prepare_dense_priors(
    frames: tuple[FrameInputs, ...],
    provider: DenseSPNetProvider,
    policy: DensePriorPolicy,
    *,
    progress_callback: Callable[[int, int, FrameInputs], None] | None = None,
) -> dict[int, DenseGeometryPrior]:
    """Materialize aligned dense priors before a Gaussian model occupies CUDA."""
    priors: dict[int, DenseGeometryPrior] = {}
    for completed, frame in enumerate(frames, start=1):
        priors[frame.index] = build_dense_geometry_prior(
            provider.dense_evidence_for(frame),
            frame.mapping,
            policy,
        )
        if progress_callback is not None:
            progress_callback(completed, len(frames), frame)
    return priors


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] * 0.5)])


def build_dense_geometry_prior(
    evidence: DepthEvidence,
    mapping: MappingObservation,
    policy: DensePriorPolicy,
) -> DenseGeometryPrior:
    """Align a dense SPNet depth field to sparse LiDAR with a bilinear scale grid."""
    if evidence.source_type != SourceType.SPNET_COMPLETED:
        raise ValueError("Dense geometry prior requires SPNET_COMPLETED evidence")
    if evidence.depth_m.shape != mapping.depth_m.shape:
        raise ValueError("Dense evidence and mapping dimensions must match")
    raw_prediction = np.asarray(evidence.depth_m, dtype=np.float32)
    raw_valid = (
        np.asarray(evidence.valid_mask, dtype=np.bool_)
        & np.isfinite(raw_prediction)
        & (raw_prediction > 0)
    )
    raw = np.where(raw_valid, raw_prediction, 0).astype(np.float32)
    lidar = np.asarray(mapping.depth_m, dtype=np.float32)
    mapping_confidence = np.asarray(mapping.confidences, dtype=np.float32)
    correspondence = raw_valid & (lidar > 0) & (mapping_confidence > 0)
    grid = np.ones((policy.grid_height, policy.grid_width), dtype=np.float32)
    fitted = np.zeros_like(grid, dtype=np.bool_)
    height, width = raw.shape
    support_rows, support_columns = np.nonzero(correspondence)
    support_ratios = lidar[correspondence] / raw[correspondence]
    support_confidence = mapping_confidence[correspondence].astype(np.float64)
    for row in range(policy.grid_height):
        top = row * height // policy.grid_height
        bottom = (row + 1) * height // policy.grid_height
        for column in range(policy.grid_width):
            left = column * width // policy.grid_width
            right = (column + 1) * width // policy.grid_width
            mask = correspondence[top:bottom, left:right]
            if int(mask.sum()) >= policy.min_lidar_support:
                ratios = lidar[top:bottom, left:right][mask] / raw[top:bottom, left:right][mask]
                weights = mapping_confidence[top:bottom, left:right][mask].astype(np.float64)
            elif support_ratios.shape[0] >= policy.min_lidar_support:
                center_row = (top + bottom - 1) * .5
                center_column = (left + right - 1) * .5
                distance_squared = (
                    (support_rows - center_row) ** 2
                    + (support_columns - center_column) ** 2
                )
                cutoff = np.partition(
                    distance_squared,
                    policy.min_lidar_support - 1,
                )[policy.min_lidar_support - 1]
                nearer = np.flatnonzero(distance_squared < cutoff)
                tied = np.flatnonzero(distance_squared == cutoff)
                tied_order = np.lexsort((
                    support_columns[tied],
                    support_rows[tied],
                ))
                needed = policy.min_lidar_support - nearer.shape[0]
                nearest = np.concatenate((nearer, tied[tied_order[:needed]]))
                ratios = support_ratios[nearest]
                spatial = 1 / (np.sqrt(distance_squared[nearest]) + 1)
                weights = support_confidence[nearest] * spatial
            else:
                continue
            weights **= policy.confidence_exponent
            grid[row, column] = _weighted_median(ratios.astype(np.float64), weights)
            fitted[row, column] = True
    scale = F.interpolate(
        torch.from_numpy(grid).view(1, 1, *grid.shape),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    fitted_support = F.interpolate(
        torch.from_numpy(fitted.astype(np.float32)).view(1, 1, *fitted.shape),
        size=(height, width),
        mode="nearest",
    )[0, 0].numpy().astype(np.bool_)
    fitted_aligned = (raw * scale).astype(np.float32)
    valid = raw_valid & fitted_support
    confidence = np.where(valid, evidence.confidence, 0).astype(np.float32)
    support_count = int(correspondence.sum())
    if support_count:
        pre_mae = float(np.abs(raw[correspondence] - lidar[correspondence]).mean())
        fitted_mae = float(
            np.abs(
                fitted_aligned[correspondence] - lidar[correspondence]
            ).mean()
        )
    else:
        pre_mae = fitted_mae = 0.0
    selected_alignment = (
        "fitted-grid" if bool(fitted.any()) else "identity-fallback"
    )
    aligned = fitted_aligned
    if (
        getattr(policy, "alignment_variant", None)
        == DENSE_ALIGNMENT_NONWORSENING_V2
        and (not bool(fitted.any()) or fitted_mae > pre_mae)
    ):
        grid = np.ones_like(grid)
        aligned = raw.copy()
        selected_alignment = (
            "identity" if bool(fitted.any()) else "identity-fallback"
        )
    post_mae = (
        float(np.abs(aligned[correspondence] - lidar[correspondence]).mean())
        if support_count
        else 0.0
    )
    return DenseGeometryPrior(
        raw.copy(),
        aligned,
        valid.astype(np.bool_),
        confidence,
        grid,
        DensePriorDiagnostics(
            lidar_support=support_count,
            fitted_grid_cells=int(fitted.sum()),
            pre_alignment_mae_m=pre_mae,
            post_alignment_mae_m=post_mae,
            fallback=not bool(fitted.any()),
            fitted_alignment_mae_m=fitted_mae,
            selected_alignment=selected_alignment,
        ),
    )
