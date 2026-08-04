from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import torch

from ..foundation.config import GaussianInitializationConfig, GrowthConfig
from ..foundation.contracts import DepthEvidence, GaussianAppendBatch, GrowthInputs, SourceType
from ..core_input import MappingFrame
from .geometry import backproject_depth
from .losses import alpha_normalized_depth
from ..foundation.source_policy import SOURCE_DESCRIPTORS, descriptor_for_type, descriptors_for_types, source_policy_value

if TYPE_CHECKING:
    from .rendering import RenderOutput


_SPATIAL_DEDUP_NEIGHBOR_OFFSETS = torch.tensor(
    tuple(product((-1, 0, 1), repeat=3)), dtype=torch.int64,
)
_SPATIAL_DEDUP_QUERY_CHUNK = 65536
_REJECTION_REASONS = frozenset({
    "frame_bias",
    "depth_out_of_range",
    "below_confidence",
    "gate_blocked",
    "duplicate_2d",
    "duplicate_3d",
    "duplicate_3d_existing",
    "quota_exhausted",
})


@dataclass(frozen=True)
class GrowthStats:
    """Per-source growth accounting with mutually exclusive rejection buckets."""

    by_source: dict[str, dict[str, object]]


@dataclass(frozen=True)
class FrameNeeds:
    """哪些像素需要新增高斯——只由渲染结果决定，不知道有几个证据源。

    coverage 只表示 alpha 不足；RGB 残差是诊断量，不再成为 coverage 的硬门。
    conflict 依赖逐源的深度与容差，因此只在这里提供它的公共部分。
    """

    target_rgb: torch.Tensor
    coverage: torch.Tensor          # alpha 不足 —— 需要补面
    low_alpha: torch.Tensor         # 仅 alpha 不足，coverage 的上界，用于归因
    rendered_depth: torch.Tensor    # alpha 归一化深度
    rendered_depth_valid: torch.Tensor


@dataclass(frozen=True)
class GrowthResult:
    batch: GaussianAppendBatch
    stats: GrowthStats


def _spatially_blocked(
    query_points: torch.Tensor,
    reference_points: torch.Tensor,
    threshold_m: float,
) -> torch.Tensor:
    """Return queries within ``threshold_m`` of a reference point.

    A voxel grid only narrows the search. The final Euclidean distance check
    avoids both false positives inside one voxel and false negatives across a
    voxel boundary.
    """
    query_count = int(query_points.shape[0])
    reference_count = int(reference_points.shape[0])
    blocked = torch.zeros(query_count, dtype=torch.bool, device=query_points.device)
    if query_count == 0 or reference_count == 0:
        return blocked

    query_voxels = torch.floor(query_points / threshold_m).to(torch.int64)
    reference_voxels = torch.floor(reference_points / threshold_m).to(torch.int64)
    offsets = _SPATIAL_DEDUP_NEIGHBOR_OFFSETS.to(device=query_points.device)
    threshold_squared = threshold_m * threshold_m

    for start in range(0, query_count, _SPATIAL_DEDUP_QUERY_CHUNK):
        end = min(start + _SPATIAL_DEDUP_QUERY_CHUNK, query_count)
        query_voxels_chunk = query_voxels[start:end]
        neighbor_voxels = (
            query_voxels_chunk[:, None, :] + offsets[None, :, :]
        ).reshape(-1, 3)
        all_voxels = torch.cat((reference_voxels, neighbor_voxels), dim=0)
        _, inverse = torch.unique(all_voxels, dim=0, return_inverse=True)
        reference_cell_ids = inverse[:reference_count]
        neighbor_cell_ids = inverse[reference_count:].reshape(end - start, -1)

        cell_counts = torch.bincount(
            reference_cell_ids,
            minlength=int(inverse.max()) + 1,
        )
        neighbor_counts = cell_counts[neighbor_cell_ids.reshape(-1)]
        if not bool((neighbor_counts > 0).any()):
            continue

        reference_order = torch.argsort(reference_cell_ids)
        cell_offsets = torch.cat((
            torch.zeros(1, dtype=torch.int64, device=query_points.device),
            torch.cumsum(cell_counts, dim=0),
        ))
        neighbor_starts = cell_offsets[neighbor_cell_ids.reshape(-1)]
        pair_neighbor_ids = torch.repeat_interleave(
            torch.arange(neighbor_cell_ids.numel(), device=query_points.device),
            neighbor_counts,
        )
        if pair_neighbor_ids.numel() == 0:
            continue
        pair_starts = torch.repeat_interleave(neighbor_starts, neighbor_counts)
        local_offsets = (
            torch.arange(pair_neighbor_ids.numel(), device=query_points.device)
            - torch.repeat_interleave(
                torch.cumsum(neighbor_counts, dim=0) - neighbor_counts,
                neighbor_counts,
            )
        )
        reference_indices = reference_order[pair_starts + local_offsets]
        query_indices = pair_neighbor_ids // neighbor_cell_ids.shape[1]
        distances_squared = (
            query_points[start + query_indices] - reference_points[reference_indices]
        ).square().sum(dim=1)
        hits = distances_squared <= threshold_squared
        if bool(hits.any()):
            blocked_chunk = torch.zeros(
                end - start, dtype=torch.uint8, device=query_points.device,
            )
            blocked_chunk.scatter_reduce_(
                0,
                query_indices[hits],
                torch.ones_like(query_indices[hits], dtype=torch.uint8),
                reduce="amax",
                include_self=True,
            )
            blocked[start:end] = blocked_chunk.to(torch.bool)
    return blocked


def _voxel_duplicates_after_first(
    points: torch.Tensor,
    voxel_size_m: float,
) -> torch.Tensor:
    """Return later candidates that share a quantized voxel with an earlier one.

    Candidate order is source-priority followed by raster order. Retaining the
    first row per voxel therefore preserves the growth arbitration order
    without materializing candidate pairs.
    """
    point_count = int(points.shape[0])
    duplicates = torch.zeros(point_count, dtype=torch.bool, device=points.device)
    if point_count == 0:
        return duplicates

    voxels = torch.floor(points / voxel_size_m).to(torch.int64)
    _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
    indices = torch.arange(point_count, dtype=torch.int64, device=points.device)
    first_indices = torch.full(
        (int(inverse.max()) + 1,),
        point_count,
        dtype=torch.int64,
        device=points.device,
    )
    first_indices.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
    return indices != first_indices[inverse]


def _empty_batch(device: torch.device) -> GaussianAppendBatch:
    return GaussianAppendBatch(
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 1), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 4), dtype=torch.float32, device=device),
        torch.empty((0,), dtype=torch.uint8, device=device),
        torch.empty((0,), dtype=torch.float32, device=device),
    )


def _increment_rejection(stats: dict[str, object], reason: str, count: int) -> None:
    if reason not in _REJECTION_REASONS:
        raise ValueError(f"Unknown growth rejection reason: {reason}")
    if count:
        reasons = stats["rejection_reasons"]
        assert isinstance(reasons, dict)
        reasons[reason] = int(reasons.get(reason, 0)) + count


class GrowthBuilder:
    """Build a batch of source-owned candidates without per-candidate Python objects."""

    def __init__(
        self,
        config: GrowthConfig,
        *,
        device: str | torch.device,
        gaussian_initialization: GaussianInitializationConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.opacity_logit = gaussian_initialization.opacity_logit
        self.scale_clamp_min = gaussian_initialization.scale_clamp_min
        self.scale_anisotropy = torch.as_tensor(
            gaussian_initialization.initial_scale_anisotropy,
            dtype=torch.float32,
            device=self.device,
        )

    def _frame_needs(self, frame: MappingFrame, rendered: RenderOutput) -> FrameNeeds:
        """L1：由渲染结果推出帧级需求。纯函数，与证据源无关。"""
        target_rgb = torch.as_tensor(frame.rgb, dtype=torch.float32, device=self.device)
        rendered_depth = alpha_normalized_depth(rendered.accumulated_depth, rendered.alpha)
        low_alpha = torch.isfinite(rendered.alpha) & (rendered.alpha < self.config.coverage_alpha_threshold)
        return FrameNeeds(
            target_rgb=target_rgb,
            coverage=low_alpha,
            low_alpha=low_alpha,
            rendered_depth=rendered_depth,
            rendered_depth_valid=torch.isfinite(rendered_depth) & (rendered_depth > 0),
        )

    def _conflict_need(
        self,
        needs: FrameNeeds,
        depth: torch.Tensor,
        valid: torch.Tensor,
        source_name: str,
    ) -> torch.Tensor:
        """L1 的逐源分量：已覆盖但渲染深度与该源证据显著冲突。"""
        residual = source_policy_value(self.config.residual_thresholds, source_name)
        tolerance = torch.maximum(torch.full_like(depth, residual.absolute_m), residual.relative * depth)
        return needs.rendered_depth_valid & valid & ((needs.rendered_depth - depth).abs() > tolerance)

    @staticmethod
    def _select_quota_rows(rows: torch.Tensor, quota: int) -> torch.Tensor:
        """Select rows with the registered deterministic raster-uniform policy."""
        if int(rows.numel()) <= quota:
            return rows
        if quota == 0:
            return rows[:0]
        return rows[torch.linspace(
            0, int(rows.numel()) - 1, quota, device=rows.device,
        ).long()]

    @staticmethod
    def _arbitrate_2d(
        gated: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        descriptors: tuple,
        stats: dict[str, dict[str, object]],
    ) -> tuple[tuple[SourceType, torch.Tensor], ...]:
        """L2：同一像素只允许一个源播种，按显式声明的优先级裁决。

        优先级来自 descriptor.priority，不再是"谁先被循环到谁占坑"的副作用；
        返回显式有序的 tuple，使下游 3D 去重的"先到先得"跟随同一套顺序。
        """
        keeps: list[tuple[SourceType, torch.Tensor]] = []
        occupied: torch.Tensor | None = None
        for descriptor in sorted(descriptors, key=lambda item: item.priority):
            entry = gated.get(int(descriptor.source_type))
            if entry is None:
                continue
            gate = entry[0]
            if occupied is None:
                occupied = torch.zeros_like(gate)
            keep = gate & ~occupied
            _increment_rejection(stats[descriptor.name], "duplicate_2d", int((gate & occupied).sum()))
            occupied |= keep
            keeps.append((descriptor.source_type, keep))
        return tuple(keeps)

    def _apply_quota(
        self,
        source_types: torch.Tensor,
        descriptors: tuple,
        stats: dict[str, dict[str, object]],
    ) -> torch.Tensor | None:
        """L2：每源独立配额。返回保留掩码，None 表示不限量。

        配额必须逐源独立：共享一个总量时先填的源会吃光额度，把低优先级源饿死
        （实测 SPNet 接受量 -97%、PSNR -3.1dB）。
        """
        quotas = self.config.max_new_per_commit
        if quotas is None:
            return None
        within_quota = torch.zeros_like(source_types, dtype=torch.bool)
        for descriptor in descriptors:
            rows = torch.nonzero(source_types == int(descriptor.source_type), as_tuple=False).flatten()
            quota = int(source_policy_value(quotas, descriptor.name))
            if int(rows.numel()) > quota:
                _increment_rejection(stats[descriptor.name], "quota_exhausted", int(rows.numel()) - quota)
                rows = self._select_quota_rows(rows, quota)
            within_quota[rows] = True
        return within_quota

    def build(
        self,
        frame: MappingFrame,
        rendered: RenderOutput,
        *,
        extra_evidences: tuple[DepthEvidence, ...] = (),
        existing_points: torch.Tensor | None = None,
    ) -> GrowthResult:
        growth_inputs = GrowthInputs((*frame.growth.evidences, *extra_evidences))
        active_descriptors = descriptors_for_types(evidence.source_type for evidence in growth_inputs.evidences)
        expected = (frame.intrinsics.height, frame.intrinsics.width)
        if any(evidence.depth_m.shape != expected for evidence in growth_inputs.evidences):
            raise ValueError("GrowthInputs evidences must match frame shape")
        if tuple(rendered.accumulated_depth.shape) != expected or tuple(rendered.alpha.shape) != expected or tuple(rendered.rgb.shape) != (*expected, 3):
            raise ValueError("Rendered output dimensions do not match frame")
        if existing_points is not None:
            existing_points = torch.as_tensor(
                existing_points, dtype=torch.float32, device=self.device,
            )
            if existing_points.ndim != 2 or existing_points.shape[1] != 3:
                raise ValueError("Existing Gaussian points must have shape Nx3")
            if not bool(torch.isfinite(existing_points).all()):
                raise ValueError("Existing Gaussian points must be finite")
        needs = self._frame_needs(frame, rendered)
        stats = self._empty_stats(active_descriptors)
        gated: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        reference_depth: torch.Tensor | None = None
        reference_base: torch.Tensor | None = None
        for evidence in growth_inputs.evidences:
            descriptor = descriptor_for_type(evidence.source_type)
            source_stats = stats[descriptor.name]
            depth = torch.as_tensor(evidence.depth_m, dtype=torch.float32, device=self.device)
            valid = torch.as_tensor(evidence.valid_mask, dtype=torch.bool, device=self.device)
            confidence = torch.as_tensor(evidence.confidence, dtype=torch.float32, device=self.device)
            base = valid & torch.isfinite(depth) & (depth > 0)
            source_stats["proposed"] = int(base.sum())
            if descriptor.priority == 0:
                reference_depth, reference_base = depth, base
            elif reference_base is not None:
                overlap = base & reference_base
                overlap_count = int(overlap.sum())
                if overlap_count >= self.config.frame_bias_min_overlap_px:
                    bias = (depth[overlap] - reference_depth[overlap]).mean()
                    reference_mean = reference_depth[overlap].mean()
                    tolerance = max(
                        self.config.frame_bias_threshold.absolute_m,
                        self.config.frame_bias_threshold.relative * float(reference_mean),
                    )
                    if float(bias.abs()) > tolerance:
                        _increment_rejection(source_stats, "frame_bias", int(base.sum()))
                        continue
            in_range = base & (depth >= self.config.min_candidate_depth_m) & (depth <= self.config.max_candidate_depth_m)
            _increment_rejection(source_stats, "depth_out_of_range", int((base & ~in_range).sum()))
            confidence_threshold = source_policy_value(
                self.config.confidence_thresholds, descriptor.name,
            )
            confidence_valid = in_range & torch.isfinite(confidence) & (confidence >= confidence_threshold)
            _increment_rejection(source_stats, "below_confidence", int((in_range & ~confidence_valid).sum()))
            conflict = self._conflict_need(needs, depth, valid, descriptor.name)
            gate = confidence_valid & (needs.coverage | conflict)
            blocked = confidence_valid & ~gate
            _increment_rejection(source_stats, "gate_blocked", int(blocked.sum()))
            gated[int(evidence.source_type)] = (gate, depth, confidence)

        proposals: list[tuple[SourceType, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for source_type, keep in self._arbitrate_2d(gated, active_descriptors, stats):
            pixels = torch.nonzero(keep, as_tuple=False)
            if pixels.numel() == 0:
                continue
            depth, confidence = gated[source_type][1], gated[source_type][2]
            points, _, proposal_depth = backproject_depth(depth, frame.intrinsics, frame.pose, keep)
            proposals.append((SourceType(source_type), pixels, points, proposal_depth, confidence[keep]))

        if not proposals:
            return GrowthResult(_empty_batch(self.device), GrowthStats(self._finish_stats(stats)))
        source_types = torch.cat([torch.full((pixels.shape[0],), int(source_type), dtype=torch.uint8, device=self.device) for source_type, pixels, _, _, _ in proposals])
        pixels = torch.cat([item[1] for item in proposals])
        points = torch.cat([item[2] for item in proposals])
        camera_depths = torch.cat([item[3] for item in proposals])
        confidences = torch.cat([item[4] for item in proposals])
        duplicate_threshold = self.config.candidate_duplicate_3d_threshold_m
        if existing_points is not None:
            existing_duplicate = _spatially_blocked(
                points, existing_points, duplicate_threshold,
            )
            for descriptor in active_descriptors:
                duplicate_count = int((
                    existing_duplicate
                    & (source_types == int(descriptor.source_type))
                ).sum())
                _increment_rejection(
                    stats[descriptor.name],
                    "duplicate_3d_existing",
                    duplicate_count,
                )
            keep = ~existing_duplicate
            source_types = source_types[keep]
            pixels = pixels[keep]
            points = points[keep]
            camera_depths = camera_depths[keep]
            confidences = confidences[keep]
            if points.shape[0] == 0:
                return GrowthResult(
                    _empty_batch(self.device),
                    GrowthStats(self._finish_stats(stats)),
                )

        duplicate = _voxel_duplicates_after_first(points, duplicate_threshold)
        for descriptor in active_descriptors:
            duplicate_count = int((
                duplicate & (source_types == int(descriptor.source_type))
            ).sum())
            _increment_rejection(
                stats[descriptor.name], "duplicate_3d", duplicate_count,
            )
        keep = ~duplicate
        source_types = source_types[keep]
        pixels = pixels[keep]
        points = points[keep]
        camera_depths = camera_depths[keep]
        confidences = confidences[keep]
        within_quota = self._apply_quota(source_types, active_descriptors, stats)
        if within_quota is not None:
            source_types = source_types[within_quota]
            pixels = pixels[within_quota]
            points = points[within_quota]
            camera_depths = camera_depths[within_quota]
            confidences = confidences[within_quota]
        for descriptor in active_descriptors:
            source_type = descriptor.source_type
            name = descriptor.name
            stats[name]["accepted"] = int((source_types == int(source_type)).sum())
        colors = needs.target_rgb[pixels[:, 0], pixels[:, 1]]
        focal = (frame.intrinsics.fx + frame.intrinsics.fy) * 0.5
        base_scales = (camera_depths / focal).clamp_min(self.scale_clamp_min)
        batch = GaussianAppendBatch(
            points, colors, torch.full((points.shape[0], 1), self.opacity_logit, dtype=torch.float32, device=self.device),
            torch.log((base_scales.unsqueeze(1) * self.scale_anisotropy).clamp_min(self.scale_clamp_min)),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device).repeat(points.shape[0], 1),
            source_types, confidences,
        )
        return GrowthResult(batch, GrowthStats(self._finish_stats(stats)))

    @staticmethod
    def _empty_stats(descriptors=SOURCE_DESCRIPTORS) -> dict[str, dict[str, object]]:
        return {descriptor.name: {"proposed": 0, "accepted": 0, "rejected": 0, "rejection_reasons": {}} for descriptor in descriptors}

    @staticmethod
    def _finish_stats(stats: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        for source_stats in stats.values():
            source_stats["rejected"] = int(source_stats["proposed"]) - int(source_stats["accepted"])
            reasons = source_stats["rejection_reasons"]
            if not set(reasons) <= _REJECTION_REASONS:
                raise RuntimeError("Growth rejection reasons contain an unknown bucket")
            if int(source_stats["rejected"]) != sum(reasons.values()):
                raise RuntimeError("Growth rejection statistics are inconsistent")
        return stats
