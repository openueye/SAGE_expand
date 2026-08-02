from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..foundation.config import GaussianInitializationConfig, GrowthConfig
from ..foundation.contracts import DepthEvidence, FrameInputs, GaussianAppendBatch, GrowthInputs, SourceType
from .geometry import backproject_depth
from .losses import alpha_normalized_depth
from ..foundation.source_policy import SOURCE_DESCRIPTORS, descriptor_for_type, descriptors_for_types, source_policy_value

if TYPE_CHECKING:
    from .rendering import RenderOutput


@dataclass(frozen=True)
class GrowthStats:
    by_source: dict[str, dict[str, object]]


@dataclass(frozen=True)
class FrameNeeds:
    """哪些像素需要新增高斯——只由渲染结果与目标图像决定，不知道有几个证据源。

    coverage 与 conflict 是动机不同的两条通道，必须分开保留：合并成一条 bool
    之后就无法回答"这个候选是补空洞还是修冲突"，也无法量化某个分量挡掉了多少。
    conflict 依赖逐源的深度与容差，因此只在这里提供它的公共部分。
    """

    target_rgb: torch.Tensor
    coverage: torch.Tensor          # alpha 不足且 RGB 无法解释 —— 需要补面
    low_alpha: torch.Tensor         # 仅 alpha 不足，coverage 的上界，用于归因
    rendered_depth: torch.Tensor    # alpha 归一化深度
    rendered_depth_valid: torch.Tensor


@dataclass(frozen=True)
class GrowthResult:
    batch: GaussianAppendBatch
    stats: GrowthStats


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

    def _frame_needs(self, frame: FrameInputs, rendered: RenderOutput) -> FrameNeeds:
        """L1：由渲染结果推出帧级需求。纯函数，与证据源无关。"""
        target_rgb = torch.as_tensor(frame.rgb, dtype=torch.float32, device=self.device)
        rgb_residual = (rendered.rgb - target_rgb).abs().mean(dim=-1)
        rendered_depth = alpha_normalized_depth(rendered.accumulated_depth, rendered.alpha)
        low_alpha = torch.isfinite(rendered.alpha) & (rendered.alpha < self.config.coverage_alpha_threshold)
        rgb_unexplained = torch.isfinite(rendered.rgb).all(dim=-1) & (rgb_residual > self.config.rgb_residual_threshold)
        return FrameNeeds(
            target_rgb=target_rgb,
            coverage=low_alpha & rgb_unexplained,
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
    def _arbitrate_2d(
        gated: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        descriptors: tuple,
        stats: dict[str, dict[str, object]],
    ) -> dict[int, torch.Tensor]:
        """L2：同一像素只允许一个源播种，按显式声明的优先级裁决。

        优先级来自 descriptor.priority，不再是"谁先被循环到谁占坑"的副作用；
        返回的字典也按优先级排序，使下游 3D 去重的"先到先得"跟随同一套顺序。
        """
        keeps: dict[int, torch.Tensor] = {}
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
            keeps[int(descriptor.source_type)] = keep
        return keeps

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
                # 均匀抽样而不是按光栅顺序截断，否则超额的源只会保留图像上半部
                rows = rows[torch.linspace(0, int(rows.numel()) - 1, quota, device=rows.device).long()]
            within_quota[rows] = True
        return within_quota

    def build(
        self,
        frame: FrameInputs,
        rendered: RenderOutput,
        *,
        extra_evidences: tuple[DepthEvidence, ...] = (),
    ) -> GrowthResult:
        growth_inputs = GrowthInputs((*frame.growth.evidences, *extra_evidences))
        active_descriptors = descriptors_for_types(evidence.source_type for evidence in growth_inputs.evidences)
        expected = (frame.intrinsics.height, frame.intrinsics.width)
        if any(evidence.depth_m.shape != expected for evidence in growth_inputs.evidences):
            raise ValueError("GrowthInputs evidences must match frame shape")
        if tuple(rendered.accumulated_depth.shape) != expected or tuple(rendered.alpha.shape) != expected or tuple(rendered.rgb.shape) != (*expected, 3):
            raise ValueError("Rendered output dimensions do not match frame")
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
            # 把旧的单一 gate_blocked 拆成互斥两类，便于量化 RGB 残差项挡掉了多少
            # 本该补洞的低 alpha 像素。两者之和仍等于原先的 gate_blocked。
            blocked = confidence_valid & ~gate
            _increment_rejection(source_stats, "rgb_residual_suppressed", int((blocked & needs.low_alpha).sum()))
            _increment_rejection(source_stats, "gate_blocked", int((blocked & ~needs.low_alpha).sum()))
            gated[int(evidence.source_type)] = (gate, depth, confidence)

        proposals: list[tuple[SourceType, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for source_type, keep in self._arbitrate_2d(gated, active_descriptors, stats).items():
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
        voxel = torch.floor(points / self.config.candidate_duplicate_3d_threshold_m).to(torch.int64)
        _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
        indices = torch.arange(points.shape[0], device=self.device, dtype=torch.int64)
        first = torch.full((int(inverse.max()) + 1,), points.shape[0], dtype=torch.int64, device=self.device)
        first.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
        keep = indices == first[inverse]
        offset = 0
        for source_type, source_pixels, _, _, _ in proposals:
            end = offset + source_pixels.shape[0]
            duplicate_count = int((~keep[offset:end]).sum())
            _increment_rejection(stats[descriptor_for_type(source_type).name], "duplicate_3d", duplicate_count)
            offset = end
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
            if int(source_stats["rejected"]) != sum(reasons.values()):
                raise RuntimeError("Growth rejection statistics are inconsistent")
        return stats
