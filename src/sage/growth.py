from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GaussianInitializationConfig, GrowthConfig
from .contracts import DepthEvidence, FrameInputs, GaussianAppendBatch, GrowthInputs, SourceType
from .geometry import backproject_depth
from .rendering import RenderOutput
from .source_policy import SOURCE_DESCRIPTORS, descriptor_for_type, descriptors_for_types, source_policy_value


@dataclass(frozen=True)
class GrowthStats:
    by_source: dict[str, dict[str, object]]


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
        target_rgb = torch.as_tensor(frame.rgb, dtype=torch.float32, device=self.device)
        rgb_residual = (rendered.rgb - target_rgb).abs().mean(dim=-1)
        coverage_gate = torch.isfinite(rendered.alpha) & (rendered.alpha < self.config.coverage_alpha_threshold)
        coverage_gate &= torch.isfinite(rendered.rgb).all(dim=-1) & (rgb_residual > self.config.rgb_residual_threshold)
        stats = self._empty_stats(active_descriptors)
        proposals: list[tuple[SourceType, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        occupied = torch.zeros(expected, dtype=torch.bool, device=self.device)
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
            residual = source_policy_value(self.config.residual_thresholds, descriptor.name)
            rendered_depth = rendered.accumulated_depth
            rendered_depth_valid = torch.isfinite(rendered_depth) & (rendered_depth > 0)
            tolerance = torch.maximum(torch.full_like(depth, residual.absolute_m), residual.relative * depth)
            geometry_gate = rendered_depth_valid & valid & ((rendered_depth - depth).abs() > tolerance)
            gate = confidence_valid & (coverage_gate | geometry_gate)
            _increment_rejection(source_stats, "gate_blocked", int((confidence_valid & ~gate).sum()))
            if not bool(gate.any()):
                continue
            suppressed = occupied
            local_keep = gate & ~suppressed
            _increment_rejection(source_stats, "duplicate_2d", int((gate & suppressed).sum()))
            occupied |= local_keep
            pixels = torch.nonzero(local_keep, as_tuple=False)
            if pixels.numel() == 0:
                continue
            points, _, proposal_depth = backproject_depth(depth, frame.intrinsics, frame.pose, local_keep)
            proposals.append((evidence.source_type, pixels, points, proposal_depth, confidence[local_keep]))

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
        for descriptor in active_descriptors:
            source_type = descriptor.source_type
            name = descriptor.name
            stats[name]["accepted"] = int((source_types == int(source_type)).sum())
        rgb = torch.as_tensor(frame.rgb, dtype=torch.float32, device=self.device)
        colors = rgb[pixels[:, 0], pixels[:, 1]]
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
