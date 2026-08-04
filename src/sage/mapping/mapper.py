from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from ..data.providers.spnet import SPNetEvidenceProvider
from ..data.providers.spnet_cache import DenseSPNetFrame
from ..engine.geometry import (
    build_optimization_schedule,
    is_mapping_frame,
    select_current_anchored_global_views,
)
from ..engine.growth import GrowthBuilder
from ..engine.losses import mapping_loss, mapping_training_loss
from ..engine.model import TrainableGaussians
from ..engine.rendering import RenderOutput, Renderer
from ..foundation.config import (
    GLOBAL_CURRENT_ANCHORED_VARIANT,
    GaussianInitializationConfig,
    GrowthConfig,
    MappingConfig,
    MappingLossConfig,
    PruningConfig,
)
from ..foundation.contracts import DepthEvidence, SourceType
from ..core_input import MappingFrame
from ..foundation.source_policy import SOURCE_DESCRIPTORS, descriptor_for_type, descriptors_for_types, opacity_keep_mask, source_counts


def should_invoke_spnet(
    frame_index: int,
    mapping: MappingConfig,
) -> bool:
    return (
        frame_index > 0
        and is_mapping_frame(frame_index, map_every=mapping.map_every)
    )


def expected_spnet_invocations(
    processed_frames: int,
    mapping: MappingConfig,
) -> int:
    """Derive expected calls after EOF from the independently observed run length."""
    if type(processed_frames) is not int or processed_frames < 0:
        raise ValueError("processed_frames must be a non-negative integer")
    return sum(
        should_invoke_spnet(
            frame_index,
            mapping,
        )
        for frame_index in range(processed_frames)
    )


def mapping_spnet_evidence(
    provider: SPNetEvidenceProvider,
    frame: MappingFrame,
) -> tuple[DepthEvidence | None, DenseSPNetFrame | None]:
    """Use bundled dense evidence when supported, preserving legacy providers."""
    bundle_for = getattr(provider, "evidence_bundle_for", None)
    if bundle_for is None:
        return provider.evidence_for(frame), None
    bundle = bundle_for(frame)
    return bundle.growth_evidence, DenseSPNetFrame(
        frame.index,
        frame.stem,
        bundle.dense_evidence.depth_m.copy(),
    )


@dataclass(frozen=True)
class PruningResult:
    pruned: int
    pruned_by_source: dict[str, int]
    pruned_by_reason: dict[str, int]
    newborn_protected: int
    newborn_protected_by_source: dict[str, int]
    mature_opacity_removed: int
    mature_opacity_removed_by_source: dict[str, int]
    newborn_protected_ids: torch.Tensor
    spnet_age_protected: int
    spnet_age_protected_by_source: dict[str, int]
    spnet_age_protected_ids: torch.Tensor


@dataclass(frozen=True)
class MappingCommit:
    frame_index: int
    frame_stem: str
    timestamp_ns: int
    active_window: tuple[int, ...]
    optimization_views: tuple[int, ...]
    optimizer_step: int
    mapping_commit_ordinal: int
    pruning_completed_steps: int
    pruning_event_executed: bool
    gaussian_count: int
    added: int
    added_by_source: dict[str, int]
    proposed_by_source: dict[str, int]
    accepted_by_source: dict[str, int]
    rejected_by_source: dict[str, int]
    rejection_reasons_by_source: dict[str, dict[str, int]]
    pruned: int
    pruned_by_source: dict[str, int]
    pruned_by_reason: dict[str, int]
    newborn_protected: int
    newborn_protected_by_source: dict[str, int]
    mature_opacity_removed: int
    mature_opacity_removed_by_source: dict[str, int]
    remaining_by_source: dict[str, int]
    spnet_available: bool
    final_loss: float
    psnr: float | None
    ssim: float | None
    lpips: float | None
    fps: float
    final_image_loss: float
    final_depth_loss: float
    final_depth_coverage_loss: float
    final_depth_valid_pixels: int
    final_depth_mean_alpha: float
    spnet_invoked: bool = False
    spnet_mode: str = "online"
    spnet_valid_pixels: int = 0
    spnet_inference_seconds: float = 0.0
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MappingRun:
    model: TrainableGaussians
    commits: tuple[MappingCommit, ...]
    processed_frames: int
    duration_seconds: float
    fps: float
    spnet_identity: object | None = None
    peak_cuda_memory_bytes: int | None = None
    spnet_expected_invocations: int = 0
    spnet_actual_invocations: int = 0
    spnet_inference_seconds: tuple[float, ...] = ()
    optimization_variant: str = GLOBAL_CURRENT_ANCHORED_VARIANT
    optimizer_lifecycle: str = "persistent"
    optimizer_final_step: int | None = None
    optimizer_append_migrations: int = 0
    optimizer_prune_migrations: int = 0
    spnet_anchor_source_types: tuple[str, ...] = ()
    dense_spnet_frames: tuple[DenseSPNetFrame, ...] = ()


@dataclass(frozen=True)
class _CachedFrameTargets:
    rgb: torch.Tensor
    depth: torch.Tensor
    source_masks: dict[str, torch.Tensor]


class MappingEngine:
    def __init__(
        self,
        config: MappingConfig,
        pruning: PruningConfig,
        growth: GrowthConfig,
        *,
        device: str | torch.device = "cuda",
        renderer: Renderer | None = None,
        metric_evaluator: Callable[[torch.Tensor, torch.Tensor], object] | None = None,
        spnet_provider: SPNetEvidenceProvider,
        gaussian_initialization: GaussianInitializationConfig,
        loss_policy: MappingLossConfig,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.pruning = pruning
        self.growth = growth
        self.spnet_min_prune_age = pruning.spnet_min_prune_age
        self.device = torch.device(device)
        if renderer is None:
            raise RuntimeError("Production mapping requires the gsplat CUDA renderer")
        self.renderer = renderer
        self.metric_evaluator = metric_evaluator
        self.gaussian_initialization = gaussian_initialization
        self.loss_policy = loss_policy
        self.spnet_provider = spnet_provider
        self.rng = np.random.default_rng(seed)
        self.growth_builder = GrowthBuilder(
            growth, device=self.device, gaussian_initialization=gaussian_initialization
        )

    def _render(self, model: TrainableGaussians, frame: MappingFrame) -> RenderOutput:
        """Render once at the raw seam shared by growth, loss, and evaluation."""
        return self.renderer(model, frame)

    def _targets(
        self,
        frame: MappingFrame,
        cache: dict[int, _CachedFrameTargets],
    ) -> _CachedFrameTargets:
        """Return GPU-resident loss targets cached for the run's lifetime.

        Optimization views are resampled with replacement across iterations and
        commits. Cache every immutable target consumed by the mapping loss so a
        repeated keyframe does not cross the NumPy-to-CUDA seam again.
        """
        targets = cache.get(frame.index)
        if targets is None:
            source_types = torch.as_tensor(
                frame.mapping.source_types,
                dtype=torch.uint8,
                device=self.device,
            )
            targets = _CachedFrameTargets(
                rgb=torch.as_tensor(
                    frame.rgb, dtype=torch.float32, device=self.device,
                ),
                depth=torch.as_tensor(
                    frame.mapping.depth_m,
                    dtype=torch.float32,
                    device=self.device,
                ),
                source_masks={
                    "center": (
                        source_types == int(SourceType.LIDAR_CENTER)
                    ),
                    "fused": (
                        source_types == int(SourceType.LIDAR_FUSED)
                    ),
                },
            )
            cache[frame.index] = targets
        return targets

    @staticmethod
    def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
        steps = {
            int(state["step"].item())
            for state in optimizer.state.values()
            if isinstance(state, dict) and "step" in state and torch.is_tensor(state["step"])
        }
        if len(steps) != 1:
            raise RuntimeError("Optimizer step state is inconsistent")
        return steps.pop()

    def run(self, frames: Iterable[MappingFrame]) -> MappingRun:
        run_started = perf_counter()
        frame_iterator = iter(frames)
        try:
            first_frame = next(frame_iterator)
        except StopIteration as exc:
            raise ValueError("Mapping requires at least one frame") from exc
        model = TrainableGaussians.from_frame(
            first_frame,
            device=self.device,
            gaussian_initialization=self.gaussian_initialization,
        )
        persistent_optimizer = torch.optim.Adam(
            model.parameter_groups(self.config.learning_rates), lr=0.0, eps=1e-15
        )
        optimizer_append_migrations = 0
        optimizer_prune_migrations = 0
        keyframes: list[int] = []
        keyframe_by_index: dict[int, MappingFrame] = {}
        target_cache: dict[int, _CachedFrameTargets] = {}
        commits: list[MappingCommit] = []
        actual_spnet_invocations = 0
        spnet_inference_seconds: list[float] = []
        spnet_anchor_source_types: set[str] = set()
        dense_spnet_frames: list[DenseSPNetFrame] = []
        processed_frames = 0
        commit_ordinal_by_frame_index: dict[int, int] = {}
        for frame in chain((first_frame,), frame_iterator):
            processed_frames += 1
            should_map = is_mapping_frame(frame.index, map_every=self.config.map_every)
            if should_map:
                mapping_commit_number = len(commits) + 1
                historical = [index for index in keyframes if index < frame.index]
                active_window = select_current_anchored_global_views(
                    historical,
                    frame.index,
                    max_views=self.config.iterations,
                    rng=self.rng,
                )
                optimization_views = list(build_optimization_schedule(
                    active_window,
                    frame.index,
                    steps=self.config.iterations,
                    rng=self.rng,
                ))
                optimizer = persistent_optimizer
                growth_batch = None
                growth_stats = None
                spnet_invoked = False
                spnet_valid_pixels = 0
                spnet_inference_elapsed = 0.0
                provider_identity = self.spnet_provider.identity
                commit_spnet_mode = provider_identity.mode
                spnet_available = any(
                    evidence.source_type == SourceType.SPNET_COMPLETED
                    for evidence in frame.growth.evidences
                )
                invoke_spnet = should_invoke_spnet(
                    frame.index,
                    self.config,
                )
                if frame.index > 0:
                    anchor_types = {
                        descriptor_for_type(evidence.source_type).name
                        for evidence in frame.growth.evidences
                        if evidence.source_type in {
                            SourceType.LIDAR_CENTER,
                        }
                    }
                    if len(anchor_types) != 1:
                        raise ValueError("SPNet run requires exactly one profile center anchor source")
                    spnet_anchor_source_types.update(anchor_types)
                    extra_evidences = ()
                    if invoke_spnet:
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        provider_started = perf_counter()
                        provider_evidence, dense_spnet_frame = mapping_spnet_evidence(
                            self.spnet_provider,
                            frame,
                        )
                        if dense_spnet_frame is not None:
                            dense_spnet_frames.append(dense_spnet_frame)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        spnet_inference_elapsed = perf_counter() - provider_started
                        spnet_inference_seconds.append(spnet_inference_elapsed)
                        spnet_invoked = True
                        actual_spnet_invocations += 1
                        if provider_evidence is not None:
                            if provider_evidence.source_type != SourceType.SPNET_COMPLETED:
                                raise ValueError("SPNet provider must return SPNET_COMPLETED evidence")
                            extra_evidences = (provider_evidence,)
                            spnet_available = True
                            spnet_valid_pixels = int(provider_evidence.valid_mask.sum())
                    with torch.no_grad():
                        growth_result = self.growth_builder.build(
                            frame,
                            self._render(model, frame),
                            extra_evidences=extra_evidences,
                            existing_points=model.means3d.detach(),
                        )
                    growth_batch = growth_result.batch
                    growth_stats = growth_result.stats
                    added_count = model.append_batch(
                        growth_batch,
                        created_at=frame.index,
                        optimizer=optimizer,
                    )
                    if added_count:
                        optimizer_append_migrations += 1
                final_loss = 0.0
                active_descriptors = descriptors_for_types(model.source_types)
                pruned_by_source = {descriptor.name: 0 for descriptor in active_descriptors}
                pruned_by_reason = {"opacity_only": 0, "scale_only": 0, "opacity_and_scale": 0}
                newborn_protected_by_source = {descriptor.name: 0 for descriptor in active_descriptors}
                mature_opacity_removed_by_source = {descriptor.name: 0 for descriptor in active_descriptors}
                newborn_protected_id_chunks: list[torch.Tensor] = []
                pruned = 0
                newborn_protected = 0
                mature_opacity_removed = 0
                pruning_event_executed = False
                step_count = len(optimization_views)
                for iteration in range(step_count):
                    selected_index = optimization_views[iteration]
                    selected_frame = frame if selected_index == frame.index else keyframe_by_index[selected_index]
                    optimizer.zero_grad(set_to_none=True)
                    output = self._render(model, selected_frame)
                    targets = self._targets(selected_frame, target_cache)
                    loss = mapping_training_loss(
                        output.rgb,
                        targets.rgb,
                        output.accumulated_depth,
                        selected_frame.mapping,
                        rendered_alpha=output.alpha,
                        policy=self.loss_policy,
                        target_depth=targets.depth,
                        source_masks=targets.source_masks,
                        validate_rgb=False,
                    )
                    torch._assert_async(
                        torch.isfinite(loss),
                        f"Non-finite mapping loss at frame {selected_frame.stem}",
                    )
                    loss.backward()
                    optimizer.step()
                    completed_steps = iteration + 1
                    if (completed_steps <= self.config.prune_stop_after
                            and completed_steps % self.config.prune_every == 0):
                        prune_result = self._prune_once(
                            model,
                            optimizer,
                            current_frame_index=frame.index,
                            mapping_commit_ordinal=mapping_commit_number,
                            commit_ordinal_by_frame_index=commit_ordinal_by_frame_index,
                        )
                        pruning_event_executed = True
                        if prune_result.pruned:
                            optimizer_prune_migrations += 1
                        pruned += prune_result.pruned
                        for name, count in prune_result.pruned_by_source.items():
                            pruned_by_source[name] += count
                        for name, count in prune_result.pruned_by_reason.items():
                            pruned_by_reason[name] += count
                        mature_opacity_removed += prune_result.mature_opacity_removed
                        if prune_result.newborn_protected_ids.numel():
                            newborn_protected_id_chunks.append(prune_result.newborn_protected_ids)
                        for name, count in prune_result.mature_opacity_removed_by_source.items():
                            mature_opacity_removed_by_source[name] += count
                with torch.no_grad():
                    final_output = self._render(model, frame)
                    final_targets = self._targets(frame, target_cache)
                    _, final_terms = mapping_loss(
                        final_output.rgb,
                        final_targets.rgb,
                        final_output.accumulated_depth,
                        frame.mapping,
                        rendered_alpha=final_output.alpha,
                        policy=self.loss_policy,
                        collect_diagnostics=True,
                        target_depth=final_targets.depth,
                        source_masks=final_targets.source_masks,
                        validate_rgb=False,
                    )
                final_loss = float(final_terms["total"].detach())
                if newborn_protected_id_chunks:
                    protected_ids = torch.unique(torch.cat(newborn_protected_id_chunks))
                    protected_mask = torch.isin(model.gaussian_ids, protected_ids)
                    newborn_protected = int(protected_mask.sum())
                    newborn_protected_by_source = source_counts(
                        model.source_types[protected_mask],
                        descriptors=descriptors_for_types(model.source_types),
                    )
                quality = None
                if self.metric_evaluator is not None:
                    with torch.no_grad():
                        quality = self.metric_evaluator(
                            final_output.rgb,
                            final_targets.rgb,
                        )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed = perf_counter() - run_started
                optimizer_step = self._optimizer_step(optimizer)
                added_by_source = {descriptor.name: int((growth_batch.source_types == int(descriptor.source_type)).sum()) for descriptor in active_descriptors} if growth_batch is not None else {descriptor.name: 0 for descriptor in active_descriptors}
                proposed_by_source = {name: int(values["proposed"]) for name, values in (growth_stats.by_source.items() if growth_stats is not None else {})}
                accepted_by_source = {name: int(values["accepted"]) for name, values in (growth_stats.by_source.items() if growth_stats is not None else {})}
                rejected_by_source = {name: int(values["rejected"]) for name, values in (growth_stats.by_source.items() if growth_stats is not None else {})}
                rejection_reasons = {name: dict(values["rejection_reasons"]) for name, values in (growth_stats.by_source.items() if growth_stats is not None else {})}
                for descriptor in active_descriptors:
                    proposed_by_source.setdefault(descriptor.name, 0)
                    accepted_by_source.setdefault(descriptor.name, 0)
                    rejected_by_source.setdefault(descriptor.name, 0)
                    rejection_reasons.setdefault(descriptor.name, {})
                commits.append(MappingCommit(
                    frame_index=frame.index,
                    frame_stem=frame.stem,
                    timestamp_ns=frame.timestamp_ns,
                    active_window=tuple(active_window),
                    optimization_views=tuple(optimization_views),
                    optimizer_step=optimizer_step,
                    mapping_commit_ordinal=mapping_commit_number,
                    pruning_completed_steps=step_count,
                    pruning_event_executed=pruning_event_executed,
                    gaussian_count=model.count,
                    added=sum(added_by_source.values()),
                    added_by_source=added_by_source,
                    proposed_by_source=proposed_by_source,
                    accepted_by_source=accepted_by_source,
                    rejected_by_source=rejected_by_source,
                    rejection_reasons_by_source=rejection_reasons,
                    pruned=pruned,
                    pruned_by_source=pruned_by_source,
                    pruned_by_reason=pruned_by_reason,
                    newborn_protected=newborn_protected,
                    newborn_protected_by_source=newborn_protected_by_source,
                    mature_opacity_removed=mature_opacity_removed,
                    mature_opacity_removed_by_source=mature_opacity_removed_by_source,
                    remaining_by_source=source_counts(model.source_types),
                    spnet_available=spnet_available,
                    final_loss=final_loss,
                    psnr=getattr(quality, "psnr", None),
                    ssim=getattr(quality, "ssim", None),
                    lpips=getattr(quality, "lpips", None),
                    fps=processed_frames / elapsed,
                    final_image_loss=float(final_terms["image"].detach()),
                    final_depth_loss=float(final_terms["depth"].detach()),
                    final_depth_coverage_loss=float(final_terms["depth_coverage"].detach()),
                    final_depth_valid_pixels=int(final_terms["depth_valid_pixels"].detach()),
                    final_depth_mean_alpha=float(final_terms["depth_mean_alpha"].detach()),
                    spnet_invoked=spnet_invoked,
                    spnet_mode=commit_spnet_mode,
                    spnet_valid_pixels=spnet_valid_pixels,
                    spnet_inference_seconds=spnet_inference_elapsed,
                    diagnostics={
                        "loss": {
                            "photo": float(final_terms["photo"].detach()),
                            "geo_center": float(final_terms["geo_center"].detach()),
                            "geo_fused": float(final_terms["geo_fused"].detach()),
                            "hit_center": float(final_terms["hit_center"].detach()),
                            "hit_fused": float(final_terms["hit_fused"].detach()),
                            "depth_coverage": float(final_terms["depth_coverage"].detach()),
                            "total": final_loss,
                        },
                        "depth": {
                            "valid_pixels": int(final_terms["depth_valid_pixels"].detach()),
                            "valid_center_pixels": int(final_terms["depth_valid_center_pixels"].detach()),
                            "valid_fused_pixels": int(final_terms["depth_valid_fused_pixels"].detach()),
                            "center_mae": float(final_terms["depth_center_mae"].detach()),
                            "fused_mae": float(final_terms["depth_fused_mae"].detach()),
                        },
                        "alpha": {
                            "center_mean": float(final_terms["alpha_center_mean"].detach()),
                            "center_p10": float(final_terms["alpha_center_p10"].detach()),
                            "center_p50": float(final_terms["alpha_center_p50"].detach()),
                            "fused_mean": float(final_terms["alpha_fused_mean"].detach()),
                            "fused_p10": float(final_terms["alpha_fused_p10"].detach()),
                            "fused_p50": float(final_terms["alpha_fused_p50"].detach()),
                            "below_a0_fraction": float(final_terms["alpha_below_a0_fraction"].detach()),
                        },
                    },
                ))
                commit_ordinal_by_frame_index[frame.index] = mapping_commit_number
            if frame.index == 0 or (frame.index + 1) % self.config.keyframe_every == 0:
                keyframes.append(frame.index)
                keyframe_by_index[frame.index] = frame
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        duration = perf_counter() - run_started
        optimizer_final_step = self._optimizer_step(persistent_optimizer)
        expected_spnet_invocations_for_run = expected_spnet_invocations(
            processed_frames,
            self.config,
        )
        return MappingRun(
            model,
            tuple(commits),
            processed_frames,
            duration,
            processed_frames / duration,
            spnet_expected_invocations=expected_spnet_invocations_for_run,
            spnet_actual_invocations=actual_spnet_invocations,
            spnet_inference_seconds=tuple(spnet_inference_seconds),
            spnet_identity=self.spnet_provider.identity,
            optimization_variant=self.config.optimization_variant,
            optimizer_lifecycle="persistent",
            optimizer_final_step=optimizer_final_step,
            optimizer_append_migrations=optimizer_append_migrations,
            optimizer_prune_migrations=optimizer_prune_migrations,
            spnet_anchor_source_types=tuple(sorted(spnet_anchor_source_types)),
            dense_spnet_frames=tuple(dense_spnet_frames),
        )

    @staticmethod
    def _age_in_mapping_commits(
        created_at: torch.Tensor,
        *,
        current_frame_index: int,
        mapping_commit_ordinal: int,
        commit_ordinal_by_frame_index: dict[int, int],
    ) -> torch.Tensor:
        """Vectorized age-in-mapping-commits, matching the row-event sidecar's age=0
        special case (current-commit rows, including the frame-0 bootstrap population).

        created_at only ever takes values from the small set of past commit frame
        indices, so this gathers through a tiny frame_index -> commit_ordinal lookup
        table instead of sorting the full (up to ~1M-row) Gaussian population.
        """
        device = created_at.device
        frame_indices = list(commit_ordinal_by_frame_index.keys()) + [current_frame_index]
        ordinals = list(commit_ordinal_by_frame_index.values()) + [mapping_commit_ordinal]
        lookup = torch.zeros(max(frame_indices) + 1, dtype=torch.int32, device=device)
        lookup[torch.tensor(frame_indices, dtype=torch.int64, device=device)] = torch.tensor(
            ordinals, dtype=torch.int32, device=device,
        )
        return mapping_commit_ordinal - lookup[created_at]

    def _prune_once(
        self,
        model: TrainableGaussians,
        optimizer: torch.optim.Optimizer,
        *,
        current_frame_index: int,
        mapping_commit_ordinal: int,
        commit_ordinal_by_frame_index: dict[int, int],
    ) -> PruningResult:
        opacity_keep = opacity_keep_mask(model.opacities, model.source_types, self.pruning.opacity_thresholds)
        scale_keep = torch.ones_like(opacity_keep)
        ceiling = self.pruning.spnet_scale_ceiling_m
        if ceiling is not None:
            is_spnet = model.source_types == int(SourceType.SPNET_COMPLETED)
            over_ceiling = is_spnet & (model.scales.detach().max(dim=1).values > ceiling)
            scale_keep = ~over_ceiling
        policy_keep = opacity_keep & scale_keep
        newborn = (current_frame_index > 0) & (model.created_at == current_frame_index)
        row_age = self._age_in_mapping_commits(
            model.created_at,
            current_frame_index=current_frame_index,
            mapping_commit_ordinal=mapping_commit_ordinal,
            commit_ordinal_by_frame_index=commit_ordinal_by_frame_index,
        )
        spnet_age_protected_mask = (
            (model.source_types == int(SourceType.SPNET_COMPLETED))
            & ~policy_keep
            & (row_age >= 1)
            & (row_age < self.spnet_min_prune_age)
        )
        keep = newborn | policy_keep | spnet_age_protected_mask
        if not bool(keep.any()):
            keep[torch.argmax(model.opacities.detach().reshape(-1))] = True
        removed_mask = ~keep
        active_descriptors = descriptors_for_types(model.source_types)
        by_source = {descriptor.name: int((removed_mask & (model.source_types == int(descriptor.source_type))).sum()) for descriptor in active_descriptors}
        opacity_only_mask = removed_mask & ~opacity_keep & scale_keep
        scale_only_mask = removed_mask & opacity_keep & ~scale_keep
        opacity_and_scale_mask = removed_mask & ~opacity_keep & ~scale_keep
        by_reason = {
            "opacity_only": int(opacity_only_mask.sum()),
            "scale_only": int(scale_only_mask.sum()),
            "opacity_and_scale": int(opacity_and_scale_mask.sum()),
        }
        newborn_protected_mask = newborn & ~policy_keep
        mature_removed_mask = removed_mask & ~newborn
        mature_opacity_mask = mature_removed_mask & ~opacity_keep
        removed = int(removed_mask.sum())
        newborn_protected = int(newborn_protected_mask.sum())
        newborn_protected_ids = model.gaussian_ids[newborn_protected_mask].detach()
        mature_opacity_removed = int(mature_opacity_mask.sum())
        newborn_protected_by_source = {
            descriptor.name: int((newborn_protected_mask & (model.source_types == int(descriptor.source_type))).sum())
            for descriptor in active_descriptors
        }
        mature_opacity_removed_by_source = {
            descriptor.name: int((mature_opacity_mask & (model.source_types == int(descriptor.source_type))).sum())
            for descriptor in active_descriptors
        }
        spnet_age_protected = int(spnet_age_protected_mask.sum())
        spnet_age_protected_ids = model.gaussian_ids[spnet_age_protected_mask].detach()
        spnet_age_protected_by_source = {
            descriptor.name: int((spnet_age_protected_mask & (model.source_types == int(descriptor.source_type))).sum())
            for descriptor in active_descriptors
        }
        if (sum(by_source.values()) != removed or sum(by_reason.values()) != removed
                or sum(newborn_protected_by_source.values()) != newborn_protected
                or sum(mature_opacity_removed_by_source.values()) != mature_opacity_removed
                or sum(spnet_age_protected_by_source.values()) != spnet_age_protected):
            raise RuntimeError("Pruning statistics are inconsistent with the final keep mask")
        model.prune_rows(keep, optimizer=optimizer)
        return PruningResult(
            removed,
            by_source,
            by_reason,
            newborn_protected,
            newborn_protected_by_source,
            mature_opacity_removed,
            mature_opacity_removed_by_source,
            newborn_protected_ids,
            spnet_age_protected,
            spnet_age_protected_by_source,
            spnet_age_protected_ids,
        )
