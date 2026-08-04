"""Atomic orchestration for the published appearance refinement."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch

from ..artifacts import load_checkpoint
from ..core_input import CoreObservationAssembler
from ..data.providers.spnet import OnlineSPNetProvider
from ..data.providers.spnet_cache import (
    DenseSPNetCache,
    ReusingDenseSPNetProvider,
    load_dense_spnet_cache,
)
from ..engine.losses import mapping_loss, photometric_loss
from ..engine.geometry import is_mapping_frame
from ..engine.metrics import ImageMetricEvaluator
from ..engine.model import TrainableGaussians
from ..engine.ply_export import write_gaussian_ply
from ..engine.rendering import (
    CachedRenderer,
    RenderOutput,
    capture_renderer_identity,
    prepare_render_static,
)
from ..foundation.artifact_versions import (
    APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
    CHECKPOINT_VERSION,
)
from ..foundation.code_identity import repository_code_identity
from ..foundation.config import (
    MappingLossConfig,
    SageConfig,
)
from ..foundation.contracts import DenseGeometryPrior, SourceType
from ..core_input import MappingFrame
from ..foundation.hashing import sha256_file
from ..foundation.identity_schema import validate_dataset_identity
from ..input.frame import source_frame_label
from .appearance_config import (
    AppearanceRefinementConfig,
    refinement_config_identity,
)
from .appearance_refinement import (
    AppearanceExposureNuisance,
    AppearanceObjective,
    AppearanceRefiner,
)
from .dense_geometry_config import DenseGeometryPolicy
from .dense_geometry_objective import (
    DenseNormalStatic,
    dense_normal_objective,
    dense_prior_support_counts,
    prepare_dense_normal_static,
)
from .dense_geometry_prior import build_dense_geometry_prior
from .stage2_cache import (
    Stage2DensePriorCache,
    Stage2DensePriorCacheWriter,
    Stage2InputCache,
    default_stage2_cache_path,
    remove_stage2_dense_prior_cache,
    STAGE2_CACHE_SCHEMA,
)


APPEARANCE_RUN_SCHEMA = "sage-refinement-run-v1"
_DENSE_PRIOR_PROGRESS_EVERY = 25
_REFINEMENT_PROGRESS_EVERY = 500
_TOPOLOGY_TENSORS = (
    "gaussian_ids",
    "created_at",
    "source_types",
    "source_confidences",
)
_STAGE2_CPU_LRU_FRAMES = 4
_STAGE2_GPU_LRU_FRAMES = 2


class _MemoryMappingFrames:
    """Legacy fallback retaining the historical in-memory refinement input."""

    def __init__(self, frames: tuple[MappingFrame, ...]) -> None:
        if not frames:
            raise ValueError("Appearance refinement requires mapping frames")
        self._frames = {frame.index: frame for frame in frames}
        self.frame_indices = tuple(self._frames)

    def load(self, frame_index: int) -> MappingFrame:
        return self._frames[frame_index]


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _report_count_progress(
    label: str,
    completed: int,
    total: int,
    *,
    started: float,
) -> None:
    elapsed = perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"SAGE {label}: {completed}/{total} "
        f"({100 * completed / total:.1f}%) | "
        f"elapsed {_format_duration(elapsed)} | "
        f"ETA {_format_duration(remaining)}",
        flush=True,
    )


def preflight_appearance_runtime(
    config: SageConfig,
    refinement: AppearanceRefinementConfig,
    *,
    device: str,
) -> dict[str, object]:
    """Validate the concrete refinement runtime without consuming scene data."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(
            "Appearance refinement requires an available CUDA device"
        )
    renderer_identity = capture_renderer_identity()
    metric_evaluator = None
    provider = None
    try:
        metric_evaluator = ImageMetricEvaluator(
            device,
            model_root=config.model_root,
        )
        provider = OnlineSPNetProvider(
            config.growth_sources.spnet,
            device=device,
            model_root=config.model_root,
        )
        return {
            "device": str(device),
            "renderer": renderer_identity,
            "metric": asdict(metric_evaluator.identity),
            "spnet": provider.identity.payload(),
            "refinement_config": refinement.payload(),
        }
    finally:
        del provider, metric_evaluator
        torch.cuda.empty_cache()


def verify_frozen_tensors(
    model: TrainableGaussians,
    source: dict[str, object],
    names: tuple[str, ...],
) -> dict[str, bool]:
    """Require every declared frozen tensor to remain bitwise identical."""
    verification = {
        name: torch.equal(
            getattr(model, name).detach().cpu(),
            source[name],
        )
        for name in names
    }
    changed = sorted(
        name for name, unchanged in verification.items() if not unchanged
    )
    if changed:
        raise RuntimeError(
            "SAGE changed frozen tensors: "
            + ", ".join(changed)
        )
    return verification


def _appearance_objective_from_render(
    output: RenderOutput,
    frame: MappingFrame,
    refinement: AppearanceRefinementConfig,
    loss_policy: MappingLossConfig,
    *,
    dense_policy: DenseGeometryPolicy,
    photometric_rgb: torch.Tensor | None = None,
    cached: dict[str, object],
) -> AppearanceObjective:
    optimized_rgb = (
        output.rgb if photometric_rgb is None else photometric_rgb
    )
    if optimized_rgb.shape != output.rgb.shape:
        raise ValueError(
            "Photometric RGB must match the native render shape"
        )
    target = cached["target_rgb"]
    if not isinstance(target, torch.Tensor):
        raise ValueError("Cached appearance RGB target is invalid")
    photo, _, _ = photometric_loss(
        optimized_rgb,
        target,
        ssim_weight=refinement.ssim_weight,
    )
    _, mapping_terms = mapping_loss(
        optimized_rgb,
        target,
        output.accumulated_depth,
        frame.mapping,
        rendered_alpha=output.alpha,
        policy=loss_policy,
        target_depth=cached.get("target_depth"),
        source_masks=cached.get("source_masks"),
        include_image=False,
        include_diagnostics=False,
        validate_rgb=False,
    )
    lidar_depth = mapping_terms["depth"]
    terms = {
        "photo": photo,
        "lidar_depth": lidar_depth,
    }
    total = photo + refinement.lidar_depth_weight * lidar_depth
    dense_static = cached.get("dense_normal_static")
    if not isinstance(dense_static, DenseNormalStatic):
        raise ValueError("Cached dense-normal target is invalid")
    dense_result = dense_normal_objective(
        output.accumulated_depth,
        output.alpha,
        frame.intrinsics,
        dense_policy,
        dense_static,
    )
    dense_normal = dense_result.loss
    terms["dense_normal"] = dense_normal
    total = total + refinement.dense_normal_weight * dense_normal
    diagnostics = {
        "dense_valid_normal_pixels": dense_result.valid_pixels,
        "dense_normal_weight_mass": dense_result.weight_mass,
    }
    return AppearanceObjective(
        total=total,
        terms=terms,
        diagnostics=diagnostics,
    )


def _dense_prior_record(
    frame_indices: tuple[int, ...],
    load_frame,
    load_prior,
    *,
    provider_identity: dict[str, object],
    alignment_variant: str | None = None,
    max_relative_depth_jump: float | None = None,
    prediction_cache: dict[str, object] | None = None,
) -> dict[str, object]:
    records = []
    for frame_index in frame_indices:
        frame = load_frame(frame_index)
        prior = load_prior(frame_index)
        valid_depth, valid_normals = dense_prior_support_counts(
            prior,
            frame.intrinsics,
            max_relative_depth_jump=max_relative_depth_jump,
        )
        if valid_normals < 1:
            raise ValueError(
                f"Dense prior frame {frame.index} has no valid "
                "target-normal support"
            )
        record = {
            "frame_index": frame.index,
            "lidar_support": prior.diagnostics.lidar_support,
            "fitted_grid_cells": prior.diagnostics.fitted_grid_cells,
            "pre_alignment_mae_m": (
                prior.diagnostics.pre_alignment_mae_m
            ),
            "post_alignment_mae_m": (
                prior.diagnostics.post_alignment_mae_m
            ),
            "fallback": prior.diagnostics.fallback,
            "valid_depth_pixels": valid_depth,
            "valid_normal_pixels": valid_normals,
            "scale_grid": prior.scale_grid.tolist(),
        }
        if alignment_variant is not None:
            if max_relative_depth_jump is None:
                raise ValueError(
                    "Corrected dense prior provenance requires a "
                    "relative-depth-jump bound"
                )
            record.update({
                "alignment_variant": alignment_variant,
                "fitted_alignment_mae_m": (
                    prior.diagnostics.fitted_alignment_mae_m
                ),
                "selected_alignment": (
                    prior.diagnostics.selected_alignment
                ),
                "max_relative_depth_jump": max_relative_depth_jump,
            })
        records.append(record)
    payload = {
        "provider_identity": deepcopy(provider_identity),
        "frames": records,
    }
    if prediction_cache is not None:
        payload["prediction_cache"] = deepcopy(prediction_cache)
    return payload


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def appearance_checkpoint_payload(
    source: dict[str, object],
    model: TrainableGaussians,
    config: AppearanceRefinementConfig,
    *,
    source_checkpoint_sha256: str,
    refinement_config_sha256: str,
    selected_frame_indices: tuple[int, ...],
    milestones: tuple[dict[str, object], ...],
    producer_code: dict[str, object] | None = None,
    dense_prior_record: dict[str, object] | None = None,
    exposure_nuisance_record: dict[str, object] | None = None,
) -> dict[str, object]:
    source_version = source.get("checkpoint_version")
    if source_version != CHECKPOINT_VERSION:
        raise ValueError(
            "SAGE source must be a baseline checkpoint"
        )
    if (
        not _valid_sha256(source_checkpoint_sha256)
        or not _valid_sha256(refinement_config_sha256)
    ):
        raise ValueError("Appearance refinement source hashes must be SHA-256")
    if dense_prior_record is None:
        raise ValueError(
            "SAGE requires dense prior provenance"
        )
    if exposure_nuisance_record is None:
        raise ValueError(
            "SAGE requires exposure nuisance provenance"
        )
    payload = dict(source)
    payload["checkpoint_version"] = (
        APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
    )
    for name in model.PARAMETER_NAMES + (
        "gaussian_ids",
        "created_at",
        "source_types",
        "source_confidences",
    ):
        payload[name] = getattr(model, name).detach().cpu()
    refinement_payload = {
        "schema_version": config.schema_version,
        "source_checkpoint_version": source_version,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "refinement_config_sha256": refinement_config_sha256,
        "config": config.payload(),
        "optimized_parameters": list(config.optimized_parameters),
        "iteration_count": config.iterations,
        "selected_frame_indices": list(selected_frame_indices),
        "milestones": [deepcopy(item) for item in milestones],
        "producer_code": producer_code or repository_code_identity(),
        "publication_ready": True,
    }
    refinement_payload["dense_prior"] = deepcopy(dense_prior_record)
    refinement_payload.update({
        "nuisance_parameters": list(config.nuisance_parameters),
        "exposure_nuisance": deepcopy(exposure_nuisance_record),
    })
    payload["appearance_refinement"] = refinement_payload
    return payload


def _consume_mapping_frames(
    config: SageConfig,
    checkpoint_identity: dict[str, object],
) -> tuple[Stage2InputCache | _MemoryMappingFrames, dict[str, object]]:
    """Load Stage 1's mapping cohort without replaying a ROSBAG.

    New mapping runs always publish the transient cache beside their structure
    output.  A legacy direct refinement invocation may still fall back to its
    historical adapter pass; online-window-v2 rejects that fallback because it
    is single-use by contract.
    """
    cache_path = default_stage2_cache_path(config.output_dir)
    if cache_path.is_dir():
        cache = Stage2InputCache(cache_path, cpu_lru_frames=_STAGE2_CPU_LRU_FRAMES)
        cache.validate_checkpoint_identity(checkpoint_identity)
        return cache, {
            "adapter_type": "stage2_transient_cache",
            "identities": cache.input_identities.payload(),
            "accepted_frames": len(cache.frame_indices),
            "cache": {
                "path": str(cache_path),
                "schema_version": STAGE2_CACHE_SCHEMA,
                "cpu_lru_frames": _STAGE2_CPU_LRU_FRAMES,
            },
        }
    if config.input.payload.get("execution") == "online-window-v2":
        raise ValueError(
            "online-window-v2 refinement requires the Stage 1 transient input cache; "
            "it will not replay a ROSBAG"
        )
    resolved = config.input.create_adapter().preflight()
    assembler = CoreObservationAssembler(resolved.contract.canonical.sources)
    frames = tuple(
        frame
        for frame in assembler.frames(resolved.frames())
        if is_mapping_frame(frame.index, map_every=config.mapping.map_every)
    )
    if not frames:
        raise ValueError("Input emitted no mapping frames for appearance refinement")
    # The completed frame pass settled the canonical sequence identity.  Doing
    # this validation before the pass would consume the ROS bag once merely to
    # calculate that identity, then consume it again to build refinement data.
    validate_dataset_identity(checkpoint_identity, resolved.identities)
    return _MemoryMappingFrames(frames), {
        "adapter_type": resolved.contract.adapter_type,
        "identities": resolved.identities.payload(),
        "accepted_frames": resolved.report.accepted_frames,
    }


def _load_mapping_dense_cache(
    source_checkpoint: Path,
    *,
    source_checkpoint_sha256: str,
) -> tuple[DenseSPNetCache, dict[str, object]] | None:
    """Load a structure-stage prediction cache only through its manifest hash."""
    manifest_path = source_checkpoint.with_name("run_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read structure manifest: {manifest_path}") from exc
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    checkpoint_record = artifacts.get("checkpoint") if isinstance(artifacts, dict) else None
    if (
        not isinstance(checkpoint_record, dict)
        or checkpoint_record.get("sha256") != source_checkpoint_sha256
    ):
        raise ValueError("Structure manifest does not bind the source checkpoint")
    cache_record = artifacts.get("dense_spnet_cache")
    if cache_record is None:
        return None
    if (
        not isinstance(cache_record, dict)
        or set(cache_record) != {"path", "sha256", "frame_count"}
        or cache_record.get("path") != "spnet_dense.pt"
        or type(cache_record.get("frame_count")) is not int
        or cache_record["frame_count"] < 1
        or not _valid_sha256(cache_record.get("sha256"))
    ):
        raise ValueError("Structure dense SPNet cache manifest is invalid")
    cache_path = source_checkpoint.with_name("spnet_dense.pt")
    if not cache_path.is_file() or sha256_file(cache_path) != cache_record["sha256"]:
        raise ValueError("Structure dense SPNet cache hash does not match")
    cache = load_dense_spnet_cache(cache_path)
    if len(cache.frames) != cache_record["frame_count"]:
        raise ValueError("Structure dense SPNet cache frame count does not match")
    return cache, {
        "schema_version": "sage-dense-spnet-reuse-v1",
        "source_manifest_sha256": sha256_file(manifest_path),
        "cache_sha256": cache_record["sha256"],
        "cached_frame_indices": [frame.frame_index for frame in cache.frames],
    }


def run_appearance_refinement(
    config: SageConfig,
    checkpoint: Path,
    refinement: AppearanceRefinementConfig,
    output: Path,
    *,
    device: str,
) -> Path:
    """Run the appearance phase of complete SAGE training."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(
            "Appearance refinement requires an available CUDA device"
        )
    destination = Path(output).resolve()
    source_checkpoint = Path(checkpoint).resolve()
    base_config_sha256 = sha256_file(config.config_path)
    # The training identity, not the config file: a run must stay reusable when
    # only the input section differs between two equivalent canonical paths.
    training_identity = config.training_config_identity()
    refinement_sha256 = refinement_config_identity(training_identity, refinement)
    if destination.exists():
        raise ValueError(
            f"Refusing an existing appearance output path: {destination}"
        )
    source_payload = load_checkpoint(source_checkpoint)
    if (
        source_payload["identity_snapshot"]["training_config_identity"]
        != training_identity
    ):
        raise ValueError(
            "Source checkpoint configuration does not match "
            "the appearance base config"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    started = perf_counter()
    producer_code = repository_code_identity()
    source_sha256 = sha256_file(source_checkpoint)
    dense_prior_writer: Stage2DensePriorCacheWriter | None = None
    try:
        checkpoint_identity = source_payload["identity_snapshot"]
        mapping_frames, training_input = _consume_mapping_frames(
            config,
            checkpoint_identity,
        )
        frame_indices = mapping_frames.frame_indices
        provider = OnlineSPNetProvider(
            config.growth_sources.spnet,
            device=device,
            model_root=config.model_root,
        )
        cache_result = _load_mapping_dense_cache(
            source_checkpoint,
            source_checkpoint_sha256=source_sha256,
        )
        dense_provider = provider
        cache_provenance = None
        if cache_result is not None:
            dense_cache, cache_provenance = cache_result
            dense_provider = ReusingDenseSPNetProvider(dense_cache, provider)
        dense_prior_started = perf_counter()

        def report_dense_prior(
            completed: int,
            total: int,
            _frame: MappingFrame,
        ) -> None:
            if (
                completed % _DENSE_PRIOR_PROGRESS_EVERY == 0
                or completed == total
            ):
                _report_count_progress(
                    "dense priors",
                    completed,
                    total,
                    started=dense_prior_started,
                )

        print(
            f"SAGE dense priors: preparing {len(frame_indices)} mapping frames",
            flush=True,
        )
        if isinstance(mapping_frames, Stage2InputCache):
            # A hard interruption can leave only this derived cache published.
            # It is not a Stage 2 artifact and must not block an idempotent
            # retry against the preserved input handoff.
            remove_stage2_dense_prior_cache(mapping_frames)
            dense_prior_writer = Stage2DensePriorCacheWriter(mapping_frames)
            for completed, frame_index in enumerate(frame_indices, start=1):
                frame = mapping_frames.load(frame_index)
                dense_prior_writer.write(
                    frame_index,
                    build_dense_geometry_prior(
                        dense_provider.dense_evidence_for(frame),
                        frame.mapping,
                        refinement.dense_prior,
                    ),
                )
                report_dense_prior(completed, len(frame_indices), frame)
            dense_prior_writer.finalize()
            dense_priors = Stage2DensePriorCache(
                mapping_frames,
                cpu_lru_frames=_STAGE2_CPU_LRU_FRAMES,
            )
            load_prior = dense_priors.load
        else:
            dense_priors_by_frame = {}
            for completed, frame_index in enumerate(frame_indices, start=1):
                frame = mapping_frames.load(frame_index)
                dense_priors_by_frame[frame_index] = build_dense_geometry_prior(
                    dense_provider.dense_evidence_for(frame),
                    frame.mapping,
                    refinement.dense_prior,
                )
                report_dense_prior(completed, len(frame_indices), frame)
            load_prior = dense_priors_by_frame.__getitem__
        dense_prior_seconds = perf_counter() - dense_prior_started
        dense_prior_provenance = _dense_prior_record(
            frame_indices,
            mapping_frames.load,
            load_prior,
            provider_identity=dense_provider.identity.payload(),
            alignment_variant=refinement.dense_prior.alignment_variant,
            max_relative_depth_jump=(
                refinement.dense_normal_max_relative_depth_jump
            ),
            prediction_cache=(
                {
                    **cache_provenance,
                    "reused_frames": dense_provider.reused_frames,
                    "live_frames": dense_provider.live_frames,
                }
                if isinstance(dense_provider, ReusingDenseSPNetProvider)
                and cache_provenance is not None
                else None
            ),
        )
        del dense_provider, provider
        torch.cuda.empty_cache()
        exposure_nuisance = AppearanceExposureNuisance(
            frame_indices,
            refinement,
            device=device,
        )
        model = TrainableGaussians.from_checkpoint(
            source_payload,
            device=device,
        )
        optimizes_geometry = any(
            name in refinement.optimized_parameters
            for name in ("means3d", "log_scales", "rotations")
        )
        render_static = (
            None if optimizes_geometry else prepare_render_static(model)
        )
        renderer = CachedRenderer()
        dense_policy = DenseGeometryPolicy(
            dense_depth_weight=0.0,
            dense_normal_weight=1.0,
            normal_smoothness_weight=0.0,
            edge_weight_gamma=refinement.dense_normal_edge_weight_gamma,
            alpha_support_a0=refinement.dense_normal_alpha_support_a0,
            max_relative_depth_jump=(
                refinement.dense_normal_max_relative_depth_jump
            ),
        )
        appearance_cache: OrderedDict[int, dict[str, object]] = OrderedDict()

        def appearance_targets(frame_index: int) -> dict[str, object]:
            cached = appearance_cache.get(frame_index)
            if cached is not None:
                appearance_cache.move_to_end(frame_index)
                return cached
            frame = mapping_frames.load(frame_index)
            target_rgb = torch.as_tensor(
                frame.rgb,
                dtype=torch.float32,
                device=device,
            )
            source_types = torch.as_tensor(
                frame.mapping.source_types,
                dtype=torch.uint8,
                device=device,
            )
            cached = {
                "target_rgb": target_rgb,
                "target_depth": torch.as_tensor(
                    frame.mapping.depth_m,
                    dtype=torch.float32,
                    device=device,
                ),
                "source_masks": {
                    "center": source_types == int(SourceType.LIDAR_CENTER),
                    "fused": source_types == int(SourceType.LIDAR_FUSED),
                },
                "dense_normal_static": prepare_dense_normal_static(
                    load_prior(frame_index),
                    target_rgb,
                    frame.intrinsics,
                    refinement.dense_prior,
                    dense_policy,
                ),
            }
            appearance_cache[frame_index] = cached
            while len(appearance_cache) > _STAGE2_GPU_LRU_FRAMES:
                appearance_cache.popitem(last=False)
            return cached

        def objective(
            item: TrainableGaussians,
            frame_index: int,
            _step: int,
        ) -> torch.Tensor | AppearanceObjective:
            frame = mapping_frames.load(frame_index)
            output = renderer(item, frame, static=render_static)
            photometric_rgb = (
                exposure_nuisance.apply(output.rgb, frame_index)
                if exposure_nuisance is not None
                else None
            )
            return _appearance_objective_from_render(
                output,
                frame,
                refinement,
                config.loss,
                dense_policy=dense_policy,
                photometric_rgb=photometric_rgb,
                cached=appearance_targets(frame_index),
            )

        refinement_started = perf_counter()

        def report_refinement_progress(
            step: int,
            total: int,
            frame_index: int,
            loss: torch.Tensor,
        ) -> None:
            elapsed = perf_counter() - refinement_started
            rate = step / elapsed if elapsed > 0 else 0.0
            remaining = (total - step) / rate if rate > 0 else 0.0
            print(
                f"SAGE refinement: {step}/{total} "
                f"({100 * step / total:.1f}%) | "
                f"{source_frame_label(mapping_frames.load(frame_index).canonical)} | "
                f"loss {loss.item():.6g} | "
                f"{rate:.2f} step/s | elapsed {_format_duration(elapsed)} | "
                f"ETA {_format_duration(remaining)}",
                flush=True,
            )

        print(
            f"SAGE refinement: optimizing {refinement.iterations} steps",
            flush=True,
        )
        result = AppearanceRefiner(refinement).optimize(
            model,
            frame_indices,
            objective,
            exposure_nuisance=exposure_nuisance,
            progress_every=_REFINEMENT_PROGRESS_EVERY,
            progress_callback=report_refinement_progress,
        )
        optimization_seconds = perf_counter() - refinement_started
        frozen_names = _TOPOLOGY_TENSORS + tuple(
            name
            for name in TrainableGaussians.PARAMETER_NAMES
            if name not in refinement.optimized_parameters
        )
        frozen_verification = verify_frozen_tensors(
            model,
            source_payload,
            frozen_names,
        )
        payload = appearance_checkpoint_payload(
            source_payload,
            model,
            refinement,
            source_checkpoint_sha256=source_sha256,
            refinement_config_sha256=refinement_sha256,
            selected_frame_indices=result.selected_frame_indices,
            milestones=result.milestones,
            producer_code=producer_code,
            dense_prior_record=dense_prior_provenance,
            exposure_nuisance_record=result.exposure_nuisance,
        )
        model.to("cpu")
        torch.cuda.empty_cache()
        checkpoint_path = staging / "appearance_checkpoint.pt"
        torch.save(payload, checkpoint_path)
        write_gaussian_ply(staging / "appearance_map.ply", model)
        manifest = {
            "schema_version": APPEARANCE_RUN_SCHEMA,
            "producer_code": producer_code,
            "publication_ready": True,
            "source_checkpoint": {
                "path": str(source_checkpoint),
                "sha256": source_sha256,
                "version": source_payload["checkpoint_version"],
            },
            "base_config": {
                "path": str(config.config_path),
                "sha256": base_config_sha256,
            },
            "refinement_config": {
                "path": str(config.config_path),
                "sha256": refinement_sha256,
                "payload": refinement.payload(),
            },
            "mapping_frame_indices": list(frame_indices),
            "selected_frame_indices": list(result.selected_frame_indices),
            "optimizer_steps": list(result.steps),
            "milestones": list(result.milestones),
            **(
                {"dense_prior": dense_prior_provenance}
                if dense_prior_provenance is not None
                else {}
            ),
            **(
                {"exposure_nuisance": result.exposure_nuisance}
                if result.exposure_nuisance is not None
                else {}
            ),
            "frozen_tensor_verification": frozen_verification,
            "input": training_input,
            "duration_seconds": perf_counter() - started,
            "timings": {
                "dense_prior_seconds": dense_prior_seconds,
                "optimization_seconds": optimization_seconds,
            },
            "artifacts": {
                "checkpoint": "appearance_checkpoint.pt",
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "map_ply": "appearance_map.ply",
            },
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if sha256_file(config.config_path) != base_config_sha256:
            raise RuntimeError("Appearance base config changed during refinement")
        staging.rename(destination)
    except BaseException:
        if dense_prior_writer is not None:
            dense_prior_writer.abort()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
