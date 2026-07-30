"""Atomic orchestration for the published appearance refinement."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch

from .appearance_config import AppearanceRefinementConfig
from .appearance_refinement import (
    AppearanceExposureNuisance,
    AppearanceObjective,
    AppearanceRefiner,
)
from .artifact_identity import (
    normalize_dataset_identity,
    validate_dataset_identity,
)
from .artifact_versions import (
    APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
    CHECKPOINT_VERSION,
)
from .artifacts import load_checkpoint, write_model_ply
from .code_identity import repository_code_identity
from .config import (
    ALL_ACCEPTED_FRAME_LIMIT,
    MappingLossConfig,
    SageConfig,
)
from .contracts import DenseGeometryPrior, FrameInputs
from .dense_geometry_objective import (
    dense_geometry_objective,
    dense_prior_support_counts,
    prepare_dense_geometry_static,
)
from .dense_geometry_prior import prepare_dense_priors
from .evaluation import (
    EvaluationDepthPolicy,
    aggregate_evaluation_frame_reports,
    evaluate_frames,
)
from .frame_source import frame_source_for_config
from .dense_geometry_config import DenseGeometryPolicy
from .hashing import sha256_file
from .losses import mapping_loss, photometric_loss
from .metrics import ImageMetricEvaluator
from .model import TrainableGaussians
from .providers.spnet import OnlineSPNetProvider
from .rendering import (
    RenderOutput,
    capture_renderer_identity,
    prepare_render_static,
    render,
)


APPEARANCE_RUN_SCHEMA = "sage-refinement-run-v1"
_GEOMETRY_AND_TOPOLOGY_TENSORS = (
    "means3d",
    "log_scales",
    "rotations",
    "gaussian_ids",
    "created_at",
    "source_types",
    "source_confidences",
)


def is_mapping_frame(frame_index: int, *, map_every: int) -> bool:
    """Return whether a source frame belongs to the frozen mapping cohort."""
    return frame_index == 0 or (frame_index + 1) % map_every == 0


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
        metric_evaluator = ImageMetricEvaluator(device)
        provider = OnlineSPNetProvider(
            config.growth_sources.spnet,
            device=device,
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
    frame: FrameInputs,
    refinement: AppearanceRefinementConfig,
    loss_policy: MappingLossConfig,
    dense_prior: DenseGeometryPrior | None = None,
    *,
    photometric_rgb: torch.Tensor | None = None,
    cached: dict[str, object] | None = None,
) -> torch.Tensor | AppearanceObjective:
    optimized_rgb = (
        output.rgb if photometric_rgb is None else photometric_rgb
    )
    if optimized_rgb.shape != output.rgb.shape:
        raise ValueError(
            "Photometric RGB must match the native render shape"
        )
    target = (
        cached["target_rgb"]
        if cached is not None
        else torch.as_tensor(
            frame.rgb,
            dtype=optimized_rgb.dtype,
            device=optimized_rgb.device,
        )
    )
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
        target_depth=(
            cached.get("target_depth")
            if cached is not None
            else None
        ),
        source_masks=(
            cached.get("source_masks")
            if cached is not None
            else None
        ),
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
    if dense_prior is None:
        raise ValueError(
            "SAGE objective requires an aligned dense prior"
        )
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
    _, dense_terms = dense_geometry_objective(
        output.accumulated_depth,
        output.alpha,
        dense_prior,
        target,
        frame.intrinsics,
        refinement.dense_prior,
        dense_policy,
        static=(
            cached.get("dense_static")
            if cached is not None
            else None
        ),
    )
    dense_normal = dense_terms["normal"]
    terms["dense_normal"] = dense_normal
    total = total + refinement.dense_normal_weight * dense_normal
    diagnostics = {
        "dense_valid_normal_pixels": dense_terms["valid_normal_pixels"],
        "dense_normal_weight_mass": dense_terms["normal_weight_mass"],
    }
    return AppearanceObjective(
        total=total,
        terms=terms,
        diagnostics=diagnostics,
    )


def _dense_prior_record(
    frames: tuple[FrameInputs, ...],
    priors: dict[int, DenseGeometryPrior],
    *,
    provider_identity: dict[str, object],
    alignment_variant: str | None = None,
    max_relative_depth_jump: float | None = None,
) -> dict[str, object]:
    frame_indices = [frame.index for frame in frames]
    if set(priors) != set(frame_indices):
        raise ValueError(
            "Dense prior frames must exactly match appearance mapping frames"
        )
    records = []
    for frame in frames:
        prior = priors[frame.index]
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
    return {
        "provider_identity": deepcopy(provider_identity),
        "frames": records,
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def evaluation_cohorts(
    report: dict[str, object],
    *,
    map_every: int,
    policy: EvaluationDepthPolicy,
) -> dict[str, dict[str, object]]:
    """Split one all-frame render report into disjoint mapping/held-out cohorts."""
    frames = report.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Appearance evaluation report lacks frame records")
    mapping = [
        frame
        for frame in frames
        if is_mapping_frame(int(frame["index"]), map_every=map_every)
    ]
    held_out = [
        frame
        for frame in frames
        if not is_mapping_frame(int(frame["index"]), map_every=map_every)
    ]
    if not mapping or not held_out:
        raise ValueError(
            "Appearance evaluation requires non-empty mapping and held-out cohorts"
        )
    return {
        "all": report,
        "mapping": aggregate_evaluation_frame_reports(
            mapping,
            policy=policy,
        ),
        "held_out": aggregate_evaluation_frame_reports(
            held_out,
            policy=policy,
        ),
    }


def _metric_values(frame: dict[str, object]) -> dict[str, float]:
    image = frame["image"]
    geometry = frame["geometry"]
    return {
        "psnr": float(image["psnr"]),
        "ssim": float(image["ssim"]),
        "lpips": float(image["lpips"]),
        "depth_mae_m": float(geometry["depth"]["total"]["mae_m"]),
    }


def compare_evaluation_cohorts(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Report continuous deltas and per-frame directions without a gate."""
    comparison: dict[str, object] = {}
    for cohort in ("all", "mapping", "held_out"):
        baseline_report = baseline[cohort]
        candidate_report = candidate[cohort]
        baseline_frames = {
            int(frame["index"]): frame
            for frame in baseline_report["frames"]
        }
        candidate_frames = {
            int(frame["index"]): frame
            for frame in candidate_report["frames"]
        }
        if set(baseline_frames) != set(candidate_frames):
            raise ValueError(
                f"Appearance {cohort} evaluation frame sets do not match"
            )
        directions = {
            "psnr": 1,
            "ssim": 1,
            "lpips": -1,
            "depth_mae_m": -1,
        }
        per_metric = {}
        for name, direction in directions.items():
            wins = ties = losses = 0
            for index in baseline_frames:
                before = _metric_values(baseline_frames[index])[name]
                after = _metric_values(candidate_frames[index])[name]
                signed = direction * (after - before)
                if signed > 0:
                    wins += 1
                elif signed < 0:
                    losses += 1
                else:
                    ties += 1
            per_metric[name] = {
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        baseline_aggregate = baseline_report["aggregate"]
        candidate_aggregate = candidate_report["aggregate"]
        comparison[cohort] = {
            "frame_count": len(baseline_frames),
            "aggregate_delta": {
                "psnr": (
                    candidate_aggregate["image"]["psnr"]
                    - baseline_aggregate["image"]["psnr"]
                ),
                "ssim": (
                    candidate_aggregate["image"]["ssim"]
                    - baseline_aggregate["image"]["ssim"]
                ),
                "lpips": (
                    candidate_aggregate["image"]["lpips"]
                    - baseline_aggregate["image"]["lpips"]
                ),
                "depth_mae_m": (
                    candidate_aggregate["depth"]["total"]["mae_m"]
                    - baseline_aggregate["depth"]["total"]["mae_m"]
                ),
                "alpha_mean": (
                    candidate_aggregate["alpha"]["total"]["mean"]
                    - baseline_aggregate["alpha"]["total"]["mean"]
                ),
            },
            "per_frame": per_metric,
        }
    return comparison


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


def appearance_milestone_state_payload(
    model: TrainableGaussians,
    optimizer_state: dict[str, object],
    *,
    config: AppearanceRefinementConfig,
    optimizer_step: int,
    source_checkpoint_sha256: str,
    refinement_config_sha256: str,
    producer_code: dict[str, object],
) -> dict[str, object]:
    """Build a diagnostic milestone explicitly unsuitable for resumption."""
    if (
        type(optimizer_step) is not int
        or optimizer_step < 1
        or not _valid_sha256(source_checkpoint_sha256)
        or not _valid_sha256(refinement_config_sha256)
    ):
        raise ValueError("Appearance milestone identity is invalid")
    return {
        "schema_version": "sage-appearance-milestone-state-v1",
        "status": "incomplete",
        "resumable": False,
        "optimizer_step": optimizer_step,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "refinement_config_sha256": refinement_config_sha256,
        "producer_code": deepcopy(producer_code),
        "model_tensors": {
            name: getattr(model, name).detach().cpu()
            for name in model.PARAMETER_NAMES + (
                "gaussian_ids",
                "created_at",
                "source_types",
                "source_confidences",
            )
        },
        "optimizer_state": deepcopy(optimizer_state),
    }


def _consume_mapping_frames(
    config: SageConfig,
    checkpoint_identity: dict[str, object],
) -> tuple[tuple[object, ...], dict[str, object]]:
    source = frame_source_for_config(
        config.scene,
        frame_limit=ALL_ACCEPTED_FRAME_LIMIT,
        non_formal=True,
    )
    prepared = False
    try:
        validate_dataset_identity(
            checkpoint_identity,
            normalize_dataset_identity(source.start_identity()),
        )
        frames = tuple(
            frame
            for frame in source.frames()
            if is_mapping_frame(
                frame.index,
                map_every=config.mapping.map_every,
            )
        )
        if not frames:
            raise ValueError(
                "Input emitted no mapping frames for appearance refinement"
            )
        receipt = source.prepare_close()
        prepared = True
        source.commit()
        return frames, receipt
    except BaseException as exc:
        if prepared:
            source.rollback_commit(exc)
        else:
            source.abort(exc)
        raise


def _evaluate_all_frames(
    config: SageConfig,
    model: TrainableGaussians,
    checkpoint_identity: dict[str, object],
    *,
    evaluator: ImageMetricEvaluator,
    policy: EvaluationDepthPolicy,
) -> tuple[dict[str, object], dict[str, object]]:
    source = frame_source_for_config(
        config.scene,
        frame_limit=ALL_ACCEPTED_FRAME_LIMIT,
        non_formal=True,
    )
    prepared = False
    try:
        validate_dataset_identity(
            checkpoint_identity,
            normalize_dataset_identity(source.start_identity()),
        )
        with torch.no_grad():
            report = evaluate_frames(
                model,
                source.frames(),
                renderer=render,
                image_metrics=evaluator,
                policy=policy,
                map_every=1,
            )
        receipt = source.prepare_close()
        prepared = True
        source.commit()
        return report, receipt
    except BaseException as exc:
        if prepared:
            source.rollback_commit(exc)
        else:
            source.abort(exc)
        raise


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
    refinement_sha256 = base_config_sha256
    if destination.exists():
        raise ValueError(
            f"Refusing an existing appearance output path: {destination}"
        )
    source_payload = load_checkpoint(source_checkpoint)
    if (
        source_payload["identity_snapshot"]["config_sha256"]
        != base_config_sha256
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
    try:
        checkpoint_identity = source_payload["identity_snapshot"]
        mapping_frames, training_receipt = _consume_mapping_frames(
            config,
            checkpoint_identity,
        )
        provider = OnlineSPNetProvider(
            config.growth_sources.spnet,
            device=device,
        )
        dense_priors = prepare_dense_priors(
            mapping_frames,
            provider,
            refinement.dense_prior,
        )
        dense_prior_provenance = _dense_prior_record(
            mapping_frames,
            dense_priors,
            provider_identity=provider.identity.payload(),
            alignment_variant=refinement.dense_prior.alignment_variant,
            max_relative_depth_jump=(
                refinement.dense_normal_max_relative_depth_jump
            ),
        )
        del provider
        torch.cuda.empty_cache()
        frame_by_index = {
            frame.index: frame
            for frame in mapping_frames
        }
        exposure_nuisance = AppearanceExposureNuisance(
            tuple(frame_by_index),
            refinement,
            device=device,
        )
        model = TrainableGaussians.from_checkpoint(
            source_payload,
            device=device,
        )
        render_static = prepare_render_static(model)
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
        appearance_cache: dict[int, dict[str, object]] = {}
        for frame in mapping_frames:
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
            appearance_cache[frame.index] = {
                "target_rgb": target_rgb,
                "target_depth": torch.as_tensor(
                    frame.mapping.depth_m,
                    dtype=torch.float32,
                    device=device,
                ),
                "source_masks": {
                    "center": (source_types == 0) | (source_types == 3),
                    "fused5": (source_types == 1) | (source_types == 4),
                },
                "dense_static": prepare_dense_geometry_static(
                    dense_priors[frame.index],
                    target_rgb,
                    frame.intrinsics,
                    refinement.dense_prior,
                    dense_policy,
                ),
            }

        def objective(
            item: TrainableGaussians,
            frame_index: int,
            _step: int,
        ) -> torch.Tensor | AppearanceObjective:
            frame = frame_by_index[frame_index]
            output = render(item, frame, static=render_static)
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
                dense_priors.get(frame_index),
                photometric_rgb=photometric_rgb,
                cached=appearance_cache[frame_index],
            )

        result = AppearanceRefiner(refinement).optimize(
            model,
            tuple(frame_by_index),
            objective,
            exposure_nuisance=exposure_nuisance,
        )
        frozen_names = _GEOMETRY_AND_TOPOLOGY_TENSORS + (
            ("opacity_logits",)
            if "opacity_logits" not in refinement.optimized_parameters
            else ()
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
        write_model_ply(staging / "appearance_map.ply", model)
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
            "mapping_frame_indices": list(frame_by_index),
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
            "completion_receipts": {"training_frames": training_receipt},
            "duration_seconds": perf_counter() - started,
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
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
