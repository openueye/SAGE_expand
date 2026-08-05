"""Dataset-agnostic evaluation of a final SAGE checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch

from ..artifacts import load_checkpoint
from ..core_input import CoreObservationAssembler
from ..engine.evaluation import (
    EvaluationDepthPolicy,
    aggregate_evaluation_frame_reports,
    evaluate_frame,
    evaluate_frames,
)
from ..engine.metrics import ImageMetricEvaluator
from ..engine.model import TrainableGaussians
from ..engine.render_export import EvaluationRenderExporter
from ..engine.rendering import render
from ..foundation.artifact_versions import (
    APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
    CHECKPOINT_VERSION,
)
from ..foundation.code_identity import repository_code_identity
from ..foundation.config import SageConfig
from ..foundation.hashing import sha256_file
from ..foundation.identity_schema import validate_dataset_identity
from ..input.prefetch import BoundedResultStream
from ..refinement.appearance_config import (
    AppearanceRefinementConfig,
    refinement_config_identity,
)


_EVALUATION_PROGRESS_EVERY = 50
STAGE1_MAPPING = "stage1_mapping"
STAGE2_REFINEMENT = "stage2_refinement"
DEFAULT_CHECKPOINT_STAGES = (STAGE2_REFINEMENT,)
CHECKPOINT_STAGES = (STAGE1_MAPPING, STAGE2_REFINEMENT)


@dataclass(frozen=True)
class _EvaluableCheckpoint:
    stage: str
    path: Path
    sha256: str
    payload: dict[str, object]
    identity: dict[str, object]


def _checkpoint_stage(payload: dict[str, object]) -> str:
    version = payload.get("checkpoint_version")
    if version == CHECKPOINT_VERSION:
        return STAGE1_MAPPING
    if version == APPEARANCE_REFINEMENT_CHECKPOINT_VERSION:
        return STAGE2_REFINEMENT
    raise ValueError("Unsupported SAGE checkpoint version")


def _validate_checkpoint_policy(stages: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(stages)
    if not selected or len(set(selected)) != len(selected) or any(
        stage not in CHECKPOINT_STAGES for stage in selected
    ):
        raise ValueError(f"evaluation checkpoint stages must be a unique subset of {CHECKPOINT_STAGES}")
    return selected


def _prepare_checkpoint(
    config: SageConfig,
    checkpoint: Path,
    *,
    refinement_config: AppearanceRefinementConfig | None,
    allowed_checkpoint_stages: tuple[str, ...],
) -> _EvaluableCheckpoint:
    source = Path(checkpoint).resolve()
    payload = load_checkpoint(source)
    stage = _checkpoint_stage(payload)
    allowed = _validate_checkpoint_policy(allowed_checkpoint_stages)
    if stage not in allowed:
        raise ValueError(
            f"Evaluation policy accepts {allowed}, but checkpoint is {stage}. "
            "Set evaluation.checkpoint_stages explicitly to evaluate it."
        )
    training_identity = config.training_config_identity()
    identity = payload.get("identity_snapshot")
    if (
        not isinstance(identity, dict)
        or identity.get("training_config_identity") != training_identity
    ):
        raise ValueError("Checkpoint and SAGE configuration do not match")
    if stage == STAGE2_REFINEMENT:
        if refinement_config is None:
            raise ValueError("Evaluating a refined checkpoint requires refinement configuration")
        expected_refinement_sha256 = refinement_config_identity(training_identity, refinement_config)
        refinement = payload.get("appearance_refinement")
        if (
            not isinstance(refinement, dict)
            or refinement.get("refinement_config_sha256") != expected_refinement_sha256
        ):
            raise ValueError("Refined checkpoint and SAGE configuration do not match")
    return _EvaluableCheckpoint(
        stage=stage,
        path=source,
        sha256=sha256_file(source),
        payload=payload,
        identity=identity,
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_evaluation(
    config: SageConfig,
    checkpoint: Path,
    output: Path,
    *,
    device: str,
    refinement_config: AppearanceRefinementConfig | None = None,
    allow_stage1_checkpoint: bool = False,
    allowed_checkpoint_stages: tuple[str, ...] = DEFAULT_CHECKPOINT_STAGES,
) -> Path:
    """Evaluate every accepted frame emitted by either supported input."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("SAGE evaluation requires an available CUDA device")
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError(f"Refusing an existing evaluation path: {destination}")
    permitted = tuple(allowed_checkpoint_stages)
    if allow_stage1_checkpoint and STAGE1_MAPPING not in permitted:
        permitted = (*permitted, STAGE1_MAPPING)
    evaluated = _prepare_checkpoint(
        config,
        checkpoint,
        refinement_config=refinement_config,
        allowed_checkpoint_stages=permitted,
    )
    source_checkpoint = evaluated.path
    checkpoint_payload = evaluated.payload
    identity = evaluated.identity
    is_stage1 = evaluated.stage == STAGE1_MAPPING
    config_sha256 = sha256_file(config.config_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ))
    started = perf_counter()
    try:
        if is_stage1:
            print(
                "SAGE evaluation: STAGE-1 (mapping-only) checkpoint — "
                "results are NOT final-quality",
                flush=True,
            )
        model = TrainableGaussians.from_checkpoint(
            checkpoint_payload,
            device=device,
        )
        evaluator = ImageMetricEvaluator(device, model_root=config.model_root)
        policy = EvaluationDepthPolicy(
            depth_policy=config.mapping.evaluation_depth_policy,
            min_alpha=config.mapping.evaluation_min_alpha,
            epsilon=config.mapping.evaluation_epsilon,
            alpha_support_a0=config.mapping.evaluation_alpha_support_a0,
            hit_target_center=config.mapping.evaluation_hit_target_center,
            hit_target_fused=config.mapping.evaluation_hit_target_fused,
        )
        resolved = config.input.create_adapter().preflight()
        assembler = CoreObservationAssembler(resolved.contract.canonical.sources)
        stream = BoundedResultStream(
            lambda: assembler.frames(resolved.frames()),
            identity={"adapter": resolved.contract.adapter_type},
            queue_capacity=config.input.prefetch_depth,
        )
        evaluation_started = perf_counter()

        def report_evaluation_progress(
            completed: int,
            _frame: object,
        ) -> None:
            if completed % _EVALUATION_PROGRESS_EVERY == 0:
                print(
                    f"SAGE evaluation: {completed} frames rendered | "
                    f"elapsed {_format_duration(perf_counter() - evaluation_started)}",
                    flush=True,
                )

        render_exporter = EvaluationRenderExporter(
            staging, keyframe_every=config.mapping.keyframe_every,
        )
        try:
            print("SAGE evaluation: rendering accepted frames", flush=True)
            with torch.no_grad():
                result = evaluate_frames(
                    model,
                    stream.frames(),
                    renderer=render,
                    image_metrics=evaluator,
                    policy=policy,
                    map_every=1,
                    render_export=render_exporter,
                    progress_callback=report_evaluation_progress,
                )
            stream.close()
            # ``ResolvedInput.frames()`` settles the canonical sequence
            # identity at EOF.  Validate only after this evaluation stream has
            # been consumed: asking for ``resolved.identities`` above would
            # trigger a second full ROS bag replay solely to calculate it.
            validate_dataset_identity(identity, resolved.identities)
        except BaseException:
            stream.abort("SAGE evaluation failed")
            raise
        print(
            f"SAGE evaluation: complete ({result['frame_count']} frames) | "
            f"elapsed {_format_duration(perf_counter() - evaluation_started)}",
            flush=True,
        )
        model.to("cpu")
        report = {
            "producer_code": repository_code_identity(),
            "checkpoint": {
                "path": str(source_checkpoint),
                "sha256": evaluated.sha256,
                "stage": evaluated.stage,
            },
            "config": {
                "path": str(config.config_path),
                "sha256": config_sha256,
            },
            "input": {
                "adapter_type": resolved.contract.adapter_type,
                "identities": resolved.identities.payload(),
                "canonical": resolved.contract.canonical.payload(),
            },
            "duration_seconds": perf_counter() - started,
            "result": result,
        }
        report_path = staging / "evaluation.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "producer_code": report["producer_code"],
            "checkpoint": report["checkpoint"],
            "config": report["config"],
            "input": report["input"],
            "artifacts": {
                "report": report_path.name,
                "report_sha256": sha256_file(report_path),
            },
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if is_stage1:
            (staging / "STAGE1_CHECKPOINT_ONLY").write_text(
                "This evaluation used a stage-1 mapping checkpoint, not the "
                "final refined checkpoint. Results are not final-quality.\n",
                encoding="utf-8",
            )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def run_evaluations(
    config: SageConfig,
    checkpoints: dict[str, Path],
    outputs: dict[str, Path],
    *,
    device: str,
    refinement_config: AppearanceRefinementConfig | None = None,
    allowed_checkpoint_stages: tuple[str, ...] = DEFAULT_CHECKPOINT_STAGES,
) -> dict[str, Path]:
    """Evaluate selected checkpoints over one shared canonical frame stream.

    Each model renders sequentially for the current frame.  Thus enabling
    Stage 1 and Stage 2 comparison doubles rendering work but never rereads or
    reassembles the ROSBAG, and each report is still published separately.
    """
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("SAGE evaluation requires an available CUDA device")
    allowed = _validate_checkpoint_policy(allowed_checkpoint_stages)
    if set(checkpoints) != set(outputs) or not checkpoints:
        raise ValueError("Evaluation checkpoints and outputs must have the same non-empty stages")
    prepared = {
        requested_stage: _prepare_checkpoint(
            config,
            checkpoint,
            refinement_config=refinement_config,
            allowed_checkpoint_stages=allowed,
        )
        for requested_stage, checkpoint in checkpoints.items()
    }
    if any(context.stage != requested_stage for requested_stage, context in prepared.items()):
        raise ValueError("Evaluation checkpoint key does not match its checkpoint stage")
    destinations = {stage: Path(path).resolve() for stage, path in outputs.items()}
    if any(path.exists() for path in destinations.values()):
        existing = next(path for path in destinations.values() if path.exists())
        raise ValueError(f"Refusing an existing evaluation path: {existing}")
    for destination in destinations.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
    staging = {
        stage: Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        for stage, destination in destinations.items()
    }
    models: dict[str, TrainableGaussians] = {}
    started = perf_counter()
    try:
        for stage, context in prepared.items():
            if stage == STAGE1_MAPPING:
                print(
                    "SAGE evaluation: STAGE-1 (mapping-only) checkpoint — "
                    "results are NOT final-quality",
                    flush=True,
                )
            models[stage] = TrainableGaussians.from_checkpoint(context.payload, device=device)
        evaluator = ImageMetricEvaluator(device, model_root=config.model_root)
        policy = EvaluationDepthPolicy(
            depth_policy=config.mapping.evaluation_depth_policy,
            min_alpha=config.mapping.evaluation_min_alpha,
            epsilon=config.mapping.evaluation_epsilon,
            alpha_support_a0=config.mapping.evaluation_alpha_support_a0,
            hit_target_center=config.mapping.evaluation_hit_target_center,
            hit_target_fused=config.mapping.evaluation_hit_target_fused,
        )
        resolved = config.input.create_adapter().preflight()
        assembler = CoreObservationAssembler(resolved.contract.canonical.sources)
        stream = BoundedResultStream(
            lambda: assembler.frames(resolved.frames()),
            identity={"adapter": resolved.contract.adapter_type},
            queue_capacity=config.input.prefetch_depth,
        )
        frame_reports: dict[str, list[dict[str, object]]] = {
            stage: [] for stage in prepared
        }
        render_exporters = {
            stage: EvaluationRenderExporter(
                staging[stage], keyframe_every=config.mapping.keyframe_every,
            )
            for stage in prepared
        }
        evaluation_started = perf_counter()
        try:
            print("SAGE evaluation: rendering accepted frames", flush=True)
            with torch.no_grad():
                for completed, frame in enumerate(stream.frames(), start=1):
                    for stage, context in prepared.items():
                        frame_reports[stage].append(evaluate_frame(
                            models[stage],
                            frame,
                            renderer=render,
                            image_metrics=evaluator,
                            policy=policy,
                            render_export=render_exporters[stage],
                        ))
                    if completed % _EVALUATION_PROGRESS_EVERY == 0:
                        print(
                            f"SAGE evaluation: {completed} frames rendered for "
                            f"{len(prepared)} checkpoint(s) | elapsed "
                            f"{_format_duration(perf_counter() - evaluation_started)}",
                            flush=True,
                        )
            stream.close()
            for context in prepared.values():
                validate_dataset_identity(context.identity, resolved.identities)
        except BaseException:
            stream.abort("SAGE evaluation failed")
            raise
        config_sha256 = sha256_file(config.config_path)
        input_payload = {
            "adapter_type": resolved.contract.adapter_type,
            "identities": resolved.identities.payload(),
            "canonical": resolved.contract.canonical.payload(),
        }
        producer_code = repository_code_identity()
        for stage, context in prepared.items():
            result = aggregate_evaluation_frame_reports(frame_reports[stage], policy=policy)
            report = {
                "producer_code": producer_code,
                "checkpoint": {
                    "path": str(context.path),
                    "sha256": context.sha256,
                    "stage": context.stage,
                },
                "config": {"path": str(config.config_path), "sha256": config_sha256},
                "input": input_payload,
                "duration_seconds": perf_counter() - started,
                "result": result,
            }
            report_path = staging[stage] / "evaluation.json"
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            manifest = {
                "producer_code": producer_code,
                "checkpoint": report["checkpoint"],
                "config": report["config"],
                "input": input_payload,
                "artifacts": {"report": report_path.name, "report_sha256": sha256_file(report_path)},
            }
            (staging[stage] / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            if context.stage == STAGE1_MAPPING:
                (staging[stage] / "STAGE1_CHECKPOINT_ONLY").write_text(
                    "This evaluation used a stage-1 mapping checkpoint, not the "
                    "refined checkpoint. Results are not final-quality.\n",
                    encoding="utf-8",
                )
        for model in models.values():
            model.to("cpu")
        for stage, destination in destinations.items():
            staging[stage].rename(destination)
        return destinations
    except BaseException:
        for path in staging.values():
            shutil.rmtree(path, ignore_errors=True)
        raise


__all__ = [
    "CHECKPOINT_STAGES",
    "DEFAULT_CHECKPOINT_STAGES",
    "STAGE1_MAPPING",
    "STAGE2_REFINEMENT",
    "run_evaluation",
    "run_evaluations",
]
