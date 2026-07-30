"""Dataset-agnostic evaluation of a final SAGE checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import torch

from .artifacts import load_checkpoint
from .artifact_identity import (
    normalize_dataset_identity,
    validate_dataset_identity,
)
from .artifact_versions import APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
from .code_identity import repository_code_identity
from .config import ALL_ACCEPTED_FRAME_LIMIT, SageConfig
from .evaluation import EvaluationDepthPolicy, evaluate_frames
from .frame_source import frame_source_for_config
from .hashing import sha256_file
from .metrics import ImageMetricEvaluator
from .model import TrainableGaussians
from .rendering import render


_EVALUATION_PROGRESS_EVERY = 50


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
) -> Path:
    """Evaluate every accepted frame emitted by either supported input."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("SAGE evaluation requires an available CUDA device")
    destination = Path(output).resolve()
    source_checkpoint = Path(checkpoint).resolve()
    if destination.exists():
        raise ValueError(f"Refusing an existing evaluation path: {destination}")

    checkpoint_payload = load_checkpoint(source_checkpoint)
    if (
        checkpoint_payload["checkpoint_version"]
        != APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
    ):
        raise ValueError("SAGE evaluation requires the final checkpoint")
    config_sha256 = sha256_file(config.config_path)
    identity = checkpoint_payload.get("identity_snapshot")
    refinement = checkpoint_payload.get("appearance_refinement")
    if (
        not isinstance(identity, dict)
        or identity.get("config_sha256") != config_sha256
        or not isinstance(refinement, dict)
        or refinement.get("refinement_config_sha256") != config_sha256
    ):
        raise ValueError(
            "Final checkpoint and SAGE configuration do not match"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ))
    source = None
    prepared = False
    started = perf_counter()
    try:
        model = TrainableGaussians.from_checkpoint(
            checkpoint_payload,
            device=device,
        )
        evaluator = ImageMetricEvaluator(device)
        policy = EvaluationDepthPolicy(
            depth_policy=config.mapping.evaluation_depth_policy,
            min_alpha=config.mapping.evaluation_min_alpha,
            epsilon=config.mapping.evaluation_epsilon,
            alpha_support_a0=config.mapping.evaluation_alpha_support_a0,
            hit_target_center=config.mapping.evaluation_hit_target_center,
            hit_target_fused5=config.mapping.evaluation_hit_target_fused5,
        )
        source = frame_source_for_config(
            config.scene,
            frame_limit=ALL_ACCEPTED_FRAME_LIMIT,
            non_formal=True,
        )
        dataset_identity = normalize_dataset_identity(
            source.start_identity()
        )
        validate_dataset_identity(identity, dataset_identity)
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

        print("SAGE evaluation: rendering accepted frames", flush=True)
        with torch.no_grad():
            result = evaluate_frames(
                model,
                source.frames(),
                renderer=render,
                image_metrics=evaluator,
                policy=policy,
                map_every=1,
                progress_callback=report_evaluation_progress,
            )
        print(
            f"SAGE evaluation: complete ({result['frame_count']} frames) | "
            f"elapsed {_format_duration(perf_counter() - evaluation_started)}",
            flush=True,
        )
        receipt = source.prepare_close()
        prepared = True
        source.commit()
        model.to("cpu")
        report = {
            "producer_code": repository_code_identity(),
            "checkpoint": {
                "path": str(source_checkpoint),
                "sha256": sha256_file(source_checkpoint),
            },
            "config": {
                "path": str(config.config_path),
                "sha256": config_sha256,
            },
            "completion_receipt": receipt,
            "dataset_identity": dataset_identity,
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
            "completion_receipt": receipt,
            "dataset_identity": dataset_identity,
            "artifacts": {
                "report": report_path.name,
                "report_sha256": sha256_file(report_path),
            },
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except BaseException as exc:
        if source is not None:
            if prepared:
                source.rollback_commit(exc)
            else:
                source.abort(exc)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


__all__ = ["run_evaluation"]
