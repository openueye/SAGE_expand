"""Orchestration for complete SAGE training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from .artifacts import load_checkpoint
from .evaluation.evaluation_run import (
    STAGE1_MAPPING,
    STAGE2_REFINEMENT,
    run_evaluations,
)
from .execution import (
    EXECUTION_RECEIPT_SCHEMA_VERSION,
    publish_json_atomic,
    validate_execution_receipt,
)
from .foundation.artifact_versions import APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
from .foundation.config import SageConfig
from .foundation.hashing import sha256_file
from .foundation.identity_schema import validate_dataset_identity
from .mapping.mapping_worker import run_formal_training, run_training_preflight
from .method_config import SageMethodConfig
from .refinement.appearance_config import (
    AppearanceRefinementConfig,
    refinement_config_identity,
)
from .refinement.refinement_run import (
    preflight_appearance_runtime,
    run_appearance_refinement,
)
from .refinement.stage2_cache import (
    Stage2InputCache,
    default_stage2_cache_path,
    remove_stage2_cache,
)


def _current_input_identity(config: SageConfig):
    """Obtain identity without replaying a completed Stage 1 input stream."""
    cache_path = default_stage2_cache_path(config.output_dir)
    if cache_path.is_dir():
        return Stage2InputCache(cache_path).input_identities
    resolved = config.input.create_adapter().preflight()
    if resolved.contract.canonical.frame_count is None:
        raise ValueError(
            "online-window-v2 resume requires its Stage 1 transient input cache"
        )
    return resolved.identities


def _remove_orphaned_staging(output: Path) -> None:
    """Remove incomplete atomic-publication directories from a prior run."""
    if not output.is_dir():
        return
    prefixes = (
        ".structure.staging-",
        ".final.staging-",
        ".evaluation.staging-",
    )
    for candidate in output.iterdir():
        if not candidate.name.startswith(prefixes):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(
                f"Invalid SAGE staging artifact: {candidate}"
            )
        shutil.rmtree(candidate)


def _remove_failed_structure_receipt(destination: Path) -> None:
    """Allow a clean retry after a recorded failed structure attempt."""
    receipt_path = destination / "structure.execution.json"
    if not receipt_path.exists():
        return
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid partial structure receipt: {receipt_path}"
        ) from exc
    artifacts = receipt.get("artifacts") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version")
        != EXECUTION_RECEIPT_SCHEMA_VERSION
        or type(receipt.get("exit_code")) is not int
        or receipt["exit_code"] == 0
        or not isinstance(artifacts, dict)
        or any(
            isinstance(value, dict) and value.get("exists") is True
            for value in artifacts.values()
        )
    ):
        raise ValueError(
            f"Structure receipt is not a clean failed attempt: {receipt_path}"
        )
    receipt_path.unlink()


def _structure_output_is_resumable(
    output: Path,
    *,
    config: SageConfig,
) -> bool:
    checkpoint = output / "checkpoint.pt"
    manifest_path = output / "run_manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("config") != config.manifest_dict()
        ):
            return False
        validate_dataset_identity(
            manifest.get("identity_snapshot"),
            _current_input_identity(config),
        )
        validate_execution_receipt(
            output.with_name(f"{output.name}.execution.json"),
            manifest_path=manifest_path,
            config_path=config.config_path,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def _final_output_is_resumable(
    output: Path,
    *,
    config: SageConfig,
    refinement_config: AppearanceRefinementConfig,
    source_checkpoint: Path,
) -> bool:
    checkpoint = output / "appearance_checkpoint.pt"
    manifest_path = output / "run_manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        return False
    try:
        payload = load_checkpoint(checkpoint)
        refinement = payload.get("appearance_refinement")
        training_identity = config.training_config_identity()
        expected_refinement_sha256 = refinement_config_identity(
            training_identity, refinement_config
        )
        if (
            payload.get("checkpoint_version")
            != APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
            or not isinstance(refinement, dict)
            or refinement.get("refinement_config_sha256")
            != expected_refinement_sha256
            or refinement.get("source_checkpoint_sha256")
            != sha256_file(source_checkpoint)
        ):
            return False
        validate_dataset_identity(
            payload.get("identity_snapshot"),
            _current_input_identity(config),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            isinstance(manifest, dict)
            and manifest.get("artifacts", {}).get("checkpoint_sha256")
            == sha256_file(checkpoint)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _evaluation_output_is_resumable(
    output: Path,
    *,
    config: SageConfig,
    checkpoint: Path,
) -> bool:
    manifest_path = output / "run_manifest.json"
    report_path = output / "evaluation.json"
    if not manifest_path.is_file() or not report_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        current_identity = _current_input_identity(config)
        validate_dataset_identity(
            load_checkpoint(checkpoint).get("identity_snapshot"),
            current_identity,
        )
        identities = current_identity.payload()
        return (
            isinstance(manifest, dict)
            and isinstance(report, dict)
            and manifest.get("checkpoint", {}).get("sha256")
            == sha256_file(checkpoint)
            and manifest.get("config", {}).get("sha256")
            == sha256_file(config.config_path)
            and manifest.get("input", {}).get("identities") == identities
            and report.get("input", {}).get("identities") == identities
            and manifest.get("artifacts", {}).get("report_sha256")
            == sha256_file(report_path)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _run_structure_optimization(
    method: SageMethodConfig,
    output: Path,
    *,
    device: str,
) -> Path:
    status = run_formal_training(
        config_path=method.path,
        output=output,
        device=device,
    )
    if status:
        raise RuntimeError(f"SAGE structure optimization failed: {status}")
    checkpoint = output.resolve() / "checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(
            f"SAGE structure optimization did not publish {checkpoint}"
        )
    return output.resolve()


def preflight_training(
    *,
    method: SageMethodConfig,
    output: Path,
    device: str,
) -> int:
    """Validate all runtime phases and the concrete input without training."""
    destination = Path(output).resolve()
    structure_output = destination / "structure"
    config = method.resolve(structure_output)
    preflight_appearance_runtime(
        config,
        method.refinement_config(),
        device=device,
    )
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"SAGE output is not a directory: {destination}")
        if _structure_output_is_resumable(
            structure_output,
            config=config,
        ) and not (
            (destination / "run_manifest.json").exists()
            or (destination / "final").exists()
            or (destination / "evaluation").exists()
        ):
            print(
                f"SAGE preflight: validated structure checkpoint at "
                f"{structure_output}",
                flush=True,
            )
            return 0
        if any(destination.iterdir()):
            raise ValueError(
                "Existing SAGE output is not empty or a validated "
                f"recoverable structure checkpoint: {destination}"
            )
    return run_training_preflight(
        config_path=method.path,
        output=structure_output,
        device=device,
    )


def run_training(
    *,
    method: SageMethodConfig,
    output: Path,
    device: str,
) -> Path:
    """Run or resume structure, appearance, and final evaluation."""
    destination = Path(output).resolve()
    structure_output = destination / "structure"
    final_output = destination / "final"
    evaluation_output = destination / "evaluation"
    config = method.resolve(structure_output)
    config_sha256 = sha256_file(method.path)

    if (destination / "run_manifest.json").exists():
        raise ValueError(f"SAGE output is already complete: {destination}")
    allowed = {
        "structure",
        "structure.execution.json",
        "final",
        "evaluation",
        ".stage2-input-cache",
    }
    if destination.exists():
        _remove_orphaned_staging(destination)
        unknown = {path.name for path in destination.iterdir()} - allowed
        if unknown:
            raise ValueError(
                "SAGE output contains unknown partial artifacts: "
                + ", ".join(sorted(unknown))
            )
    else:
        destination.mkdir(parents=True)

    structure_ready = _structure_output_is_resumable(
        structure_output,
        config=config,
    )
    if structure_output.exists() and not structure_ready:
        raise ValueError(
            f"Invalid partial structure output: {structure_output}"
        )
    if not structure_ready:
        print("SAGE stage 1/3: structure mapping", flush=True)
        _remove_failed_structure_receipt(destination)
        _run_structure_optimization(
            method,
            structure_output,
            device=device,
        )
    if sha256_file(method.path) != config_sha256:
        raise RuntimeError(
            f"SAGE configuration changed during training: {method.path}"
        )

    source_checkpoint = structure_output / "checkpoint.pt"
    final_checkpoint = final_output / "appearance_checkpoint.pt"
    final_ready = _final_output_is_resumable(
        final_output,
        config=config,
        refinement_config=method.refinement_config(),
        source_checkpoint=source_checkpoint,
    )
    if final_output.exists() and not final_ready:
        raise ValueError(f"Invalid partial final output: {final_output}")
    if not final_ready:
        print("SAGE stage 2/3: appearance refinement", flush=True)
        run_appearance_refinement(
            config,
            source_checkpoint,
            method.refinement_config(),
            final_output,
            device=device,
        )
    if not final_checkpoint.is_file():
        raise RuntimeError(
            f"SAGE appearance phase did not publish {final_checkpoint}"
        )

    evaluation_stages = method.evaluation_checkpoint_stages()
    evaluation_checkpoints = {
        STAGE1_MAPPING: source_checkpoint,
        STAGE2_REFINEMENT: final_checkpoint,
    }
    evaluation_outputs = {
        stage: (
            evaluation_output
            if evaluation_stages == (STAGE2_REFINEMENT,)
            else evaluation_output / stage
        )
        for stage in evaluation_stages
    }
    evaluation_ready = {
        stage: _evaluation_output_is_resumable(
            path,
            config=config,
            checkpoint=evaluation_checkpoints[stage],
        )
        for stage, path in evaluation_outputs.items()
    }
    if any(path.exists() for path in evaluation_outputs.values()) and not all(evaluation_ready.values()):
        raise ValueError("Evaluation outputs are partially complete or invalid")
    if not all(evaluation_ready.values()):
        print("SAGE stage 3/3: streaming evaluation", flush=True)
        run_evaluations(
            config,
            {stage: evaluation_checkpoints[stage] for stage in evaluation_stages},
            evaluation_outputs,
            device=device,
            refinement_config=method.refinement_config(),
            allowed_checkpoint_stages=evaluation_stages,
        )
    # This cache is only the Stage 1 → Stage 2 retry handoff.  Keep it through
    # a Stage 3 failure, then remove it after the complete pipeline succeeds.
    remove_stage2_cache(default_stage2_cache_path(structure_output))
    if sha256_file(method.path) != config_sha256:
        raise RuntimeError(
            f"SAGE configuration changed during the run: {method.path}"
        )
    manifest = {
        "config": {
            "path": str(method.path),
            "sha256": config_sha256,
        },
        # Recorded, not recomputed: the manifest must state what this run
        # actually consumed, and re-resolving the input would decode it again.
        "input": json.loads(
            (structure_output / "run_manifest.json").read_text(encoding="utf-8")
        )["input"],
        "artifacts": {
            "structure_checkpoint": str(
                source_checkpoint.relative_to(destination)
            ),
            "structure_checkpoint_sha256": sha256_file(source_checkpoint),
            "final_checkpoint": str(
                final_checkpoint.relative_to(destination)
            ),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "evaluation": (
                str(evaluation_output.relative_to(destination))
                if evaluation_stages == (STAGE2_REFINEMENT,)
                else {
                    stage: str(path.relative_to(destination))
                    for stage, path in evaluation_outputs.items()
                }
            ),
        },
    }
    publish_json_atomic(destination / "run_manifest.json", manifest)
    return destination


__all__ = ["preflight_training", "run_training"]
