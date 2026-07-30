"""Orchestration for complete SAGE training and evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

from .artifact_identity import (
    normalize_dataset_identity,
    validate_dataset_identity,
)
from .artifact_versions import APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
from .artifacts import load_checkpoint
from .config import ALL_ACCEPTED_FRAME_LIMIT, SageConfig
from .evaluation_run import run_evaluation
from .execution import (
    EXECUTION_RECEIPT_SCHEMA_VERSION,
    publish_json_atomic,
    validate_execution_receipt,
)
from .frame_source import frame_source_for_config
from .hashing import sha256_file
from .mapping_worker import run_formal_training, run_training_preflight
from .method_config import SageInput, SageMethodConfig
from .refinement_run import (
    preflight_appearance_runtime,
    run_appearance_refinement,
)


def _current_dataset_identity(config: SageConfig) -> dict[str, object]:
    source = frame_source_for_config(
        config.scene,
        frame_limit=ALL_ACCEPTED_FRAME_LIMIT,
        non_formal=True,
    )
    try:
        return normalize_dataset_identity(source.start_identity())
    finally:
        source.abort("dataset identity check complete")


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
            _current_dataset_identity(config),
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
    source_checkpoint: Path,
) -> bool:
    checkpoint = output / "appearance_checkpoint.pt"
    manifest_path = output / "run_manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        return False
    try:
        payload = load_checkpoint(checkpoint)
        refinement = payload.get("appearance_refinement")
        config_sha256 = sha256_file(config.config_path)
        if (
            payload.get("checkpoint_version")
            != APPEARANCE_REFINEMENT_CHECKPOINT_VERSION
            or not isinstance(refinement, dict)
            or refinement.get("refinement_config_sha256") != config_sha256
            or refinement.get("source_checkpoint_sha256")
            != sha256_file(source_checkpoint)
        ):
            return False
        validate_dataset_identity(
            payload.get("identity_snapshot"),
            _current_dataset_identity(config),
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
        current_identity = _current_dataset_identity(config)
        validate_dataset_identity(
            load_checkpoint(checkpoint).get("identity_snapshot"),
            current_identity,
        )
        return (
            isinstance(manifest, dict)
            and isinstance(report, dict)
            and manifest.get("checkpoint", {}).get("sha256")
            == sha256_file(checkpoint)
            and manifest.get("config", {}).get("sha256")
            == sha256_file(config.config_path)
            and manifest.get("dataset_identity") == current_identity
            and report.get("dataset_identity") == current_identity
            and manifest.get("artifacts", {}).get("report_sha256")
            == sha256_file(report_path)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _run_structure_optimization(
    method: SageMethodConfig,
    source: SageInput,
    output: Path,
    *,
    device: str,
) -> Path:
    status = run_formal_training(
        config_path=method.path,
        output=output,
        input_format=source.internal_format,
        data_root=source.root,
        calibration=source.calibration,
        write_through=source.write_through,
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


def _runtime_config(config: SageConfig) -> SageConfig:
    """Disable mapping-only write-through for replayed downstream phases."""
    return replace(
        config,
        scene=replace(config.scene, write_through_dir=None),
    )


def preflight_training(
    *,
    method: SageMethodConfig,
    source: SageInput,
    output: Path,
    device: str,
) -> int:
    """Validate all runtime phases and the concrete input without training."""
    destination = Path(output).resolve()
    structure_output = destination / "structure"
    config = method.resolve(source, structure_output)
    preflight_appearance_runtime(
        _runtime_config(config),
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
        input_format=source.internal_format,
        data_root=source.root,
        calibration=source.calibration,
        write_through=source.write_through,
        device=device,
    )


def run_training(
    *,
    method: SageMethodConfig,
    source: SageInput,
    output: Path,
    device: str,
) -> Path:
    """Run or resume structure, appearance, and final evaluation."""
    destination = Path(output).resolve()
    structure_output = destination / "structure"
    final_output = destination / "final"
    evaluation_output = destination / "evaluation"
    config = method.resolve(source, structure_output)
    runtime = _runtime_config(config)
    config_sha256 = sha256_file(method.path)

    if (destination / "run_manifest.json").exists():
        raise ValueError(f"SAGE output is already complete: {destination}")
    allowed = {
        "structure",
        "structure.execution.json",
        "final",
        "evaluation",
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
            source,
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
        config=runtime,
        source_checkpoint=source_checkpoint,
    )
    if final_output.exists() and not final_ready:
        raise ValueError(f"Invalid partial final output: {final_output}")
    if not final_ready:
        print("SAGE stage 2/3: appearance refinement", flush=True)
        run_appearance_refinement(
            runtime,
            source_checkpoint,
            method.refinement_config(),
            final_output,
            device=device,
        )
    if not final_checkpoint.is_file():
        raise RuntimeError(
            f"SAGE appearance phase did not publish {final_checkpoint}"
        )

    evaluation_ready = _evaluation_output_is_resumable(
        evaluation_output,
        config=runtime,
        checkpoint=final_checkpoint,
    )
    if evaluation_output.exists() and not evaluation_ready:
        raise ValueError(
            f"Invalid partial evaluation output: {evaluation_output}"
        )
    if not evaluation_ready:
        print("SAGE stage 3/3: final evaluation", flush=True)
        run_evaluation(
            runtime,
            final_checkpoint,
            evaluation_output,
            device=device,
        )
    if sha256_file(method.path) != config_sha256:
        raise RuntimeError(
            f"SAGE configuration changed during the run: {method.path}"
        )
    manifest = {
        "config": {
            "path": str(method.path),
            "sha256": config_sha256,
        },
        "input": {
            "kind": source.kind,
            "root": str(source.root),
        },
        "artifacts": {
            "structure_checkpoint": str(
                source_checkpoint.relative_to(destination)
            ),
            "structure_checkpoint_sha256": sha256_file(source_checkpoint),
            "final_checkpoint": str(
                final_checkpoint.relative_to(destination)
            ),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "evaluation": str(evaluation_output.relative_to(destination)),
        },
    }
    publish_json_atomic(destination / "run_manifest.json", manifest)
    return destination


__all__ = ["preflight_training", "run_training"]
