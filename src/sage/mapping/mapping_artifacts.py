from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

from ..data.providers.spnet_cache import write_dense_spnet_cache
from ..engine.ply_export import write_gaussian_ply
from ..foundation.artifact_versions import CHECKPOINT_VERSION
from ..foundation.config import SageConfig
from ..foundation.hashing import sha256_file
from ..input.adapter import ResolvedInput
from ..foundation.source_policy import SOURCE_POLICY_VERSION, source_counts
from .mapper import MappingCommit, MappingRun
from .run_identity import RunInputIdentity


RUN_MANIFEST_SCHEMA_VERSION = "sage-run-manifest-v1"

_DIAGNOSTIC_FIELDS = {
    "loss": {"photo", "geo_center", "geo_fused", "hit_center", "hit_fused", "depth_coverage", "total"},
    "depth": {"valid_pixels", "valid_center_pixels", "valid_fused_pixels", "center_mae", "fused_mae"},
    "alpha": {"center_mean", "center_p10", "center_p50", "fused_mean", "fused_p10", "fused_p50", "below_a0_fraction"},
}


def _validate_commit_diagnostics(commit: dict[str, object]) -> None:
    if "diagnostics" not in commit:
        return
    diagnostics = commit["diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(_DIAGNOSTIC_FIELDS):
        raise ValueError("SAGE run manifest commit diagnostics shape is invalid")
    for group, fields in _DIAGNOSTIC_FIELDS.items():
        values = diagnostics[group]
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError("SAGE run manifest commit diagnostics shape is invalid")
        if any(
            type(value) not in {int, float} or not np.isfinite(value)
            for value in values.values()
        ):
            raise ValueError("SAGE run manifest commit diagnostics must be finite scalars")
@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    checkpoint: Path
    manifest: Path
    ply: Path


def _checkpoint_payload(
    run: MappingRun,
    identity: RunInputIdentity,
) -> dict[str, object]:
    model = run.model
    tensors = {
        name: getattr(model, name).detach().cpu()
        for name in (
            "means3d", "colors", "opacity_logits", "log_scales", "rotations",
            "gaussian_ids", "created_at", "source_types", "source_confidences",
        )
    }
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "source_policy_version": identity.source_policy_version,
        **tensors,
        "last_commit": asdict(run.commits[-1]) if run.commits else None,
        "identity_snapshot": identity.identity_snapshot(),
        "optimization": {
            "variant": run.optimization_variant,
            "optimizer_lifecycle": run.optimizer_lifecycle,
            "optimizer_final_step": run.optimizer_final_step,
            "append_state_migrations": run.optimizer_append_migrations,
            "prune_state_migrations": run.optimizer_prune_migrations,
        },
    }


def _validate_run(run: MappingRun) -> None:
    if len(run.dense_spnet_frames) not in {0, run.spnet_actual_invocations}:
        raise ValueError("Dense SPNet cache count must match mapping invocations")
    previous = None
    for commit in run.commits:
        _validate_commit_diagnostics(asdict(commit))
        if sum(commit.added_by_source.values()) != commit.added or sum(commit.pruned_by_source.values()) != commit.pruned:
            raise ValueError("Run commit source totals do not match scalar totals")
        if sum(commit.remaining_by_source.values()) != commit.gaussian_count:
            raise ValueError("Run commit source totals do not match Gaussian count")
        if previous is not None and previous + commit.added - commit.pruned != commit.gaussian_count:
            raise ValueError("Run commit Gaussian transition is inconsistent")
        previous = commit.gaussian_count


def write_run_artifacts(
    config: SageConfig,
    run: MappingRun,
    *,
    input_identity: RunInputIdentity,
    resolved: ResolvedInput,
    runtime_metrics: dict[str, object] | None = None,
) -> RunArtifacts:
    """Publish the mapping run atomically, with its input evidence beside it."""
    if run.model.means3d.device.type == "cpu":
        raise ValueError("Published mapping artifacts require CUDA execution")
    declared = resolved.contract.canonical.frame_count
    if run.processed_frames != declared:
        raise ValueError(
            f"Mapping consumed {run.processed_frames} frames but the resolved input froze {declared}"
        )
    output = config.output_dir
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _validate_run(run)
        checkpoint = staging / "checkpoint.pt"
        ply = staging / "map.ply"
        manifest = staging / "run_manifest.json"
        dense_spnet_cache = staging / "spnet_dense.pt"
        torch.save(_checkpoint_payload(run, input_identity), checkpoint)
        if run.dense_spnet_frames:
            write_dense_spnet_cache(
                dense_spnet_cache,
                run.dense_spnet_frames,
                run.spnet_identity,
            )
        write_gaussian_ply(ply, run.model)
        (staging / "input_contract.json").write_text(
            json.dumps(resolved.contract.payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "input_preflight.json").write_text(
            json.dumps(resolved.report.payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_payload = {
            "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint)},
            "map_ply": {"path": ply.name, "sha256": sha256_file(ply)},
            "input_contract": {"path": "input_contract.json"},
            "input_preflight": {"path": "input_preflight.json"},
        }
        if dense_spnet_cache.is_file():
            artifact_payload["dense_spnet_cache"] = {
                "path": dense_spnet_cache.name,
                "sha256": sha256_file(dense_spnet_cache),
                "frame_count": len(run.dense_spnet_frames),
            }
        payload = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "producer_code": input_identity.producer_code_payload(),
            "source_policy_version": input_identity.source_policy_version,
            "config": config.manifest_dict(),
            "config_sha256": input_identity.config_sha256,
            "identity_snapshot": input_identity.identity_snapshot(),
            "dependencies": input_identity.dependencies.payload(),
            "input": {
                "adapter_type": resolved.contract.adapter_type,
                "identities": resolved.identities.payload(),
                "canonical": resolved.contract.canonical.payload(),
                "accepted_frames": resolved.report.accepted_frames,
                "rejected_frames": len(resolved.report.rejected_frames),
            },
            "commits": [asdict(commit) for commit in run.commits],
            "final_gaussian_count": run.model.count,
            "final_by_source": source_counts(run.model.source_types),
            "optimization": {
                "variant": run.optimization_variant,
                "optimizer_lifecycle": run.optimizer_lifecycle,
                "optimizer_final_step": run.optimizer_final_step,
                "append_state_migrations": run.optimizer_append_migrations,
                "prune_state_migrations": run.optimizer_prune_migrations,
            },
            "spnet_execution": {
                "mode": run.spnet_identity.mode,
                "expected_invocations": run.spnet_expected_invocations,
                "actual_invocations": run.spnet_actual_invocations,
                "anchor_source_types": list(run.spnet_anchor_source_types),
                "adapter": _execution_adapter_payload(run.spnet_identity),
                "inference_seconds": list(run.spnet_inference_seconds),
                "peak_cuda_memory_bytes": run.peak_cuda_memory_bytes,
                "dense_cache_frames": len(run.dense_spnet_frames),
            },
            "runtime": {
                "mapping_wall_seconds": float(run.duration_seconds),
                "mapping_fps": float(run.fps),
                **deepcopy(dict(runtime_metrics or {})),
            },
            "environment": {
                "python": sys.version.split()[0], "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "device": str(run.model.means3d.device),
                "locks": dict(input_identity.environment_locks),
            },
            "artifacts": artifact_payload,
        }
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        input_identity.validate_unchanged(config)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RunArtifacts(output, output / "checkpoint.pt", output / "run_manifest.json", output / "map.ply")


def _execution_adapter_payload(spnet_identity: object) -> dict[str, object]:
    payload = getattr(spnet_identity, "execution_adapter_payload", None)
    if not callable(payload):
        raise ValueError("SPNet execution identity lacks execution_adapter_payload()")
    value = payload()
    if not isinstance(value, dict):
        raise ValueError("SPNet execution adapter identity must be an object")
    return deepcopy(value)
