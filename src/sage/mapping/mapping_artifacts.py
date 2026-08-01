from __future__ import annotations

from collections.abc import Callable
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
from ..data.scene import sha256_file
from ..engine.ply_export import write_gaussian_ply
from ..foundation.artifact_versions import CHECKPOINT_VERSION
from ..foundation.receipt_contract import validate_sage_completion_receipt
from ..foundation.config import SageConfig
from ..foundation.source_policy import SOURCE_POLICY_VERSION, source_counts
from .mapper import MappingCommit, MappingRun
from .run_identity import RunInputIdentity


RUN_MANIFEST_SCHEMA_VERSION = "sage-run-manifest-v1"

_DIAGNOSTIC_FIELDS = {
    "loss": {"photo", "geo_center", "geo_fused5", "hit_center", "hit_fused5", "total"},
    "depth": {"valid_pixels", "valid_center_pixels", "valid_fused5_pixels", "center_mae", "fused5_mae"},
    "alpha": {"center_mean", "center_p10", "center_p50", "fused5_mean", "fused5_p10", "fused5_p50", "below_a0_fraction"},
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
    completion_receipt: dict[str, object],
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
        "completion_receipt": deepcopy(completion_receipt),
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
    completion_receipt: dict[str, object] | None = None,
    commit_source: Callable[[], None] | None = None,
    rollback_source: Callable[[BaseException], None] | None = None,
) -> RunArtifacts:
    if completion_receipt is None:
        raise ValueError("Run artifact publication requires an explicit FrameSource completion receipt")
    adapter = input_identity.frame_source_identity.get("adapter")
    if adapter == "rosbag-fixed-lag-v1" and (
        commit_source is None or rollback_source is None
    ):
        raise ValueError(
            "Streaming artifact publication requires commit and rollback callbacks"
        )
    if run.model.means3d.device.type == "cpu":
        raise ValueError("Published mapping artifacts require CUDA execution")
    output = config.output_dir
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _validate_run(run)
        receipt = validate_sage_completion_receipt(
            completion_receipt,
            expected_adapter=str(adapter) if adapter is not None else None,
            expected_source_mode=input_identity.source_mode,
            expected_emitted=run.processed_frames,
            require_bag_exhausted=adapter == "rosbag-fixed-lag-v1",
        )
        receipt["runtime"] = {
            **(
                receipt.get("runtime", {})
                if isinstance(receipt.get("runtime"), dict) else {}
            ),
            "mapping_wall_seconds": float(run.duration_seconds),
            "mapping_fps": float(run.fps),
            "first_frame_wait_seconds": (
                receipt.get("runtime", {}).get("first_frame_wait_seconds", "unmeasured")
                if isinstance(receipt.get("runtime"), dict) else "unmeasured"
            ),
        }
        validate_sage_completion_receipt(
            receipt,
            expected_adapter=str(adapter) if adapter is not None else None,
            expected_source_mode=input_identity.source_mode,
            expected_emitted=run.processed_frames,
            require_bag_exhausted=adapter == "rosbag-fixed-lag-v1",
        )
        receipt["spnet_anchor_source_types"] = list(run.spnet_anchor_source_types)
        checkpoint = staging / "checkpoint.pt"
        ply = staging / "map.ply"
        manifest = staging / "run_manifest.json"
        dense_spnet_cache = staging / "spnet_dense.pt"
        torch.save(_checkpoint_payload(run, input_identity, receipt), checkpoint)
        if run.dense_spnet_frames:
            write_dense_spnet_cache(
                dense_spnet_cache,
                run.dense_spnet_frames,
                run.spnet_identity,
            )
        write_gaussian_ply(ply, run.model)
        artifact_payload = {
            "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint)},
            "map_ply": {"path": ply.name, "sha256": sha256_file(ply)},
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
            "completion_receipt": receipt,
            "dependencies": input_identity.dependencies.payload(),
            "scene": {
                "resize": {
                    "width": config.scene.resize_width, "height": config.scene.resize_height,
                } if config.scene.resize_width is not None else None,
                "source_mode": input_identity.source_mode,
                "source_policy_version": input_identity.source_policy_version,
                "prepared_manifest_sha256": input_identity.prepared_manifest_sha256,
                "source_manifest_sha256": input_identity.source_manifest_sha256,
                "transform_contract_sha256": input_identity.transform_contract_sha256,
                "content_sha256": input_identity.scene_content_sha256,
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
                "mode": config.growth_sources.spnet.mode,
                "expected_invocations": run.spnet_expected_invocations,
                "actual_invocations": run.spnet_actual_invocations,
                "anchor_source_types": list(run.spnet_anchor_source_types),
                "adapter": _execution_adapter_payload(run.spnet_identity),
                "inference_seconds": list(run.spnet_inference_seconds),
                "peak_cuda_memory_bytes": run.peak_cuda_memory_bytes,
                "dense_cache_frames": len(run.dense_spnet_frames),
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
        source_committed = False
        if commit_source is not None:
            commit_source()
            source_committed = True
        try:
            staging.rename(output)
        except BaseException as exc:
            if source_committed and rollback_source is not None:
                rollback_source(exc)
            raise
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
