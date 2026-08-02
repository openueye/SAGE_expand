"""Checkpoint schema validation for SAGE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data.providers.spnet import validate_spnet_identity_payload
from .foundation.identity_schema import DependencyIdentity, _valid_environment_locks
from .foundation.artifact_versions import (
    APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
    CHECKPOINT_VERSION,
    SOURCE_CHECKPOINT_VERSIONS,
)
from .foundation.receipt_contract import validate_sage_completion_receipt
from .foundation.source_policy import SOURCE_POLICY_VERSION
from .refinement.appearance_config import (
    APPEARANCE_REFINEMENT_SCHEMA,
    SCALAR_EXPOSURE_NUISANCE_SCHEMA,
    AppearanceRefinementConfig,
)


_GAUSSIAN_TENSORS = {
    "means3d",
    "colors",
    "opacity_logits",
    "log_scales",
    "rotations",
    "gaussian_ids",
    "created_at",
    "source_types",
    "source_confidences",
}
_BASE_FIELDS = {
    "checkpoint_version",
    "source_policy_version",
    *_GAUSSIAN_TENSORS,
    "last_commit",
    "identity_snapshot",
    "optimization",
    "completion_receipt",
}
_SOURCE_COMMIT_FIELDS = {
    "frame_index",
    "frame_stem",
    "timestamp_ns",
    "active_window",
    "optimization_views",
    "optimizer_step",
    "mapping_commit_ordinal",
    "pruning_completed_steps",
    "pruning_event_executed",
    "gaussian_count",
    "added",
    "added_by_source",
    "proposed_by_source",
    "accepted_by_source",
    "rejected_by_source",
    "rejection_reasons_by_source",
    "pruned",
    "pruned_by_source",
    "pruned_by_reason",
    "newborn_protected",
    "newborn_protected_by_source",
    "mature_opacity_removed",
    "mature_opacity_removed_by_source",
    "scale_removed",
    "scale_removed_by_source",
    "remaining_by_source",
    "spnet_available",
    "final_loss",
    "psnr",
    "ssim",
    "lpips",
    "fps",
    "final_image_loss",
    "final_depth_loss",
    "final_depth_coverage_loss",
    "final_depth_valid_pixels",
    "final_depth_mean_alpha",
    "spnet_invoked",
    "spnet_mode",
    "spnet_valid_pixels",
    "spnet_inference_seconds",
    "diagnostics",
}
_DIAGNOSTIC_FIELDS = {
    "loss": {
        "photo",
        "geo_center",
        "geo_fused5",
        "hit_center",
        "hit_fused5",
        "depth_coverage",
        "total",
    },
    "depth": {
        "valid_pixels",
        "valid_center_pixels",
        "valid_fused5_pixels",
        "center_mae",
        "fused5_mae",
    },
    "alpha": {
        "center_mean",
        "center_p10",
        "center_p50",
        "fused5_mean",
        "fused5_p10",
        "fused5_p50",
        "below_a0_fraction",
    },
}


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_commit(commit: object) -> None:
    if commit is None:
        return
    if not isinstance(commit, dict) or set(commit) != _SOURCE_COMMIT_FIELDS:
        raise ValueError("SAGE checkpoint last_commit shape is invalid")
    diagnostics = commit["diagnostics"]
    if (
        not isinstance(diagnostics, dict)
        or set(diagnostics) != set(_DIAGNOSTIC_FIELDS)
    ):
        raise ValueError("SAGE checkpoint commit diagnostics are invalid")
    for group, fields in _DIAGNOSTIC_FIELDS.items():
        values = diagnostics[group]
        if (
            not isinstance(values, dict)
            or set(values) != fields
            or any(
                type(value) not in {int, float}
                or not np.isfinite(value)
                for value in values.values()
            )
        ):
            raise ValueError("SAGE checkpoint commit diagnostics are invalid")


def _validate_exposure(
    value: object,
    *,
    config: AppearanceRefinementConfig,
    expected_frame_indices: list[int],
) -> None:
    expected = {
        "schema_version",
        "gauge_variant",
        "reference_frame_index",
        "log_gain_bound",
        "bias_bound",
        "anchor_weight",
        "frames",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value["schema_version"] != SCALAR_EXPOSURE_NUISANCE_SCHEMA
        or value["gauge_variant"] != config.exposure_gauge_variant
        or value["log_gain_bound"] != config.exposure_log_gain_bound
        or value["bias_bound"] != config.exposure_bias_bound
        or value["anchor_weight"] != config.exposure_anchor_weight
        or not isinstance(value["frames"], list)
        or not value["frames"]
    ):
        raise ValueError(
            "Appearance refinement exposure nuisance is invalid"
        )
    frame_fields = {
        "frame_index",
        "log_gain",
        "gain",
        "bias",
        "reference",
    }
    indices: list[int] = []
    reference_count = 0
    for frame in value["frames"]:
        if not isinstance(frame, dict) or set(frame) != frame_fields:
            raise ValueError(
                "Appearance refinement exposure nuisance frame is invalid"
            )
        try:
            log_gain = float(frame["log_gain"])
            gain = float(frame["gain"])
            bias = float(frame["bias"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Appearance refinement exposure nuisance values are invalid"
            ) from exc
        reference = frame["reference"]
        if (
            type(frame["frame_index"]) is not int
            or frame["frame_index"] < 0
            or type(reference) is not bool
            or not np.isfinite([log_gain, gain, bias]).all()
            or abs(log_gain) > config.exposure_log_gain_bound
            or abs(bias) > config.exposure_bias_bound
            or gain <= 0
            or not np.isclose(
                gain,
                np.exp(log_gain),
                rtol=1e-6,
                atol=1e-8,
            )
        ):
            raise ValueError(
                "Appearance refinement exposure nuisance frame is invalid"
            )
        if reference:
            reference_count += 1
            if (
                frame["frame_index"] != value["reference_frame_index"]
                or log_gain != 0.0
                or gain != 1.0
                or bias != 0.0
            ):
                raise ValueError(
                    "Appearance refinement exposure nuisance reference "
                    "is invalid"
                )
        indices.append(frame["frame_index"])
    if (
        indices != expected_frame_indices
        or indices != sorted(indices)
        or len(indices) != len(set(indices))
        or reference_count != 1
        or value["reference_frame_index"] != indices[0]
    ):
        raise ValueError(
            "Appearance refinement exposure nuisance frame set is invalid"
        )


def _validate_dense_prior(
    value: object,
    *,
    config: AppearanceRefinementConfig,
) -> list[int]:
    if (
        not isinstance(value, dict)
        or set(value) not in (
            {"provider_identity", "frames"},
            {"provider_identity", "frames", "prediction_cache"},
        )
        or not isinstance(value["frames"], list)
        or not value["frames"]
    ):
        raise ValueError(
            "Appearance refinement dense prior provenance is invalid"
        )
    try:
        provider_grids = validate_spnet_identity_payload(
            value["provider_identity"]
        )
    except ValueError as exc:
        raise ValueError(
            "Appearance refinement dense prior provider identity is invalid"
        ) from exc
    expected = {
        "frame_index",
        "lidar_support",
        "fitted_grid_cells",
        "pre_alignment_mae_m",
        "post_alignment_mae_m",
        "fallback",
        "valid_depth_pixels",
        "valid_normal_pixels",
        "scale_grid",
        "alignment_variant",
        "fitted_alignment_mae_m",
        "selected_alignment",
        "max_relative_depth_jump",
    }
    indices: list[int] = []
    for frame in value["frames"]:
        if not isinstance(frame, dict) or set(frame) != expected:
            raise ValueError(
                "Appearance refinement dense prior frame is invalid"
            )
        counts = (
            frame["lidar_support"],
            frame["fitted_grid_cells"],
            frame["valid_depth_pixels"],
            frame["valid_normal_pixels"],
        )
        try:
            residuals = np.asarray(
                [
                    frame["pre_alignment_mae_m"],
                    frame["post_alignment_mae_m"],
                    frame["fitted_alignment_mae_m"],
                ],
                dtype=np.float64,
            )
            grid = np.asarray(frame["scale_grid"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Appearance refinement dense prior values are invalid"
            ) from exc
        if (
            type(frame["frame_index"]) is not int
            or frame["frame_index"] < 0
            or any(type(count) is not int or count < 0 for count in counts)
            or frame["valid_normal_pixels"] > frame["valid_depth_pixels"]
            or frame["valid_normal_pixels"] < 1
            or type(frame["fallback"]) is not bool
            or residuals.shape != (3,)
            or not np.isfinite(residuals).all()
            or (residuals < 0).any()
            or residuals[1] > residuals[0]
            or grid.shape
            != (
                config.dense_prior.grid_height,
                config.dense_prior.grid_width,
            )
            or not np.isfinite(grid).all()
            or (grid <= 0).any()
            or frame["alignment_variant"]
            != config.dense_prior.alignment_variant
            or frame["selected_alignment"]
            not in {"fitted-grid", "identity", "identity-fallback"}
            or frame["max_relative_depth_jump"]
            != config.dense_normal_max_relative_depth_jump
        ):
            raise ValueError(
                "Appearance refinement dense prior frame is invalid"
            )
        indices.append(frame["frame_index"])
    if indices != sorted(set(indices)):
        raise ValueError(
            "Appearance refinement dense prior frames must be unique "
            "and sorted"
        )
    if [record.frame_index for record in provider_grids] != indices:
        raise ValueError(
            "Appearance refinement dense prior provider frames do not match"
        )
    cache = value.get("prediction_cache")
    if cache is not None:
        expected_cache = {
            "schema_version",
            "source_manifest_sha256",
            "cache_sha256",
            "cached_frame_indices",
            "reused_frames",
            "live_frames",
        }
        cached_indices = (
            cache.get("cached_frame_indices")
            if isinstance(cache, dict)
            else None
        )
        if (
            not isinstance(cache, dict)
            or set(cache) != expected_cache
            or cache.get("schema_version") != "sage-dense-spnet-reuse-v1"
            or not _is_hex_digest(cache.get("source_manifest_sha256"), 64)
            or not _is_hex_digest(cache.get("cache_sha256"), 64)
            or not isinstance(cached_indices, list)
            or cached_indices != sorted(set(cached_indices))
            or any(index not in indices for index in cached_indices)
            or type(cache.get("reused_frames")) is not int
            or type(cache.get("live_frames")) is not int
            or cache["reused_frames"] != len(cached_indices)
            or cache["reused_frames"] + cache["live_frames"] != len(indices)
        ):
            raise ValueError(
                "Appearance refinement dense prediction cache is invalid"
            )
    return indices


def _validate_refinement(value: object) -> None:
    expected = {
        "schema_version",
        "source_checkpoint_version",
        "source_checkpoint_sha256",
        "refinement_config_sha256",
        "config",
        "optimized_parameters",
        "iteration_count",
        "selected_frame_indices",
        "milestones",
        "producer_code",
        "publication_ready",
        "dense_prior",
        "nuisance_parameters",
        "exposure_nuisance",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(
            "Appearance refinement checkpoint contract is invalid"
        )
    try:
        config = AppearanceRefinementConfig.from_payload(value["config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Appearance refinement checkpoint config is invalid"
        ) from exc
    selected = value["selected_frame_indices"]
    milestones = value["milestones"]
    producer = value["producer_code"]
    if (
        value["schema_version"] != APPEARANCE_REFINEMENT_SCHEMA
        or value["schema_version"] != config.schema_version
        or value["source_checkpoint_version"] not in SOURCE_CHECKPOINT_VERSIONS
        or not _is_hex_digest(value["source_checkpoint_sha256"], 64)
        or not _is_hex_digest(value["refinement_config_sha256"], 64)
        or value["optimized_parameters"]
        != list(config.optimized_parameters)
        or value["iteration_count"] != config.iterations
        or not isinstance(selected, list)
        or len(selected) != config.iterations
        or any(type(index) is not int or index < 0 for index in selected)
        or not isinstance(milestones, list)
        or len(milestones) != len(config.milestone_steps)
        or [
            item.get("step")
            for item in milestones
            if isinstance(item, dict)
        ]
        != list(config.milestone_steps)
        or not isinstance(producer, dict)
        or set(producer)
        != {
            "repository",
            "commit",
            "worktree_dirty",
            "source_state_sha256",
        }
        or not isinstance(producer["repository"], str)
        or not isinstance(producer["worktree_dirty"], bool)
        or not _is_hex_digest(producer["commit"], 40)
        or not _is_hex_digest(producer["source_state_sha256"], 64)
        or value["publication_ready"] is not True
        or value["nuisance_parameters"]
        != list(config.nuisance_parameters)
    ):
        raise ValueError(
            "Appearance refinement checkpoint contract is invalid"
        )
    frame_indices = _validate_dense_prior(
        value["dense_prior"],
        config=config,
    )
    _validate_exposure(
        value["exposure_nuisance"],
        config=config,
        expected_frame_indices=frame_indices,
    )


def _validate_common_checkpoint(payload: dict[str, object]) -> None:
    if payload.get("source_policy_version") != SOURCE_POLICY_VERSION:
        raise ValueError("Unsupported SAGE source policy version")
    if any(
        not isinstance(payload.get(name), torch.Tensor)
        for name in _GAUSSIAN_TENSORS
    ):
        raise ValueError("SAGE checkpoint is missing a Gaussian tensor")
    snapshot = payload.get("identity_snapshot")
    snapshot_fields = {
        "config_sha256",
        "prepared_manifest_sha256",
        "transform_contract_sha256",
        "scene_content_sha256",
        "source_mode",
        "source_policy_version",
        "producer_code",
        "dependencies",
        "frame_source",
        "environment_locks",
    }
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != snapshot_fields
        or snapshot["source_policy_version"] != SOURCE_POLICY_VERSION
        or not isinstance(snapshot["dependencies"], dict)
        or not _valid_environment_locks(snapshot["environment_locks"])
    ):
        raise ValueError("SAGE checkpoint lacks dependency identity")
    DependencyIdentity(**snapshot["dependencies"])
    producer = snapshot["producer_code"]
    if (
        not isinstance(producer, dict)
        or set(producer) != {"repository", "commit", "worktree_dirty"}
        or not isinstance(producer["repository"], str)
        or not _is_hex_digest(producer["commit"], 40)
        or not isinstance(producer["worktree_dirty"], bool)
    ):
        raise ValueError("SAGE checkpoint producer identity is invalid")
    _validate_commit(payload.get("last_commit"))
    optimization = payload.get("optimization")
    if (
        not isinstance(optimization, dict)
        or set(optimization)
        != {
            "variant",
            "optimizer_lifecycle",
            "optimizer_final_step",
            "append_state_migrations",
            "prune_state_migrations",
        }
    ):
        raise ValueError("SAGE checkpoint optimization identity is invalid")
    frame_source = snapshot.get("frame_source")
    adapter = (
        frame_source.get("adapter")
        if isinstance(frame_source, dict)
        else None
    )
    validate_sage_completion_receipt(
        payload.get("completion_receipt"),
        expected_adapter=adapter,
        expected_source_mode=snapshot.get("source_mode"),
        require_bag_exhausted=adapter == "rosbag-fixed-lag-v1",
    )


def load_checkpoint(path: Path) -> dict[str, object]:
    """Load either a source SAGE checkpoint or a published refined checkpoint."""
    try:
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot read SAGE checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("SAGE checkpoint must be an object")
    version = payload.get("checkpoint_version")
    expected = set(_BASE_FIELDS)
    if version == APPEARANCE_REFINEMENT_CHECKPOINT_VERSION:
        expected.add("appearance_refinement")
    if (
        version
        not in SOURCE_CHECKPOINT_VERSIONS
        | {APPEARANCE_REFINEMENT_CHECKPOINT_VERSION}
        or set(payload) != expected
    ):
        raise ValueError(
            "SAGE checkpoint fields do not match a published schema"
        )
    _validate_common_checkpoint(payload)
    if version == APPEARANCE_REFINEMENT_CHECKPOINT_VERSION:
        _validate_refinement(payload["appearance_refinement"])
    return payload
