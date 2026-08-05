"""Canonical SAGE method configuration and runtime input adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .foundation.config import (
    ALPHA_NORMALIZED_DEPTH_POLICY,
    ALL_ACCEPTED_FRAME_POLICY,
    FROZEN_MAPPING_LOSS_VARIANT,
    NATIVE_FULL_FRAME_PAD_CROP_V1,
    GLOBAL_CURRENT_ANCHORED_VARIANT,
    GaussianInitializationConfig,
    GrowthConfig,
    GrowthSourcesConfig,
    MappingConfig,
    MappingLossConfig,
    PruningConfig,
    ResidualThreshold,
    SPNetOnlineConfig,
    InputConfig,
    SageConfig,
)
from .refinement.appearance_config import (
    APPEARANCE_REFINEMENT_SCHEMA,
    FIRST_MAPPING_FRAME_EXPOSURE_GAUGE,
    SEEDED_RANDOM_MAPPING_FRAME_VARIANT,
    AppearanceRefinementConfig,
)
from .refinement.dense_geometry_config import (
    DENSE_ALIGNMENT_NONWORSENING_V2,
    DensePriorPolicy,
)


_ROOT_FIELDS = {
    "seed",
    "input",
    "spnet",
    "mapping",
    "refinement",
    "evaluation",
}
_OPTIONAL_ROOT_FIELDS = {"runtime"}
_RUNTIME_FIELDS = {"model_root", "require_clean_worktree", "prefetch_depth"}
_EVALUATION_FIELDS = {
    "min_alpha", "epsilon", "alpha_support", "hit_target_center", "hit_target_fused",
}
_EVALUATION_OPTIONAL_FIELDS = {"checkpoint_stages"}
_DEFAULT_EVALUATION_CHECKPOINT_STAGES = ("stage2_refinement",)
_EVALUATION_CHECKPOINT_STAGES = {"stage1_mapping", "stage2_refinement"}


def _section(
    payload: dict[str, Any],
    name: str,
    fields: set[str],
) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(
            f"SAGE {name} fields must be exactly: {', '.join(sorted(fields))}"
        )
    return value


def _section_with_optional(
    payload: dict[str, Any],
    name: str,
    required: set[str],
    optional: set[str],
) -> dict[str, Any]:
    """Validate field membership only; omitted optional fields stay absent.

    Callers pass the result through `_optional_kwargs` so the target
    dataclass's own default applies to whatever was left out, instead of a
    synthetic fill value that may not match that field's real default.
    """
    value = payload.get(name)
    if (
        not isinstance(value, dict)
        or not required <= set(value) <= required | optional
    ):
        raise ValueError(
            f"SAGE {name} must define: {', '.join(sorted(required))}; "
            f"optional fields: {', '.join(sorted(optional))}"
        )
    return dict(value)


def _optional_kwargs(payload: dict[str, Any], names: set[str]) -> dict[str, Any]:
    """Keyword arguments for exactly the fields present in `payload`."""
    return {name: payload[name] for name in names if name in payload}


def _runtime_section(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("runtime", {})
    if not isinstance(value, dict) or set(value) - _RUNTIME_FIELDS:
        raise ValueError(
            "SAGE runtime fields must be limited to: "
            + ", ".join(sorted(_RUNTIME_FIELDS))
        )
    return value


def _evaluation_section(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("evaluation")
    if (
        not isinstance(value, dict)
        or not _EVALUATION_FIELDS <= set(value) <= _EVALUATION_FIELDS | _EVALUATION_OPTIONAL_FIELDS
    ):
        raise ValueError(
            "SAGE evaluation must define: " + ", ".join(sorted(_EVALUATION_FIELDS))
            + "; optional fields: checkpoint_stages"
        )
    return dict(value)


@dataclass(frozen=True)
class SageMethodConfig:
    """The single configuration interface for complete SAGE training."""

    path: Path
    seed: int
    input: dict[str, Any]
    spnet: dict[str, Any]
    mapping: dict[str, Any]
    refinement: dict[str, Any]
    evaluation: dict[str, Any]
    runtime: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "SageMethodConfig":
        source = Path(path).resolve()
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot read SAGE configuration: {source}") from exc
        if (
            not isinstance(payload, dict)
            or not _ROOT_FIELDS <= set(payload) <= _ROOT_FIELDS | _OPTIONAL_ROOT_FIELDS
        ):
            raise ValueError(
                "SAGE configuration must define: "
                + ", ".join(sorted(_ROOT_FIELDS))
                + "; optional fields: "
                + ", ".join(sorted(_OPTIONAL_ROOT_FIELDS))
            )
        if type(payload["seed"]) is not int or payload["seed"] < 0:
            raise ValueError("SAGE seed must be a non-negative integer")
        input_config = payload["input"]
        if not isinstance(input_config, dict) or not isinstance(input_config.get("type"), str):
            raise ValueError("SAGE input must be an object declaring a type")
        spnet = _section_with_optional(
            payload,
            "spnet",
            set(),
            {"model_id", "enabled", "depth_scale_m", "confidence", "sample_stride"},
        )
        mapping = _section(
            payload,
            "mapping",
            {
                "map_every",
                "keyframe_every",
                "iterations",
                "prune_every",
                "prune_stop_after",
                "learning_rates",
                "initial_opacity",
                "scale_clamp_min",
                "initial_scale_anisotropy",
                "growth",
                "pruning",
                "loss",
            },
        )
        refinement = _section_with_optional(
            payload,
            "refinement",
            set(),
            {
                "iterations",
                "color_learning_rate",
                "opacity_learning_rate",
                "means3d_learning_rate",
                "log_scales_learning_rate",
                "rotations_learning_rate",
                "opacity_anchor_weight",
                "lidar_depth_weight",
                "dense_normal_weight",
                "dense_normal_edge_weight_gamma",
                "dense_normal_alpha_support",
                "dense_normal_max_relative_depth_jump",
                "dense_prior",
                "exposure_learning_rate",
                "exposure_log_gain_bound",
                "exposure_bias_bound",
                "exposure_anchor_weight",
                "ssim_weight",
                "milestones",
            },
        )
        evaluation = _evaluation_section(payload)
        runtime = _runtime_section(payload)
        config = cls(
            path=source,
            seed=payload["seed"],
            input=input_config,
            spnet=spnet,
            mapping=mapping,
            refinement=refinement,
            evaluation=evaluation,
            runtime=runtime,
        )
        config._validate()
        return config

    def _validate(self) -> None:
        self.runtime_model_root()
        self.runtime_require_clean_worktree()
        self.refinement_config()
        self.evaluation_checkpoint_stages()
        self._structure_parts()

    def runtime_model_root(self) -> Path | None:
        value = self.runtime.get("model_root")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("SAGE runtime.model_root must be a non-empty path")
        root = Path(value).expanduser()
        if not root.is_absolute():
            root = self.path.parent / root
        return root.resolve()

    def runtime_require_clean_worktree(self) -> bool:
        value = self.runtime.get("require_clean_worktree", False)
        if type(value) is not bool:
            raise ValueError("SAGE runtime.require_clean_worktree must be a boolean")
        return value

    def evaluation_checkpoint_stages(self) -> tuple[str, ...]:
        value = self.evaluation.get("checkpoint_stages", _DEFAULT_EVALUATION_CHECKPOINT_STAGES)
        if (
            not isinstance(value, (list, tuple))
            or not value
            or len(set(value)) != len(value)
            or any(not isinstance(stage, str) or stage not in _EVALUATION_CHECKPOINT_STAGES for stage in value)
        ):
            raise ValueError(
                "evaluation.checkpoint_stages must be a non-empty unique list of "
                "stage1_mapping and/or stage2_refinement"
            )
        return tuple(value)

    def _structure_parts(self) -> dict[str, object]:
        growth = _section_with_optional(
            self.mapping,
            "growth",
            {"confidence_thresholds"},
            {
                "coverage_alpha_threshold",
                "min_candidate_depth_m",
                "max_candidate_depth_m",
                "candidate_duplicate_3d_threshold_m",
                "residual_thresholds",
                "max_new_per_commit",
            },
        )
        pruning = _section(
            self.mapping,
            "pruning",
            {"opacity_thresholds"},
        )
        loss = _section_with_optional(
            self.mapping,
            "loss",
            set(),
            {
                "image_weight",
                "ssim_weight",
                "depth_weight",
                "depth_coverage_weight",
                "alpha_support",
                "depth_coverage_threshold",
                "epsilon",
            },
        )
        residuals = growth.get("residual_thresholds")
        if residuals is not None and not isinstance(residuals, dict):
            raise ValueError("SAGE growth residual_thresholds must be an object")
        loss_kwargs = _optional_kwargs(loss, {
            "image_weight", "ssim_weight", "depth_weight",
            "depth_coverage_weight", "depth_coverage_threshold", "epsilon",
        })
        if "alpha_support" in loss:
            loss_kwargs["alpha_support_a0"] = loss["alpha_support"]
        return {
            "spnet": SPNetOnlineConfig(
                model_id=self.spnet.get("model_id", ""),
                adapter_policy=NATIVE_FULL_FRAME_PAD_CROP_V1,
                **_optional_kwargs(self.spnet, {"enabled", "depth_scale_m", "confidence", "sample_stride"}),
            ),
            "growth": GrowthConfig(
                confidence_thresholds=growth["confidence_thresholds"],
                residual_thresholds=(
                    {name: ResidualThreshold(**threshold) for name, threshold in residuals.items()}
                    if residuals is not None else None
                ),
                depth_policy=ALPHA_NORMALIZED_DEPTH_POLICY,
                # 读原始 payload 而不是 growth[...]，避开 _section_with_optional 的
                # 0.0 缺省填充：不写该键表示不限量，写了就必须是完整的每源配额。
                max_new_per_commit=self.mapping["growth"].get("max_new_per_commit"),
                **_optional_kwargs(growth, {
                    "coverage_alpha_threshold", "min_candidate_depth_m",
                    "max_candidate_depth_m", "candidate_duplicate_3d_threshold_m",
                }),
            ),
            "pruning": PruningConfig(**pruning),
            "loss": MappingLossConfig(variant=FROZEN_MAPPING_LOSS_VARIANT, **loss_kwargs),
        }

    def _input(self) -> InputConfig:
        return InputConfig(
            payload=self.input,
            base_dir=self.path.parent,
            require_clean_worktree=self.runtime_require_clean_worktree(),
            prefetch_depth=int(self.runtime.get("prefetch_depth", 4)),
        )

    def resolve(self, output: Path) -> SageConfig:
        parts = self._structure_parts()
        evaluation = self.evaluation
        return SageConfig(
            config_path=self.path,
            output_dir=Path(output).resolve(),
            seed=self.seed,
            input=self._input(),
            growth_sources=GrowthSourcesConfig(spnet=parts["spnet"]),
            growth=parts["growth"],
            pruning=parts["pruning"],
            mapping=MappingConfig(
                frame_policy=ALL_ACCEPTED_FRAME_POLICY,
                map_every=self.mapping["map_every"],
                keyframe_every=self.mapping["keyframe_every"],
                iterations=self.mapping["iterations"],
                prune_every=self.mapping["prune_every"],
                prune_stop_after=self.mapping["prune_stop_after"],
                learning_rates=self.mapping["learning_rates"],
                optimization_variant=GLOBAL_CURRENT_ANCHORED_VARIANT,
                evaluation_depth_policy=ALPHA_NORMALIZED_DEPTH_POLICY,
                evaluation_min_alpha=evaluation["min_alpha"],
                evaluation_epsilon=evaluation["epsilon"],
                evaluation_alpha_support_a0=evaluation["alpha_support"],
                evaluation_hit_target_center=evaluation["hit_target_center"],
                evaluation_hit_target_fused=evaluation["hit_target_fused"],
            ),
            gaussian_initialization=GaussianInitializationConfig(
                opacity=self.mapping["initial_opacity"],
                scale_clamp_min=self.mapping["scale_clamp_min"],
                initial_scale_anisotropy=self.mapping["initial_scale_anisotropy"],
            ),
            loss=parts["loss"],
            model_root=self.runtime_model_root(),
        )

    def refinement_config(self) -> AppearanceRefinementConfig:
        refinement = self.refinement
        dense_payload = refinement.get("dense_prior")
        dense_fields = {"grid_height", "grid_width", "min_lidar_support", "confidence_exponent"}
        if dense_payload is not None and (
            not isinstance(dense_payload, dict) or not set(dense_payload) <= dense_fields
        ):
            raise ValueError(
                "SAGE refinement dense_prior fields must be a subset of: "
                + ", ".join(sorted(dense_fields))
            )
        kwargs = _optional_kwargs(refinement, {
            "iterations", "color_learning_rate", "opacity_learning_rate",
            "means3d_learning_rate", "log_scales_learning_rate", "rotations_learning_rate",
            "opacity_anchor_weight", "lidar_depth_weight", "dense_normal_weight",
            "dense_normal_edge_weight_gamma", "dense_normal_max_relative_depth_jump",
            "exposure_learning_rate", "exposure_log_gain_bound", "exposure_bias_bound",
            "exposure_anchor_weight", "ssim_weight",
        })
        if "dense_normal_alpha_support" in refinement:
            kwargs["dense_normal_alpha_support_a0"] = refinement["dense_normal_alpha_support"]
        if "milestones" in refinement:
            kwargs["milestone_steps"] = tuple(refinement["milestones"])
        return AppearanceRefinementConfig(
            schema_version=APPEARANCE_REFINEMENT_SCHEMA,
            seed=self.seed,
            sampling_variant=SEEDED_RANDOM_MAPPING_FRAME_VARIANT,
            exposure_gauge_variant=FIRST_MAPPING_FRAME_EXPOSURE_GAUGE,
            dense_prior=DensePriorPolicy(
                **(dense_payload or {}),
                alignment_variant=DENSE_ALIGNMENT_NONWORSENING_V2,
            ),
            **kwargs,
        )
