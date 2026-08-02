"""Canonical SAGE method configuration and runtime input adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .data.scene import PreparedScene
from .foundation.config import (
    ALPHA_NORMALIZED_DEPTH_POLICY,
    ALL_ACCEPTED_FRAME_POLICY,
    FROZEN_MAPPING_LOSS_VARIANT,
    NATIVE_FULL_FRAME_PAD_CROP_V1,
    ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT,
    GaussianInitializationConfig,
    GrowthConfig,
    GrowthSourcesConfig,
    MappingConfig,
    MappingLossConfig,
    PruningConfig,
    ResidualThreshold,
    SPNetOnlineConfig,
    SageConfig,
    SceneConfig,
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
_RUNTIME_FIELDS = {"model_root", "require_clean_worktree"}


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
    value = payload.get(name)
    if (
        not isinstance(value, dict)
        or not required <= set(value) <= required | optional
    ):
        raise ValueError(
            f"SAGE {name} must define: {', '.join(sorted(required))}; "
            f"optional fields: {', '.join(sorted(optional))}"
        )
    return {**{field: 0.0 for field in optional}, **value}


def _runtime_section(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("runtime", {})
    if not isinstance(value, dict) or set(value) - _RUNTIME_FIELDS:
        raise ValueError(
            "SAGE runtime fields must be limited to: "
            + ", ".join(sorted(_RUNTIME_FIELDS))
        )
    return value


@dataclass(frozen=True)
class SageInput:
    """One concrete input adapter selected for a complete SAGE run."""

    kind: str
    root: Path
    calibration: Path | None = None
    write_through: Path | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"rosbag", "prepared_scene"}:
            raise ValueError("SAGE input must be rosbag or prepared_scene")
        object.__setattr__(self, "root", Path(self.root).resolve())
        if self.kind == "rosbag":
            if self.calibration is None:
                raise ValueError("ROSBAG input requires camera/LiDAR calibration")
            object.__setattr__(
                self,
                "calibration",
                Path(self.calibration).resolve(),
            )
            if self.write_through is not None:
                object.__setattr__(
                    self,
                    "write_through",
                    Path(self.write_through).resolve(),
                )
        elif self.calibration is not None or self.write_through is not None:
            raise ValueError(
                "Prepared Scene input cannot declare ROSBAG-only paths"
            )

    @property
    def internal_format(self) -> str:
        return "odin-rosbag" if self.kind == "rosbag" else "prepared-scene"


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
        input_config = _section(
            payload,
            "input",
            {"resize_width", "resize_height", "stream_queue_size"},
        )
        spnet = _section(
            payload,
            "spnet",
            {"model_id", "depth_scale_m", "confidence", "sample_stride"},
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
            {
                "iterations",
                "color_learning_rate",
                "opacity_learning_rate",
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
            {
                "means3d_learning_rate",
                "log_scales_learning_rate",
                "rotations_learning_rate",
            },
        )
        evaluation = _section(
            payload,
            "evaluation",
            {
                "frame_selection",
                "min_alpha",
                "epsilon",
                "alpha_support",
                "hit_target_center",
                "hit_target_fused",
            },
        )
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
        if self.evaluation["frame_selection"] != "all":
            raise ValueError("SAGE evaluation must use all accepted input frames")
        self.runtime_model_root()
        self.runtime_require_clean_worktree()
        self.refinement_config()
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

    def _structure_parts(self) -> dict[str, object]:
        growth = _section_with_optional(
            self.mapping,
            "growth",
            {
                "coverage_alpha_threshold",
                "min_candidate_depth_m",
                "max_candidate_depth_m",
                "candidate_duplicate_3d_threshold_m",
                "confidence_thresholds",
                "residual_thresholds",
            },
            {"max_new_per_commit"},
        )
        pruning = _section(
            self.mapping,
            "pruning",
            {"opacity_thresholds", "spnet_min_prune_age", "spnet_scale_ceiling_m"},
        )
        loss = _section(
            self.mapping,
            "loss",
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
        residuals = growth["residual_thresholds"]
        if not isinstance(residuals, dict):
            raise ValueError("SAGE growth residual_thresholds must be an object")
        return {
            "spnet": SPNetOnlineConfig(
                model_id=self.spnet["model_id"],
                adapter_policy=NATIVE_FULL_FRAME_PAD_CROP_V1,
                depth_scale_m=self.spnet["depth_scale_m"],
                confidence=self.spnet["confidence"],
                sample_stride=self.spnet["sample_stride"],
            ),
            "growth": GrowthConfig(
                coverage_alpha_threshold=growth["coverage_alpha_threshold"],
                min_candidate_depth_m=growth["min_candidate_depth_m"],
                max_candidate_depth_m=growth["max_candidate_depth_m"],
                candidate_duplicate_3d_threshold_m=(
                    growth["candidate_duplicate_3d_threshold_m"]
                ),
                confidence_thresholds=growth["confidence_thresholds"],
                residual_thresholds={
                    name: ResidualThreshold(**threshold)
                    for name, threshold in residuals.items()
                },
                depth_policy=ALPHA_NORMALIZED_DEPTH_POLICY,
                # 读原始 payload 而不是 growth[...]，避开 _section_with_optional 的
                # 0.0 缺省填充：不写该键表示不限量，写了就必须是完整的每源配额。
                max_new_per_commit=self.mapping["growth"].get("max_new_per_commit"),
            ),
            "pruning": PruningConfig(**pruning),
            "loss": MappingLossConfig(
                variant=FROZEN_MAPPING_LOSS_VARIANT,
                image_weight=loss["image_weight"],
                ssim_weight=loss["ssim_weight"],
                depth_weight=loss["depth_weight"],
                depth_coverage_weight=loss["depth_coverage_weight"],
                alpha_support_a0=loss["alpha_support"],
                depth_coverage_threshold=loss["depth_coverage_threshold"],
                epsilon=loss["epsilon"],
            ),
        }

    def _scene(self, source: SageInput) -> SceneConfig:
        require_clean_worktree = self.runtime_require_clean_worktree()
        if source.kind == "prepared_scene":
            scene = PreparedScene.from_root(source.root).config
            return replace(
                scene,
                resize_width=self.input["resize_width"],
                resize_height=self.input["resize_height"],
                stream_queue_size=self.input["stream_queue_size"],
                require_clean_worktree=require_clean_worktree,
            )
        return SceneConfig(
            scene_dir=None,
            prepared_scene_dir=None,
            center_depth_dir=None,
            fused5_depth_dir=None,
            fused5_mask_dir=None,
            resize_width=self.input["resize_width"],
            resize_height=self.input["resize_height"],
            enabled_depth_sources=(
                "LIDAR_SLAM_CENTER",
                "LIDAR_SLAM_FUSED5",
            ),
            input_adapter="rosbag-fixed-lag-v1",
            rosbag_dir=source.root,
            calibration_path=source.calibration,
            write_through_dir=source.write_through,
            stream_queue_size=self.input["stream_queue_size"],
            preparation_profile="odin1-slam-world-native-v1",
            source_mode="SLAM_WORLD",
            fusion_policy="slam-world-centered-5-v1",
            require_clean_worktree=require_clean_worktree,
        )

    def resolve(
        self,
        source: SageInput,
        output: Path,
    ) -> SageConfig:
        parts = self._structure_parts()
        evaluation = self.evaluation
        return SageConfig(
            config_path=self.path,
            output_dir=Path(output).resolve(),
            seed=self.seed,
            scene=self._scene(source),
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
                optimization_variant=ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT,
                evaluation_depth_policy=ALPHA_NORMALIZED_DEPTH_POLICY,
                evaluation_min_alpha=evaluation["min_alpha"],
                evaluation_epsilon=evaluation["epsilon"],
                evaluation_alpha_support_a0=evaluation["alpha_support"],
                evaluation_hit_target_center=evaluation["hit_target_center"],
                evaluation_hit_target_fused5=evaluation["hit_target_fused"],
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
        dense = _section(
            self.refinement,
            "dense_prior",
            {
                "grid_height",
                "grid_width",
                "min_lidar_support",
                "confidence_exponent",
                "robust_epsilon_m",
            },
        )
        return AppearanceRefinementConfig(
            schema_version=APPEARANCE_REFINEMENT_SCHEMA,
            iterations=self.refinement["iterations"],
            seed=self.seed,
            sampling_variant=SEEDED_RANDOM_MAPPING_FRAME_VARIANT,
            color_learning_rate=self.refinement["color_learning_rate"],
            opacity_learning_rate=self.refinement["opacity_learning_rate"],
            means3d_learning_rate=self.refinement["means3d_learning_rate"],
            log_scales_learning_rate=(
                self.refinement["log_scales_learning_rate"]
            ),
            rotations_learning_rate=(
                self.refinement["rotations_learning_rate"]
            ),
            opacity_anchor_weight=self.refinement["opacity_anchor_weight"],
            lidar_depth_weight=self.refinement["lidar_depth_weight"],
            dense_normal_weight=self.refinement["dense_normal_weight"],
            dense_normal_edge_weight_gamma=(
                self.refinement["dense_normal_edge_weight_gamma"]
            ),
            dense_normal_alpha_support_a0=(
                self.refinement["dense_normal_alpha_support"]
            ),
            dense_normal_max_relative_depth_jump=(
                self.refinement["dense_normal_max_relative_depth_jump"]
            ),
            dense_prior=DensePriorPolicy(
                **dense,
                alignment_variant=DENSE_ALIGNMENT_NONWORSENING_V2,
            ),
            exposure_learning_rate=self.refinement["exposure_learning_rate"],
            exposure_log_gain_bound=self.refinement["exposure_log_gain_bound"],
            exposure_bias_bound=self.refinement["exposure_bias_bound"],
            exposure_anchor_weight=self.refinement["exposure_anchor_weight"],
            exposure_gauge_variant=FIRST_MAPPING_FRAME_EXPOSURE_GAUGE,
            ssim_weight=self.refinement["ssim_weight"],
            milestone_steps=tuple(self.refinement["milestones"]),
        )
