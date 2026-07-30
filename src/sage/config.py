from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
from typing import Any

from .source_policy import (
    SLAM_SOURCE_DESCRIPTORS,
    SLAM_SOURCE_NAMES,
    SOURCE_DESCRIPTORS,
    SOURCE_NAMES,
    materialize_source_policy,
)


SCHEMA_VERSION = "sage-gs-v1"
ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT = "odin-global-current-anchored-v1"
NATIVE_FULL_FRAME_PAD_CROP_V1 = "native-full-frame-pad-crop-v1"
RAW_ACCUMULATED_DEPTH_POLICY = "raw-accumulated-v1"
ALPHA_NORMALIZED_DEPTH_POLICY = "alpha-normalized-v1"
FROZEN_MAPPING_LOSS_VARIANT = "loss-v2-alpha-soft-v1"
ALL_ACCEPTED_FRAME_POLICY = "all-accepted"
ALL_ACCEPTED_FRAME_LIMIT = -1




@dataclass(frozen=True)
class SceneConfig:
    scene_dir: Path | None
    prepared_scene_dir: Path | None
    raw_depth_dir: Path | None
    fused5_depth_dir: Path | None
    fused5_mask_dir: Path | None
    resize_width: int | None = None
    resize_height: int | None = None
    enabled_depth_sources: tuple[str, ...] = (
        "LIDAR_SLAM_CENTER", "LIDAR_SLAM_FUSED5",
    )
    center_depth_dir: Path | None = None
    input_adapter: str = "prepared-scene-v2"
    rosbag_dir: Path | None = None
    calibration_path: Path | None = None
    write_through_dir: Path | None = None
    stream_queue_size: int = 1
    preparation_profile: str | None = None
    source_mode: str | None = None
    fusion_policy: str | None = None
    confirm_raw_offset_time_seconds_from_scan_start: bool = False
    confirm_base_from_lidar_identity: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.confirm_raw_offset_time_seconds_from_scan_start) is not bool
            or type(self.confirm_base_from_lidar_identity) is not bool
        ):
            raise ValueError("RAW semantics confirmations must be JSON booleans")
        if self.input_adapter not in {"prepared-scene-v2", "rosbag-fixed-lag-v1"}:
            raise ValueError("input_adapter must be prepared-scene-v2 or rosbag-fixed-lag-v1")
        if (self.resize_width is None) != (self.resize_height is None):
            raise ValueError("resize_width and resize_height must be configured together")
        if any(value is not None and value < 1 for value in (self.resize_width, self.resize_height)):
            raise ValueError("Resize dimensions must be positive")
        sources = tuple(self.enabled_depth_sources)
        if sources not in {
            ("LIDAR_RAW",), ("LIDAR_RAW", "LIDAR_FUSED5"),
            ("LIDAR_SLAM_CENTER",), ("LIDAR_SLAM_CENTER", "LIDAR_SLAM_FUSED5"),
        }:
            raise ValueError("enabled_depth_sources must select exactly one raw or SLAM LiDAR source family")
        object.__setattr__(self, "enabled_depth_sources", sources)
        if self.input_adapter == "prepared-scene-v2":
            if self.scene_dir is None or self.prepared_scene_dir is None:
                raise ValueError("Prepared Scene adapter requires scene_dir and prepared_scene_dir")
            if self.fused5_depth_dir is None or self.fused5_mask_dir is None:
                raise ValueError("Prepared Scene adapter requires fused5 artifact roots")
            if (self.raw_depth_dir is None) == (self.center_depth_dir is None):
                raise ValueError("Prepared Scene must configure exactly one of raw_depth_dir or center_depth_dir")
            if sources[0].startswith("LIDAR_SLAM") and self.center_depth_dir is None:
                raise ValueError("SLAM source profiles require source-neutral center_depth_dir")
        else:
            if self.rosbag_dir is None or self.calibration_path is None:
                raise ValueError("Streaming adapter requires rosbag_dir and calibration")
            if any(value is not None for value in (
                self.scene_dir, self.prepared_scene_dir, self.raw_depth_dir,
                self.center_depth_dir, self.fused5_depth_dir, self.fused5_mask_dir,
            )):
                raise ValueError("Streaming adapter cannot declare Prepared Scene artifact roots")
            if type(self.stream_queue_size) is not int or self.stream_queue_size < 1:
                raise ValueError("stream_queue_size must be a positive integer")
            if not self.preparation_profile or not self.source_mode or not self.fusion_policy:
                raise ValueError("Streaming adapter requires preparation_profile, source_mode and fusion_policy")
            expected = {
                "odin1-calibrated-native-v1": ("LIDAR_RAW", "raw-centered-5-v1"),
                "odin1-slam-world-native-v1": ("SLAM_WORLD", "slam-world-centered-5-v1"),
            }.get(self.preparation_profile)
            if expected is None or (self.source_mode, self.fusion_policy) != expected:
                raise ValueError("Streaming source profile, source_mode and fusion_policy do not match")
            expected_sources = {
                "LIDAR_RAW": {"LIDAR_RAW", "LIDAR_FUSED5"},
                "SLAM_WORLD": {"LIDAR_SLAM_CENTER", "LIDAR_SLAM_FUSED5"},
            }[self.source_mode]
            if not set(sources).issubset(expected_sources):
                raise ValueError("Streaming enabled_depth_sources do not match source_mode")
            if self.source_mode == "LIDAR_RAW" and (
                not self.confirm_raw_offset_time_seconds_from_scan_start
                or not self.confirm_base_from_lidar_identity
            ):
                raise ValueError("Formal RAW streaming requires both raw semantics confirmations")

    @property
    def center_depth_path(self) -> Path:
        path = self.center_depth_dir if self.center_depth_dir is not None else self.raw_depth_dir
        if path is None:
            raise ValueError("Streaming scene has no materialized center depth root")
        return path


@dataclass(frozen=True)
class GrowthSourcesConfig:
    spnet: "SPNetDisabledConfig | SPNetOnlineConfig" = field(default_factory=lambda: SPNetDisabledConfig())

    def __post_init__(self) -> None:
        if not isinstance(self.spnet, (SPNetDisabledConfig, SPNetOnlineConfig)):
            raise ValueError("growth_sources.spnet must be disabled or online")


@dataclass(frozen=True)
class SPNetDisabledConfig:
    mode: str = field(default="disabled", init=False)


@dataclass(frozen=True)
class SPNetOnlineConfig:
    model_id: str
    adapter_policy: str | None = None
    depth_scale_m: float = 200.0
    confidence: float = 0.4
    sample_stride: int = 8
    mode: str = field(default="online", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("SPNet model_id must be a non-empty logical identifier")
        if self.adapter_policy != NATIVE_FULL_FRAME_PAD_CROP_V1:
            raise ValueError("SPNet requires an explicit supported adapter_policy")
        if type(self.sample_stride) is not int or self.sample_stride < 1:
            raise ValueError("SPNet sample_stride must be a positive integer")
        try:
            depth_scale_m = float(self.depth_scale_m)
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("SPNet depth_scale_m and confidence must be numeric") from exc
        if not math.isfinite(depth_scale_m) or depth_scale_m <= 0:
            raise ValueError("SPNet depth_scale_m must be finite and positive")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("SPNet confidence must be finite and within [0, 1]")
        object.__setattr__(self, "depth_scale_m", depth_scale_m)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class GaussianInitializationConfig:
    opacity: float

    def __post_init__(self) -> None:
        try:
            opacity = float(self.opacity)
        except (TypeError, ValueError) as exc:
            raise ValueError("gaussian_initialization.opacity must be numeric") from exc
        if not math.isfinite(opacity) or not 0 < opacity < 1:
            raise ValueError("gaussian_initialization.opacity must be finite and within (0, 1)")
        object.__setattr__(self, "opacity", opacity)

    @property
    def opacity_logit(self) -> float:
        logit = math.log(self.opacity) - math.log1p(-self.opacity)
        if not math.isfinite(logit):
            raise ValueError("gaussian_initialization.opacity logit must be finite")
        return logit


@dataclass(frozen=True)
class ResidualThreshold:
    absolute_m: float
    relative: float

    def __post_init__(self) -> None:
        try:
            absolute_m = float(self.absolute_m)
            relative = float(self.relative)
        except (TypeError, ValueError) as exc:
            raise ValueError("Residual thresholds must be numeric") from exc
        if (not math.isfinite(absolute_m) or not math.isfinite(relative)
                or absolute_m < 0 or relative < 0):
            raise ValueError("Residual thresholds must be non-negative")
        object.__setattr__(self, "absolute_m", absolute_m)
        object.__setattr__(self, "relative", relative)


@dataclass(frozen=True)
class GrowthConfig:
    coverage_alpha_threshold: float = 0.99
    rgb_residual_threshold: float = 0.05
    min_candidate_depth_m: float = 0.1
    max_candidate_depth_m: float = 200.0
    candidate_duplicate_3d_threshold_m: float = 0.05
    confidence_thresholds: dict[str, float] | None = None
    residual_thresholds: dict[str, ResidualThreshold] | None = None
    depth_policy: str = RAW_ACCUMULATED_DEPTH_POLICY

    def __post_init__(self) -> None:
        if self.depth_policy != RAW_ACCUMULATED_DEPTH_POLICY:
            raise ValueError(f"growth depth_policy must be {RAW_ACCUMULATED_DEPTH_POLICY}")
        confidence = self.confidence_thresholds if self.confidence_thresholds is not None else {
            descriptor.name: descriptor.min_candidate_confidence
            for descriptor in SOURCE_DESCRIPTORS
        }
        residual = self.residual_thresholds if self.residual_thresholds is not None else {
            "LIDAR_RAW": ResidualThreshold(0.15, 0.03),
            "LIDAR_FUSED5": ResidualThreshold(0.25, 0.05),
            "SPNET_BLIND": ResidualThreshold(0.35, 0.07),
        }
        if not isinstance(confidence, dict):
            raise ValueError("confidence_thresholds must be an object")
        try:
            confidence = {str(name): float(value) for name, value in confidence.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence_thresholds must contain numeric values") from exc
        if (frozenset(confidence) not in {frozenset(SOURCE_NAMES), frozenset(SLAM_SOURCE_NAMES)}
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidence.values())):
            raise ValueError("confidence_thresholds must define all sources within [0, 1]")
        if not isinstance(residual, dict) or frozenset(residual) not in {frozenset(SOURCE_NAMES), frozenset(SLAM_SOURCE_NAMES)}:
            raise ValueError("residual_thresholds must define all sources")
        if not all(isinstance(value, ResidualThreshold) for value in residual.values()):
            raise ValueError("residual_thresholds must contain threshold objects")
        if not 0 <= self.coverage_alpha_threshold <= 1 or self.rgb_residual_threshold < 0:
            raise ValueError("Growth rendered gates are invalid")
        if self.min_candidate_depth_m <= 0 or self.max_candidate_depth_m < self.min_candidate_depth_m:
            raise ValueError("Candidate depth range is invalid")
        if self.candidate_duplicate_3d_threshold_m <= 0:
            raise ValueError("Candidate duplicate threshold must be positive")
        object.__setattr__(self, "confidence_thresholds", dict(confidence))
        object.__setattr__(self, "residual_thresholds", dict(residual))


@dataclass(frozen=True)
class PruningConfig:
    opacity_thresholds: dict[str, float] | None = None
    spnet_min_prune_age: int = 1

    def __post_init__(self) -> None:
        values = self.opacity_thresholds if self.opacity_thresholds is not None else {
            descriptor.name: descriptor.default_opacity_threshold
            for descriptor in SOURCE_DESCRIPTORS
        }
        if not isinstance(values, dict):
            raise ValueError("opacity_thresholds must be an object")
        try:
            values = {str(name): float(value) for name, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("opacity_thresholds must contain numeric values") from exc
        if (frozenset(values) not in {frozenset(SOURCE_NAMES), frozenset(SLAM_SOURCE_NAMES)}
                or any(not math.isfinite(value) or value < 0 for value in values.values())):
            raise ValueError("opacity_thresholds must define all sources with non-negative values")
        object.__setattr__(self, "opacity_thresholds", dict(values))
        if type(self.spnet_min_prune_age) is not int or self.spnet_min_prune_age < 1:
            raise ValueError("spnet_min_prune_age must be a positive integer")


@dataclass(frozen=True)
class MappingLossConfig:
    variant: str = FROZEN_MAPPING_LOSS_VARIANT
    image_weight: float = 1.0
    ssim_weight: float = 0.2
    depth_weight: float = 0.005
    alpha_support_a0: float = 0.01

    def __post_init__(self) -> None:
        if self.variant != FROZEN_MAPPING_LOSS_VARIANT:
            raise ValueError(f"SAGE-GS v1 requires loss variant {FROZEN_MAPPING_LOSS_VARIANT}")
        for name in ("image_weight", "ssim_weight", "depth_weight", "alpha_support_a0"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.image_weight <= 0:
            raise ValueError("image_weight must be positive")
        if not 0 <= self.ssim_weight <= 1:
            raise ValueError("ssim_weight must be within [0, 1]")
        if self.depth_weight < 0:
            raise ValueError("depth_weight must be non-negative")
        if not 0 < self.alpha_support_a0 <= 1:
            raise ValueError("alpha_support_a0 must be finite and within (0, 1]")


@dataclass(frozen=True)
class MappingConfig:
    frame_policy: str = ALL_ACCEPTED_FRAME_POLICY
    map_every: int = 1
    keyframe_every: int = 5
    iterations: int = 60
    prune_every: int = 20
    prune_stop_after: int = 20
    learning_rates: dict[str, float] | None = None
    optimization_variant: str = ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT
    evaluation_depth_policy: str = ALPHA_NORMALIZED_DEPTH_POLICY
    evaluation_min_alpha: float = 0.01
    evaluation_epsilon: float = 1e-6
    evaluation_alpha_support_a0: float = 0.01
    evaluation_hit_target_center: float = 0.5
    evaluation_hit_target_fused5: float = 0.35

    def __post_init__(self) -> None:
        if self.frame_policy != ALL_ACCEPTED_FRAME_POLICY:
            raise ValueError(f"SAGE-GS v1 requires frame_policy={ALL_ACCEPTED_FRAME_POLICY}")
        if self.evaluation_depth_policy not in {
            RAW_ACCUMULATED_DEPTH_POLICY, ALPHA_NORMALIZED_DEPTH_POLICY,
        }:
            raise ValueError("evaluation depth policy is unsupported")
        for name in (
            "evaluation_min_alpha",
            "evaluation_alpha_support_a0",
            "evaluation_hit_target_center",
            "evaluation_hit_target_fused5",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and within (0, 1]")
            object.__setattr__(self, name, value)
        epsilon = float(self.evaluation_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("evaluation_epsilon must be finite and positive")
        object.__setattr__(self, "evaluation_epsilon", epsilon)
        if self.optimization_variant != ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT:
            raise ValueError(
                f"SAGE-GS v1 requires optimization_variant={ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT}"
            )
        if any(value < 1 for value in (self.map_every, self.keyframe_every, self.iterations)):
            raise ValueError("Mapping intervals and iterations must be positive")
        if self.prune_every < 1 or self.prune_stop_after < 0:
            raise ValueError("Pruning schedule is invalid")
        rates = self.learning_rates if self.learning_rates is not None else {
            "means3d": 0.00016, "colors": 0.0025, "opacity_logits": 0.05,
            "log_scales": 0.001, "rotations": 0.001,
        }
        if not isinstance(rates, dict):
            raise ValueError("learning_rates must be an object")
        try:
            rates = {str(name): float(value) for name, value in rates.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("learning_rates must contain numeric values") from exc
        expected = {"means3d", "colors", "opacity_logits", "log_scales", "rotations"}
        if (set(rates) != expected
                or any(not math.isfinite(value) or value <= 0 for value in rates.values())):
            raise ValueError("learning_rates must define all Gaussian parameters with positive values")
        object.__setattr__(self, "learning_rates", dict(rates))


@dataclass(frozen=True)
class SageConfig:
    config_path: Path
    output_dir: Path
    seed: int
    scene: SceneConfig
    growth_sources: GrowthSourcesConfig
    growth: GrowthConfig
    pruning: PruningConfig
    mapping: MappingConfig
    gaussian_initialization: GaussianInitializationConfig
    loss: MappingLossConfig
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Live config must use schema_version={SCHEMA_VERSION}")
        descriptors = (
            SLAM_SOURCE_DESCRIPTORS
            if self.scene.enabled_depth_sources[0] == "LIDAR_SLAM_CENTER"
            else SOURCE_DESCRIPTORS
        )
        confidence = materialize_source_policy(self.growth.confidence_thresholds, descriptors)
        residuals = materialize_source_policy(self.growth.residual_thresholds, descriptors)
        opacity = materialize_source_policy(self.pruning.opacity_thresholds, descriptors)
        if confidence != self.growth.confidence_thresholds or residuals != self.growth.residual_thresholds:
            object.__setattr__(self, "growth", replace(
                self.growth,
                confidence_thresholds=confidence,
                residual_thresholds=residuals,
            ))
        if opacity != self.pruning.opacity_thresholds:
            object.__setattr__(self, "pruning", PruningConfig(
                opacity_thresholds=opacity,
                spnet_min_prune_age=self.pruning.spnet_min_prune_age,
            ))

    def manifest_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return convert(asdict(value)) if hasattr(value, "__dataclass_fields__") else value

        payload = {"schema_version": self.schema_version, "run": {"output_dir": self.output_dir, "seed": self.seed}, "scene": self.scene, "growth_sources": self.growth_sources, "growth": self.growth, "pruning": self.pruning, "mapping": self.mapping}
        payload["gaussian_initialization"] = self.gaussian_initialization
        payload["loss"] = self.loss
        converted = convert(payload)
        scene = converted["scene"]
        if scene.get("center_depth_dir") is None:
            scene.pop("center_depth_dir", None)
        else:
            scene.pop("raw_depth_dir", None)
        if self.scene.input_adapter == "prepared-scene-v2":
            for name in (
                "input_adapter", "rosbag_dir", "calibration_path", "write_through_dir",
                "stream_queue_size", "preparation_profile", "source_mode", "fusion_policy",
                "confirm_raw_offset_time_seconds_from_scan_start",
                "confirm_base_from_lidar_identity",
            ):
                scene.pop(name, None)
        else:
            scene["calibration"] = scene.pop("calibration_path")
            for name in ("scene_dir", "prepared_scene_dir", "raw_depth_dir", "center_depth_dir", "fused5_depth_dir", "fused5_mask_dir"):
                if scene.get(name) is None:
                    scene.pop(name, None)
        return converted
