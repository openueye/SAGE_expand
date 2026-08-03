from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .source_policy import (
    SOURCE_DESCRIPTORS,
    SOURCE_NAMES,
    materialize_source_policy,
)


SCHEMA_VERSION = "sage-gs-v1"
ODIN_GLOBAL_CURRENT_ANCHORED_VARIANT = "odin-global-current-anchored-v1"
NATIVE_FULL_FRAME_PAD_CROP_V1 = "native-full-frame-pad-crop-v1"
RAW_ACCUMULATED_DEPTH_POLICY = "raw-accumulated-v1"
ALPHA_NORMALIZED_DEPTH_POLICY = "alpha-normalized-v1"
FROZEN_MAPPING_LOSS_VARIANT = "loss-v3-alpha-detached-support-coverage-v1"
ALL_ACCEPTED_FRAME_POLICY = "all-accepted"
ALL_ACCEPTED_FRAME_LIMIT = -1




@dataclass(frozen=True)
class InputConfig:
    """The `input:` section, verbatim, plus where relative paths resolve from.

    SAGE Core never reads this: it hands the payload to the adapter factory and
    then works only with the resolved canonical frames.
    """

    payload: Mapping[str, Any]
    base_dir: Path
    require_clean_worktree: bool = False
    prefetch_depth: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping) or not isinstance(self.payload.get("type"), str):
            raise ValueError("input must be an object declaring a type")
        if type(self.require_clean_worktree) is not bool:
            raise ValueError("require_clean_worktree must be a JSON boolean")
        if type(self.prefetch_depth) is not int or self.prefetch_depth < 1:
            raise ValueError("input prefetch_depth must be a positive integer")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(self, "base_dir", Path(self.base_dir).resolve())

    @property
    def adapter_type(self) -> str:
        return str(self.payload["type"])

    def create_adapter(self):
        from ..input.factory import create_input_adapter

        return create_input_adapter(self.payload, base_dir=self.base_dir)

    def manifest_payload(self) -> dict[str, Any]:
        return {"base_dir": str(self.base_dir), **_jsonable(dict(self.payload))}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class GrowthSourcesConfig:
    spnet: "SPNetOnlineConfig"

    def __post_init__(self) -> None:
        if not isinstance(self.spnet, SPNetOnlineConfig):
            raise ValueError("growth_sources.spnet must be online")


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
    scale_clamp_min: float = 1e-4
    initial_scale_anisotropy: tuple[float, float, float] = (0.95, 1.05, 1.20)

    def __post_init__(self) -> None:
        try:
            opacity = float(self.opacity)
        except (TypeError, ValueError) as exc:
            raise ValueError("gaussian_initialization.opacity must be numeric") from exc
        if not math.isfinite(opacity) or not 0 < opacity < 1:
            raise ValueError("gaussian_initialization.opacity must be finite and within (0, 1)")
        try:
            scale_clamp_min = float(self.scale_clamp_min)
        except (TypeError, ValueError) as exc:
            raise ValueError("gaussian_initialization.scale_clamp_min must be numeric") from exc
        if not math.isfinite(scale_clamp_min) or scale_clamp_min <= 0:
            raise ValueError("gaussian_initialization.scale_clamp_min must be finite and positive")
        try:
            anisotropy = tuple(float(value) for value in self.initial_scale_anisotropy)
        except (TypeError, ValueError) as exc:
            raise ValueError("gaussian_initialization.initial_scale_anisotropy must be three positive floats") from exc
        if len(anisotropy) != 3 or any(not math.isfinite(value) or value <= 0 for value in anisotropy):
            raise ValueError("gaussian_initialization.initial_scale_anisotropy must be three positive finite values")
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "scale_clamp_min", scale_clamp_min)
        object.__setattr__(self, "initial_scale_anisotropy", anisotropy)

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
    coverage_alpha_threshold: float = 0.85
    min_candidate_depth_m: float = 0.1
    max_candidate_depth_m: float = 200.0
    candidate_duplicate_3d_threshold_m: float = 0.05
    confidence_thresholds: dict[str, float] | None = None
    residual_thresholds: dict[str, ResidualThreshold] | None = None
    depth_policy: str = ALPHA_NORMALIZED_DEPTH_POLICY
    frame_bias_threshold: ResidualThreshold = ResidualThreshold(0.10, 0.02)
    frame_bias_min_overlap_px: int = 50
    # 每个 mapping commit 每个源允许接受的候选上限；None 表示不限量（历史行为）。
    # 必须按源给配额而不是给一个全局总量：共享总量时优先级会退化成词典序，
    # 低优先级源（SPNet）会被 LiDAR 吃光额度而饿死。
    max_new_per_commit: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.depth_policy != ALPHA_NORMALIZED_DEPTH_POLICY:
            raise ValueError(
                "growth depth_policy must be "
                f"{ALPHA_NORMALIZED_DEPTH_POLICY}"
            )
        confidence = self.confidence_thresholds if self.confidence_thresholds is not None else {
            descriptor.name: descriptor.min_candidate_confidence
            for descriptor in SOURCE_DESCRIPTORS
        }
        residual = self.residual_thresholds if self.residual_thresholds is not None else {
            "LIDAR_CENTER": ResidualThreshold(0.15, 0.03),
            "LIDAR_FUSED": ResidualThreshold(0.25, 0.05),
            "SPNET_COMPLETED": ResidualThreshold(0.35, 0.07),
        }
        if not isinstance(confidence, dict):
            raise ValueError("confidence_thresholds must be an object")
        try:
            confidence = {str(name): float(value) for name, value in confidence.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence_thresholds must contain numeric values") from exc
        if (frozenset(confidence) != frozenset(SOURCE_NAMES)
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidence.values())):
            raise ValueError("confidence_thresholds must define all sources within [0, 1]")
        if not isinstance(residual, dict) or frozenset(residual) != frozenset(SOURCE_NAMES):
            raise ValueError("residual_thresholds must define all sources")
        if not all(isinstance(value, ResidualThreshold) for value in residual.values()):
            raise ValueError("residual_thresholds must contain threshold objects")
        if not 0 <= self.coverage_alpha_threshold <= 1:
            raise ValueError("Growth rendered gates are invalid")
        if self.min_candidate_depth_m <= 0 or self.max_candidate_depth_m < self.min_candidate_depth_m:
            raise ValueError("Candidate depth range is invalid")
        if self.candidate_duplicate_3d_threshold_m <= 0:
            raise ValueError("Candidate duplicate threshold must be positive")
        if not isinstance(self.frame_bias_threshold, ResidualThreshold):
            raise ValueError("frame_bias_threshold must be a ResidualThreshold")
        if type(self.frame_bias_min_overlap_px) is not int or self.frame_bias_min_overlap_px < 1:
            raise ValueError("frame_bias_min_overlap_px must be a positive integer")
        quotas = self.max_new_per_commit
        if quotas is not None:
            if (not isinstance(quotas, dict)
                    or frozenset(quotas) != frozenset(SOURCE_NAMES)
                    or any(type(value) is not int or value < 0 for value in quotas.values())):
                raise ValueError(
                    "max_new_per_commit must define a non-negative integer quota for all sources, or be null"
                )
        object.__setattr__(self, "confidence_thresholds", dict(confidence))
        object.__setattr__(self, "residual_thresholds", dict(residual))


@dataclass(frozen=True)
class PruningConfig:
    opacity_thresholds: dict[str, float] | None = None
    spnet_min_prune_age: int = 1
    spnet_scale_ceiling_m: float | None = None

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
        if (frozenset(values) != frozenset(SOURCE_NAMES)
                or any(not math.isfinite(value) or value < 0 for value in values.values())):
            raise ValueError("opacity_thresholds must define all sources with non-negative values")
        object.__setattr__(self, "opacity_thresholds", dict(values))
        if type(self.spnet_min_prune_age) is not int or self.spnet_min_prune_age < 1:
            raise ValueError("spnet_min_prune_age must be a positive integer")
        if self.spnet_scale_ceiling_m is not None:
            ceiling = float(self.spnet_scale_ceiling_m)
            if not math.isfinite(ceiling) or ceiling <= 0:
                raise ValueError("spnet_scale_ceiling_m must be finite and positive")
            object.__setattr__(self, "spnet_scale_ceiling_m", ceiling)


@dataclass(frozen=True)
class MappingLossConfig:
    variant: str = FROZEN_MAPPING_LOSS_VARIANT
    image_weight: float = 1.0
    ssim_weight: float = 0.2
    depth_weight: float = 0.005
    depth_coverage_weight: float = 0.05
    alpha_support_a0: float = 0.85
    depth_coverage_threshold: float = 0.85
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.variant != FROZEN_MAPPING_LOSS_VARIANT:
            raise ValueError(f"SAGE-GS v1 requires loss variant {FROZEN_MAPPING_LOSS_VARIANT}")
        for name in (
            "image_weight",
            "ssim_weight",
            "depth_weight",
            "depth_coverage_weight",
            "alpha_support_a0",
            "depth_coverage_threshold",
            "epsilon",
        ):
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
        if self.depth_coverage_weight < 0:
            raise ValueError("depth_coverage_weight must be non-negative")
        if not 0 < self.alpha_support_a0 <= 1:
            raise ValueError("alpha_support_a0 must be finite and within (0, 1]")
        if not 0 < self.depth_coverage_threshold <= 1:
            raise ValueError(
                "depth_coverage_threshold must be finite and within (0, 1]"
            )
        if self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")


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
    evaluation_hit_target_fused: float = 0.35

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
            "evaluation_hit_target_fused",
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
    input: InputConfig
    growth_sources: GrowthSourcesConfig
    growth: GrowthConfig
    pruning: PruningConfig
    mapping: MappingConfig
    gaussian_initialization: GaussianInitializationConfig
    loss: MappingLossConfig
    model_root: Path | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Live config must use schema_version={SCHEMA_VERSION}")
        if self.model_root is not None:
            object.__setattr__(self, "model_root", Path(self.model_root).expanduser().resolve())
        confidence = materialize_source_policy(self.growth.confidence_thresholds, SOURCE_DESCRIPTORS)
        residuals = materialize_source_policy(self.growth.residual_thresholds, SOURCE_DESCRIPTORS)
        opacity = materialize_source_policy(self.pruning.opacity_thresholds, SOURCE_DESCRIPTORS)
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
                spnet_scale_ceiling_m=self.pruning.spnet_scale_ceiling_m,
            ))

    def training_config_identity(self) -> str:
        """Digest of everything except the input.

        Checkpoint reuse must survive swapping a ROSBAG for the Prepared Scene
        exported from it, so the input section is deliberately excluded: the
        canonical input identities cover that side separately.
        """
        payload = {
            name: value for name, value in self.manifest_dict().items()
            if name not in {"input", "run"}
        }
        payload["seed"] = self.seed
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    def manifest_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return convert(asdict(value)) if hasattr(value, "__dataclass_fields__") else value

        payload = {
            "schema_version": self.schema_version,
            "run": {"output_dir": self.output_dir, "seed": self.seed},
            "runtime": {
                "model_root": self.model_root,
                "require_clean_worktree": self.input.require_clean_worktree,
            },
            "input": self.input.manifest_payload(),
            "growth_sources": self.growth_sources,
            "growth": self.growth,
            "pruning": self.pruning,
            "mapping": self.mapping,
            "gaussian_initialization": self.gaussian_initialization,
            "loss": self.loss,
        }
        return convert(payload)
