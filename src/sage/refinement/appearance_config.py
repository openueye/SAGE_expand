"""Strict configuration for the published SAGE refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .dense_geometry_config import (
    DENSE_ALIGNMENT_NONWORSENING_V2,
    DensePriorPolicy,
)


APPEARANCE_REFINEMENT_SCHEMA = "sage-refinement-v1"
SEEDED_RANDOM_MAPPING_FRAME_VARIANT = "seeded-random-mapping-frame-v1"
FIRST_MAPPING_FRAME_EXPOSURE_GAUGE = (
    "first-mapping-frame-identity-v1"
)
SCALAR_EXPOSURE_NUISANCE_SCHEMA = "sage-scalar-exposure-nuisance-v1"

_FIELDS = {
    "schema_version",
    "iterations",
    "seed",
    "sampling_variant",
    "color_learning_rate",
    "opacity_learning_rate",
    "means3d_learning_rate",
    "log_scales_learning_rate",
    "rotations_learning_rate",
    "opacity_anchor_weight",
    "lidar_depth_weight",
    "dense_normal_weight",
    "dense_normal_edge_weight_gamma",
    "dense_normal_alpha_support_a0",
    "dense_normal_max_relative_depth_jump",
    "dense_prior",
    "exposure_learning_rate",
    "exposure_log_gain_bound",
    "exposure_bias_bound",
    "exposure_anchor_weight",
    "exposure_gauge_variant",
    "ssim_weight",
    "milestone_steps",
}
_GEOMETRY_LEARNING_RATE_FIELDS = {
    "means3d_learning_rate",
    "log_scales_learning_rate",
    "rotations_learning_rate",
}


def _finite_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Appearance refinement {name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Appearance refinement {name} must be numeric"
        ) from exc
    if (
        not math.isfinite(number)
        or number < 0
        or (positive and number == 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(
            f"Appearance refinement {name} must be finite and {qualifier}"
        )
    return number


@dataclass(frozen=True)
class AppearanceRefinementConfig:
    schema_version: str
    iterations: int
    seed: int
    sampling_variant: str
    color_learning_rate: float
    opacity_learning_rate: float
    means3d_learning_rate: float
    log_scales_learning_rate: float
    rotations_learning_rate: float
    opacity_anchor_weight: float
    lidar_depth_weight: float
    dense_normal_weight: float
    dense_normal_edge_weight_gamma: float
    dense_normal_alpha_support_a0: float
    dense_normal_max_relative_depth_jump: float
    dense_prior: DensePriorPolicy
    exposure_learning_rate: float
    exposure_log_gain_bound: float
    exposure_bias_bound: float
    exposure_anchor_weight: float
    exposure_gauge_variant: str
    ssim_weight: float
    milestone_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != APPEARANCE_REFINEMENT_SCHEMA:
            raise ValueError(
                "Appearance refinement schema_version is unsupported"
            )
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError(
                "Appearance refinement iterations must be a positive integer"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError(
                "Appearance refinement seed must be a non-negative integer"
            )
        if self.sampling_variant != SEEDED_RANDOM_MAPPING_FRAME_VARIANT:
            raise ValueError(
                "Appearance refinement sampling_variant must be "
                f"{SEEDED_RANDOM_MAPPING_FRAME_VARIANT}"
            )
        positive_fields = (
            "color_learning_rate",
            "opacity_learning_rate",
            "opacity_anchor_weight",
            "lidar_depth_weight",
            "dense_normal_weight",
            "dense_normal_edge_weight_gamma",
            "dense_normal_alpha_support_a0",
            "dense_normal_max_relative_depth_jump",
            "exposure_learning_rate",
            "exposure_log_gain_bound",
            "exposure_bias_bound",
        )
        for name in positive_fields:
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name, positive=True),
            )
        for name in _GEOMETRY_LEARNING_RATE_FIELDS:
            object.__setattr__(
                self,
                name,
                _finite_number(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "exposure_anchor_weight",
            _finite_number(
                self.exposure_anchor_weight,
                "exposure_anchor_weight",
            ),
        )
        ssim_weight = _finite_number(self.ssim_weight, "ssim_weight")
        if ssim_weight > 1:
            raise ValueError(
                "Appearance refinement ssim_weight must be within [0, 1]"
            )
        object.__setattr__(self, "ssim_weight", ssim_weight)
        if not 0 < self.dense_normal_alpha_support_a0 <= 1:
            raise ValueError(
                "Appearance refinement dense_normal_alpha_support_a0 must "
                "be within (0, 1]"
            )
        if (
            not isinstance(self.dense_prior, DensePriorPolicy)
            or self.dense_prior.alignment_variant
            != DENSE_ALIGNMENT_NONWORSENING_V2
        ):
            raise ValueError(
                "Appearance refinement requires the nonworsening aligned "
                "dense prior"
            )
        if (
            self.exposure_gauge_variant
            != FIRST_MAPPING_FRAME_EXPOSURE_GAUGE
        ):
            raise ValueError(
                "Appearance refinement requires the first-mapping-frame "
                "exposure gauge"
            )
        milestones = tuple(self.milestone_steps)
        if (
            any(type(step) is not int for step in milestones)
            or tuple(sorted(set(milestones))) != milestones
            or any(step < 1 or step > self.iterations for step in milestones)
            or not milestones
            or milestones[-1] != self.iterations
        ):
            raise ValueError(
                "Appearance refinement milestone_steps must be sorted, "
                "unique, within [1, iterations], and end at iterations"
            )
        object.__setattr__(self, "milestone_steps", milestones)

    @property
    def optimized_parameters(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, learning_rate in self.learning_rates.items()
            if learning_rate > 0
        )

    @property
    def learning_rates(self) -> dict[str, float]:
        return {
            "means3d": self.means3d_learning_rate,
            "colors": self.color_learning_rate,
            "opacity_logits": self.opacity_learning_rate,
            "log_scales": self.log_scales_learning_rate,
            "rotations": self.rotations_learning_rate,
        }

    @property
    def nuisance_parameters(self) -> tuple[str, ...]:
        return ("exposure_log_gains", "exposure_biases")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
    ) -> "AppearanceRefinementConfig":
        if not isinstance(payload, dict):
            raise ValueError("Appearance refinement config must be an object")
        unknown = sorted(set(payload) - _FIELDS)
        missing = sorted(
            _FIELDS - _GEOMETRY_LEARNING_RATE_FIELDS - set(payload)
        )
        if unknown:
            raise ValueError(
                "Unknown appearance refinement fields: "
                + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "Appearance refinement is missing required fields: "
                + ", ".join(missing)
            )
        normalized_payload = {
            **{name: 0.0 for name in _GEOMETRY_LEARNING_RATE_FIELDS},
            **payload,
        }
        milestones = normalized_payload["milestone_steps"]
        dense_prior = normalized_payload["dense_prior"]
        if not isinstance(milestones, list):
            raise ValueError(
                "Appearance refinement milestone_steps must be a list"
            )
        if not isinstance(dense_prior, dict):
            raise ValueError(
                "Appearance refinement dense_prior must be an object"
            )
        try:
            policy = DensePriorPolicy(**dense_prior)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Appearance refinement dense_prior is invalid"
            ) from exc
        return cls(
            **{
                **normalized_payload,
                "dense_prior": policy,
                "milestone_steps": tuple(milestones),
            }
        )

    @classmethod
    def load(cls, path: Path) -> "AppearanceRefinementConfig":
        source = Path(path)
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Cannot read appearance refinement config: {source}"
            ) from exc
        return cls.from_payload(payload)

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["milestone_steps"] = list(self.milestone_steps)
        return payload


def refinement_config_identity(
    training_identity: str,
    refinement: AppearanceRefinementConfig,
) -> str:
    """Digest binding a mapping checkpoint's identity to the refinement config.

    Reuse checks must invalidate a final output whenever either the mapping
    stage identity or the appearance refinement parameters (dense prior,
    learning rates, milestones, ...) change.
    """
    payload = {
        "training_config_identity": training_identity,
        "refinement_config": refinement.payload(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
