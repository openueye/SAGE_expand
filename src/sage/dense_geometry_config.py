"""Configuration policies for the published dense-geometry constraint."""

from __future__ import annotations

from dataclasses import dataclass
import math


DENSE_ALIGNMENT_NONWORSENING_V2 = (
    "identity-nonworsening-bilinear-scale-grid-v2"
)


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


def _positive(value: object, name: str) -> float:
    number = _nonnegative(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DensePriorPolicy:
    grid_height: int
    grid_width: int
    min_lidar_support: int
    confidence_exponent: float
    robust_epsilon_m: float
    alignment_variant: str = DENSE_ALIGNMENT_NONWORSENING_V2

    def __post_init__(self) -> None:
        for name in ("grid_height", "grid_width", "min_lidar_support"):
            object.__setattr__(
                self,
                name,
                _positive_integer(
                    getattr(self, name),
                    f"dense_prior.{name}",
                ),
            )
        for name in ("confidence_exponent", "robust_epsilon_m"):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), f"dense_prior.{name}"),
            )
        if self.alignment_variant != DENSE_ALIGNMENT_NONWORSENING_V2:
            raise ValueError(
                "dense_prior.alignment_variant must use the published "
                "nonworsening alignment"
            )


@dataclass(frozen=True)
class DenseGeometryPolicy:
    dense_depth_weight: float
    dense_normal_weight: float
    normal_smoothness_weight: float
    edge_weight_gamma: float
    alpha_support_a0: float
    max_relative_depth_jump: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "dense_depth_weight",
            "dense_normal_weight",
            "normal_smoothness_weight",
            "edge_weight_gamma",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), f"geometry.{name}"),
            )
        alpha = _positive(
            self.alpha_support_a0,
            "geometry.alpha_support_a0",
        )
        if alpha > 1:
            raise ValueError(
                "geometry.alpha_support_a0 must be within (0, 1]"
            )
        object.__setattr__(self, "alpha_support_a0", alpha)
        if self.max_relative_depth_jump is not None:
            object.__setattr__(
                self,
                "max_relative_depth_jump",
                _positive(
                    self.max_relative_depth_jump,
                    "geometry.max_relative_depth_jump",
                ),
            )

    @property
    def total_weight(self) -> float:
        return (
            self.dense_depth_weight
            + self.dense_normal_weight
            + self.normal_smoothness_weight
        )
