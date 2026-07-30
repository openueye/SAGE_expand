"""Deterministic appearance optimization for SAGE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .appearance_config import (
    SCALAR_EXPOSURE_NUISANCE_SCHEMA,
    AppearanceRefinementConfig,
)
from .model import TrainableGaussians


@dataclass(frozen=True)
class AppearanceRefinementResult:
    steps: tuple[dict[str, float | int], ...]
    milestones: tuple[dict[str, object], ...]
    selected_frame_indices: tuple[int, ...]
    exposure_nuisance: dict[str, object] | None = None


@dataclass(frozen=True)
class AppearanceObjective:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.total, torch.Tensor)
            or self.total.ndim != 0
            or not isinstance(self.terms, dict)
            or not self.terms
            or "photo" not in self.terms
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, torch.Tensor)
                or value.ndim != 0
                for name, value in self.terms.items()
            )
            or not isinstance(self.diagnostics, dict)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, torch.Tensor)
                or value.ndim != 0
                for name, value in self.diagnostics.items()
            )
        ):
            raise ValueError(
                "Appearance objective requires a scalar total and named "
                "scalar terms including photo"
            )


def _distribution(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "p50": float(torch.quantile(flat, 0.5)),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def _signed_distribution(
    values: torch.Tensor,
) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


class AppearanceExposureNuisance(torch.nn.Module):
    """Bounded per-frame scalar exposure outside the Gaussian map."""

    def __init__(
        self,
        frame_indices: tuple[int, ...],
        config: AppearanceRefinementConfig,
        *,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        if (
            not frame_indices
            or len(frame_indices) != len(set(frame_indices))
            or any(
                type(index) is not int or index < 0
                for index in frame_indices
            )
        ):
            raise ValueError(
                "Scalar exposure nuisance requires unique non-negative "
                "frame indices"
            )
        self.config = config
        self.frame_indices = tuple(frame_indices)
        self.reference_frame_index = frame_indices[0]
        self._position = {
            frame_index: position
            for position, frame_index in enumerate(frame_indices)
        }
        count = len(frame_indices) - 1
        self.raw_log_gains = torch.nn.Parameter(
            torch.zeros(count, dtype=torch.float32, device=device)
        )
        self.raw_biases = torch.nn.Parameter(
            torch.zeros(count, dtype=torch.float32, device=device)
        )

    def values(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            position = self._position[frame_index]
        except KeyError as exc:
            raise ValueError(
                f"Exposure nuisance does not own frame {frame_index}"
            ) from exc
        if position == 0:
            zero = self.raw_log_gains.new_zeros(())
            return zero, zero
        nuisance_position = position - 1
        log_gain = self.config.exposure_log_gain_bound * torch.tanh(
            self.raw_log_gains[nuisance_position]
        )
        bias = self.config.exposure_bias_bound * torch.tanh(
            self.raw_biases[nuisance_position]
        )
        return log_gain, bias

    def apply(
        self,
        image: torch.Tensor,
        frame_index: int,
    ) -> torch.Tensor:
        log_gain, bias = self.values(frame_index)
        return torch.exp(log_gain) * image + bias

    def _effective_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self.raw_log_gains.new_zeros(1)
        log_gains = torch.cat((
            zero,
            self.config.exposure_log_gain_bound
            * torch.tanh(self.raw_log_gains),
        ))
        biases = torch.cat((
            zero,
            self.config.exposure_bias_bound * torch.tanh(self.raw_biases),
        ))
        return log_gains, biases

    def anchor_loss(self) -> torch.Tensor:
        log_gains, biases = self._effective_values()
        return log_gains.square().mean() + biases.square().mean()

    def summary(self) -> dict[str, object]:
        log_gains, biases = self._effective_values()
        return {
            "schema_version": SCALAR_EXPOSURE_NUISANCE_SCHEMA,
            "frame_count": len(self.frame_indices),
            "reference_frame_index": self.reference_frame_index,
            "log_gain": _signed_distribution(log_gains),
            "gain": _signed_distribution(torch.exp(log_gains)),
            "bias": _signed_distribution(biases),
        }

    def record(self) -> dict[str, object]:
        log_gains, biases = self._effective_values()
        log_gain_values = log_gains.detach().cpu().tolist()
        bias_values = biases.detach().cpu().tolist()
        return {
            "schema_version": SCALAR_EXPOSURE_NUISANCE_SCHEMA,
            "gauge_variant": self.config.exposure_gauge_variant,
            "reference_frame_index": self.reference_frame_index,
            "log_gain_bound": self.config.exposure_log_gain_bound,
            "bias_bound": self.config.exposure_bias_bound,
            "anchor_weight": self.config.exposure_anchor_weight,
            "frames": [
                {
                    "frame_index": frame_index,
                    "log_gain": float(log_gain),
                    "gain": float(np.exp(log_gain)),
                    "bias": float(bias),
                    "reference": position == 0,
                }
                for position, (frame_index, log_gain, bias) in enumerate(
                    zip(
                        self.frame_indices,
                        log_gain_values,
                        bias_values,
                    )
                )
            ],
        }


class AppearanceRefiner:
    """Optimize configured appearance fields while freezing geometry and poses."""

    def __init__(self, config: AppearanceRefinementConfig) -> None:
        self.config = config

    def optimize(
        self,
        model: TrainableGaussians,
        frame_indices: tuple[int, ...],
        objective: Callable[
            [TrainableGaussians, int, int],
            torch.Tensor | AppearanceObjective,
        ],
        *,
        exposure_nuisance: AppearanceExposureNuisance | None = None,
        milestone_callback: Callable[
            [TrainableGaussians, dict[str, object], dict[str, object]],
            None,
        ]
        | None = None,
    ) -> AppearanceRefinementResult:
        if (
            not frame_indices
            or len(frame_indices) != len(set(frame_indices))
            or any(type(index) is not int or index < 0 for index in frame_indices)
        ):
            raise ValueError(
                "Appearance refinement frame_indices must be unique "
                "non-negative integers"
            )
        if (
            exposure_nuisance is None
            or exposure_nuisance.frame_indices != frame_indices
        ):
            raise ValueError(
                "SAGE requires scalar exposure nuisance for the "
                "exact mapping frame set"
            )

        random = np.random.default_rng(self.config.seed)
        selected = tuple(
            frame_indices[int(random.integers(0, len(frame_indices)))]
            for _ in range(self.config.iterations)
        )
        baseline_colors = model.colors.detach().clone()
        baseline_opacities = model.opacities.detach().clone()
        original_requires_grad = {
            name: getattr(model, name).requires_grad
            for name in model.PARAMETER_NAMES
        }
        learning_rates = {
            "colors": self.config.color_learning_rate,
            "opacity_logits": self.config.opacity_learning_rate,
        }
        parameter_groups = [
                {
                    "params": [getattr(model, name)],
                    "lr": learning_rates[name],
                    "name": name,
                }
                for name in self.config.optimized_parameters
        ]
        if exposure_nuisance is not None:
            parameter_groups.extend((
                {
                    "params": [exposure_nuisance.raw_log_gains],
                    "lr": self.config.exposure_learning_rate,
                    "name": "exposure_log_gains",
                },
                {
                    "params": [exposure_nuisance.raw_biases],
                    "lr": self.config.exposure_learning_rate,
                    "name": "exposure_biases",
                },
            ))
        optimizer = torch.optim.Adam(
            parameter_groups,
            lr=0.0,
            eps=1e-15,
        )
        # Full per-step histories force several CUDA-to-host synchronizations
        # and are not consumed by training or evaluation. Keep milestone
        # diagnostics only; they are enough to verify that optimization is
        # active without putting logging in the 8k-step hot path.
        history: list[dict[str, float | int]] = []
        milestones: list[dict[str, object]] = []
        try:
            for name in model.PARAMETER_NAMES:
                getattr(model, name).requires_grad_(
                    name in self.config.optimized_parameters
                )
            for step, frame_index in enumerate(selected, start=1):
                optimizer.zero_grad(set_to_none=True)
                objective_result = objective(model, frame_index, step - 1)
                if isinstance(objective_result, AppearanceObjective):
                    objective_loss = objective_result.total
                    objective_terms = objective_result.terms
                    objective_diagnostics = objective_result.diagnostics
                else:
                    objective_loss = objective_result
                    objective_terms = {"photo": objective_result}
                    objective_diagnostics = {}
                required = (
                    "dense_valid_normal_pixels",
                    "dense_normal_weight_mass",
                )
                if any(
                    name not in objective_diagnostics
                    for name in required
                ):
                    raise ValueError(
                        "SAGE dense-normal objective has zero "
                        f"support at step {step} for frame {frame_index}"
                    )
                optimizes_opacity = (
                    "opacity_logits" in self.config.optimized_parameters
                )
                opacity_anchor_loss = (
                    (
                        model.opacities - baseline_opacities
                    ).square().mean()
                    if optimizes_opacity
                    else objective_loss.new_zeros(())
                )
                exposure_anchor_loss = (
                    exposure_nuisance.anchor_loss()
                    if exposure_nuisance is not None
                    else objective_loss.new_zeros(())
                )
                loss = (
                    objective_loss
                    + self.config.opacity_anchor_weight
                    * opacity_anchor_loss
                    + self.config.exposure_anchor_weight
                    * exposure_anchor_loss
                )
                if objective_loss.ndim != 0 or loss.ndim != 0:
                    raise ValueError(
                        "Appearance refinement objective must return a scalar"
                    )
                validation = torch.stack(
                    tuple(
                        value.detach()
                        for value in (
                            *objective_terms.values(),
                            *objective_diagnostics.values(),
                            loss,
                        )
                    )
                )
                support = torch.stack(
                    tuple(
                        objective_diagnostics[name].detach()
                        for name in required
                    )
                )
                torch._assert_async(
                    torch.isfinite(validation).all()
                    & (support > 0).all(),
                    "Appearance refinement produced a non-finite objective "
                    "or lost dense-normal support",
                )
                loss.backward()
                optimizer.step()
                if step in self.config.milestone_steps:
                    milestone_values = {
                        "loss": loss.detach(),
                        **{
                            f"{name}_loss": value.detach()
                            for name, value in objective_terms.items()
                        },
                        **{
                            name: value.detach()
                            for name, value in objective_diagnostics.items()
                        },
                        "opacity_anchor_loss": opacity_anchor_loss.detach(),
                        "exposure_anchor_loss": exposure_anchor_loss.detach(),
                    }
                    numbers = torch.stack(
                        tuple(milestone_values.values())
                    ).cpu().tolist()
                    history.append({
                        "step": step,
                        "frame_index": frame_index,
                        **{
                            name: float(value)
                            for name, value in zip(
                                milestone_values,
                                numbers,
                            )
                        },
                    })
                    milestone = {
                        "step": step,
                        "gaussian_count": model.count,
                        "color_displacement": _distribution(
                            (model.colors.detach() - baseline_colors)
                            .abs()
                            .amax(dim=1)
                        ),
                    }
                    if optimizes_opacity:
                        milestone["opacity_displacement"] = _distribution(
                            (
                                model.opacities.detach()
                                - baseline_opacities
                            ).abs().reshape(-1)
                        )
                    if exposure_nuisance is not None:
                        milestone["exposure_nuisance"] = (
                            exposure_nuisance.summary()
                        )
                    if milestone_callback is not None:
                        milestone_callback(
                            model,
                            milestone,
                            optimizer.state_dict(),
                        )
                    milestones.append(milestone)
        finally:
            for name, requires_grad in original_requires_grad.items():
                getattr(model, name).requires_grad_(requires_grad)
        return AppearanceRefinementResult(
            tuple(history),
            tuple(milestones),
            selected,
            (
                exposure_nuisance.record()
                if exposure_nuisance is not None
                else None
            ),
        )
