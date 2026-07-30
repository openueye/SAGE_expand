"""Evaluation-only depth, alpha, coverage, and hit metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import torch

from .config import ALPHA_NORMALIZED_DEPTH_POLICY, RAW_ACCUMULATED_DEPTH_POLICY
from .contracts import FrameInputs, MappingObservation, SourceType
from .losses import alpha_normalized_depth
from .rendering import RenderOutput


EVALUATION_SCHEMA_VERSION = "sage-evaluation-v1"


@dataclass(frozen=True)
class EvaluationDepthPolicy:
    """A fixed evaluation representation independent of training support."""

    depth_policy: str = ALPHA_NORMALIZED_DEPTH_POLICY
    min_alpha: float = 0.01
    epsilon: float = 1e-6
    alpha_support_a0: float = 0.01
    hit_target_center: float = 0.5
    hit_target_fused5: float = 0.35

    def __post_init__(self) -> None:
        if self.depth_policy not in {
            RAW_ACCUMULATED_DEPTH_POLICY, ALPHA_NORMALIZED_DEPTH_POLICY,
        }:
            raise ValueError("Unsupported evaluation depth policy")
        if not torch.isfinite(torch.tensor(self.min_alpha)) or not 0 < self.min_alpha <= 1:
            raise ValueError("Evaluation alpha threshold must be finite and within (0, 1]")
        if not torch.isfinite(torch.tensor(self.epsilon)) or not 0 < self.epsilon:
            raise ValueError("Evaluation epsilon must be finite and positive")
        if not torch.isfinite(torch.tensor(self.alpha_support_a0)) or not 0 < self.alpha_support_a0 <= 1:
            raise ValueError("Evaluation alpha-support threshold must be finite and within (0, 1]")
        if any(
            not torch.isfinite(torch.tensor(target)) or not 0 <= target <= 1
            for target in (self.hit_target_center, self.hit_target_fused5)
        ):
            raise ValueError("Evaluation hit targets must be finite and within [0, 1]")

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _source_masks(target_mapping: MappingObservation, device: torch.device) -> dict[str, torch.Tensor]:
    source_types = torch.as_tensor(target_mapping.source_types, dtype=torch.uint8, device=device)
    return {
        "center": (source_types == int(SourceType.LIDAR_RAW))
        | (source_types == int(SourceType.LIDAR_SLAM_CENTER)),
        "fused5": (source_types == int(SourceType.LIDAR_FUSED5))
        | (source_types == int(SourceType.LIDAR_SLAM_FUSED5)),
    }


def _scalar_count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _quantile(values: torch.Tensor, q: float) -> float | None:
    return None if values.numel() == 0 else float(torch.quantile(values, q))


def _alpha_summary(
    alpha: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    a0: float,
    include_samples: bool,
) -> dict[str, Any]:
    finite_mask = observation_mask & torch.isfinite(alpha)
    values = alpha[finite_mask]
    observation_count = _scalar_count(observation_mask)
    finite_count = int(values.numel())
    result: dict[str, Any] = {
        "observation_count": observation_count,
        "finite_count": finite_count,
        "missing_alpha_count": observation_count - finite_count,
        "mean": None if finite_count == 0 else float(values.mean()),
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "below_a0_count": _scalar_count(finite_mask & (alpha < a0)),
        "below_a0_ratio": (
            _scalar_count(finite_mask & (alpha < a0)) / observation_count
            if observation_count else None
        ),
    }
    if include_samples:
        result["_samples"] = values.detach()
    return result


def _hit_summary(
    alpha: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    target: float,
) -> dict[str, Any]:
    finite_mask = observation_mask & torch.isfinite(alpha)
    violation = finite_mask & (alpha < target)
    observation_count = _scalar_count(observation_mask)
    return {
        "target": target,
        "observation_count": observation_count,
        "finite_count": _scalar_count(finite_mask),
        "missing_alpha_count": observation_count - _scalar_count(finite_mask),
        "violation_count": _scalar_count(violation),
        "violation_ratio": _scalar_count(violation) / observation_count if observation_count else None,
    }


def _depth_summary(
    depth: torch.Tensor,
    target_depth: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    alpha: torch.Tensor,
    min_alpha: float,
) -> dict[str, Any]:
    render_valid = (
        observation_mask
        & torch.isfinite(depth) & (depth > 0)
        & torch.isfinite(alpha) & (alpha >= min_alpha)
    )
    error = (depth - target_depth).abs()
    observation_count = _scalar_count(observation_mask)
    render_valid_count = _scalar_count(render_valid)
    error_sum = float(error[render_valid].sum())
    return {
        "absolute_error_sum_m": error_sum,
        "observation_count": observation_count,
        "render_valid_count": render_valid_count,
        "missing_support_count": observation_count - render_valid_count,
        "mae_m": error_sum / render_valid_count if render_valid_count else None,
        "coverage": render_valid_count / observation_count if observation_count else None,
    }


def evaluate_render_output(
    output: RenderOutput,
    target_mapping: MappingObservation,
    *,
    policy: EvaluationDepthPolicy,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Evaluate one raw render with one fixed geometry/mask policy.

    This function never reads a training-loss variant or support mask. Invalid
    rendered pixels are excluded from error numerators and reported explicitly;
    a source with no valid render has a null MAE rather than a zero MAE.
    """
    if not isinstance(target_mapping, MappingObservation):
        raise ValueError("Evaluation requires a MappingObservation target")
    if target_mapping.depth_m.shape != output.depth.shape:
        raise ValueError("Evaluation target depth must match rendered depth")
    target_depth = torch.as_tensor(
        target_mapping.depth_m, dtype=output.depth.dtype, device=output.depth.device,
    )
    if policy.depth_policy == ALPHA_NORMALIZED_DEPTH_POLICY:
        depth = alpha_normalized_depth(output.accumulated_depth, output.alpha, epsilon=policy.epsilon)
    else:
        depth = output.accumulated_depth
    masks = _source_masks(target_mapping, depth.device)
    known_source = masks["center"] | masks["fused5"]
    observation = torch.isfinite(target_depth) & (target_depth > 0) & known_source
    depth_metrics = {
        name: _depth_summary(
            depth, target_depth, observation & source_mask,
            alpha=output.alpha, min_alpha=policy.min_alpha,
        )
        for name, source_mask in masks.items()
    }
    depth_metrics["total"] = _depth_summary(
        depth, target_depth, observation, alpha=output.alpha, min_alpha=policy.min_alpha,
    )
    alpha_metrics = {
        name: _alpha_summary(
            output.alpha, observation & source_mask, a0=policy.alpha_support_a0,
            include_samples=include_samples,
        )
        for name, source_mask in masks.items()
    }
    alpha_metrics["total"] = _alpha_summary(
        output.alpha, observation, a0=policy.alpha_support_a0, include_samples=include_samples,
    )
    hit_metrics = {
        "center": _hit_summary(
            output.alpha, observation & masks["center"], target=policy.hit_target_center,
        ),
        "fused5": _hit_summary(
            output.alpha, observation & masks["fused5"], target=policy.hit_target_fused5,
        ),
    }
    return {
        "evaluation_depth_policy": policy.payload(),
        "depth": depth_metrics,
        "alpha": alpha_metrics,
        "hit_margin": hit_metrics,
    }


def _sum_optional(rows: list[dict[str, Any]], name: str) -> int:
    return sum(int(row[name]) for row in rows)


def _aggregate_geometry(frames: list[dict[str, Any]]) -> dict[str, Any]:
    depth: dict[str, Any] = {}
    alpha: dict[str, Any] = {}
    for source in ("center", "fused5", "total"):
        depth_rows = [frame["geometry"]["depth"][source] for frame in frames]
        error_sum = sum(float(row["absolute_error_sum_m"]) for row in depth_rows)
        observed = _sum_optional(depth_rows, "observation_count")
        valid = _sum_optional(depth_rows, "render_valid_count")
        depth[source] = {
            "absolute_error_sum_m": error_sum,
            "observation_count": observed,
            "render_valid_count": valid,
            "missing_support_count": observed - valid,
            "mae_m": error_sum / valid if valid else None,
            "coverage": valid / observed if observed else None,
        }

        alpha_rows = [frame["geometry"]["alpha"][source] for frame in frames]
        finite = _sum_optional(alpha_rows, "finite_count")
        alpha_sum = sum(
            float(row["mean"]) * int(row["finite_count"])
            for row in alpha_rows
            if row["mean"] is not None
        )
        alpha_observed = _sum_optional(alpha_rows, "observation_count")
        below = _sum_optional(alpha_rows, "below_a0_count")
        alpha[source] = {
            "observation_count": alpha_observed,
            "finite_count": finite,
            "missing_alpha_count": alpha_observed - finite,
            "mean": alpha_sum / finite if finite else None,
            "below_a0_count": below,
            "below_a0_ratio": below / alpha_observed if alpha_observed else None,
        }

    hit: dict[str, Any] = {}
    for source in ("center", "fused5"):
        rows = [frame["geometry"]["hit_margin"][source] for frame in frames]
        observed = _sum_optional(rows, "observation_count")
        finite = _sum_optional(rows, "finite_count")
        violations = _sum_optional(rows, "violation_count")
        hit[source] = {
            "target": rows[0]["target"],
            "observation_count": observed,
            "finite_count": finite,
            "missing_alpha_count": observed - finite,
            "violation_count": violations,
            "violation_ratio": violations / observed if observed else None,
        }
    return {"depth": depth, "alpha": alpha, "hit_margin": hit}


def evaluate_frames(
    model: object,
    frames: Iterable[FrameInputs],
    *,
    renderer: Callable[[object, FrameInputs], RenderOutput],
    image_metrics: Callable[[torch.Tensor, torch.Tensor], object],
    policy: EvaluationDepthPolicy,
    map_every: int,
) -> dict[str, Any]:
    """Run the frozen image and geometry protocol over an ordered frame stream."""
    if type(map_every) is not int or map_every < 1:
        raise ValueError("Evaluation map_every must be a positive integer")
    frame_reports: list[dict[str, Any]] = []
    for frame in frames:
        if frame.index != 0 and (frame.index + 1) % map_every != 0:
            continue
        output = renderer(model, frame)
        image = image_metrics(
            output.rgb,
            torch.as_tensor(frame.rgb, dtype=torch.float32, device=output.rgb.device),
        )
        frame_reports.append({
            "index": frame.index,
            "stem": frame.stem,
            "image": {
                "psnr": float(getattr(image, "psnr")),
                "ssim": float(getattr(image, "ssim")),
                "lpips": float(getattr(image, "lpips")),
            },
            "geometry": evaluate_render_output(output, frame.mapping, policy=policy),
        })
    return aggregate_evaluation_frame_reports(
        frame_reports,
        policy=policy,
    )


def aggregate_evaluation_frame_reports(
    frame_reports: Iterable[dict[str, Any]],
    *,
    policy: EvaluationDepthPolicy,
) -> dict[str, Any]:
    """Aggregate an already-rendered frame subset without rerendering it."""
    reports = list(frame_reports)
    if not reports:
        raise ValueError("Evaluation frame subset must be non-empty")
    count = len(reports)
    image = {
        name: sum(float(frame["image"][name]) for frame in reports) / count
        for name in ("psnr", "ssim", "lpips")
    }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "frame_count": count,
        "evaluation_depth_policy": policy.payload(),
        "aggregate": {
            "image": image,
            **_aggregate_geometry(reports),
        },
        "frames": reports,
    }


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationDepthPolicy",
    "aggregate_evaluation_frame_reports",
    "evaluate_frames",
    "evaluate_render_output",
]
