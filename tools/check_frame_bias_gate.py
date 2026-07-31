"""Standalone CPU self-check for the growth-time frame bias gate.

Not wired into any test runner (this repo has none for src/sage) — run it
directly: `python tools/check_frame_bias_gate.py`.

Covers the three cases the frame-level source bias gate must handle:
1. reference (priority 0) and a secondary source agree -> no frame_bias reject.
2. secondary source is globally biased with enough overlap -> whole evidence
   rejected under "frame_bias", zero candidates from that source.
3. secondary source is biased but overlap is below the configured minimum ->
   gate does not fire (falls through to normal per-pixel gating).
"""

from __future__ import annotations

import numpy as np
import torch

from sage.foundation.config import GaussianInitializationConfig, GrowthConfig
from sage.foundation.contracts import (
    CameraIntrinsics,
    DepthEvidence,
    FrameInputs,
    GrowthInputs,
    MappingObservation,
    Pose,
    SourceType,
)
from sage.engine.growth import GrowthBuilder
from sage.engine.rendering import RenderOutput

HEIGHT = WIDTH = 8
REFERENCE_DEPTH_M = 5.0


def _make_frame(secondary_depth_m: float, secondary_valid_count: int) -> FrameInputs:
    intrinsics = CameraIntrinsics(WIDTH, HEIGHT, 32.0, 32.0, WIDTH / 2, HEIGHT / 2)
    pose = Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    reference_depth = np.full((HEIGHT, WIDTH), REFERENCE_DEPTH_M, dtype=np.float32)
    reference_valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    reference_confidence = np.ones((HEIGHT, WIDTH), dtype=np.float32)

    secondary_depth = np.full((HEIGHT, WIDTH), secondary_depth_m, dtype=np.float32)
    secondary_valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    secondary_valid.flat[:secondary_valid_count] = True
    secondary_confidence = np.where(secondary_valid, 1.0, 0.0).astype(np.float32)

    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    mapping_depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mapping_sources = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    mapping_confidence = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    return FrameInputs(
        index=1,
        stem="frame-bias-check",
        timestamp_ns=0,
        intrinsics=intrinsics,
        pose=pose,
        rgb=rgb,
        mapping=MappingObservation(mapping_depth, mapping_sources, mapping_confidence),
        growth=GrowthInputs((
            DepthEvidence(SourceType.LIDAR_SLAM_CENTER, reference_depth, reference_valid, reference_confidence),
            DepthEvidence(SourceType.SPNET_BLIND, secondary_depth, secondary_valid, secondary_confidence),
        )),
    )


def _make_rendered() -> RenderOutput:
    # alpha=0 (uncovered) + rgb far from target keeps coverage_gate open for every pixel.
    return RenderOutput(
        rgb=torch.full((HEIGHT, WIDTH, 3), 0.9, dtype=torch.float32),
        depth=torch.zeros((HEIGHT, WIDTH), dtype=torch.float32),
        alpha=torch.zeros((HEIGHT, WIDTH), dtype=torch.float32),
    )


def _spnet_reasons(builder: GrowthBuilder, frame: FrameInputs) -> dict[str, int]:
    result = builder.build(frame, _make_rendered())
    return result.stats.by_source["SPNET_BLIND"]["rejection_reasons"]


def main() -> None:
    growth_config = GrowthConfig()
    builder = GrowthBuilder(
        growth_config,
        device="cpu",
        gaussian_initialization=GaussianInitializationConfig(opacity=0.5),
    )
    min_overlap = growth_config.frame_bias_min_overlap_px
    total_px = HEIGHT * WIDTH
    assert min_overlap < total_px, "test grid too small for configured min overlap"

    # 1) agreement -> gate must not fire.
    agreeing_frame = _make_frame(REFERENCE_DEPTH_M, total_px)
    reasons = _spnet_reasons(builder, agreeing_frame)
    assert reasons.get("frame_bias", 0) == 0, f"unexpected frame_bias reject on agreement: {reasons}"

    # 2) systematic bias with enough overlap -> whole evidence rejected.
    biased_depth = REFERENCE_DEPTH_M + growth_config.frame_bias_threshold.absolute_m + 0.2
    biased_frame = _make_frame(biased_depth, total_px)
    reasons = _spnet_reasons(builder, biased_frame)
    assert reasons.get("frame_bias", 0) == total_px, f"frame_bias did not reject the full evidence: {reasons}"

    # 3) same bias, overlap below the configured minimum -> gate must not fire.
    sparse_frame = _make_frame(biased_depth, min_overlap - 1)
    reasons = _spnet_reasons(builder, sparse_frame)
    assert reasons.get("frame_bias", 0) == 0, f"frame_bias fired despite insufficient overlap: {reasons}"

    print("check_frame_bias_gate: all 3 scenarios passed")


if __name__ == "__main__":
    main()
