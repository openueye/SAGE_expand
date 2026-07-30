from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch


class SourceType(IntEnum):
    LIDAR_RAW = 0
    LIDAR_FUSED5 = 1
    SPNET_BLIND = 2
    LIDAR_SLAM_CENTER = 3
    LIDAR_SLAM_FUSED5 = 4


INVALID_SOURCE_TYPE = np.uint8(255)


def _lidar_family(values: np.ndarray) -> str | None:
    source_types = np.asarray(values)
    has_raw = bool(np.isin(source_types, [
        int(SourceType.LIDAR_RAW), int(SourceType.LIDAR_FUSED5),
    ]).any())
    has_slam = bool(np.isin(source_types, [
        int(SourceType.LIDAR_SLAM_CENTER), int(SourceType.LIDAR_SLAM_FUSED5),
    ]).any())
    if has_raw and has_slam:
        raise ValueError("Source types cannot mix raw and SLAM LiDAR families")
    return "LIDAR_RAW" if has_raw else "SLAM_WORLD" if has_slam else None


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1 or min(self.fx, self.fy) <= 0:
            raise ValueError("Camera dimensions and focal lengths must be positive")


@dataclass(frozen=True)
class Pose:
    """Finalized external camera pose T_wc."""

    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float

    @property
    def translation(self) -> tuple[float, float, float]:
        return self.tx, self.ty, self.tz

    def __post_init__(self) -> None:
        values = np.asarray((*self.translation, self.qx, self.qy, self.qz, self.qw), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Pose must contain finite values")
        if np.linalg.norm(values[3:]) <= 0:
            raise ValueError("Pose quaternion must be non-zero")


def _validate_image_shape(name: str, value: np.ndarray, shape: tuple[int, int]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


@dataclass(frozen=True)
class MappingObservation:
    depth_m: np.ndarray
    source_types: np.ndarray
    confidences: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        sources = np.asarray(self.source_types)
        confidences = np.asarray(self.confidences)
        if depth.dtype != np.float32:
            raise ValueError("Mapping depth must be float32")
        if sources.dtype != np.uint8:
            raise ValueError("Mapping source_types must be uint8")
        if confidences.dtype != np.float32:
            raise ValueError("Mapping confidences must be float32")
        if depth.ndim != 2:
            raise ValueError("Mapping depth must be a 2-D array")
        _validate_image_shape("Mapping source_types", sources, depth.shape)
        _validate_image_shape("Mapping confidences", confidences, depth.shape)
        if not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError("Mapping depth must be finite and non-negative")
        if not np.isfinite(confidences).all() or ((confidences < 0) | (confidences > 1)).any():
            raise ValueError("Mapping confidences must be finite and within [0, 1]")
        valid_sources = np.isin(sources, [
            int(SourceType.LIDAR_RAW), int(SourceType.LIDAR_FUSED5),
            int(SourceType.LIDAR_SLAM_CENTER), int(SourceType.LIDAR_SLAM_FUSED5),
            int(INVALID_SOURCE_TYPE),
        ])
        if not valid_sources.all():
            raise ValueError("Mapping source_types contains an unsupported source ID")
        _lidar_family(sources)
        invalid = sources == INVALID_SOURCE_TYPE
        if np.any(invalid & ((depth != 0) | (confidences != 0))):
            raise ValueError("Invalid mapping pixels must have zero depth and confidence")

    @property
    def source_family(self) -> str | None:
        return _lidar_family(self.source_types)


@dataclass(frozen=True)
class DepthEvidence:
    source_type: SourceType
    depth_m: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        valid = np.asarray(self.valid_mask)
        confidence = np.asarray(self.confidence)
        if not isinstance(self.source_type, SourceType):
            raise ValueError("DepthEvidence source_type must be a SourceType")
        if depth.dtype != np.float32 or confidence.dtype != np.float32 or valid.dtype != np.bool_:
            raise ValueError("DepthEvidence arrays must be float32, bool, and float32")
        if depth.ndim != 2:
            raise ValueError("DepthEvidence depth must be a 2-D array")
        _validate_image_shape("DepthEvidence valid_mask", valid, depth.shape)
        _validate_image_shape("DepthEvidence confidence", confidence, depth.shape)
        if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("DepthEvidence confidence must be finite and within [0, 1]")
        if np.any(~valid & (confidence != 0)):
            raise ValueError("DepthEvidence confidence must be zero outside valid_mask")
        if np.any(valid & (~np.isfinite(depth) | (depth <= 0))):
            raise ValueError("DepthEvidence valid pixels must have finite positive depth")


@dataclass(frozen=True)
class DensePriorDiagnostics:
    lidar_support: int
    fitted_grid_cells: int
    pre_alignment_mae_m: float
    post_alignment_mae_m: float
    fallback: bool
    fitted_alignment_mae_m: float = 0.0
    selected_alignment: str = "fitted-grid"

    def __post_init__(self) -> None:
        if type(self.lidar_support) is not int or self.lidar_support < 0:
            raise ValueError("Dense prior LiDAR support must be a non-negative integer")
        if type(self.fitted_grid_cells) is not int or self.fitted_grid_cells < 0:
            raise ValueError("Dense prior fitted cells must be a non-negative integer")
        values = np.asarray(
            (
                self.pre_alignment_mae_m,
                self.post_alignment_mae_m,
                self.fitted_alignment_mae_m,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Dense prior residuals must be finite and non-negative")
        if type(self.fallback) is not bool:
            raise ValueError("Dense prior fallback must be boolean")
        if self.selected_alignment not in {
            "fitted-grid",
            "identity",
            "identity-fallback",
        }:
            raise ValueError("Dense prior selected alignment is invalid")


@dataclass(frozen=True)
class DenseGeometryPrior:
    raw_depth_m: np.ndarray
    aligned_depth_m: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray
    scale_grid: np.ndarray
    diagnostics: DensePriorDiagnostics

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_depth_m)
        aligned = np.asarray(self.aligned_depth_m)
        valid = np.asarray(self.valid_mask)
        confidence = np.asarray(self.confidence)
        grid = np.asarray(self.scale_grid)
        if raw.dtype != np.float32 or aligned.dtype != np.float32:
            raise ValueError("Dense prior depths must be float32")
        if valid.dtype != np.bool_ or confidence.dtype != np.float32:
            raise ValueError("Dense prior mask/confidence must be bool/float32")
        if raw.ndim != 2 or aligned.shape != raw.shape or valid.shape != raw.shape or confidence.shape != raw.shape:
            raise ValueError("Dense prior image fields must share a 2-D shape")
        if grid.dtype != np.float32 or grid.ndim != 2 or not grid.size:
            raise ValueError("Dense prior scale grid must be a non-empty float32 matrix")
        if not np.isfinite(raw).all() or not np.isfinite(aligned).all() or not np.isfinite(grid).all():
            raise ValueError("Dense prior values must be finite")
        if (raw < 0).any() or (aligned < 0).any() or (grid <= 0).any():
            raise ValueError("Dense prior depths must be non-negative and scales positive")
        if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("Dense prior confidence must be within [0, 1]")
        if np.any(~valid & (confidence != 0)):
            raise ValueError("Dense prior confidence must be zero outside valid support")
        if not isinstance(self.diagnostics, DensePriorDiagnostics):
            raise ValueError("Dense prior diagnostics have an invalid type")


@dataclass(frozen=True)
class GrowthInputs:
    evidences: tuple[DepthEvidence, ...]

    def __post_init__(self) -> None:
        source_types = [evidence.source_type for evidence in self.evidences]
        if len(source_types) != len(set(source_types)):
            raise ValueError("GrowthInputs may contain each source type at most once")
        priorities = {
            SourceType.LIDAR_RAW: 0,
            SourceType.LIDAR_SLAM_CENTER: 0,
            SourceType.LIDAR_FUSED5: 1,
            SourceType.LIDAR_SLAM_FUSED5: 1,
            SourceType.SPNET_BLIND: 2,
        }
        if source_types != sorted(source_types, key=priorities.__getitem__):
            raise ValueError("GrowthInputs must be sorted by source priority")
        raw_family = {SourceType.LIDAR_RAW, SourceType.LIDAR_FUSED5}
        slam_family = {SourceType.LIDAR_SLAM_CENTER, SourceType.LIDAR_SLAM_FUSED5}
        if set(source_types) & raw_family and set(source_types) & slam_family:
            raise ValueError("GrowthInputs cannot mix raw and SLAM LiDAR source families")
        shapes = {evidence.depth_m.shape for evidence in self.evidences}
        if len(shapes) > 1:
            raise ValueError("GrowthInputs evidences must share the same shape")

    @property
    def source_family(self) -> str | None:
        return _lidar_family(np.asarray([int(item.source_type) for item in self.evidences]))


@dataclass(frozen=True)
class FrameInputs:
    index: int
    stem: str
    timestamp_ns: int
    intrinsics: CameraIntrinsics
    pose: Pose
    rgb: np.ndarray
    mapping: MappingObservation
    growth: GrowthInputs

    def __post_init__(self) -> None:
        expected = (self.intrinsics.height, self.intrinsics.width)
        if self.rgb.shape != (*expected, 3) or self.rgb.dtype != np.float32:
            raise ValueError("Frame RGB must be float32 HxWx3 matching intrinsics")
        if not np.isfinite(self.rgb).all() or ((self.rgb < 0) | (self.rgb > 1)).any():
            raise ValueError("Frame RGB must be finite and within [0, 1]")
        _validate_image_shape("Mapping depth", self.mapping.depth_m, expected)
        for evidence in self.growth.evidences:
            _validate_image_shape("Growth evidence", evidence.depth_m, expected)
        if (
            self.mapping.source_family is not None
            and self.growth.source_family is not None
            and self.mapping.source_family != self.growth.source_family
        ):
            raise ValueError("Frame mapping and growth cannot use different LiDAR source families")


@dataclass(frozen=True)
class GaussianAppendBatch:
    means3d: torch.Tensor
    colors: torch.Tensor
    opacity_logits: torch.Tensor
    log_scales: torch.Tensor
    rotations: torch.Tensor
    source_types: torch.Tensor
    source_confidences: torch.Tensor
