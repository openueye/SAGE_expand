"""SE(3) helpers and the `A_from_B` transform chain.

Every matrix in this module reads `A_from_B`: it maps a point expressed in B
into A. There is no second convention anywhere in the input layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Published extrinsics are quoted to about five decimals, so a real rigid
# transform is only orthonormal to ~1e-5. Tighter than this rejects valid
# calibration files; looser stops catching a genuinely wrong matrix.
ROTATION_TOLERANCE = 1e-4


def rigid_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} must have a homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=ROTATION_TOLERANCE):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=ROTATION_TOLERANCE):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("Quaternion must be finite and non-zero")
    x, y, z, w = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm == 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


def pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    left = np.asarray(q0, dtype=np.float64)
    right = np.asarray(q1, dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (np.sin((1.0 - alpha) * angle) / scale) * left + (np.sin(alpha * angle) / scale) * right


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    reference_from_pose: np.ndarray

    def __post_init__(self) -> None:
        if type(self.timestamp_ns) is not int:
            raise ValueError("PoseSample timestamp_ns must be an integer")
        object.__setattr__(
            self, "reference_from_pose", rigid_matrix(self.reference_from_pose, "reference_from_pose"),
        )


def interpolate_pose(
    left: PoseSample, right: PoseSample, timestamp_ns: int, *, max_gap_ns: int,
) -> np.ndarray:
    """Linear translation + SLERP rotation. Never extrapolates, never clamps."""
    timestamp = int(timestamp_ns)
    if timestamp < left.timestamp_ns or timestamp > right.timestamp_ns:
        raise ValueError("Pose interpolation cannot extrapolate outside its bracket")
    if left.timestamp_ns == right.timestamp_ns:
        return left.reference_from_pose.copy()
    span = right.timestamp_ns - left.timestamp_ns
    if span > int(max_gap_ns):
        raise ValueError(
            f"Pose bracket spans {span} ns, exceeding max_pose_gap of {int(max_gap_ns)} ns"
        )
    alpha = (timestamp - left.timestamp_ns) / span
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (
        (1.0 - alpha) * left.reference_from_pose[:3, 3] + alpha * right.reference_from_pose[:3, 3]
    )
    matrix[:3, :3] = quaternion_xyzw_to_matrix(
        slerp(
            matrix_to_quaternion_xyzw(left.reference_from_pose[:3, :3]),
            matrix_to_quaternion_xyzw(right.reference_from_pose[:3, :3]),
            alpha,
        )
    )
    return matrix


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply `A_from_B` to Nx3 points expressed in B."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Points must have shape Nx3, got {values.shape}")
    if not len(values):
        return values
    return values @ np.asarray(matrix, dtype=np.float64)[:3, :3].T + matrix[:3, 3]


__all__ = [
    "PoseSample",
    "interpolate_pose",
    "matrix_to_quaternion_xyzw",
    "pose_matrix",
    "quaternion_xyzw_to_matrix",
    "rigid_matrix",
    "slerp",
    "transform_points",
]
