"""Reference-frame points to canonical camera z-depth.

Depth is always the camera optical z of a point, never a ray range and never a
reference-frame coordinate. Occlusion is resolved by a nearest z-buffer on the
output grid.
"""

from __future__ import annotations

import numpy as np

from .calibration import OutputCamera


def project_to_depth(
    points_reference: np.ndarray,
    *,
    camera_from_reference: np.ndarray,
    camera: OutputCamera,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Project points into one camera and z-buffer them onto its pixel grid."""
    depth = np.zeros((camera.height, camera.width), dtype=np.float32)
    statistics = {"points": 0, "positive_depth_points": 0, "in_image_points": 0}
    points = np.asarray(points_reference, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Projection expects Nx3 points, got {points.shape}")
    statistics["points"] = int(len(points))
    if not len(points):
        return depth, statistics
    matrix = np.asarray(camera_from_reference, dtype=np.float64)
    camera_points = points @ matrix[:3, :3].T + matrix[:3, 3]
    z = camera_points[:, 2]
    inside_range = np.isfinite(camera_points).all(axis=1) & (z >= min_depth_m) & (z <= max_depth_m)
    camera_points = camera_points[inside_range]
    z = z[inside_range]
    statistics["positive_depth_points"] = int(len(z))
    if not len(z):
        return depth, statistics
    u = np.rint(camera.fx * camera_points[:, 0] / z + camera.cx).astype(np.int64)
    v = np.rint(camera.fy * camera_points[:, 1] / z + camera.cy).astype(np.int64)
    inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    u, v, z = u[inside], v[inside], z[inside].astype(np.float32)
    statistics["in_image_points"] = int(len(z))
    if not len(z):
        return depth, statistics
    zbuffer = np.full(depth.shape, np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, (v, u), z)
    finite = np.isfinite(zbuffer)
    depth[finite] = zbuffer[finite]
    return depth, statistics


__all__ = ["project_to_depth"]
