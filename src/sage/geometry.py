from __future__ import annotations

import numpy as np
import torch

from .contracts import CameraIntrinsics, Pose


def rotation_wc(pose: Pose, *, device: torch.device | str = "cpu") -> torch.Tensor:
    quaternion = torch.tensor([pose.qx, pose.qy, pose.qz, pose.qw], dtype=torch.float32, device=device)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    x, y, z, w = quaternion.unbind()
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )).reshape(3, 3)


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
    pose: Pose,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected = (intrinsics.height, intrinsics.width)
    if tuple(depth.shape) != expected:
        raise ValueError(f"Depth dimensions must be {expected}")
    valid = torch.isfinite(depth) & (depth > 0)
    if mask is not None:
        if tuple(mask.shape) != expected:
            raise ValueError("Backprojection mask dimensions do not match depth")
        valid &= mask.to(device=depth.device, dtype=torch.bool)
    pixels = torch.nonzero(valid, as_tuple=False)
    if pixels.numel() == 0:
        return (
            torch.empty((0, 3), dtype=torch.float32, device=depth.device),
            pixels,
            torch.empty((0,), dtype=torch.float32, device=depth.device),
        )
    z = depth[pixels[:, 0], pixels[:, 1]].to(torch.float32)
    rows, cols = pixels.to(torch.float32).unbind(1)
    camera = torch.stack(((cols - intrinsics.cx) * z / intrinsics.fx,
                          (rows - intrinsics.cy) * z / intrinsics.fy, z), dim=1)
    translation = torch.tensor(pose.translation, dtype=torch.float32, device=depth.device)
    world = camera @ rotation_wc(pose, device=depth.device).T + translation
    return world, pixels, z


def world_to_camera(points: torch.Tensor, pose: Pose) -> torch.Tensor:
    translation = torch.tensor(pose.translation, dtype=torch.float32, device=points.device)
    return (points - translation) @ rotation_wc(pose, device=points.device)


def select_current_anchored_global_views(
    historical_indices: list[int] | tuple[int, ...],
    current_index: int,
    *,
    max_views: int,
    rng: np.random.Generator | None = None,
) -> tuple[int, ...]:
    """Select the current frame plus a global, past-only keyframe sample."""
    if max_views < 1:
        raise ValueError("max_views must be positive")
    historical = tuple(int(index) for index in historical_indices)
    if len(set(historical)) != len(historical):
        raise ValueError("historical_indices must not contain duplicates")
    if any(index > current_index for index in historical):
        raise ValueError("historical_indices must not contain future frames")
    historical = tuple(index for index in historical if index != current_index)
    if len(historical) + 1 <= max_views:
        return (int(current_index), *historical)
    generator = np.random.default_rng(0) if rng is None else rng
    sampled = generator.choice(
        np.asarray(historical, dtype=np.int64),
        size=max_views - 1,
        replace=False,
    )
    return (int(current_index), *(int(index) for index in sampled))


def build_optimization_schedule(
    active_window: tuple[int, ...] | list[int],
    current_index: int,
    *,
    steps: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Build an exact-length schedule anchored by the current frame."""
    window = tuple(int(index) for index in active_window)
    if steps < 1:
        raise ValueError("steps must be positive")
    if not window or window[0] != current_index:
        raise ValueError("active_window must start with the current frame")
    if len(set(window)) != len(window):
        raise ValueError("active_window must contain unique frames")
    if steps == 1:
        return (int(current_index),)
    sampled = rng.choice(
        np.asarray(window, dtype=np.int64),
        size=steps - 1,
        replace=True,
    )
    return (int(current_index), *(int(index) for index in sampled))
