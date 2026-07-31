"""Public Gaussian PLY export shared by every SAGE stage that publishes a map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from torch.nn import functional as F


def write_gaussian_ply(path: Path, model: object) -> None:
    """Export the model's Gaussians using the public Gaussian PLY layout."""
    means = model.means3d.detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh_dc = (
        (model.colors.detach() - 0.5) / 0.28209479177387814
    ).cpu().numpy()
    opacity = model.opacity_logits.detach().cpu().reshape(-1).numpy()
    scales = model.scales.detach()
    rotations = F.normalize(
        model.rotations.detach(),
        dim=1,
    ).cpu().numpy()
    float_names = (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )
    elements = np.empty(
        model.count,
        dtype=[
            *((name, "f4") for name in float_names),
            ("gaussian_id", "i4"),
            ("created_at", "i4"),
        ],
    )
    attributes = np.concatenate(
        (means, normals, sh_dc, opacity[:, None], scales.cpu().numpy(), rotations),
        axis=1,
    )
    for column, name in enumerate(float_names):
        elements[name] = attributes[:, column]
    elements["gaussian_id"] = model.gaussian_ids.detach().cpu().numpy()
    elements["created_at"] = model.created_at.detach().cpu().numpy()
    PlyData(
        [PlyElement.describe(elements, "vertex")],
        text=False,
    ).write(path)


__all__ = ["write_gaussian_ply"]
