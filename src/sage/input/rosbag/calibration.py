"""Canonical `calibration.yaml`: the single static-transform authority.

Calibration is the only runtime source of static transforms, so a bag never
silently borrows a missing edge from `/tf_static`. It also fixes the canonical
camera: the raw model plus the undistorted pinhole grid every canonical frame
is delivered on, decided once at preflight and never re-derived per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from .transforms import rigid_matrix


CALIBRATION_SCHEMA = "sage_calibration"
CALIBRATION_REVISION = 1

# Raw camera models we can undistort deterministically. `fishpoly` is the
# odd one out only in its distortion polynomial; both output plain pinhole.
DISTORTION_KEYS = {
    "pinhole_brown": ("k1", "k2", "p1", "p2", "k3"),
    "fishpoly": ("k2", "k3", "k4", "k5", "k6", "k7", "p1", "p2"),
}

_FRAME_ROLES = ("reference", "pose", "lidar", "camera")
_TIME_OFFSET_STREAMS = ("lidar", "odometry")


@dataclass(frozen=True)
class OutputCamera:
    """The canonical undistorted pinhole camera all frames are delivered on."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def payload(self) -> dict[str, object]:
        return {
            "width": self.width, "height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
        }


@dataclass(frozen=True)
class Calibration:
    path: Path
    frames: Mapping[str, str]
    raw_model: str
    raw_size: tuple[int, int]
    raw_intrinsics: Mapping[str, float]
    distortion: Mapping[str, float]
    output_camera: OutputCamera
    camera_from_lidar: np.ndarray
    pose_from_lidar: np.ndarray
    time_offsets_to_image_ns: Mapping[str, int]

    @property
    def lidar_from_camera(self) -> np.ndarray:
        return np.linalg.inv(self.camera_from_lidar)

    def payload(self) -> dict[str, object]:
        """Provenance only: canonical frames already carry K and the pose."""
        return {
            "schema_name": CALIBRATION_SCHEMA,
            "schema_revision": CALIBRATION_REVISION,
            "frames": dict(self.frames),
            "camera": {
                "raw_model": self.raw_model,
                "raw_size": list(self.raw_size),
                "raw_intrinsics": dict(self.raw_intrinsics),
                "distortion": dict(self.distortion),
                "output": self.output_camera.payload(),
                "output_model": "pinhole",
            },
            "extrinsics": {
                "camera_from_lidar": self.camera_from_lidar.tolist(),
                "pose_from_lidar": self.pose_from_lidar.tolist(),
            },
            "time_offsets_to_image_ns": dict(self.time_offsets_to_image_ns),
        }


def _require(payload: Mapping[str, Any], name: str, keys: tuple[str, ...]) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(
            f"calibration.yaml {name} must define exactly: {', '.join(sorted(keys))}"
        )
    return dict(value)


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration.yaml {name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"calibration.yaml {name} must be finite")
    return number


def load_calibration(path: Path) -> Calibration:
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read calibration: {source}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"calibration.yaml must be a mapping: {source}")
    expected = {
        "schema_name", "schema_revision", "units", "frames",
        "camera", "extrinsics", "time_offsets_to_image_ns",
    }
    if set(payload) != expected:
        raise ValueError(
            "Unsupported calibration schema.\n\n"
            "This branch accepts only the canonical SAGE calibration format. "
            f"Required top-level fields: {', '.join(sorted(expected))}."
        )
    if payload["schema_name"] != CALIBRATION_SCHEMA or payload["schema_revision"] != CALIBRATION_REVISION:
        raise ValueError(
            f"calibration.yaml must declare {CALIBRATION_SCHEMA} revision {CALIBRATION_REVISION}"
        )
    units = _require(payload, "units", ("length", "angle"))
    if units["length"] != "meter" or units["angle"] != "radian":
        raise ValueError("calibration.yaml units must be meter and radian")
    frames = _require(payload, "frames", _FRAME_ROLES)
    if any(not isinstance(value, str) or not value for value in frames.values()):
        raise ValueError("calibration.yaml frames must be non-empty names")

    camera = payload["camera"]
    if not isinstance(camera, Mapping) or set(camera) != {
        "model", "width", "height", "intrinsics", "distortion", "output",
    }:
        raise ValueError(
            "calibration.yaml camera must define: model, width, height, intrinsics, distortion, output"
        )
    model = str(camera["model"])
    if model not in DISTORTION_KEYS:
        raise ValueError(
            f"Unsupported camera model {model!r}; supported: {', '.join(sorted(DISTORTION_KEYS))}"
        )
    raw_size = (int(camera["width"]), int(camera["height"]))
    if min(raw_size) < 1:
        raise ValueError("calibration.yaml camera width and height must be positive")
    raw_intrinsics = {
        name: _finite_float(value, f"camera.intrinsics.{name}")
        for name, value in _require(camera, "intrinsics", ("fx", "fy", "cx", "cy", "skew")).items()
    }
    if min(raw_intrinsics["fx"], raw_intrinsics["fy"]) <= 0:
        raise ValueError("calibration.yaml camera focal lengths must be positive")
    distortion = {
        name: _finite_float(value, f"camera.distortion.{name}")
        for name, value in _require(camera, "distortion", DISTORTION_KEYS[model]).items()
    }
    output = _require(camera, "output", ("width", "height", "fx", "fy", "cx", "cy"))
    output_camera = OutputCamera(
        width=int(output["width"]),
        height=int(output["height"]),
        fx=_finite_float(output["fx"], "camera.output.fx"),
        fy=_finite_float(output["fy"], "camera.output.fy"),
        cx=_finite_float(output["cx"], "camera.output.cx"),
        cy=_finite_float(output["cy"], "camera.output.cy"),
    )
    if min(output_camera.width, output_camera.height) < 1 or min(output_camera.fx, output_camera.fy) <= 0:
        raise ValueError("calibration.yaml camera.output must be a positive pinhole grid")

    extrinsics = _require(payload, "extrinsics", ("camera_from_lidar", "pose_from_lidar"))
    camera_from_lidar = rigid_matrix(np.asarray(extrinsics["camera_from_lidar"], dtype=np.float64), "camera_from_lidar")
    pose_from_lidar = rigid_matrix(np.asarray(extrinsics["pose_from_lidar"], dtype=np.float64), "pose_from_lidar")
    offsets_payload = _require(payload, "time_offsets_to_image_ns", _TIME_OFFSET_STREAMS)
    offsets = {}
    for name, value in offsets_payload.items():
        if type(value) is not int:
            raise ValueError(f"calibration.yaml time_offsets_to_image_ns.{name} must be an integer")
        offsets[name] = value
    return Calibration(
        path=source,
        frames=frames,
        raw_model=model,
        raw_size=raw_size,
        raw_intrinsics=raw_intrinsics,
        distortion=distortion,
        output_camera=output_camera,
        camera_from_lidar=camera_from_lidar,
        pose_from_lidar=pose_from_lidar,
        time_offsets_to_image_ns=offsets,
    )


def _fishpoly_maps(calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    output = calibration.output_camera
    raw = calibration.raw_intrinsics
    distortion = calibration.distortion
    u, v = np.meshgrid(
        np.arange(output.width, dtype=np.float64),
        np.arange(output.height, dtype=np.float64),
    )
    x = (u - output.cx) / output.fx
    y = (v - output.cy) / output.fy
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta_d = theta * (
        1.0
        + distortion["k2"] * theta**2
        + distortion["k3"] * theta**3
        + distortion["k4"] * theta**4
        + distortion["k5"] * theta**5
        + distortion["k6"] * theta**6
        + distortion["k7"] * theta**7
    )
    scale = np.divide(theta_d, r, out=np.ones_like(theta_d), where=r > 1e-12)
    xd = scale * x
    yd = scale * y
    map_x = raw["fx"] * xd + raw["skew"] * yd + raw["cx"]
    map_y = raw["fy"] * yd + raw["cy"]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def _pinhole_brown_maps(calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    raw = calibration.raw_intrinsics
    raw_matrix = np.asarray(
        [[raw["fx"], raw["skew"], raw["cx"]], [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coefficients = np.asarray(
        [calibration.distortion[name] for name in ("k1", "k2", "p1", "p2", "k3")], dtype=np.float64,
    )
    output = calibration.output_camera
    map_x, map_y = cv2.initUndistortRectifyMap(
        raw_matrix, coefficients, None, output.matrix(),
        (output.width, output.height), cv2.CV_32FC1,
    )
    return map_x.astype(np.float32), map_y.astype(np.float32)


_MAP_BUILDERS = {"fishpoly": _fishpoly_maps, "pinhole_brown": _pinhole_brown_maps}


class ImageRectifier:
    """One-step remap from the raw camera onto the canonical pinhole grid.

    Bilinear sampling straight to the output grid: rectifying at native
    resolution and resizing afterwards would resample twice for the same K.
    """

    def __init__(self, calibration: Calibration) -> None:
        map_x, map_y = _MAP_BUILDERS[calibration.raw_model](calibration)
        width, height = calibration.raw_size
        self._source_size = (width, height)
        self._output_camera = calibration.output_camera
        # Half a pixel of slack at the border: a source coordinate of exactly
        # width-1 can land at width-1+1e-7, and dropping it would blacken the
        # outer row and column of an otherwise fully covered image.
        valid = (
            (map_x >= -0.5) & (map_x <= width - 0.5)
            & (map_y >= -0.5) & (map_y <= height - 0.5)
        )
        x = np.clip(map_x, 0, width - 1)
        y = np.clip(map_y, 0, height - 1)
        self._x0 = np.floor(x).astype(np.int32)
        self._y0 = np.floor(y).astype(np.int32)
        self._x1 = np.minimum(self._x0 + 1, width - 1)
        self._y1 = np.minimum(self._y0 + 1, height - 1)
        self._wx = (x - self._x0).astype(np.float32)[:, :, None]
        self._wy = (y - self._y0).astype(np.float32)[:, :, None]
        self._valid = valid

    @property
    def output_camera(self) -> OutputCamera:
        return self._output_camera

    @property
    def identity(self) -> dict[str, object]:
        return {
            "interpolation": "bilinear",
            "source_size": list(self._source_size),
            "output_size": [self._output_camera.width, self._output_camera.height],
            "out_of_bounds": "zero",
            "quantization": "uint8_then_float32_over_255",
        }

    def rectify_bgr(self, image: np.ndarray) -> np.ndarray:
        """Remap a BGR uint8 frame to canonical RGB uint8 on the output grid."""
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("Rectifier expects an HxWx3 image")
        if (source.shape[1], source.shape[0]) != self._source_size:
            raise ValueError(
                f"Image size {(source.shape[1], source.shape[0])} does not match calibration "
                f"{self._source_size}"
            )
        values = source.astype(np.float32)
        top = values[self._y0, self._x0] * (1.0 - self._wx) + values[self._y0, self._x1] * self._wx
        bottom = values[self._y1, self._x0] * (1.0 - self._wx) + values[self._y1, self._x1] * self._wx
        out = top * (1.0 - self._wy) + bottom * self._wy
        out[~self._valid] = 0.0
        return np.clip(out, 0, 255).astype(np.uint8)[:, :, ::-1].copy()


__all__ = ["Calibration", "ImageRectifier", "OutputCamera", "load_calibration"]
