#!/usr/bin/env python3
"""Convert a `cam_in_ex.txt` camera/LiDAR calibration to canonical calibration.yaml.

This is a data preparation tool, not a runtime profile: nothing in SAGE
dispatches on which converter produced a calibration. Run it once per dataset
and keep the result next to the bag.

    python tools/convert_calibration.py \
        --source /path/cam_in_ex.txt --output /path/calibration.yaml \
        --reference-frame odom --pose-frame odin1_base_link \
        --lidar-frame odin1_base_link --camera-frame camera \
        --output-size 800 648
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


MODEL_NAMES = {"FishPoly": "fishpoly", "Pinhole": "pinhole_brown"}
DISTORTION_KEYS = {
    "fishpoly": ("k2", "k3", "k4", "k5", "k6", "k7", "p1", "p2"),
    "pinhole_brown": ("k1", "k2", "p1", "p2", "k3"),
}


def _scalar(value: str) -> Any:
    text = value.split("#", 1)[0].strip()
    try:
        return float(text) if any(character in text for character in ".eE") else int(text)
    except ValueError:
        return text


def parse_cam_in_ex(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    matrix_values: list[float] = []
    in_matrix = False
    parameters: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(": ["):
            in_matrix = True
            continue
        if in_matrix:
            if line == "]":
                in_matrix = False
                continue
            matrix_values.extend(
                float(part) for part in line.rstrip(",").split(",") if part.strip()
            )
            continue
        if line.endswith(":") and not raw.startswith(" "):
            continue
        if ":" in line:
            key, value = (part.strip() for part in line.split(":", 1))
            parameters[key] = _scalar(value)
    if len(matrix_values) != 16:
        raise SystemExit(f"Expected one 4x4 Tcl_0 matrix in {path}")
    return np.asarray(matrix_values, dtype=np.float64).reshape(4, 4), parameters


def build(args: argparse.Namespace) -> dict[str, Any]:
    camera_from_lidar, parameters = parse_cam_in_ex(args.source)
    model = MODEL_NAMES.get(str(parameters.get("cam_model")))
    if model is None:
        raise SystemExit(f"Unsupported cam_model: {parameters.get('cam_model')!r}")
    width = int(parameters["image_width"])
    height = int(parameters["image_height"])
    if model == "fishpoly":
        intrinsics = {
            "fx": float(parameters["A11"]), "fy": float(parameters["A22"]),
            "cx": float(parameters["u0"]), "cy": float(parameters["v0"]),
            "skew": float(parameters["A12"]),
        }
    else:
        intrinsics = {
            "fx": float(parameters["fx"]), "fy": float(parameters["fy"]),
            "cx": float(parameters["cx"]), "cy": float(parameters["cy"]),
            "skew": 0.0,
        }
    distortion = {key: float(parameters.get(key, 0.0)) for key in DISTORTION_KEYS[model]}
    output_width, output_height = args.output_size
    if model == "fishpoly":
        # The FishPoly rectification is parameterized by the output grid itself:
        # a half-width focal length keeps the same field of view the producer used.
        output = {
            "width": output_width, "height": output_height,
            "fx": output_width / 2.0, "fy": output_height / 2.0,
            "cx": output_width / 2.0, "cy": output_height / 2.0,
        }
    else:
        scale_x = output_width / width
        scale_y = output_height / height
        output = {
            "width": output_width, "height": output_height,
            "fx": intrinsics["fx"] * scale_x, "fy": intrinsics["fy"] * scale_y,
            "cx": intrinsics["cx"] * scale_x, "cy": intrinsics["cy"] * scale_y,
        }
    return {
        "schema_name": "sage_calibration",
        "schema_revision": 1,
        "units": {"length": "meter", "angle": "radian"},
        "frames": {
            "reference": args.reference_frame,
            "pose": args.pose_frame,
            "lidar": args.lidar_frame,
            "camera": args.camera_frame,
        },
        "camera": {
            "model": model,
            "width": width,
            "height": height,
            "intrinsics": intrinsics,
            "distortion": distortion,
            "output": output,
        },
        "extrinsics": {
            "camera_from_lidar": camera_from_lidar.tolist(),
            "pose_from_lidar": np.asarray(
                yaml.safe_load(args.pose_from_lidar), dtype=np.float64,
            ).reshape(4, 4).tolist(),
        },
        "time_offsets_to_image_ns": {
            "lidar": args.lidar_time_offset_ns,
            "odometry": args.odometry_time_offset_ns,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-frame", required=True)
    parser.add_argument("--pose-frame", required=True)
    parser.add_argument("--lidar-frame", required=True)
    parser.add_argument("--camera-frame", required=True)
    parser.add_argument("--output-size", required=True, nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument(
        "--pose-from-lidar",
        default="[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
        help="4x4 pose_from_lidar as JSON/YAML; identity when the pose frame is the LiDAR frame",
    )
    parser.add_argument("--lidar-time-offset-ns", type=int, default=0)
    parser.add_argument("--odometry-time-offset-ns", type=int, default=0)
    args = parser.parse_args(argv)
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Canonical calibration written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
