"""Minimal ROS 2 CDR, camera, and point-cloud decoding used by SAGE."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

BASE_OFFSET = 4
DISTORTION_KEYS = ("k2", "k3", "k4", "k5", "k6", "k7", "p1", "p2")
POINTFIELD_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

@dataclass(frozen=True)
class CameraCalibration:
    path: str
    matrix_name: str
    camera_name: str
    camera_params: Dict[str, Any]
    k_like: List[List[float]]
    distortion: Dict[str, Any]
    t_camera_from_lidar: np.ndarray

def align(offset: int, alignment: int) -> int:
    return BASE_OFFSET + (((offset - BASE_OFFSET) + (alignment - 1)) & ~(alignment - 1))


def read_u8(blob: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from("<B", blob, offset)[0], offset + 1


def read_bool(blob: bytes, offset: int) -> Tuple[bool, int]:
    return struct.unpack_from("<?", blob, offset)[0], offset + 1


def read_u32(blob: bytes, offset: int) -> Tuple[int, int]:
    offset = align(offset, 4)
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_i32(blob: bytes, offset: int) -> Tuple[int, int]:
    offset = align(offset, 4)
    return struct.unpack_from("<i", blob, offset)[0], offset + 4


def read_f64(blob: bytes, offset: int) -> Tuple[float, int]:
    offset = align(offset, 8)
    return struct.unpack_from("<d", blob, offset)[0], offset + 8


def read_string(blob: bytes, offset: int) -> Tuple[str, int]:
    offset = align(offset, 4)
    length = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    raw = blob[offset : offset + length]
    offset += length
    return (raw[:-1].decode("utf-8") if length else ""), offset


def parse_header(blob: bytes) -> Dict[str, Any]:
    offset = BASE_OFFSET
    sec, offset = read_i32(blob, offset)
    nsec, offset = read_u32(blob, offset)
    frame_id, offset = read_string(blob, offset)
    return {"stamp_sec": sec, "stamp_nsec": nsec, "frame_id": frame_id, "offset": offset}


def parse_compressed_image(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    image_format, offset = read_string(blob, offset)
    data_len, offset = read_u32(blob, offset)
    data = blob[offset : offset + data_len]
    return {**header, "format": image_format, "data": data, "data_len": data_len}


def decode_compressed_image(blob: bytes) -> Dict[str, Any]:
    import cv2

    message = parse_compressed_image(blob)
    image = cv2.imdecode(np.frombuffer(message["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image.")
    return {**message, "image": image}


def parse_odometry(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    child_frame_id, offset = read_string(blob, offset)
    position = []
    for _ in range(3):
        value, offset = read_f64(blob, offset)
        position.append(value)
    orientation = []
    for _ in range(4):
        value, offset = read_f64(blob, offset)
        orientation.append(value)
    return {
        **header,
        "child_frame_id": child_frame_id,
        "position": np.asarray(position, dtype=np.float64),
        "orientation_xyzw": np.asarray(orientation, dtype=np.float64),
    }


def parse_pointcloud2(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    height, offset = read_u32(blob, offset)
    width, offset = read_u32(blob, offset)
    field_count, offset = read_u32(blob, offset)
    fields = []
    for _ in range(field_count):
        name, offset = read_string(blob, offset)
        field_offset, offset = read_u32(blob, offset)
        datatype, offset = read_u8(blob, offset)
        count, offset = read_u32(blob, offset)
        fields.append({"name": name, "offset": field_offset, "datatype": datatype, "count": count})
    is_bigendian, offset = read_bool(blob, offset)
    point_step, offset = read_u32(blob, offset)
    row_step, offset = read_u32(blob, offset)
    data_len, offset = read_u32(blob, offset)
    data = blob[offset : offset + data_len]
    return {
        **header,
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data_len": data_len,
        "data": data,
    }


def parse_camera_lidar_calibration(path: Path) -> CameraCalibration:
    matrix_name = None
    in_matrix_block = False
    matrix_values: List[float] = []
    camera_name = None
    camera_params: Dict[str, Any] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(": ["):
            matrix_name = line[:-3]
            in_matrix_block = True
            matrix_values = []
            continue
        if in_matrix_block:
            if line == "]":
                in_matrix_block = False
                continue
            cleaned = line.rstrip(",")
            matrix_values.extend(float(part.strip()) for part in cleaned.split(",") if part.strip())
            continue
        if line.endswith(":") and not raw_line.startswith(" "):
            camera_name = line[:-1]
            continue
        if camera_name and ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
            camera_params[key] = _parse_scalar(value)

    if len(matrix_values) != 16:
        raise ValueError(f"Expected one 4x4 calibration matrix in {path}, got {len(matrix_values)} values.")
    matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4)
    k_like = [
        [camera_params["A11"], camera_params["A12"], camera_params["u0"]],
        [0.0, camera_params["A22"], camera_params["v0"]],
        [0.0, 0.0, 1.0],
    ]
    distortion = {key: camera_params.get(key) for key in DISTORTION_KEYS}
    return CameraCalibration(
        path=str(path),
        matrix_name=str(matrix_name),
        camera_name=str(camera_name),
        camera_params=camera_params,
        k_like=k_like,
        distortion=distortion,
        t_camera_from_lidar=matrix,
    )


def _parse_scalar(value: str) -> Any:
    value = value.split("#", 1)[0].strip()
    if value in {"", "."}:
        return value
    try:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def build_fishpoly_rectification_maps(camera_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    k_mat = np.asarray(camera_data["K_like"], dtype=np.float64)
    src_width = int(camera_data["width"])
    src_height = int(camera_data["height"])
    a11 = float(k_mat[0, 0])
    a12 = float(k_mat[0, 1])
    a22 = float(k_mat[1, 1])
    u0 = float(k_mat[0, 2])
    v0 = float(k_mat[1, 2])
    distortion = camera_data["distortion"]
    k2 = float(distortion["k2"])
    k3 = float(distortion["k3"])
    k4 = float(distortion["k4"])
    k5 = float(distortion["k5"])
    k6 = float(distortion["k6"])
    k7 = float(distortion["k7"])

    output_width = src_width
    output_height = src_height
    fx = output_width / 2.0
    fy = output_height / 2.0
    cx = output_width / 2.0
    cy = output_height / 2.0

    u, v = np.meshgrid(np.arange(output_width, dtype=np.float64), np.arange(output_height, dtype=np.float64))
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta_d = theta * (
        1.0
        + k2 * theta**2
        + k3 * theta**3
        + k4 * theta**4
        + k5 * theta**5
        + k6 * theta**6
        + k7 * theta**7
    )
    scale = np.divide(theta_d, r, out=np.ones_like(theta_d), where=r > 1e-12)
    xd = scale * x
    yd = scale * y
    map_x = a11 * xd + a12 * yd + u0
    map_y = a22 * yd + v0
    pinhole = {"width": output_width, "height": output_height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return map_x.astype(np.float32), map_y.astype(np.float32), pinhole


def build_sampler(map_x: np.ndarray, map_y: np.ndarray, source_width: int, source_height: int) -> Dict[str, np.ndarray]:
    valid = (map_x >= 0.0) & (map_x <= source_width - 1) & (map_y >= 0.0) & (map_y <= source_height - 1)
    x0 = np.floor(np.clip(map_x, 0, source_width - 1)).astype(np.int32)
    y0 = np.floor(np.clip(map_y, 0, source_height - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (np.clip(map_x, 0, source_width - 1) - x0).astype(np.float32)
    wy = (np.clip(map_y, 0, source_height - 1) - y0).astype(np.float32)
    return {"valid": valid, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "wx": wx, "wy": wy}


def remap_bilinear(image: np.ndarray, sampler: Dict[str, np.ndarray]) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    x0 = sampler["x0"]
    x1 = sampler["x1"]
    y0 = sampler["y0"]
    y1 = sampler["y1"]
    wx = sampler["wx"][:, :, None]
    wy = sampler["wy"][:, :, None]
    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    out = top * (1.0 - wy) + bottom * wy
    out[~sampler["valid"]] = 0.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :, 0] if image.ndim == 2 else out


def _dtype_for_field(field: Dict[str, Any], endian: str) -> Tuple[np.dtype, int]:
    base = POINTFIELD_DTYPE.get(int(field["datatype"]))
    if base is None:
        raise ValueError(f"Unsupported PointField datatype: {field['datatype']}")
    dtype = np.dtype(base).newbyteorder(endian)
    return dtype, int(field.get("count", 1))


def decode_structured_fields(pointcloud: Dict[str, Any], field_names: Iterable[str]) -> Dict[str, np.ndarray]:
    selected = []
    endian = ">" if pointcloud.get("is_bigendian") else "<"
    for field in pointcloud["fields"]:
        if field["name"] not in field_names:
            continue
        dtype, count = _dtype_for_field(field, endian)
        if count == 1:
            selected.append((field["name"], dtype, int(field["offset"])))
        else:
            selected.append((field["name"], dtype, int(field["offset"]), count))
    dtype_spec = {
        "names": [entry[0] for entry in selected],
        "formats": [entry[1] if len(entry) == 3 else (entry[1], (entry[3],)) for entry in selected],
        "offsets": [entry[2] for entry in selected],
        "itemsize": int(pointcloud["point_step"]),
    }
    structured = np.frombuffer(
        pointcloud["data"],
        dtype=np.dtype(dtype_spec),
        count=int(pointcloud["width"]) * int(pointcloud["height"]),
    )
    return {name: np.asarray(structured[name]) for name in structured.dtype.names}


def decode_raw_cloud(pointcloud: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fields = decode_structured_fields(pointcloud, ("x", "y", "z", "intensity", "confidence", "offset_time"))
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    payload = {"xyz": xyz[finite]}
    for key in ("intensity", "confidence", "offset_time"):
        values = fields.get(key)
        if values is not None:
            payload[key] = np.asarray(values)[finite]
    return payload


def decode_rgb_field(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype == np.float32:
        packed = values.view(np.uint32)
    elif values.dtype == np.float64:
        packed = values.astype(np.float32).view(np.uint32)
    else:
        packed = values.astype(np.uint32, copy=False)
    rgb = np.empty((len(packed), 3), dtype=np.uint8)
    rgb[:, 0] = (packed >> 16) & 0xFF
    rgb[:, 1] = (packed >> 8) & 0xFF
    rgb[:, 2] = packed & 0xFF
    return rgb


def decode_slam_cloud(pointcloud: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fields = decode_structured_fields(pointcloud, ("x", "y", "z", "rgb"))
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32, copy=False)
    finite = np.isfinite(xyz).all(axis=1)
    payload = {"xyz": xyz[finite]}
    if "rgb" in fields:
        payload["rgb"] = decode_rgb_field(fields["rgb"])[finite]
    return payload
