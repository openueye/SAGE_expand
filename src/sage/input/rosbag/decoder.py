"""Minimal ROS 2 CDR decoding for the message types SAGE ingests.

Nothing here knows about datasets, topics or roles: it turns bytes into plain
dictionaries. Which topic carries which role is the caller's decision.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np


BASE_OFFSET = 4

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

# Raw sensor_msgs/msg/Image encodings we can turn into canonical RGB without
# guessing. Anything else must fail loudly rather than be reinterpreted.
IMAGE_ENCODINGS = {
    "rgb8": (3, np.uint8, "rgb"),
    "bgr8": (3, np.uint8, "bgr"),
    "mono8": (1, np.uint8, "mono"),
}

SUPPORTED_IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
SUPPORTED_LIDAR_TYPES = ("sensor_msgs/msg/PointCloud2",)
SUPPORTED_ODOMETRY_TYPES = ("nav_msgs/msg/Odometry",)


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


def header_timestamp_ns(message: Dict[str, Any]) -> int:
    return int(message["stamp_sec"]) * 1_000_000_000 + int(message["stamp_nsec"])


def parse_compressed_image(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    image_format, offset = read_string(blob, offset)
    data_len, offset = read_u32(blob, offset)
    return {
        **header,
        "format": image_format,
        "data": blob[offset : offset + data_len],
        "data_len": data_len,
    }


def parse_image(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    height, offset = read_u32(blob, offset)
    width, offset = read_u32(blob, offset)
    encoding, offset = read_string(blob, offset)
    is_bigendian, offset = read_u8(blob, offset)
    step, offset = read_u32(blob, offset)
    data_len, offset = read_u32(blob, offset)
    return {
        **header,
        "height": height,
        "width": width,
        "encoding": encoding,
        "is_bigendian": bool(is_bigendian),
        "step": step,
        "data": blob[offset : offset + data_len],
        "data_len": data_len,
    }


def decode_compressed_image(blob: bytes) -> np.ndarray:
    """Decode to BGR uint8, the layout OpenCV hands back."""
    message = parse_compressed_image(blob)
    image = cv2.imdecode(np.frombuffer(message["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode CompressedImage payload")
    return image


def decode_image(blob: bytes) -> np.ndarray:
    """Decode a raw Image message to BGR uint8."""
    message = parse_image(blob)
    encoding = str(message["encoding"])
    layout = IMAGE_ENCODINGS.get(encoding)
    if layout is None:
        raise ValueError(
            f"Unsupported Image encoding {encoding!r}; supported: {sorted(IMAGE_ENCODINGS)}"
        )
    channels, dtype, order = layout
    height, width, step = int(message["height"]), int(message["width"]), int(message["step"])
    expected = height * step
    if step < width * channels or int(message["data_len"]) < expected:
        raise ValueError("Image message dimensions do not match its payload")
    rows = np.frombuffer(message["data"][:expected], dtype=dtype).reshape(height, step)
    pixels = rows[:, : width * channels].reshape(height, width, channels)
    if order == "mono":
        return np.repeat(pixels, 3, axis=2).copy()
    return pixels[:, :, ::-1].copy() if order == "rgb" else pixels.copy()


def decode_image_message(message_type: str, blob: bytes) -> np.ndarray:
    if message_type == "sensor_msgs/msg/CompressedImage":
        return decode_compressed_image(blob)
    if message_type == "sensor_msgs/msg/Image":
        return decode_image(blob)
    raise ValueError(f"Unsupported image message type: {message_type}")


def compressed_image_codec(blob: bytes) -> str:
    """Record the concrete codec (jpeg/png) a CompressedImage actually carries."""
    payload = parse_compressed_image(blob)["data"]
    if payload[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if payload[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    raise ValueError("CompressedImage payload is neither JPEG nor PNG")


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
    if offset > len(blob):
        raise ValueError("PointCloud2 payload is truncated before its point data")
    return {
        **header,
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data_len": data_len,
        # memoryview, not a slice copy: indexing a bag re-reads every cloud
        # header and copying half a megabyte per message just to drop it again
        # dominated preflight.
        "data": memoryview(blob)[offset : offset + data_len],
    }


def parse_pointcloud2_prefix(prefix: bytes) -> Dict[str, Any]:
    """Parse everything but the point data, from a truncated blob prefix.

    Indexing a bag only needs the stamp, frame and layout, all of which live in
    the first few hundred bytes. Reading the full cloud back from SQLite for
    that would move gigabytes for nothing.
    """
    message = dict(parse_pointcloud2(prefix))
    message["data"] = memoryview(b"")
    return message


def pointcloud_layout(message: Dict[str, Any]) -> Dict[str, Any]:
    """Structural identity of a cloud, used to prove the layout never changes."""
    height = int(message["height"])
    width = int(message["width"])
    point_step = int(message["point_step"])
    row_step = int(message["row_step"])
    data_len = int(message["data_len"])
    if min(height, width, point_step) < 1 or row_step != width * point_step or data_len != height * row_step:
        raise ValueError("PointCloud2 payload dimensions do not match point_step and row_step")
    return {
        "fields": [
            {
                "name": str(field["name"]),
                "offset": int(field["offset"]),
                "datatype": int(field["datatype"]),
                "count": int(field["count"]),
            }
            for field in message["fields"]
        ],
        "height": height,
        "is_bigendian": bool(message["is_bigendian"]),
        "point_step": point_step,
    }


def _dtype_for_field(field: Dict[str, Any], endian: str) -> Tuple[np.dtype, int]:
    base = POINTFIELD_DTYPE.get(int(field["datatype"]))
    if base is None:
        raise ValueError(f"Unsupported PointField datatype: {field['datatype']}")
    return np.dtype(base).newbyteorder(endian), int(field.get("count", 1))


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


def decode_points_xyz(pointcloud: Dict[str, Any]) -> np.ndarray:
    """Nx3 float32 finite points. x/y/z is the only hard field requirement."""
    names = {str(field["name"]) for field in pointcloud["fields"]}
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"PointCloud2 is missing required fields: {sorted(missing)}")
    fields = decode_structured_fields(pointcloud, ("x", "y", "z"))
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32, copy=False)
    return xyz[np.isfinite(xyz).all(axis=1)]


__all__ = [
    "SUPPORTED_IMAGE_TYPES",
    "SUPPORTED_LIDAR_TYPES",
    "SUPPORTED_ODOMETRY_TYPES",
    "compressed_image_codec",
    "decode_image_message",
    "decode_points_xyz",
    "header_timestamp_ns",
    "parse_header",
    "parse_odometry",
    "parse_pointcloud2",
    "pointcloud_layout",
]
