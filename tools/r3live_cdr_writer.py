"""Minimal ROS2-CDR encoders + a matching rosbag2 SQLite (.db3) writer.

Mirrors the byte layouts that ``sage.data.rosbag.decoder`` already knows how to
parse (see ``parse_header``/``parse_compressed_image``/``parse_odometry``/
``parse_pointcloud2``), just in the write direction. Only the fields the
decoder actually reads are encoded -- trailing message fields the adapter
never touches (e.g. Odometry covariance/twist, PointCloud2 is_dense) are
omitted, since nothing downstream reads past them.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

CDR_LE_HEADER = b"\x00\x01\x00\x00"


class CdrWriter:
    """Grows a byte buffer using the same alignment rules as decoder.align()."""

    def __init__(self) -> None:
        self._buf = bytearray(CDR_LE_HEADER)

    def _align(self, alignment: int) -> None:
        # CDR alignment is relative to the payload start, i.e. right after the
        # 4-byte encapsulation header -- mirrors decoder.align()'s BASE_OFFSET.
        relative = len(self._buf) - len(CDR_LE_HEADER)
        pad = (-relative) % alignment
        self._buf.extend(b"\x00" * pad)

    def i32(self, value: int) -> "CdrWriter":
        self._align(4)
        self._buf.extend(struct.pack("<i", value))
        return self

    def u32(self, value: int) -> "CdrWriter":
        self._align(4)
        self._buf.extend(struct.pack("<I", value))
        return self

    def u8(self, value: int) -> "CdrWriter":
        self._buf.extend(struct.pack("<B", value))
        return self

    def boolean(self, value: bool) -> "CdrWriter":
        self._buf.extend(struct.pack("<?", bool(value)))
        return self

    def f64(self, value: float) -> "CdrWriter":
        self._align(8)
        self._buf.extend(struct.pack("<d", float(value)))
        return self

    def string(self, value: str) -> "CdrWriter":
        encoded = value.encode("utf-8") + b"\x00"
        self.u32(len(encoded))
        self._buf.extend(encoded)
        return self

    def bytes_field(self, value: bytes) -> "CdrWriter":
        self.u32(len(value))
        self._buf.extend(value)
        return self

    def raw(self, value: bytes) -> "CdrWriter":
        self._buf.extend(value)
        return self

    def to_bytes(self) -> bytes:
        return bytes(self._buf)


def encode_header(writer: CdrWriter, *, stamp_sec: int, stamp_nsec: int, frame_id: str) -> None:
    writer.i32(stamp_sec).u32(stamp_nsec).string(frame_id)


def encode_compressed_image(
    *, stamp_sec: int, stamp_nsec: int, frame_id: str, image_format: str, data: bytes,
) -> bytes:
    writer = CdrWriter()
    encode_header(writer, stamp_sec=stamp_sec, stamp_nsec=stamp_nsec, frame_id=frame_id)
    writer.string(image_format)
    writer.bytes_field(data)
    return writer.to_bytes()


def encode_odometry(
    *,
    stamp_sec: int,
    stamp_nsec: int,
    frame_id: str,
    child_frame_id: str,
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
) -> bytes:
    writer = CdrWriter()
    encode_header(writer, stamp_sec=stamp_sec, stamp_nsec=stamp_nsec, frame_id=frame_id)
    writer.string(child_frame_id)
    for value in position:
        writer.f64(value)
    for value in orientation_xyzw:
        writer.f64(value)
    return writer.to_bytes()


def encode_pointcloud2_xyzrgb(
    *, stamp_sec: int, stamp_nsec: int, frame_id: str, xyz: np.ndarray, rgb: np.ndarray,
) -> bytes:
    """x,y,z float32 + packed-float32 rgb, point_step=16 -- matches EXPECTED_POINT_LAYOUTS['LIDAR_WORLD']."""
    n = len(xyz)
    writer = CdrWriter()
    encode_header(writer, stamp_sec=stamp_sec, stamp_nsec=stamp_nsec, frame_id=frame_id)
    writer.u32(1).u32(n)  # height, width
    fields = (("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1), ("rgb", 12, 7, 1))
    writer.u32(len(fields))
    for name, offset, datatype, count in fields:
        writer.string(name).u32(offset).u8(datatype).u32(count)
    writer.boolean(False)  # is_bigendian
    writer.u32(16)  # point_step
    writer.u32(16 * n)  # row_step
    packed_rgb = (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    ).astype(np.uint32)
    points = np.zeros((n, 4), dtype=np.float32)
    points[:, :3] = xyz.astype(np.float32, copy=False)
    points[:, 3] = packed_rgb.view(np.float32)
    writer.bytes_field(points.tobytes())
    return writer.to_bytes()


def write_ros2_bag(
    output_dir: Path,
    *,
    topics: dict[str, str],
    messages: Iterable[tuple[str, int, bytes]],
) -> None:
    """topics: {topic_name: message_type}. messages: iterable of (topic, timestamp_ns, cdr_bytes)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "synthetic_0.db3"
    db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.executescript(
            """
            CREATE TABLE topics (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                serialization_format TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                data BLOB NOT NULL
            );
            """
        )
        topic_ids = {}
        for name, message_type in topics.items():
            cursor = connection.execute(
                "INSERT INTO topics (name, type, serialization_format) VALUES (?, ?, 'cdr')",
                (name, message_type),
            )
            topic_ids[name] = cursor.lastrowid
        connection.executemany(
            "INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)",
            ((topic_ids[topic], timestamp_ns, data) for topic, timestamp_ns, data in messages),
        )
        connection.commit()
    finally:
        connection.close()
    (output_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  relative_file_paths:\n"
        f"    - '{db_path.name}'\n",
        encoding="utf-8",
    )
