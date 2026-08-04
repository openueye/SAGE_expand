"""A small synthetic ROS 2 bag plus its canonical calibration.

Deliberately tiny and self-contained: the regression fixture must not depend on
the machine-local datasets, and it has to be regenerable from code so a change
in the CDR layout shows up as a test failure rather than a stale artifact.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from r3live_cdr_writer import (  # noqa: E402
    CdrWriter,
    encode_compressed_image,
    encode_header,
    encode_odometry,
    write_ros2_bag,
)


REFERENCE_FRAME = "odom"
POSE_FRAME = "base_link"
LIDAR_FRAME = "lidar"
CAMERA_FRAME = "camera"

LIDAR_TOPIC = "/points"
IMAGE_TOPIC = "/camera/image/compressed"
ODOMETRY_TOPIC = "/odom"

WIDTH, HEIGHT = 16, 12
FRAME_COUNT = 12
FRAME_PERIOD_NS = 100_000_000
FIRST_STAMP_NS = 1_000_000_000

# LiDAR x-forward / z-up into camera z-forward / y-down.
CAMERA_FROM_LIDAR = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def encode_pointcloud2_xyz(
    *, stamp_sec: int, stamp_nsec: int, frame_id: str, xyz: np.ndarray,
) -> bytes:
    """x/y/z float32 only: the minimum contract an adapter must accept."""
    points = np.asarray(xyz, dtype=np.float32)
    writer = CdrWriter()
    encode_header(writer, stamp_sec=stamp_sec, stamp_nsec=stamp_nsec, frame_id=frame_id)
    writer.u32(1).u32(len(points))
    fields = (("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1))
    writer.u32(len(fields))
    for name, offset, datatype, count in fields:
        writer.string(name).u32(offset).u8(datatype).u32(count)
    writer.boolean(False)
    writer.u32(12)
    writer.u32(12 * len(points))
    writer.bytes_field(points.tobytes())
    return writer.to_bytes()


def encode_image(
    *, stamp_sec: int, stamp_nsec: int, frame_id: str, image_bgr: np.ndarray,
) -> bytes:
    """A raw sensor_msgs/msg/Image in bgr8."""
    pixels = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    height, width = pixels.shape[:2]
    writer = CdrWriter()
    encode_header(writer, stamp_sec=stamp_sec, stamp_nsec=stamp_nsec, frame_id=frame_id)
    writer.u32(height).u32(width)
    writer.string("bgr8")
    writer.u8(0)
    writer.u32(width * 3)
    writer.bytes_field(pixels.tobytes())
    return writer.to_bytes()


def _split_stamp(timestamp_ns: int) -> tuple[int, int]:
    return divmod(timestamp_ns, 1_000_000_000)


def _image(index: int) -> np.ndarray:
    """A deterministic gradient that differs per frame, in BGR."""
    rows = np.linspace(0, 255, HEIGHT, dtype=np.float64)[:, None]
    columns = np.linspace(0, 255, WIDTH, dtype=np.float64)[None, :]
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 0] = np.clip(rows + index, 0, 255)
    image[:, :, 1] = np.clip(columns, 0, 255)
    image[:, :, 2] = np.clip(rows + columns, 0, 255) // 2
    return image


def _scan(index: int) -> np.ndarray:
    """A wall in front of the sensor, shifted per scan so fusion has content."""
    y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, 9), np.linspace(-0.6, 0.6, 7), indexing="ij",
    )
    x = np.full(y.size, 4.0) + 0.01 * index
    return np.stack([x, y.reshape(-1), z.reshape(-1)], axis=1).astype(np.float32)


def _reference_from_pose(index: int) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = 0.05 * index
    return matrix


def write_synthetic_bag(
    root: Path,
    *,
    image_encoding: str = "compressed",
    image_frame_id: str = CAMERA_FRAME,
    frame_count: int = FRAME_COUNT,
) -> Path:
    bag = Path(root) / "bag"
    messages: list[tuple[str, int, bytes]] = []
    # One extra pose on each side so every image timestamp is bracketed.
    for index in range(-1, frame_count + 1):
        timestamp = FIRST_STAMP_NS + index * FRAME_PERIOD_NS
        sec, nsec = _split_stamp(timestamp)
        messages.append((ODOMETRY_TOPIC, timestamp, encode_odometry(
            stamp_sec=sec, stamp_nsec=nsec,
            frame_id=REFERENCE_FRAME, child_frame_id=POSE_FRAME,
            position=_reference_from_pose(index)[:3, 3],
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        )))
    for index in range(frame_count):
        timestamp = FIRST_STAMP_NS + index * FRAME_PERIOD_NS
        sec, nsec = _split_stamp(timestamp)
        if image_encoding == "compressed":
            buffer = io.BytesIO()
            Image.fromarray(_image(index)[:, :, ::-1], mode="RGB").save(buffer, format="PNG")
            payload = encode_compressed_image(
                stamp_sec=sec, stamp_nsec=nsec, frame_id=image_frame_id,
                image_format="png", data=buffer.getvalue(),
            )
        else:
            payload = encode_image(
                stamp_sec=sec, stamp_nsec=nsec, frame_id=image_frame_id,
                image_bgr=_image(index),
            )
        messages.append((IMAGE_TOPIC, timestamp, payload))
        messages.append((LIDAR_TOPIC, timestamp, encode_pointcloud2_xyz(
            stamp_sec=sec, stamp_nsec=nsec, frame_id=LIDAR_FRAME, xyz=_scan(index),
        )))
    write_ros2_bag(bag, topics={
        LIDAR_TOPIC: "sensor_msgs/msg/PointCloud2",
        IMAGE_TOPIC: (
            "sensor_msgs/msg/CompressedImage" if image_encoding == "compressed"
            else "sensor_msgs/msg/Image"
        ),
        ODOMETRY_TOPIC: "nav_msgs/msg/Odometry",
    }, messages=messages)
    return bag


def write_calibration(root: Path) -> Path:
    path = Path(root) / "calibration.yaml"
    path.write_text(yaml.safe_dump({
        "schema_name": "sage_calibration",
        "schema_revision": 1,
        "units": {"length": "meter", "angle": "radian"},
        "frames": {
            "reference": REFERENCE_FRAME,
            "pose": POSE_FRAME,
            "lidar": LIDAR_FRAME,
            "camera": CAMERA_FRAME,
        },
        "camera": {
            "model": "pinhole_brown",
            "width": WIDTH,
            "height": HEIGHT,
            "intrinsics": {
                "fx": 10.0, "fy": 10.0, "cx": WIDTH / 2, "cy": HEIGHT / 2, "skew": 0.0,
            },
            "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
            "output": {
                "width": WIDTH, "height": HEIGHT,
                "fx": 10.0, "fy": 10.0, "cx": WIDTH / 2, "cy": HEIGHT / 2,
            },
        },
        "extrinsics": {
            "camera_from_lidar": CAMERA_FROM_LIDAR.tolist(),
            "pose_from_lidar": np.eye(4).tolist(),
        },
        "time_offsets_to_image_ns": {"lidar": 0, "odometry": 0},
    }, sort_keys=False), encoding="utf-8")
    return path


def synthetic_input(root: Path, **spec_overrides):
    """A resolvable RosbagInputSpec over a freshly written synthetic bag."""
    from sage.input.rosbag import RosbagInputSpec

    bag = write_synthetic_bag(root, **{
        key: value
        for key, value in spec_overrides.items()
        if key in {"image_encoding", "image_frame_id"}
    })
    calibration = write_calibration(root)
    return RosbagInputSpec(
        rosbag_path=bag,
        calibration_path=calibration,
        lidar_topic=LIDAR_TOPIC,
        image_topic=IMAGE_TOPIC,
        odometry_topic=ODOMETRY_TOPIC,
        points_frame="lidar_frame",
        **{
            key: value
            for key, value in spec_overrides.items()
            if key not in {"image_encoding", "image_frame_id"}
        },
    )


__all__ = [
    "CAMERA_FROM_LIDAR",
    "FRAME_COUNT",
    "encode_image",
    "encode_pointcloud2_xyz",
    "synthetic_input",
    "write_calibration",
    "write_synthetic_bag",
]
