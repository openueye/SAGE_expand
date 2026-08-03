"""Convert an R3LIVE ROS1 bag into a synthetic ROS2-bag-shaped Prepared Scene
source, compatible with the r3live-lidar-world-v1 preparation_profile.

R3LIVE bags ship raw /livox/lidar (livox_ros_driver/CustomMsg), /livox/imu,
and /camera/image_color/compressed -- no pose/odometry topic. This script:

  1. Reads the ROS1 bag (rosbags library; the bag embeds the CustomMsg
     definition, so no ROS install is required).
  2. Runs KISS-ICP (LiDAR-only odometry, no loop closure) over the raw scans
     to get a world_from_lidar pose per scan.
  3. For each camera frame, interpolates the bracketing scan poses, colorizes
     the nearest scan's points by projecting them into that frame (official
     R3LIVE camera_intrinsic/camera_dist_coeffs/camera_ext_R/t), and emits a
     colored world-frame point cloud + odometry sample at the image timestamp.
  4. Writes /r3live/cloud_slam, /r3live/odometry, /r3live/image/compressed
     into a synthetic ROS2 .db3 bag directory via r3live_cdr_writer.

Usage:
    python tools/r3live_to_bag.py --bag /path/to/hku_park_00.bag --output /path/to/out_dir [--frame-limit N]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3live_cdr_writer import encode_compressed_image, encode_odometry, encode_pointcloud2_xyzrgb, write_ros2_bag

# Official calibration: github.com/hku-mars/r3live config/r3live_config.yaml
CAMERA_K = np.asarray(
    [[863.4241, 0.0, 640.6808], [0.0, 863.4171, 518.3392], [0.0, 0.0, 1.0]], dtype=np.float64,
)
CAMERA_DIST = np.asarray([-0.1080, 0.1050, -1.2872e-04, 5.7923e-05, -0.0222], dtype=np.float64)
# r3live's published camera_ext_R/t is actually T_lidar_from_camera (p_lidar = R@p_camera + t);
# inverted here to T_camera_from_lidar, matching cam_in_ex.txt's Tcl_0 and _colorize()'s convention.
CAMERA_EXT_R = np.asarray(
    [
        [-0.00113207, -0.9999999, 0.00050462],
        [-0.01586880, -0.00048659, -0.99987400],
        [0.99987300, -0.00113994, -0.01586820],
    ],
    dtype=np.float64,
)
CAMERA_EXT_T = np.asarray([0.04748415, -0.03041842, -0.05060133], dtype=np.float64)

LIDAR_TOPIC = "/livox/lidar"
IMAGE_TOPIC = "/camera/image_color/compressed"

IMAGE_OUT_TOPIC = "/r3live/image/compressed"
ODOMETRY_OUT_TOPIC = "/r3live/odometry"
CLOUD_OUT_TOPIC = "/r3live/cloud_slam"


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = 2.0 * np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12))
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x, y, z = 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12))
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x, y, z = (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12))
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x, y, z = (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    left = q0 / np.linalg.norm(q0)
    right = q1 / np.linalg.norm(q1)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right, dot = -right, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + alpha * (right - left)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (np.sin((1.0 - alpha) * angle) / scale) * left + (np.sin(alpha * angle) / scale) * right


def _interpolate_pose(pose_a: np.ndarray, pose_b: np.ndarray, alpha: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
    q0 = _matrix_to_quaternion_xyzw(pose_a[:3, :3])
    q1 = _matrix_to_quaternion_xyzw(pose_b[:3, :3])
    x, y, z, w = _slerp(q0, q1, alpha)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return matrix


def _read_bag(bag_path: Path, frame_limit: int | None):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    lidar_scans: list[tuple[int, np.ndarray]] = []
    images: list[tuple[int, bytes]] = []
    with Reader(bag_path) as reader:
        for connection in reader.connections:
            if connection.msgtype == "livox_ros_driver/msg/CustomMsg":
                typestore.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
        connections = [c for c in reader.connections if c.topic in (LIDAR_TOPIC, IMAGE_TOPIC)]
        for connection, timestamp_ns, raw in reader.messages(connections=connections):
            if connection.topic == LIDAR_TOPIC:
                if frame_limit is not None and len(lidar_scans) >= frame_limit:
                    continue
                message = typestore.deserialize_ros1(raw, connection.msgtype)
                points = np.asarray(
                    [(point.x, point.y, point.z) for point in message.points], dtype=np.float64,
                )
                finite = np.isfinite(points).all(axis=1)
                lidar_scans.append((int(timestamp_ns), points[finite]))
            elif connection.topic == IMAGE_TOPIC:
                message = typestore.deserialize_ros1(raw, connection.msgtype)
                images.append((int(timestamp_ns), bytes(message.data)))
    lidar_scans.sort(key=lambda item: item[0])
    images.sort(key=lambda item: item[0])
    if frame_limit is not None:
        images = [item for item in images if item[0] <= lidar_scans[-1][0]] if lidar_scans else images
    return lidar_scans, images


def _run_kiss_icp(lidar_scans: list[tuple[int, np.ndarray]]) -> list[np.ndarray]:
    from kiss_icp.config import KISSConfig
    from kiss_icp.config.config import DataConfig, MappingConfig
    from kiss_icp.kiss_icp import KissICP

    data_config = DataConfig(deskew=False)
    config = KISSConfig(data=data_config, mapping=MappingConfig(voxel_size=data_config.max_range / 100.0))
    odometry = KissICP(config)
    poses: list[np.ndarray] = []
    for _, points in lidar_scans:
        timestamps = np.zeros(len(points), dtype=np.float64)
        odometry.register_frame(points, timestamps)
        poses.append(odometry.last_pose.copy())
    return poses


def _colorize(
    world_points: np.ndarray, target_pose: np.ndarray, image_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame points into the target camera pose's image; keep points landing inside it."""
    lidar_from_world = np.linalg.inv(target_pose)
    lidar_points = (lidar_from_world[:3, :3] @ world_points.T).T + lidar_from_world[:3, 3]
    camera_points = (CAMERA_EXT_R @ lidar_points.T).T + CAMERA_EXT_T
    in_front = camera_points[:, 2] > 0.05
    if not in_front.any():
        return world_points[:0], np.zeros((0, 3), dtype=np.uint8)
    pixels, _ = cv2.projectPoints(
        camera_points[in_front], np.zeros(3), np.zeros(3), CAMERA_K, CAMERA_DIST,
    )
    pixels = pixels.reshape(-1, 2)
    height, width = image_bgr.shape[:2]
    u, v = pixels[:, 0], pixels[:, 1]
    inside = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    kept_world = world_points[in_front][inside]
    kept_u = np.clip(u[inside].astype(np.int32), 0, width - 1)
    kept_v = np.clip(v[inside].astype(np.int32), 0, height - 1)
    bgr = image_bgr[kept_v, kept_u]
    rgb = bgr[:, ::-1]
    return kept_world, rgb


def convert(bag_path: Path, output_dir: Path, *, frame_limit: int | None = None) -> None:
    print(f"Reading {bag_path} ...")
    lidar_scans, images = _read_bag(bag_path, frame_limit)
    if len(lidar_scans) < 2:
        raise ValueError("Need at least two LiDAR scans to run odometry")
    print(f"{len(lidar_scans)} LiDAR scans, {len(images)} camera frames")

    print("Running KISS-ICP odometry ...")
    poses = _run_kiss_icp(lidar_scans)
    scan_timestamps = np.asarray([timestamp for timestamp, _ in lidar_scans], dtype=np.int64)

    window_radius = 2  # accumulate a centered window of scans (like SAGE's own "fused-5") for point density
    messages: list[tuple[str, int, bytes]] = []
    emitted = 0
    for image_timestamp, jpeg_bytes in images:
        right = int(np.searchsorted(scan_timestamps, image_timestamp, side="left"))
        if right <= 0 or right >= len(scan_timestamps):
            continue  # no extrapolation past odometry coverage
        left = right - 1
        span = int(scan_timestamps[right] - scan_timestamps[left])
        alpha = (image_timestamp - int(scan_timestamps[left])) / span if span else 0.0
        pose = _interpolate_pose(poses[left], poses[right], float(alpha))
        nearest_index = left if abs(image_timestamp - scan_timestamps[left]) <= abs(
            scan_timestamps[right] - image_timestamp
        ) else right

        window = range(max(0, nearest_index - window_radius), min(len(lidar_scans), nearest_index + window_radius + 1))
        world_chunks = []
        for index in window:
            points = lidar_scans[index][1]
            if points.size == 0:
                continue
            scan_pose = poses[index]
            world_chunks.append((scan_pose[:3, :3] @ points.T).T + scan_pose[:3, 3])
        if not world_chunks:
            continue
        world_points = np.concatenate(world_chunks, axis=0)

        image = np.asarray(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        world_colored, rgb = _colorize(world_points, pose, image)
        if len(world_colored) < 100:
            continue

        sec, nsec = divmod(image_timestamp, 1_000_000_000)
        messages.append((
            IMAGE_OUT_TOPIC, image_timestamp,
            encode_compressed_image(
                stamp_sec=sec, stamp_nsec=nsec, frame_id="r3live_lidar", image_format="jpeg", data=jpeg_bytes,
            ),
        ))
        quaternion = _matrix_to_quaternion_xyzw(pose[:3, :3])
        messages.append((
            ODOMETRY_OUT_TOPIC, image_timestamp,
            encode_odometry(
                stamp_sec=sec, stamp_nsec=nsec, frame_id="odom", child_frame_id="r3live_lidar",
                position=pose[:3, 3], orientation_xyzw=quaternion,
            ),
        ))
        messages.append((
            CLOUD_OUT_TOPIC, image_timestamp,
            encode_pointcloud2_xyzrgb(
                stamp_sec=sec, stamp_nsec=nsec, frame_id="odom", xyz=world_colored, rgb=rgb,
            ),
        ))
        emitted += 1

    if emitted == 0:
        raise ValueError("No camera frame fell inside odometry coverage; nothing to write")
    print(f"Writing {emitted} synced (image, odometry, cloud) triples to {output_dir}")
    write_ros2_bag(
        output_dir,
        topics={
            IMAGE_OUT_TOPIC: "sensor_msgs/msg/CompressedImage",
            ODOMETRY_OUT_TOPIC: "nav_msgs/msg/Odometry",
            CLOUD_OUT_TOPIC: "sensor_msgs/msg/PointCloud2",
        },
        messages=messages,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-limit", type=int, default=None, help="cap LiDAR scans read (smoke tests)")
    args = parser.parse_args(argv)
    convert(args.bag, args.output, frame_limit=args.frame_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
