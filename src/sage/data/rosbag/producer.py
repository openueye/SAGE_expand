from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock, get_ident
from typing import Iterator

import numpy as np
import cv2
import PIL

from .decoder import (
    build_fishpoly_rectification_maps,
    build_pinhole_rectification_maps,
    build_sampler,
    decode_compressed_image,
    decode_slam_cloud,
    parse_camera_lidar_calibration,
    parse_compressed_image,
    parse_odometry,
    parse_pointcloud2,
    remap_bilinear,
)
from .poses import pose_to_matrix

from .geometry import (
    CameraIntrinsics,
    CenteredFiveProjector,
    CenteredFiveWindow,
    FinalizedSensorFrame,
    LidarWorldNormalizer,
    PoseSample,
    PoseTrack,
    ProjectedCenter,
    RejectedCenter,
    RejectedSensorFrame,
    WorldCloudEvidence,
)
from .profiles import PreparationProfile, profile_for_name
from .writer import sha256_file


def _expected_topic_identities(profile: PreparationProfile) -> dict[str, tuple[str, str]]:
    """Message type/serialization each profile topic must declare, by role (device-agnostic)."""
    return {
        profile.source.image_topic: ("sensor_msgs/msg/CompressedImage", "cdr"),
        profile.source.odometry_topic: ("nav_msgs/msg/Odometry", "cdr"),
        profile.source.topic: ("sensor_msgs/msg/PointCloud2", "cdr"),
    }


def _bag_shards(root: Path, metadata: Path) -> tuple[Path, ...]:
    """Return ROSBAG shards in metadata-declared order when available."""
    declared = tuple(
        match.group(1)
        for match in re.finditer(
            r"^\s*-\s*['\"]?([^'\"\s]+\.db3)['\"]?\s*$",
            metadata.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    if declared:
        shards = tuple((root / name).resolve() for name in declared)
        if len(set(shards)) != len(shards) or any(
            shard.parent != root or not shard.is_file() for shard in shards
        ):
            raise ValueError("metadata.yaml declares invalid ROSBAG shard paths")
        return shards
    shards = tuple(sorted(root.glob("*.db3")))
    if not shards:
        raise ValueError("ROS bag folder must contain at least one .db3 shard")
    return shards
EXPECTED_POINT_LAYOUTS = {
    "LIDAR_WORLD": {
        "fields": (
            ("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1),
            ("rgb", 12, 7, 1),
        ),
        "height": 1,
        "is_bigendian": False,
        "point_step": 16,
    },
}


@dataclass(frozen=True)
class BagMessage:
    header_timestamp_ns: int
    storage_file_index: int
    storage_row_id: int
    data: bytes


@dataclass(frozen=True)
class BagInputs:
    messages: dict[str, tuple[BagMessage, ...]]
    topic_identities: dict[str, dict[str, object]]
    input_files: dict[str, Path]


@dataclass
class _StreamingReadConnections:
    index_queries: sqlite3.Connection
    image_iteration: sqlite3.Connection
    shards: dict[str, sqlite3.Connection]


class StreamingBagIndex:
    """Disk-backed event index for fixed-lag production.

    The index retains only event metadata and row locators. Message payloads are
    fetched from the source shard when a bounded window needs them. Each calling
    thread receives persistent SQLite read handles so fixed-lag lookups do
    not reopen a database for every payload or pose bracket.
    """

    def __init__(self, rosbag_dir: Path, profile: PreparationProfile) -> None:
        root = Path(rosbag_dir).resolve()
        metadata = root / "metadata.yaml"
        if not metadata.is_file():
            raise ValueError("ROS bag folder must contain metadata.yaml")
        shards = _bag_shards(root, metadata)
        self.root = root
        self.input_files = {"metadata.yaml": metadata}
        self.input_files.update({f"rosbag/{path.name}": path for path in shards})
        self._shards = shards
        index_fd, index_name = tempfile.mkstemp(prefix=".odin-stream-index-", suffix=".sqlite3")
        os.close(index_fd)
        self._index_path = Path(index_name)
        self._closed = False
        self._reader_lock = RLock()
        self._readers_by_thread: dict[int, _StreamingReadConnections] = {}
        self._topic_identities: dict[str, dict[str, object]] = {}
        self._source_topic = profile.source.topic
        self._image_topic = profile.source.image_topic
        self._odometry_topic = profile.source.odometry_topic
        self._source_frame_ids: set[str] = set()
        self._odom_frame_ids: set[str] = set()
        self._odom_child_frame_ids: set[str] = set()
        self._source_layouts: list[dict[str, object]] = []
        try:
            self._build(profile)
        except BaseException:
            self._index_path.unlink(missing_ok=True)
            raise

    @property
    def topic_identities(self) -> dict[str, dict[str, object]]:
        return self._topic_identities

    @property
    def source_frame_ids(self) -> set[str]:
        return set(self._source_frame_ids)

    @property
    def odom_frame_ids(self) -> set[str]:
        return set(self._odom_frame_ids)

    @property
    def odom_child_frame_ids(self) -> set[str]:
        return set(self._odom_child_frame_ids)

    @property
    def source_layout(self) -> dict[str, object]:
        if not self._source_layouts:
            raise ValueError("Streaming bag source layout is unavailable")
        return self._source_layouts[0]

    def _build(self, profile: PreparationProfile) -> None:
        required_topics = (self._image_topic, self._odometry_topic, profile.source.topic)
        expected_identities = _expected_topic_identities(profile)
        connection = sqlite3.connect(str(self._index_path))
        try:
            connection.executescript(
                """
                CREATE TABLE events (
                    topic TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    storage_file_index INTEGER NOT NULL,
                    storage_row_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    shard_name TEXT NOT NULL,
                    PRIMARY KEY (topic, storage_file_index, storage_row_id)
                );
                CREATE INDEX events_topic_time ON events(topic, timestamp_ns, storage_file_index, storage_row_id);
                """
            )
            for file_index, shard in enumerate(self._shards):
                with sqlite3.connect(str(shard)) as shard_connection:
                    topics = {
                        str(name): (int(topic_id), str(message_type), str(serialization))
                        for topic_id, name, message_type, serialization in shard_connection.execute(
                            "SELECT id, name, type, serialization_format FROM topics"
                        )
                    }
                    for topic in required_topics:
                        if topic not in topics:
                            continue
                        topic_id, message_type, serialization = topics[topic]
                        expected_type, expected_serialization = expected_identities[topic]
                        if (message_type, serialization) != (expected_type, expected_serialization):
                            raise ValueError(f"Topic identity mismatch for {topic}: {(message_type, serialization)}")
                        observed = {"type": message_type, "serialization": serialization}
                        if topic in self._topic_identities and self._topic_identities[topic] != observed:
                            raise ValueError(f"Topic identity changes across bag shards: {topic}")
                        self._topic_identities[topic] = observed
                        rows = shard_connection.execute(
                            "SELECT rowid, data FROM messages WHERE topic_id = ? ORDER BY rowid",
                            (topic_id,),
                        )
                        batch: list[tuple[object, ...]] = []
                        for row_id, data in rows:
                            payload = bytes(data)
                            timestamp_ns = _header_timestamp(topic, payload, profile)
                            self._validate_payload(topic, payload, profile)
                            batch.append((topic, timestamp_ns, file_index, int(row_id), topic_id, shard.name))
                            if len(batch) >= 512:
                                connection.executemany(
                                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", batch,
                                )
                                batch.clear()
                        if batch:
                            connection.executemany(
                                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)", batch,
                            )
            connection.commit()
        finally:
            connection.close()
        for topic in required_topics:
            if topic not in self._topic_identities:
                raise ValueError(f"Missing required profile topic: {topic}")
            with sqlite3.connect(str(self._index_path)) as check:
                count, first, last = check.execute(
                    "SELECT COUNT(*), MIN(timestamp_ns), MAX(timestamp_ns) FROM events WHERE topic = ?",
                    (topic,),
                ).fetchone()
            if not count:
                raise ValueError(f"Missing required profile topic: {topic}")
            self._topic_identities[topic]["message_count"] = int(count)
            self._topic_identities[topic]["header_timestamp_range_ns"] = [int(first), int(last)]
        expected_layout = EXPECTED_POINT_LAYOUTS[profile.source.mode]
        self._source_layouts = [{
            "fields": [
                {"name": name, "offset": offset, "datatype": datatype, "count": count}
                for name, offset, datatype, count in expected_layout["fields"]
            ],
            "height": expected_layout["height"],
            "is_bigendian": expected_layout["is_bigendian"],
            "point_step": expected_layout["point_step"],
            "frame_ids": sorted(self._source_frame_ids),
        }]

    def _validate_payload(
        self,
        topic: str,
        payload: bytes,
        profile: PreparationProfile,
    ) -> None:
        if topic == profile.source.topic:
            message = parse_pointcloud2(payload)
            layout = _point_layout(message)
            if layout != EXPECTED_POINT_LAYOUTS[profile.source.mode]:
                raise ValueError(f"PointCloud2 field layout mismatch for {profile.source.mode}")
            self._source_frame_ids.add(str(message["frame_id"]))
        elif topic == self._odometry_topic:
            message = parse_odometry(payload)
            self._odom_frame_ids.add(str(message["frame_id"]))
            self._odom_child_frame_ids.add(str(message["child_frame_id"]))

    def _readers(self) -> _StreamingReadConnections:
        thread_id = get_ident()
        with self._reader_lock:
            readers = self._readers_by_thread.get(thread_id)
            if readers is None:
                if self._closed:
                    raise RuntimeError("Streaming bag index is closed")
                readers = _StreamingReadConnections(
                    index_queries=sqlite3.connect(
                        str(self._index_path),
                        check_same_thread=False,
                    ),
                    image_iteration=sqlite3.connect(
                        str(self._index_path),
                        check_same_thread=False,
                    ),
                    shards={},
                )
                self._readers_by_thread[thread_id] = readers
            return readers

    def _shard_reader(self, shard_name: str) -> sqlite3.Connection:
        readers = self._readers()
        connection = readers.shards.get(shard_name)
        if connection is None:
            connection = sqlite3.connect(
                str(self.root / shard_name),
                check_same_thread=False,
            )
            readers.shards[shard_name] = connection
        return connection

    def _row(self, row: tuple[object, ...]) -> BagMessage:
        _, timestamp_ns, file_index, row_id, topic_id, shard_name = row
        data = self._shard_reader(str(shard_name)).execute(
            "SELECT data FROM messages WHERE topic_id = ? AND rowid = ?",
            (int(topic_id), int(row_id)),
        ).fetchone()
        if data is None:
            raise ValueError(f"Streaming bag row disappeared: {shard_name}:{row_id}")
        return BagMessage(int(timestamp_ns), int(file_index), int(row_id), bytes(data[0]))

    def _query_one(self, sql: str, params: tuple[object, ...]) -> tuple[object, ...] | None:
        return self._readers().index_queries.execute(sql, params).fetchone()

    def image_count(self) -> int:
        row = self._query_one("SELECT COUNT(*) FROM events WHERE topic = ?", (self._image_topic,))
        return int(row[0]) if row else 0

    def iter_images(self, image_start: int, image_limit: int | None) -> Iterator[tuple[int, BagMessage]]:
        stop = None if image_limit is None else image_start + image_limit
        sql = (
            "SELECT topic, timestamp_ns, storage_file_index, storage_row_id, topic_id, shard_name "
            "FROM events WHERE topic = ? ORDER BY timestamp_ns, storage_file_index, storage_row_id"
        )
        rows = self._readers().image_iteration.execute(sql, (self._image_topic,))
        for image_index, row in enumerate(rows):
            if image_index < image_start:
                continue
            if stop is not None and image_index >= stop:
                break
            yield image_index, self._row(row)

    def nearest_source(self, timestamp_ns: int, *, max_dt_ns: int) -> BagMessage | None:
        return self.nearest(self._source_topic, timestamp_ns, max_dt_ns=max_dt_ns)

    def nearest(self, topic: str, timestamp_ns: int, *, max_dt_ns: int) -> BagMessage | None:
        left = self._query_one(
            "SELECT topic, timestamp_ns, storage_file_index, storage_row_id, topic_id, shard_name "
            "FROM events WHERE topic = ? AND timestamp_ns <= ? "
            "ORDER BY timestamp_ns DESC, storage_file_index, storage_row_id LIMIT 1",
            (topic, int(timestamp_ns)),
        )
        right = self._query_one(
            "SELECT topic, timestamp_ns, storage_file_index, storage_row_id, topic_id, shard_name "
            "FROM events WHERE topic = ? AND timestamp_ns >= ? "
            "ORDER BY timestamp_ns, storage_file_index, storage_row_id LIMIT 1",
            (topic, int(timestamp_ns)),
        )
        candidates = [row for row in (left, right) if row is not None]
        if not candidates:
            return None
        row = min(candidates, key=lambda value: (
            abs(int(value[1]) - int(timestamp_ns)), int(value[2]), int(value[3]),
        ))
        if abs(int(row[1]) - int(timestamp_ns)) > int(max_dt_ns):
            return None
        return self._row(row)

    def pose_bracket(self, timestamp_ns: int) -> tuple[BagMessage | None, BagMessage | None]:
        left = self._query_one(
            "SELECT topic, timestamp_ns, storage_file_index, storage_row_id, topic_id, shard_name "
            "FROM events WHERE topic = ? AND timestamp_ns <= ? "
            "ORDER BY timestamp_ns DESC, storage_file_index, storage_row_id LIMIT 1",
            (self._odometry_topic, int(timestamp_ns)),
        )
        right = self._query_one(
            "SELECT topic, timestamp_ns, storage_file_index, storage_row_id, topic_id, shard_name "
            "FROM events WHERE topic = ? AND timestamp_ns >= ? "
            "ORDER BY timestamp_ns, storage_file_index, storage_row_id LIMIT 1",
            (self._odometry_topic, int(timestamp_ns)),
        )
        return (self._row(left) if left is not None else None, self._row(right) if right is not None else None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._reader_lock:
            readers = tuple(self._readers_by_thread.values())
            self._readers_by_thread.clear()
        for handles in readers:
            handles.index_queries.close()
            handles.image_iteration.close()
            for connection in handles.shards.values():
                connection.close()
        try:
            self._index_path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Unable to remove temporary ROSBAG event index: {self._index_path}"
            ) from exc


class IndexedPoseTrack:
    """PoseTrack-compatible resolver backed by the disk event index."""

    def __init__(self, index: StreamingBagIndex) -> None:
        self._index = index

    @property
    def coverage_ns(self) -> tuple[int, int]:
        row = self._index._query_one(
            "SELECT MIN(timestamp_ns), MAX(timestamp_ns) FROM events WHERE topic = ?",
            (self._index._odometry_topic,),
        )
        if row is None:
            raise ValueError("Odometry topic has no timestamp coverage")
        first, last = row
        if first is None or last is None:
            raise ValueError("Odometry topic has no timestamp coverage")
        return int(first), int(last)

    def _bracket(self, timestamp_ns: int, *, max_distance_ns: int | None = None) -> tuple[PoseSample, PoseSample, float]:
        left_record, right_record = self._index.pose_bracket(timestamp_ns)
        if left_record is None or right_record is None:
            raise ValueError(f"Timestamp {timestamp_ns} is outside odometry coverage {self.coverage_ns}")
        left_message = parse_odometry(left_record.data)
        right_message = parse_odometry(right_record.data)
        left = PoseSample(left_record.header_timestamp_ns, pose_to_matrix(
            left_message["position"], left_message["orientation_xyzw"],
        ))
        right = PoseSample(right_record.header_timestamp_ns, pose_to_matrix(
            right_message["position"], right_message["orientation_xyzw"],
        ))
        timestamp = int(timestamp_ns)
        if left.timestamp_ns == right.timestamp_ns:
            alpha = 0.0
        else:
            span = right.timestamp_ns - left.timestamp_ns
            alpha = (timestamp - left.timestamp_ns) / span
            if max_distance_ns is not None:
                distances = (timestamp - left.timestamp_ns, right.timestamp_ns - timestamp)
                if min(distances) > int(max_distance_ns) or span > 2 * int(max_distance_ns):
                    raise ValueError(f"Timestamp {timestamp} has no odometry bracket within {max_distance_ns} ns")
        return left, right, float(alpha)

    def interpolate(self, timestamp_ns: int, *, max_distance_ns: int | None = None) -> np.ndarray:
        left, right, alpha = self._bracket(timestamp_ns, max_distance_ns=max_distance_ns)
        if left.timestamp_ns == right.timestamp_ns:
            return left.world_from_base.copy()
        return PoseTrack((left, right)).interpolate(timestamp_ns, max_distance_ns=max_distance_ns)

    def bracket_receipt(self, timestamp_ns: int, *, max_distance_ns: int | None = None) -> dict[str, object]:
        left, right, alpha = self._bracket(timestamp_ns, max_distance_ns=max_distance_ns)
        return {
            "query_timestamp_ns": int(timestamp_ns),
            "previous_timestamp_ns": left.timestamp_ns,
            "next_timestamp_ns": right.timestamp_ns,
            "alpha": alpha,
        }


class StreamingBagRuntime:
    """Finite bag runtime that keeps payloads bounded to the active window."""

    def __init__(
        self,
        index: StreamingBagIndex,
        *,
        profile: PreparationProfile,
        calibration: Path,
        sampler: dict[str, np.ndarray],
        intrinsics: CameraIntrinsics,
        lidar_from_camera: np.ndarray,
        producer_identity: dict[str, object],
    ) -> None:
        self.index = index
        self.profile = profile
        self.calibration = calibration
        self.sampler = sampler
        self.intrinsics = intrinsics
        self.lidar_from_camera = lidar_from_camera
        self.poses = IndexedPoseTrack(index)
        self.producer_identity = producer_identity

    @property
    def input_files(self) -> dict[str, Path]:
        return self.index.input_files

    @property
    def image_count(self) -> int:
        return self.index.image_count()

    @property
    def source_layout(self) -> dict[str, object]:
        return self.index.source_layout

    @property
    def odometry_frames(self) -> dict[str, object]:
        return {
            "frame_ids": sorted(self.index.odom_frame_ids),
            "child_frame_ids": sorted(self.index.odom_child_frame_ids),
        }

    def iter_images(self, image_start: int, image_limit: int | None) -> Iterator[tuple[int, BagMessage]]:
        return self.index.iter_images(image_start, image_limit)

    def nearest_source(self, timestamp_ns: int, *, max_dt_ns: int) -> BagMessage | None:
        return self.index.nearest_source(timestamp_ns, max_dt_ns=max_dt_ns)

    def close(self) -> None:
        self.index.close()


@dataclass(frozen=True)
class PreparedBagRuntime:
    """Validated bag/calibration state shared by materialized and streaming producers."""

    profile: PreparationProfile
    bag: BagInputs
    calibration: Path
    source_messages: tuple[BagMessage, ...]
    poses: PoseTrack
    sampler: dict[str, np.ndarray]
    intrinsics: CameraIntrinsics
    lidar_from_camera: np.ndarray
    producer_identity: dict[str, object]

    @property
    def input_files(self) -> dict[str, Path]:
        return self.bag.input_files

    @property
    def image_count(self) -> int:
        return len(self.bag.messages[self.profile.source.image_topic])

    def iter_images(self, image_start: int, image_limit: int | None) -> Iterator[tuple[int, BagMessage]]:
        stop = None if image_limit is None else image_start + image_limit
        image_topic = self.profile.source.image_topic
        for image_index, image_record in tuple(enumerate(self.bag.messages[image_topic]))[image_start:stop]:
            yield image_index, image_record

    def nearest_source(self, timestamp_ns: int, *, max_dt_ns: int) -> BagMessage | None:
        return nearest_message(timestamp_ns, self.source_messages, max_dt_ns=max_dt_ns)

    def close(self) -> None:
        return None


def prepare_bag_runtime(
    rosbag_dir: Path,
    *,
    preparation_profile: str,
    calibration_path: Path | None,
    non_formal: bool,
    require_clean_worktree: bool = False,
) -> PreparedBagRuntime:
    profile = profile_for_name(preparation_profile)
    bag_root = Path(rosbag_dir).resolve()
    calibration = Path(calibration_path).resolve() if calibration_path is not None else bag_root / "cam_in_ex.txt"
    if not calibration.is_file():
        raise ValueError(f"Missing camera/LiDAR calibration: {calibration}")

    bag = read_bag_inputs(bag_root, profile)
    source_messages = bag.messages[profile.source.topic]
    source_layout = _source_layout(source_messages, profile)
    if set(source_layout["frame_ids"]) != {profile.source.frame_id}:
        raise ValueError(f"Profile source frame mismatch: {source_layout['frame_ids']}")
    poses, odometry_frames = _pose_track(bag.messages[profile.source.odometry_topic], profile)
    camera_calibration, sampler, intrinsics = _camera_setup(calibration, profile)

    identity = _code_identity()
    identity["input_contract"] = {
        "topics": bag.topic_identities,
        "source_layout": source_layout,
        "odometry_frames": odometry_frames,
        "image_order": "header-timestamp-storage-file-index-storage-row-id",
        "calibration": {
            "source_model": camera_calibration.camera_params["cam_model"],
            "matrix_name": camera_calibration.matrix_name,
            "matrix_semantics": "T_camera_from_lidar",
            "output_grid": list(profile.camera.output_grid),
        },
    }
    if require_clean_worktree and identity["dirty"]:
        raise ValueError("Clean worktree required")
    return PreparedBagRuntime(
        profile=profile,
        bag=bag,
        calibration=calibration,
        source_messages=source_messages,
        poses=poses,
        sampler=sampler,
        intrinsics=intrinsics,
        lidar_from_camera=np.linalg.inv(camera_calibration.t_camera_from_lidar),
        producer_identity=identity,
    )


def prepare_streaming_bag_runtime(
    rosbag_dir: Path,
    *,
    preparation_profile: str,
    calibration_path: Path | None,
    non_formal: bool,
    require_clean_worktree: bool = False,
) -> StreamingBagRuntime:
    profile = profile_for_name(preparation_profile)
    bag_root = Path(rosbag_dir).resolve()
    calibration = Path(calibration_path).resolve() if calibration_path is not None else bag_root / "cam_in_ex.txt"
    if not calibration.is_file():
        raise ValueError(f"Missing camera/LiDAR calibration: {calibration}")
    index = StreamingBagIndex(bag_root, profile)
    try:
        if index.source_frame_ids != {profile.source.frame_id}:
            raise ValueError(f"Profile source frame mismatch: {sorted(index.source_frame_ids)}")
        if index.odom_frame_ids != {"odom"} or index.odom_child_frame_ids != {profile.source.base_frame_id}:
            raise ValueError(
                "Odometry frame contract mismatch: "
                f"frame_id={sorted(index.odom_frame_ids)}, child_frame_id={sorted(index.odom_child_frame_ids)}"
            )
        camera_calibration, sampler, intrinsics = _camera_setup(calibration, profile)
        identity = _code_identity()
        identity["input_contract"] = {
            "topics": index.topic_identities,
            "source_layout": index.source_layout,
            "odometry_frames": {
                "frame_ids": sorted(index.odom_frame_ids),
                "child_frame_ids": sorted(index.odom_child_frame_ids),
            },
            "image_order": "header-timestamp-storage-file-index-storage-row-id",
            "calibration": {
                "source_model": camera_calibration.camera_params["cam_model"],
                "matrix_name": camera_calibration.matrix_name,
                "matrix_semantics": "T_camera_from_lidar",
                "output_grid": list(profile.camera.output_grid),
            },
        }
        if require_clean_worktree and identity["dirty"]:
            raise ValueError("Clean worktree required")
        return StreamingBagRuntime(
            index,
            profile=profile,
            calibration=calibration,
            sampler=sampler,
            intrinsics=intrinsics,
            lidar_from_camera=np.linalg.inv(camera_calibration.t_camera_from_lidar),
            producer_identity=identity,
        )
    except BaseException:
        index.close()
        raise


def _header_timestamp(topic: str, data: bytes, profile: PreparationProfile) -> int:
    if topic == profile.source.image_topic:
        message = parse_compressed_image(data)
    elif topic == profile.source.odometry_topic:
        message = parse_odometry(data)
    else:
        message = parse_pointcloud2(data)
    return int(message["stamp_sec"]) * 1_000_000_000 + int(message["stamp_nsec"])


def read_bag_inputs(rosbag_dir: Path, profile: PreparationProfile) -> BagInputs:
    root = Path(rosbag_dir).resolve()
    metadata = root / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError("ROS bag folder must contain metadata.yaml")
    shards = _bag_shards(root, metadata)
    required_topics = (profile.source.image_topic, profile.source.odometry_topic, profile.source.topic)
    expected_identities = _expected_topic_identities(profile)
    collected: dict[str, list[BagMessage]] = {topic: [] for topic in required_topics}
    identities: dict[str, dict[str, object]] = {}
    for file_index, shard in enumerate(shards):
        with sqlite3.connect(str(shard)) as connection:
            topics = {
                str(name): (int(topic_id), str(message_type), str(serialization))
                for topic_id, name, message_type, serialization in connection.execute(
                    "SELECT id, name, type, serialization_format FROM topics"
                )
            }
            for topic in required_topics:
                if topic not in topics:
                    continue
                topic_id, message_type, serialization = topics[topic]
                expected_type, expected_serialization = expected_identities[topic]
                if (message_type, serialization) != (expected_type, expected_serialization):
                    raise ValueError(f"Topic identity mismatch for {topic}: {(message_type, serialization)}")
                observed = {"type": message_type, "serialization": serialization}
                if topic in identities and identities[topic] != observed:
                    raise ValueError(f"Topic identity changes across bag shards: {topic}")
                identities[topic] = observed
                rows = connection.execute(
                    "SELECT rowid, timestamp, data FROM messages WHERE topic_id = ? ORDER BY rowid",
                    (topic_id,),
                )
                for row_id, _, data in rows:
                    payload = bytes(data)
                    collected[topic].append(BagMessage(
                        _header_timestamp(topic, payload, profile), file_index, int(row_id), payload,
                    ))
    for topic in required_topics:
        if not collected[topic]:
            raise ValueError(f"Missing required profile topic: {topic}")
        collected[topic].sort(key=lambda item: (
            item.header_timestamp_ns, item.storage_file_index, item.storage_row_id,
        ))
        identities[topic]["message_count"] = len(collected[topic])
        identities[topic]["header_timestamp_range_ns"] = [
            collected[topic][0].header_timestamp_ns,
            collected[topic][-1].header_timestamp_ns,
        ]
    input_files = {"metadata.yaml": metadata}
    input_files.update({f"rosbag/{path.name}": path for path in shards})
    return BagInputs(
        messages={name: tuple(values) for name, values in collected.items()},
        topic_identities=identities,
        input_files=input_files,
    )


def nearest_message(
    timestamp_ns: int,
    messages: tuple[BagMessage, ...],
    *,
    max_dt_ns: int,
) -> BagMessage | None:
    timestamps = np.fromiter((item.header_timestamp_ns for item in messages), dtype=np.int64)
    right = int(np.searchsorted(timestamps, int(timestamp_ns), side="left"))
    candidate_indices = tuple(index for index in (right - 1, right) if 0 <= index < len(messages))
    if not candidate_indices:
        return None
    index = min(candidate_indices, key=lambda value: (
        abs(messages[value].header_timestamp_ns - int(timestamp_ns)),
        messages[value].storage_file_index,
        messages[value].storage_row_id,
    ))
    candidate = messages[index]
    return candidate if abs(candidate.header_timestamp_ns - int(timestamp_ns)) <= int(max_dt_ns) else None


def _pose_track(
    messages: tuple[BagMessage, ...], profile: PreparationProfile,
) -> tuple[PoseTrack, dict[str, object]]:
    samples: list[PoseSample] = []
    frame_ids: set[str] = set()
    child_frame_ids: set[str] = set()
    for record in messages:
        message = parse_odometry(record.data)
        frame_ids.add(str(message["frame_id"]))
        child_frame_ids.add(str(message["child_frame_id"]))
        samples.append(PoseSample(
            record.header_timestamp_ns,
            pose_to_matrix(message["position"], message["orientation_xyzw"]),
        ))
    if frame_ids != {"odom"} or child_frame_ids != {profile.source.base_frame_id}:
        raise ValueError(
            f"Odometry frame contract mismatch: frame_id={sorted(frame_ids)}, child_frame_id={sorted(child_frame_ids)}"
        )
    return PoseTrack(tuple(samples)), {
        "frame_ids": sorted(frame_ids),
        "child_frame_ids": sorted(child_frame_ids),
    }


_CAMERA_RECTIFICATION_BUILDERS = {
    "FishPoly": build_fishpoly_rectification_maps,
    "Pinhole": build_pinhole_rectification_maps,
}


def _camera_setup(calibration_path: Path, profile: PreparationProfile):
    calibration = parse_camera_lidar_calibration(calibration_path)
    if calibration.camera_params.get("cam_model") != profile.camera.source_model:
        raise ValueError(f"Calibration must declare cam_model={profile.camera.source_model}")
    if calibration.matrix_name != "Tcl_0":
        raise ValueError("Calibration must declare Tcl_0 as T_camera_from_lidar")
    t_camera_from_lidar = np.asarray(calibration.t_camera_from_lidar, dtype=np.float64)
    rotation = t_camera_from_lidar[:3, :3]
    if (
        t_camera_from_lidar.shape != (4, 4)
        or not np.isfinite(t_camera_from_lidar).all()
        or not np.allclose(t_camera_from_lidar[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
    ):
        raise ValueError("Tcl_0 must be a finite rigid T_camera_from_lidar transform")
    raw_camera = {
        "width": int(calibration.camera_params["image_width"]),
        "height": int(calibration.camera_params["image_height"]),
        "K_like": calibration.k_like,
        "distortion": calibration.distortion,
    }
    builder = _CAMERA_RECTIFICATION_BUILDERS.get(profile.camera.source_model)
    if builder is None:
        raise ValueError(f"Unsupported camera source_model: {profile.camera.source_model}")
    map_x, map_y, pinhole = builder(raw_camera)
    if (int(pinhole["height"]), int(pinhole["width"])) != profile.camera.output_grid:
        raise ValueError("Rectification grid does not match preparation profile")
    sampler = build_sampler(map_x, map_y, raw_camera["width"], raw_camera["height"])
    intrinsics = CameraIntrinsics(
        int(pinhole["width"]), int(pinhole["height"]),
        float(pinhole["fx"]), float(pinhole["fy"]), float(pinhole["cx"]), float(pinhole["cy"]),
    )
    return calibration, sampler, intrinsics


def _point_layout(message: dict[str, object]) -> dict[str, object]:
    fields = tuple(
        (str(field["name"]), int(field["offset"]), int(field["datatype"]), int(field["count"]))
        for field in message["fields"]
    )
    height = int(message["height"])
    width = int(message["width"])
    point_step = int(message["point_step"])
    row_step = int(message["row_step"])
    data_len = int(message["data_len"])
    if min(height, width, point_step) < 1 or row_step != width * point_step or data_len != height * row_step:
        raise ValueError("PointCloud2 payload dimensions do not match point_step and row_step")
    return {
        "fields": fields,
        "height": height,
        "is_bigendian": bool(message["is_bigendian"]),
        "point_step": point_step,
    }


def _source_layout(messages: tuple[BagMessage, ...], profile: PreparationProfile) -> dict[str, object]:
    expected = EXPECTED_POINT_LAYOUTS[profile.source.mode]
    layouts = [_point_layout(parse_pointcloud2(record.data)) for record in messages]
    if any(layout != expected for layout in layouts):
        raise ValueError(f"PointCloud2 field layout mismatch for {profile.source.mode}")
    frame_ids = {str(parse_pointcloud2(record.data)["frame_id"]) for record in messages}
    return {
        "fields": [
            {"name": name, "offset": offset, "datatype": datatype, "count": count}
            for name, offset, datatype, count in expected["fields"]
        ],
        "height": expected["height"],
        "is_bigendian": expected["is_bigendian"],
        "point_step": expected["point_step"],
        "frame_ids": sorted(frame_ids),
    }


def _code_identity() -> dict[str, object]:
    from ...foundation.code_identity import repository_code_identity

    repository = Path(__file__).resolve().parents[4]
    environment_lock = repository / "conda-lock.yml"
    if not environment_lock.is_file():
        raise ValueError("Missing SAGE environment lock: conda-lock.yml")
    code = repository_code_identity(repository)
    required_distributions = ("numpy", "opencv-python", "Pillow")
    distributions = {}
    for name in required_distributions:
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD")
        if record is None:
            raise ValueError(f"Installed producer dependency has no RECORD identity: {name}")
        distributions[name] = {
            "version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
        }
    return {
        "repository": "sage",
        "repository_commit": code["commit"],
        "tracked_tree_sha256": code["source_state_sha256"],
        "dirty": code["worktree_dirty"],
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "environment_lock": {
            "path": environment_lock.name,
            "sha256": sha256_file(environment_lock),
        },
        "dependency_distributions": distributions,
        "environment_prefix": os.environ.get("CONDA_PREFIX", ""),
    }


def _build_slot(
    image_index: int,
    image_record: BagMessage,
    *,
    profile: PreparationProfile,
    source_record: BagMessage | None,
    poses: PoseTrack,
    sampler: dict[str, np.ndarray],
    intrinsics: CameraIntrinsics,
    lidar_from_camera: np.ndarray,
) -> FinalizedSensorFrame | RejectedSensorFrame:
    stem = f"{image_index:06d}"
    timestamp = image_record.header_timestamp_ns
    if source_record is None:
        return RejectedSensorFrame(image_index, stem, timestamp, "missing-profile-cloud")
    try:
        image_pose_bracket = poses.bracket_receipt(
            timestamp, max_distance_ns=int(round(profile.odometry_max_dt_ms * 1_000_000)),
        )
        world_from_base = poses.interpolate(
            timestamp, max_distance_ns=int(round(profile.odometry_max_dt_ms * 1_000_000)),
        )
    except ValueError:
        return RejectedSensorFrame(image_index, stem, timestamp, "image-pose-bracket-invalid")
    world_from_camera = world_from_base @ lidar_from_camera
    cloud_message = parse_pointcloud2(source_record.data)
    if cloud_message["frame_id"] != profile.source.frame_id:
        return RejectedSensorFrame(image_index, stem, timestamp, "source-frame-mismatch")
    payload = decode_slam_cloud(cloud_message)
    world_xyz = LidarWorldNormalizer().normalize(payload["xyz"], frame_id=str(cloud_message["frame_id"]))
    normalization_receipt = {
        "mode": "direct-world",
        "frame_id": "odom",
        "point_time_policy": "unavailable-in-topic",
        "producer_processing": "opaque-recorded-output",
    }
    normalization_receipt["image_pose_bracket"] = image_pose_bracket
    try:
        decoded = decode_compressed_image(image_record.data)["image"]
        rgb = remap_bilinear(decoded, sampler)[:, :, ::-1].copy()
    except Exception:
        return RejectedSensorFrame(image_index, stem, timestamp, "image-decode-or-rectification-failed")
    cloud_id = f"{source_record.storage_file_index:03d}-{source_record.storage_row_id:09d}"
    cloud = WorldCloudEvidence(
        image_index=image_index,
        image_timestamp_ns=timestamp,
        cloud_id=cloud_id,
        cloud_timestamp_ns=source_record.header_timestamp_ns,
        world_xyz=world_xyz,
        center_source_type=profile.source.center_source_type,
        fused_source_type=profile.source.fused_source_type,
        normalization_receipt=normalization_receipt,
    )
    return FinalizedSensorFrame(
        image_index, stem, timestamp, rgb, intrinsics, world_from_camera, cloud,
    )


def iter_projected_centers(
    runtime: PreparedBagRuntime,
    *,
    image_start: int = 0,
    image_limit: int | None = None,
    emitted_limit: int | None = None,
    receipt: dict[str, object] | None = None,
    rejected_centers: list[RejectedCenter] | None = None,
) -> Iterable[ProjectedCenter]:
    if image_start < 0 or (image_limit is not None and image_limit < 5):
        raise ValueError("Diagnostic image_start must be non-negative and image_limit must be at least five")
    local_receipt = receipt if receipt is not None else {}
    local_rejections = rejected_centers if rejected_centers is not None else []
    emitted = 0
    window = CenteredFiveWindow(CenteredFiveProjector(
        min_depth_m=runtime.profile.depth.min_depth_m,
        max_depth_m=runtime.profile.depth.max_depth_m,
        conflict_threshold_m=runtime.profile.depth.conflict_threshold_m,
    ))
    for image_index, image_record in runtime.iter_images(image_start, image_limit):
        local_receipt["selected_image_slots"] = int(local_receipt.get("selected_image_slots", 0)) + 1
        slot = _build_slot(
            image_index,
            image_record,
            profile=runtime.profile,
            source_record=runtime.nearest_source(
                image_record.header_timestamp_ns,
                max_dt_ns=int(round(runtime.profile.cloud_max_dt_ms * 1_000_000)),
            ),
            poses=runtime.poses,
            sampler=runtime.sampler,
            intrinsics=runtime.intrinsics,
            lidar_from_camera=runtime.lidar_from_camera,
        )
        key = "rejected_slots" if isinstance(slot, RejectedSensorFrame) else "finalized_slots"
        local_receipt[key] = int(local_receipt.get(key, 0)) + 1
        for result in window.push(slot):
            if isinstance(result, RejectedCenter):
                local_rejections.append(result)
            else:
                emitted += 1
                yield result
                if emitted_limit is not None and emitted >= emitted_limit:
                    local_receipt["edge_slots_not_emitted"] = min(
                        4, int(local_receipt.get("selected_image_slots", 0)),
                    )
                    local_receipt["bag_exhausted"] = False
                    local_receipt["truncated"] = True
                    local_receipt["complete"] = True
                    return
    window.finish()
    local_receipt["edge_slots_not_emitted"] = min(4, int(local_receipt.get("selected_image_slots", 0)))
    local_receipt["bag_exhausted"] = image_start == 0 and image_limit is None
    local_receipt["truncated"] = not local_receipt["bag_exhausted"]
    local_receipt["complete"] = True
