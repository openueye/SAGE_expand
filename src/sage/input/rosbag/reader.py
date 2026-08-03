"""Bag access: shard discovery, one indexing pass, on-demand payload fetch.

The index keeps only event metadata (effective timestamp and row locator) plus
the fully parsed odometry track, which is small. Cloud and image payloads stay
on disk until a frame actually needs them, so a multi-gigabyte bag never has to
fit in memory.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

from .decoder import (
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_LIDAR_TYPES,
    SUPPORTED_ODOMETRY_TYPES,
    header_timestamp_ns,
    parse_header,
    parse_odometry,
    parse_pointcloud2_prefix,
    pointcloud_layout,
)
from .transforms import PoseSample, pose_matrix


# Enough for a PointCloud2 header plus a generous field table; a cloud with a
# field list longer than this is not something we should silently accept.
_CLOUD_PREFIX_BYTES = 2048
_IMAGE_PREFIX_BYTES = 512

SUPPORTED_TYPES_BY_ROLE = {
    "lidar": SUPPORTED_LIDAR_TYPES,
    "image": SUPPORTED_IMAGE_TYPES,
    "odometry": SUPPORTED_ODOMETRY_TYPES,
}


@dataclass(frozen=True)
class MessageLocator:
    """Where a message lives and when it happened, without its payload."""

    effective_timestamp_ns: int
    header_timestamp_ns: int
    shard_index: int
    topic_id: int
    row_id: int

    @property
    def message_id(self) -> str:
        return f"{self.shard_index:03d}-{self.row_id:09d}"


def bag_shards(root: Path) -> tuple[Path, ...]:
    """Shards in metadata-declared order; declaration order is the bag's truth."""
    metadata = root / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError(f"ROS 2 bag folder must contain metadata.yaml: {root}")
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
        raise ValueError(f"ROS 2 bag folder must contain at least one .db3 shard: {root}")
    return shards


class RosbagReader:
    """One indexing pass over the three declared topics."""

    def __init__(
        self,
        rosbag_path: Path,
        *,
        topics: dict[str, str],
        time_offsets_ns: dict[str, int],
    ) -> None:
        root = Path(rosbag_path).resolve()
        if not root.is_dir():
            raise ValueError(f"ROS 2 bag folder does not exist: {root}")
        self.root = root
        self.shards = bag_shards(root)
        self.input_files = {
            "metadata.yaml": root / "metadata.yaml",
            **{f"rosbag/{path.name}": path for path in self.shards},
        }
        self._topics = dict(topics)
        # image offset is fixed at zero: it is the anchor stream by definition.
        self._offsets = {"image": 0, **{name: int(time_offsets_ns.get(name, 0)) for name in ("lidar", "odometry")}}
        self._connections: dict[int, sqlite3.Connection] = {}
        self.message_types: dict[str, str] = {}
        self.topic_identities: dict[str, dict[str, object]] = {}
        self.image_events: tuple[MessageLocator, ...] = ()
        self.lidar_events: tuple[MessageLocator, ...] = ()
        self.pose_samples: tuple[PoseSample, ...] = ()
        self._pose_timestamps: list[int] = []
        self.lidar_frame_ids: set[str] = set()
        self.image_frame_ids: set[str] = set()
        self.odometry_frame_ids: set[str] = set()
        self.odometry_child_frame_ids: set[str] = set()
        self.lidar_layout: dict[str, object] | None = None
        try:
            self._index()
        except BaseException:
            self.close()
            raise

    # -- indexing -----------------------------------------------------------

    def _connection(self, shard_index: int) -> sqlite3.Connection:
        connection = self._connections.get(shard_index)
        if connection is None:
            connection = sqlite3.connect(str(self.shards[shard_index]))
            self._connections[shard_index] = connection
        return connection

    def _index(self) -> None:
        images: list[MessageLocator] = []
        lidar: list[MessageLocator] = []
        poses: list[tuple[int, PoseSample]] = []
        collected = {"image": images, "lidar": lidar}
        for shard_index in range(len(self.shards)):
            connection = self._connection(shard_index)
            available = {
                str(name): (int(topic_id), str(message_type), str(serialization))
                for topic_id, name, message_type, serialization in connection.execute(
                    "SELECT id, name, type, serialization_format FROM topics"
                )
            }
            for role, topic in self._topics.items():
                if topic not in available:
                    continue
                topic_id, message_type, serialization = available[topic]
                self._record_topic_identity(role, topic, message_type, serialization)
                offset = self._offsets[role]
                if role == "odometry":
                    rows = connection.execute(
                        "SELECT rowid, data FROM messages WHERE topic_id = ? ORDER BY rowid",
                        (topic_id,),
                    )
                    for row_id, blob in rows:
                        message = parse_odometry(bytes(blob))
                        self.odometry_frame_ids.add(str(message["frame_id"]))
                        self.odometry_child_frame_ids.add(str(message["child_frame_id"]))
                        poses.append((
                            header_timestamp_ns(message) + offset,
                            PoseSample(
                                header_timestamp_ns(message) + offset,
                                pose_matrix(message["position"], message["orientation_xyzw"]),
                            ),
                        ))
                    continue
                prefix = _CLOUD_PREFIX_BYTES if role == "lidar" else _IMAGE_PREFIX_BYTES
                rows = connection.execute(
                    "SELECT rowid, substr(data, 1, ?) FROM messages WHERE topic_id = ? ORDER BY rowid",
                    (prefix, topic_id),
                )
                for row_id, blob in rows:
                    payload = bytes(blob)
                    if role == "lidar":
                        message = parse_pointcloud2_prefix(payload)
                        self.lidar_frame_ids.add(str(message["frame_id"]))
                        layout = pointcloud_layout(message)
                        if self.lidar_layout is None:
                            self.lidar_layout = layout
                        elif layout != self.lidar_layout:
                            raise ValueError(
                                f"PointCloud2 field layout changes inside {self._topics['lidar']}"
                            )
                        stamp = header_timestamp_ns(message)
                    else:
                        message = parse_header(payload)
                        self.image_frame_ids.add(str(message["frame_id"]))
                        stamp = header_timestamp_ns(message)
                    collected[role].append(MessageLocator(
                        stamp + offset, stamp, shard_index, topic_id, int(row_id),
                    ))
        for role, topic in self._topics.items():
            if role not in self.topic_identities:
                raise ValueError(f"Bag does not contain the declared {role} topic: {topic}")
        self.image_events = _ordered(images, self._topics["image"])
        self.lidar_events = _ordered(lidar, self._topics["lidar"])
        self.pose_samples = tuple(sample for _, sample in sorted(poses, key=lambda item: item[0]))
        if not self.pose_samples:
            raise ValueError(f"Odometry topic carries no messages: {self._topics['odometry']}")
        timestamps = [sample.timestamp_ns for sample in self.pose_samples]
        if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
            raise ValueError("Odometry timestamps must be strictly increasing")
        self._pose_timestamps = timestamps
        for role, events in (("image", self.image_events), ("lidar", self.lidar_events)):
            identity = self.topic_identities[role]
            identity["message_count"] = len(events)
            identity["effective_timestamp_range_ns"] = [
                events[0].effective_timestamp_ns, events[-1].effective_timestamp_ns,
            ]
        odometry_identity = self.topic_identities["odometry"]
        odometry_identity["message_count"] = len(self.pose_samples)
        odometry_identity["effective_timestamp_range_ns"] = [timestamps[0], timestamps[-1]]
        odometry_identity["reference_frame"] = sorted(self.odometry_frame_ids)
        odometry_identity["pose_frame"] = sorted(self.odometry_child_frame_ids)
        self.topic_identities["lidar"]["frame_ids"] = sorted(self.lidar_frame_ids)
        self.topic_identities["lidar"]["layout"] = self.lidar_layout

    def _record_topic_identity(
        self, role: str, topic: str, message_type: str, serialization: str,
    ) -> None:
        supported = SUPPORTED_TYPES_BY_ROLE[role]
        if message_type not in supported:
            raise ValueError(
                f"Unsupported {role} message type {message_type!r} on {topic}; supported: {supported}"
            )
        if serialization != "cdr":
            raise ValueError(f"Unsupported serialization {serialization!r} on {topic}")
        observed = {"name": topic, "type": message_type, "serialization": serialization}
        existing = self.topic_identities.get(role)
        if existing is not None and {
            key: existing[key] for key in ("name", "type", "serialization")
        } != observed:
            raise ValueError(f"Topic identity changes across bag shards: {topic}")
        self.topic_identities[role] = {**(existing or {}), **observed}
        self.message_types[role] = message_type

    # -- payload access -----------------------------------------------------

    def payload(self, locator: MessageLocator) -> bytes:
        row = self._connection(locator.shard_index).execute(
            "SELECT data FROM messages WHERE topic_id = ? AND rowid = ?",
            (locator.topic_id, locator.row_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"ROSBAG row disappeared: {locator.message_id}")
        return bytes(row[0])

    def pose_bracket(self, timestamp_ns: int) -> tuple[PoseSample, PoseSample] | None:
        """The two samples straddling a timestamp, or None outside coverage."""
        timestamps = self._pose_timestamps
        if timestamp_ns < timestamps[0] or timestamp_ns > timestamps[-1]:
            return None
        index = bisect.bisect_left(timestamps, timestamp_ns)
        if timestamps[index] == timestamp_ns:
            return self.pose_samples[index], self.pose_samples[index]
        return self.pose_samples[index - 1], self.pose_samples[index]

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def __enter__(self) -> "RosbagReader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _ordered(events: list[MessageLocator], topic: str) -> tuple[MessageLocator, ...]:
    if not events:
        raise ValueError(f"Topic carries no messages: {topic}")
    return tuple(sorted(events, key=lambda item: (
        item.effective_timestamp_ns, item.shard_index, item.row_id,
    )))


__all__ = ["MessageLocator", "RosbagReader", "bag_shards"]
