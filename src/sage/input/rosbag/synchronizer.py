"""Turn bag events into an accepted, frozen frame plan.

Images anchor the timeline: each image is one candidate frame at its effective
timestamp. A candidate becomes an accepted frame only if a LiDAR scan lands
inside the skew budget, the pose track brackets every timestamp the frame needs
without extrapolating, and — when fusion is enabled — its whole fusion window
is likewise resolvable.
"""

from __future__ import annotations

from dataclasses import dataclass

import bisect
import numpy as np

from ..preflight import PreflightReport, RejectedFrame, distribution
from .calibration import Calibration
from .reader import MessageLocator, RosbagReader
from .spec import RosbagInputSpec
from .transforms import interpolate_pose


@dataclass(frozen=True, eq=False)
class AssociatedFrame:
    """One candidate image with everything resolved except pixels and points."""

    candidate_index: int
    timestamp_ns: int
    image: MessageLocator
    lidar: MessageLocator
    reference_from_camera: np.ndarray
    reference_from_lidar: np.ndarray | None
    lidar_skew_ns: int
    pose_bracket_ns: tuple[int, int]


@dataclass(frozen=True)
class FramePlan:
    """The frozen result of synchronization."""

    accepted: tuple[AssociatedFrame, ...]
    window_by_frame: tuple[tuple[AssociatedFrame, ...], ...]
    candidate_count: int


def _associate_lidar(
    events: tuple[MessageLocator, ...],
    timestamps: list[int],
    anchor_ns: int,
    *,
    policy: str,
    max_skew_ns: int,
) -> MessageLocator | None:
    index = bisect.bisect_right(timestamps, anchor_ns)
    if policy == "latest_not_after_anchor":
        candidate = events[index - 1] if index else None
    else:
        options = [events[position] for position in (index - 1, index) if 0 <= position < len(events)]
        candidate = min(
            options,
            key=lambda event: (abs(event.effective_timestamp_ns - anchor_ns), event.shard_index, event.row_id),
        ) if options else None
    if candidate is None:
        return None
    if abs(candidate.effective_timestamp_ns - anchor_ns) > max_skew_ns:
        return None
    return candidate


def _pose_at(reader: RosbagReader, timestamp_ns: int, *, max_gap_ns: int) -> np.ndarray | None:
    bracket = reader.pose_bracket(timestamp_ns)
    if bracket is None:
        return None
    left, right = bracket
    try:
        return interpolate_pose(left, right, timestamp_ns, max_gap_ns=max_gap_ns)
    except ValueError:
        return None


def synchronize(
    reader: RosbagReader,
    spec: RosbagInputSpec,
    calibration: Calibration,
    report: PreflightReport,
) -> FramePlan:
    synchronization = spec.synchronization
    max_gap_ns = synchronization.max_pose_gap_ns
    lidar_timestamps = [event.effective_timestamp_ns for event in reader.lidar_events]
    pose_from_lidar = calibration.pose_from_lidar
    lidar_from_camera = calibration.lidar_from_camera
    needs_scan_pose = spec.points_frame == "lidar_frame"

    associated: dict[int, AssociatedFrame] = {}
    skews_ms: list[float] = []
    pose_spans_ms: list[float] = []
    for candidate_index, image in enumerate(reader.image_events):
        anchor = image.effective_timestamp_ns
        scan = _associate_lidar(
            reader.lidar_events, lidar_timestamps, anchor,
            policy=synchronization.lidar_association,
            max_skew_ns=synchronization.max_lidar_skew_ns,
        )
        if scan is None:
            report.reject(RejectedFrame(
                candidate_index, anchor, "no-lidar-scan-within-skew-budget",
                {"max_lidar_skew_ms": synchronization.max_lidar_skew_ms},
            ))
            continue
        reference_from_pose = _pose_at(reader, anchor, max_gap_ns=max_gap_ns)
        if reference_from_pose is None:
            report.reject(RejectedFrame(
                candidate_index, anchor, "image-timestamp-has-no-pose-bracket",
                {"max_pose_gap_ms": synchronization.max_pose_gap_ms},
            ))
            continue
        reference_from_lidar = None
        if needs_scan_pose:
            scan_pose = _pose_at(reader, scan.effective_timestamp_ns, max_gap_ns=max_gap_ns)
            if scan_pose is None:
                report.reject(RejectedFrame(
                    candidate_index, anchor, "lidar-timestamp-has-no-pose-bracket",
                    {"lidar_timestamp_ns": scan.effective_timestamp_ns},
                ))
                continue
            reference_from_lidar = scan_pose @ pose_from_lidar
        bracket = reader.pose_bracket(anchor)
        assert bracket is not None
        skews_ms.append(abs(scan.effective_timestamp_ns - anchor) / 1e6)
        pose_spans_ms.append((bracket[1].timestamp_ns - bracket[0].timestamp_ns) / 1e6)
        associated[candidate_index] = AssociatedFrame(
            candidate_index=candidate_index,
            timestamp_ns=anchor,
            image=image,
            lidar=scan,
            reference_from_camera=reference_from_pose @ pose_from_lidar @ lidar_from_camera,
            reference_from_lidar=reference_from_lidar,
            lidar_skew_ns=int(scan.effective_timestamp_ns - anchor),
            pose_bracket_ns=(bracket[0].timestamp_ns, bracket[1].timestamp_ns),
        )

    accepted: list[AssociatedFrame] = []
    windows: list[tuple[AssociatedFrame, ...]] = []
    half = spec.fusion.half_window if spec.enable_fused else 0
    max_age_ns = spec.fusion.max_age_ns
    for candidate_index in sorted(associated):
        frame = associated[candidate_index]
        window_indices = range(candidate_index - half, candidate_index + half + 1)
        window = [associated.get(index) for index in window_indices]
        if any(member is None for member in window):
            report.reject(RejectedFrame(
                frame.candidate_index, frame.timestamp_ns, "incomplete-fusion-window",
                {"window": [index for index in window_indices]},
            ))
            continue
        aged_out = [
            member.candidate_index for member in window
            if abs(member.lidar.effective_timestamp_ns - frame.timestamp_ns) > max_age_ns
        ]
        if aged_out:
            report.reject(RejectedFrame(
                frame.candidate_index, frame.timestamp_ns, "fusion-window-exceeds-max-age",
                {"max_age_ms": spec.fusion.max_age_ms, "aged_out": aged_out},
            ))
            continue
        accepted.append(frame)
        windows.append(tuple(window))
        if spec.frame_limit is not None and len(accepted) >= spec.frame_limit:
            break

    report.record_statistics("lidar_image_skew_ms", distribution(skews_ms))
    report.record_statistics("pose_bracket_span_ms", distribution(pose_spans_ms))
    report.record_counts(
        candidate_frames=len(reader.image_events), accepted_frames=len(accepted),
    )
    if not accepted:
        report.add_error(
            "No candidate image produced an accepted canonical frame; see rejections for reasons"
        )
    return FramePlan(tuple(accepted), tuple(windows), len(reader.image_events))


__all__ = ["AssociatedFrame", "FramePlan", "synchronize"]
