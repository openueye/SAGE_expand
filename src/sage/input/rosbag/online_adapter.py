"""One-pass ``online-window-v2`` ROSBAG adapter for Stage 1 mapping.

The legacy ROSBAG adapter intentionally compiles a global plan.  This module
is the explicit alternative for new runs: it performs only static topic and
calibration and compact header-index checks at startup, then resolves
synchronization, fusion windows, projection checks and the canonical sequence
digest while canonical frames are consumed once.  It never silently falls back
to the global planner.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
import statistics

import numpy as np

from ..adapter import StreamingResolvedInput
from ..contract import (
    ONLINE_CANONICAL_CONTRACT_REVISION,
    CanonicalInputContract,
    ResolvedInputContract,
)
from ..frame import FrameInputs, FrameMetadata
from ..identity import canonical_json_sha256, file_sha256
from ..preflight import PreflightReport, RejectedFrame
from .adapter import LIDAR_MOTION_COMPENSATION
from .calibration import Calibration, ImageRectifier, load_calibration
from .decoder import compressed_image_codec, decode_image_message, pointcloud_layout
from .fusion import build_observations
from .reader import MessageLocator, OnlineMessage, OnlineRosbagReader
from .spec import ADAPTER_TYPE, EXECUTION_ONLINE_WINDOW_V2, RosbagInputSpec
from .synchronizer import AssociatedFrame, _associate_lidar
from .transforms import PoseSample, interpolate_pose, pose_matrix


_PAYLOAD_CACHE_MESSAGES = 12
_PROJECTION_SAMPLE = 8


@dataclass(frozen=True)
class _ImageCandidate:
    index: int
    image: MessageLocator


@dataclass
class _RunningDistribution:
    """Constant-memory mean/max accumulator for online input evidence."""

    count: int = 0
    total: float = 0.0
    maximum: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)

    def payload(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else 0.0,
            "max": self.maximum if self.count else 0.0,
        }


class OnlineRosbagAdapter:
    """Produce a single online canonical-frame stream from a time-indexed bag."""

    def __init__(self, spec: RosbagInputSpec) -> None:
        if not isinstance(spec, RosbagInputSpec):
            raise ValueError("OnlineRosbagAdapter requires a RosbagInputSpec")
        if spec.execution != EXECUTION_ONLINE_WINDOW_V2:
            raise ValueError("OnlineRosbagAdapter requires execution=online-window-v2")
        self._spec = spec

    def preflight(self) -> StreamingResolvedInput:
        """Index compact headers; validate frame content while streaming frames."""
        spec = self._spec
        report = PreflightReport(ADAPTER_TYPE)
        calibration = load_calibration(spec.calibration_path)
        rectifier = ImageRectifier(calibration)
        reader = OnlineRosbagReader(
            spec.rosbag_path,
            topics=spec.topic_roles(),
            time_offsets_ns=dict(calibration.time_offsets_to_image_ns),
            payload_cache_messages=_PAYLOAD_CACHE_MESSAGES,
        )
        try:
            contract = self._provisional_contract(reader, calibration, rectifier)
            source_identity = self._source_identity(reader, calibration)
        finally:
            reader.close()

        def frames() -> Iterator[FrameInputs]:
            stream_reader = OnlineRosbagReader(
                spec.rosbag_path,
                topics=spec.topic_roles(),
                time_offsets_ns=dict(calibration.time_offsets_to_image_ns),
                payload_cache_messages=_PAYLOAD_CACHE_MESSAGES,
            )
            yield from self._emit(stream_reader, calibration, rectifier, report)

        def finalize(frame_count: int) -> ResolvedInputContract:
            report.raise_for_errors()
            return self._final_contract(contract, frame_count)

        return StreamingResolvedInput(
            contract=contract,
            report=report,
            source_identity=source_identity,
            _frames=frames,
            _finalize_contract=finalize,
        )

    def _adapter_details(
        self,
        reader: OnlineRosbagReader,
        calibration: Calibration,
        rectifier: ImageRectifier,
    ) -> dict[str, object]:
        return {
            "adapter_type": ADAPTER_TYPE,
            "execution": EXECUTION_ONLINE_WINDOW_V2,
            "online_admissibility": "effective-header-timestamp-sorted-metadata-v1",
            "payload_cache_messages": _PAYLOAD_CACHE_MESSAGES,
            "topics": {
                role: dict(identity)
                for role, identity in sorted(reader.topic_identities.items())
            },
            "points_frame": self._spec.points_frame,
            "allow_empty_image_frame_id": self._spec.allow_empty_image_frame_id,
            "synchronization": self._spec.synchronization.payload(),
            "fusion": self._spec.fusion.payload(),
            "depth": self._spec.depth.payload(),
            "lidar_motion_compensation": LIDAR_MOTION_COMPENSATION,
            "calibration": calibration.payload(),
            "image_processing": rectifier.identity,
        }

    def _provisional_contract(
        self,
        reader: OnlineRosbagReader,
        calibration: Calibration,
        rectifier: ImageRectifier,
    ) -> ResolvedInputContract:
        canonical = CanonicalInputContract(
            frame_count=None,
            reference_frame=calibration.frames["reference"],
            camera_frame=calibration.frames["camera"],
            image_size=(rectifier.output_camera.width, rectifier.output_camera.height),
            sources=self._spec.sources,
            fusion=self._spec.fusion.payload(),
            schema_revision=ONLINE_CANONICAL_CONTRACT_REVISION,
        )
        return ResolvedInputContract(
            adapter_type=ADAPTER_TYPE,
            canonical=canonical,
            adapter_details=self._adapter_details(reader, calibration, rectifier),
        )

    @staticmethod
    def _final_contract(
        provisional: ResolvedInputContract,
        frame_count: int,
    ) -> ResolvedInputContract:
        canonical = provisional.canonical
        return ResolvedInputContract(
            adapter_type=provisional.adapter_type,
            canonical=CanonicalInputContract(
                frame_count=frame_count,
                reference_frame=canonical.reference_frame,
                camera_frame=canonical.camera_frame,
                image_size=canonical.image_size,
                sources=canonical.sources,
                fusion=canonical.fusion,
                camera_model=canonical.camera_model,
                depth_definition=canonical.depth_definition,
                length_unit=canonical.length_unit,
                pose_convention=canonical.pose_convention,
                schema_revision=ONLINE_CANONICAL_CONTRACT_REVISION,
            ),
            adapter_details=provisional.adapter_details,
        )

    def _source_identity(
        self, reader: OnlineRosbagReader, calibration: Calibration,
    ) -> str:
        """Fast provenance fingerprint; canonical frames bind consumed bytes.

        Hashing every multi-GB DB file before mapping is intentionally excluded
        from online mode.  The canonical sequence digest remains the semantic
        checkpoint gate; this source identity is audit-only by design.
        """
        return canonical_json_sha256({
            "scheme": "rosbag-stat-fingerprint-v1",
            "input_files": {
                label: {
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for label, path in sorted(reader.input_files.items())
            },
            "calibration_sha256": file_sha256(calibration.path),
            "topics": self._spec.topic_roles(),
            "frame_limit": self._spec.frame_limit if self._spec.frame_limit is not None else -1,
            "execution": EXECUTION_ONLINE_WINDOW_V2,
        })

    def _emit(
        self,
        reader: OnlineRosbagReader,
        calibration: Calibration,
        rectifier: ImageRectifier,
        report: PreflightReport,
    ) -> Iterator[FrameInputs]:
        spec = self._spec
        pending: deque[_ImageCandidate] = deque()
        lidars: deque[MessageLocator] = deque()
        poses: deque[PoseSample] = deque()
        associated: dict[int, AssociatedFrame | None] = {}
        candidate_count = 0
        accepted_count = 0
        next_window_center = 0
        event_count = {"image": 0, "lidar": 0, "odometry": 0}
        lidar_layout: dict[str, object] | None = None
        skews_ms = _RunningDistribution()
        pose_spans_ms = _RunningDistribution()
        positive_depth: list[int] = []
        in_image: list[int] = []
        center_coverage: list[float] = []
        fused_coverage: list[float] = []
        emitted_first = False
        current_time: int | None = None
        stopped = False

        try:
            for event in reader.events():
                event_count[event.role] += 1
                current_time = event.locator.effective_timestamp_ns
                if event.role == "odometry":
                    pose = self._pose(event, calibration, report)
                    if poses and pose.timestamp_ns <= poses[-1].timestamp_ns:
                        report.add_error("Odometry timestamps must be strictly increasing")
                    else:
                        poses.append(pose)
                elif event.role == "lidar":
                    layout = pointcloud_layout(event.header)
                    if lidar_layout is None:
                        lidar_layout = layout
                    elif layout != lidar_layout:
                        report.add_error(
                            f"PointCloud2 field layout changes inside {spec.lidar_topic}"
                        )
                    if event.header.get("frame_id") != self._expected_lidar_frame(calibration):
                        report.add_error(
                            "PointCloud2 header.frame_id does not match the declared points_frame"
                        )
                    lidars.append(event.locator)
                else:
                    image_frame_id = event.header.get("frame_id")
                    if (
                        image_frame_id != calibration.frames["camera"]
                        and not (
                            self._spec.allow_empty_image_frame_id
                            and image_frame_id == ""
                        )
                    ):
                        report.add_error(
                            "Image header.frame_id does not match calibration camera frame"
                        )
                    pending.append(_ImageCandidate(candidate_count, event.locator))
                    candidate_count += 1

                for candidate_index, frame in self._finalize_due_candidates(
                    pending=pending,
                    lidars=lidars,
                    poses=poses,
                    associated=associated,
                    report=report,
                    calibration=calibration,
                    force=False,
                    watermark_ns=current_time,
                    skews_ms=skews_ms,
                    pose_spans_ms=pose_spans_ms,
                ):
                    output, next_window_center = self._resolve_window(
                        reader=reader,
                        candidate_index=candidate_index,
                        frame=frame,
                        associated=associated,
                        next_window_center=next_window_center,
                        calibration=calibration,
                        rectifier=rectifier,
                        report=report,
                        projection_stats=(
                            positive_depth,
                            in_image,
                            center_coverage,
                            fused_coverage,
                        ),
                        emitted_first=emitted_first,
                        emitted_count=accepted_count,
                    )
                    self._trim_associated(
                        associated, next_window_center=next_window_center, spec=spec,
                    )
                    emitted_first = emitted_first or output is not None
                    if output is not None:
                        yield output
                        accepted_count += 1
                        if spec.frame_limit is not None and accepted_count >= spec.frame_limit:
                            stopped = True
                            break
                self._trim_buffers(pending, lidars, poses, spec, watermark_ns=current_time)
                if stopped:
                    break

            if not stopped:
                for candidate_index, frame in self._finalize_due_candidates(
                    pending=pending,
                    lidars=lidars,
                    poses=poses,
                    associated=associated,
                    report=report,
                    calibration=calibration,
                    force=True,
                    watermark_ns=current_time,
                    skews_ms=skews_ms,
                    pose_spans_ms=pose_spans_ms,
                ):
                    output, next_window_center = self._resolve_window(
                        reader=reader,
                        candidate_index=candidate_index,
                        frame=frame,
                        associated=associated,
                        next_window_center=next_window_center,
                        calibration=calibration,
                        rectifier=rectifier,
                        report=report,
                        projection_stats=(
                            positive_depth,
                            in_image,
                            center_coverage,
                            fused_coverage,
                        ),
                        emitted_first=emitted_first,
                        emitted_count=accepted_count,
                    )
                    self._trim_associated(
                        associated, next_window_center=next_window_center, spec=spec,
                    )
                    emitted_first = emitted_first or output is not None
                    if output is not None:
                        yield output
                        accepted_count += 1
                self._reject_trailing_windows(
                    associated, next_window_center, candidate_count, report, spec,
                )
            self._record_online_report(
                report=report,
                event_count=event_count,
                candidate_count=candidate_count,
                accepted_count=accepted_count,
                skews_ms=skews_ms,
                pose_spans_ms=pose_spans_ms,
                positive_depth=positive_depth,
                in_image=in_image,
                center_coverage=center_coverage,
                fused_coverage=fused_coverage,
            )
        finally:
            reader.close()

    def _expected_lidar_frame(self, calibration: Calibration) -> str:
        return (
            calibration.frames["reference"]
            if self._spec.points_frame == "reference_frame"
            else calibration.frames["lidar"]
        )

    def _pose(
        self, event: OnlineMessage, calibration: Calibration, report: PreflightReport,
    ) -> PoseSample:
        frame_id = event.header.get("frame_id")
        child_frame_id = event.header.get("child_frame_id")
        if frame_id != calibration.frames["reference"]:
            report.add_error("Odometry header.frame_id does not match calibration reference frame")
        if child_frame_id != calibration.frames["pose"]:
            report.add_error("Odometry child_frame_id does not match calibration pose frame")
        return PoseSample(
            event.locator.effective_timestamp_ns,
            pose_matrix(event.header["position"], event.header["orientation_xyzw"]),
        )

    def _finalize_due_candidates(
        self,
        *,
        pending: deque[_ImageCandidate],
        lidars: deque[MessageLocator],
        poses: deque[PoseSample],
        associated: dict[int, AssociatedFrame | None],
        report: PreflightReport,
        calibration: Calibration,
        force: bool,
        watermark_ns: int | None,
        skews_ms: _RunningDistribution,
        pose_spans_ms: _RunningDistribution,
    ) -> Iterator[tuple[int, AssociatedFrame | None]]:
        spec = self._spec
        while pending:
            candidate = pending[0]
            anchor = candidate.image.effective_timestamp_ns
            ready_at = anchor + spec.synchronization.max_lidar_skew_ns
            last_pose = poses[-1].timestamp_ns if poses else None
            # Do not let a missing/stopped odometry topic retain an unbounded
            # image backlog. Once the effective-time watermark has passed the
            # allowed pose gap, a future pose cannot form a valid bracket and
            # the normal association path will record this candidate's reject.
            if not force and (
                watermark_ns is None
                or watermark_ns < ready_at
                or (
                    (last_pose is None or last_pose < ready_at)
                    and watermark_ns < anchor + spec.synchronization.max_pose_gap_ns
                )
            ):
                break
            pending.popleft()
            frame = self._associate_candidate(
                candidate, lidars, poses, calibration, report,
            )
            associated[candidate.index] = frame
            if frame is not None:
                skews_ms.add(abs(frame.lidar_skew_ns) / 1e6)
                pose_spans_ms.add((frame.pose_bracket_ns[1] - frame.pose_bracket_ns[0]) / 1e6)
            yield candidate.index, frame

    def _associate_candidate(
        self,
        candidate: _ImageCandidate,
        lidars: deque[MessageLocator],
        poses: deque[PoseSample],
        calibration: Calibration,
        report: PreflightReport,
    ) -> AssociatedFrame | None:
        spec = self._spec
        anchor = candidate.image.effective_timestamp_ns
        lidar_events = tuple(lidars)
        scan = _associate_lidar(
            lidar_events,
            [event.effective_timestamp_ns for event in lidar_events],
            anchor,
            policy=spec.synchronization.lidar_association,
            max_skew_ns=spec.synchronization.max_lidar_skew_ns,
        )
        if scan is None:
            report.reject(RejectedFrame(
                candidate.index, anchor, "no-lidar-scan-within-skew-budget",
                {"max_lidar_skew_ms": spec.synchronization.max_lidar_skew_ms},
            ))
            return None
        image_pose, image_bracket = self._pose_at(poses, anchor, spec.synchronization.max_pose_gap_ns)
        if image_pose is None or image_bracket is None:
            report.reject(RejectedFrame(
                candidate.index, anchor, "image-timestamp-has-no-pose-bracket",
                {"max_pose_gap_ms": spec.synchronization.max_pose_gap_ms},
            ))
            return None
        reference_from_lidar = None
        if spec.points_frame == "lidar_frame":
            lidar_pose, _ = self._pose_at(
                poses, scan.effective_timestamp_ns, spec.synchronization.max_pose_gap_ns,
            )
            if lidar_pose is None:
                report.reject(RejectedFrame(
                    candidate.index, anchor, "lidar-timestamp-has-no-pose-bracket",
                    {"lidar_timestamp_ns": scan.effective_timestamp_ns},
                ))
                return None
            reference_from_lidar = lidar_pose @ calibration.pose_from_lidar
        return AssociatedFrame(
            candidate_index=candidate.index,
            timestamp_ns=anchor,
            image=candidate.image,
            lidar=scan,
            reference_from_camera=image_pose @ calibration.pose_from_lidar @ calibration.lidar_from_camera,
            reference_from_lidar=reference_from_lidar,
            lidar_skew_ns=int(scan.effective_timestamp_ns - anchor),
            pose_bracket_ns=(image_bracket[0].timestamp_ns, image_bracket[1].timestamp_ns),
        )

    @staticmethod
    def _pose_at(
        poses: deque[PoseSample], timestamp_ns: int, max_gap_ns: int,
    ) -> tuple[np.ndarray | None, tuple[PoseSample, PoseSample] | None]:
        values = tuple(poses)
        if not values or timestamp_ns < values[0].timestamp_ns or timestamp_ns > values[-1].timestamp_ns:
            return None, None
        timestamps = [sample.timestamp_ns for sample in values]
        index = bisect.bisect_left(timestamps, timestamp_ns)
        left, right = (
            (values[index], values[index])
            if timestamps[index] == timestamp_ns
            else (values[index - 1], values[index])
        )
        try:
            return interpolate_pose(left, right, timestamp_ns, max_gap_ns=max_gap_ns), (left, right)
        except ValueError:
            return None, None

    def _resolve_window(
        self,
        *,
        reader: OnlineRosbagReader,
        candidate_index: int,
        frame: AssociatedFrame | None,
        associated: dict[int, AssociatedFrame | None],
        next_window_center: int,
        calibration: Calibration,
        rectifier: ImageRectifier,
        report: PreflightReport,
        projection_stats: tuple[list[int], list[int], list[float], list[float]],
        emitted_first: bool,
        emitted_count: int,
    ) -> tuple[FrameInputs | None, int]:
        half = self._spec.fusion.half_window if self._spec.enable_fused else 0
        if candidate_index != next_window_center + half:
            return None, next_window_center
        center_index = next_window_center
        window_indices = range(center_index - half, center_index + half + 1)
        members = [associated.get(index) for index in window_indices]
        next_window_center += 1
        center = associated.get(center_index)
        if center is None:
            return None, next_window_center
        if any(member is None for member in members):
            report.reject(RejectedFrame(
                center.candidate_index, center.timestamp_ns, "incomplete-fusion-window",
                {"window": list(window_indices)},
            ))
            return None, next_window_center
        window = tuple(member for member in members if member is not None)
        aged_out = [
            member.candidate_index
            for member in window
            if abs(member.lidar.effective_timestamp_ns - center.timestamp_ns)
            > self._spec.fusion.max_age_ns
        ]
        if aged_out:
            report.reject(RejectedFrame(
                center.candidate_index, center.timestamp_ns, "fusion-window-exceeds-max-age",
                {"max_age_ms": self._spec.fusion.max_age_ms, "aged_out": aged_out},
            ))
            return None, next_window_center
        observations = build_observations(
            reader, self._spec, calibration, center, window, camera=rectifier.output_camera,
        )
        positive_depth, in_image, center_coverage, fused_coverage = projection_stats
        if len(positive_depth) < _PROJECTION_SAMPLE:
            positive_depth.append(int(observations.statistics["positive_depth_points"]))
            in_image.append(int(observations.statistics["in_image_points"]))
            center_coverage.append(observations.center.coverage)
            fused_coverage.append(
                observations.fused.coverage if observations.fused is not None else 0.0
            )
        if not emitted_first and observations.center.coverage <= 0:
            report.add_error(
                "The first accepted frame carries no LIDAR_CENTER depth; mapping "
                "initialization has no reproducible starting frame"
            )
        image_payload = reader.payload(center.image)
        rgb = rectifier.rectify_bgr(decode_image_message(reader.message_types["image"], image_payload))
        diagnostics: dict[str, float | int | str] = {
            "candidate_index": center.candidate_index,
            "lidar_skew_ns": center.lidar_skew_ns,
            "pose_bracket_span_ns": center.pose_bracket_ns[1] - center.pose_bracket_ns[0],
            **{key: int(value) for key, value in observations.statistics.items()},
        }
        if reader.message_types["image"] == "sensor_msgs/msg/CompressedImage":
            diagnostics["image_codec"] = compressed_image_codec(image_payload)
        return FrameInputs(
            frame_index=emitted_count,
            timestamp_ns=int(center.timestamp_ns),
            image_rgb=np.asarray(rgb, dtype=np.float32) / 255.0,
            intrinsics=rectifier.output_camera.matrix(),
            reference_from_camera=center.reference_from_camera,
            lidar_center=observations.center,
            lidar_fused=observations.fused,
            metadata=FrameMetadata(
                source_timestamp_ns={
                    "image": int(center.image.header_timestamp_ns),
                    "lidar": int(center.lidar.header_timestamp_ns),
                },
                frame_names={
                    "reference": calibration.frames["reference"],
                    "camera": calibration.frames["camera"],
                    "lidar": calibration.frames["lidar"],
                    "pose": calibration.frames["pose"],
                },
                diagnostics=diagnostics,
            ),
        ), next_window_center

    @staticmethod
    def _trim_buffers(
        pending: deque[_ImageCandidate],
        lidars: deque[MessageLocator],
        poses: deque[PoseSample],
        spec: RosbagInputSpec,
        *,
        watermark_ns: int | None,
    ) -> None:
        # A lidar scan associated to the next image (pending, or not yet
        # arrived when pending is momentarily empty between two images) can
        # carry a timestamp up to max_lidar_skew_ns earlier than that image,
        # and its own pose bracket can reach a further max_pose_gap_ns behind
        # the scan timestamp. Neither buffer may be trimmed past that margin,
        # or association fails even though the full stream covers it. When
        # pending is empty there is no next-image timestamp yet, so the most
        # recently seen event (watermark_ns) is the best available anchor for
        # the same margin -- collapsing to a bounded floor here (as the code
        # used to) discards history a not-yet-arrived image may still need.
        anchor = pending[0].image.effective_timestamp_ns if pending else watermark_ns
        if anchor is None:
            return
        lidar_keep_after = anchor - spec.synchronization.max_lidar_skew_ns
        while len(lidars) > 1 and lidars[1].effective_timestamp_ns < lidar_keep_after:
            lidars.popleft()
        pose_keep_after = (
            anchor
            - spec.synchronization.max_lidar_skew_ns
            - spec.synchronization.max_pose_gap_ns
        )
        while len(poses) > 2 and poses[1].timestamp_ns <= pose_keep_after:
            poses.popleft()

    @staticmethod
    def _trim_associated(
        associated: dict[int, AssociatedFrame | None],
        *,
        next_window_center: int,
        spec: RosbagInputSpec,
    ) -> None:
        """Retain only candidates that a future centered fusion window can use."""
        half = spec.fusion.half_window if spec.enable_fused else 0
        first_needed = next_window_center - half
        for candidate_index in tuple(associated):
            if candidate_index < first_needed:
                del associated[candidate_index]

    @staticmethod
    def _reject_trailing_windows(
        associated: dict[int, AssociatedFrame | None],
        next_window_center: int,
        candidate_count: int,
        report: PreflightReport,
        spec: RosbagInputSpec,
    ) -> None:
        if not spec.enable_fused:
            return
        half = spec.fusion.half_window
        for center_index in range(next_window_center, candidate_count):
            center = associated.get(center_index)
            if center is not None:
                report.reject(RejectedFrame(
                    center.candidate_index, center.timestamp_ns, "incomplete-fusion-window",
                    {"window": list(range(center_index - half, center_index + half + 1))},
                ))

    @staticmethod
    def _record_online_report(
        *,
        report: PreflightReport,
        event_count: dict[str, int],
        candidate_count: int,
        accepted_count: int,
        skews_ms: _RunningDistribution,
        pose_spans_ms: _RunningDistribution,
        positive_depth: list[int],
        in_image: list[int],
        center_coverage: list[float],
        fused_coverage: list[float],
    ) -> None:
        for role, count in event_count.items():
            if count == 0:
                report.add_error(f"Declared {role} topic carries no messages")
        report.record_counts(candidate_frames=candidate_count, accepted_frames=accepted_count)
        report.record_statistics("lidar_image_skew_ms", skews_ms.payload())
        report.record_statistics("pose_bracket_span_ms", pose_spans_ms.payload())
        if positive_depth:
            report.record_statistics("projection", {
                "sampled_frames": len(positive_depth),
                "median_positive_depth_points": int(statistics.median(positive_depth)),
                "median_in_image_points": int(statistics.median(in_image)),
            })
            report.record_statistics("coverage", {
                "lidar_center": float(statistics.median(center_coverage)),
                "lidar_fused": float(statistics.median(fused_coverage)),
            })
            if statistics.median(in_image) <= 0:
                report.add_error(
                    "No accepted frame projects any LiDAR point into the image; check the extrinsics "
                    "direction, the declared points_frame and the calibration units"
                )
            if statistics.median(center_coverage) <= 0:
                report.add_error("Accepted LIDAR_CENTER depth coverage is zero")
        else:
            report.add_error("No candidate image produced an accepted canonical frame")


__all__ = ["OnlineRosbagAdapter"]
