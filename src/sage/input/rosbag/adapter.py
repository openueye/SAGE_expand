"""GenericRosbagAdapter: a ROS 2 bag compiled into canonical frames.

The adapter is a facade over reader → decoder → synchronizer → transforms →
projection → fusion. It owns no training policy, writes nothing, and holds no
dataset knowledge: everything it does is derived from the spec and the
canonical calibration.
"""

from __future__ import annotations

from collections.abc import Iterator
import statistics as statistics_module

import numpy as np

from ..adapter import ResolvedInput
from ..contract import CanonicalInputContract, ResolvedInputContract
from ..frame import FrameInputs, FrameMetadata
from ..identity import CanonicalSequenceDigest, InputIdentities, canonical_json_sha256, file_sha256
from ..preflight import PreflightReport
from .calibration import ImageRectifier, load_calibration
from .decoder import compressed_image_codec, decode_image_message
from .fusion import build_observations
from .reader import RosbagReader
from .spec import ADAPTER_TYPE, RosbagInputSpec
from .synchronizer import FramePlan, synchronize


# Projection sanity is checked on a sample of accepted frames rather than all
# of them: it exists to catch a swapped extrinsic or a wrong unit, and that
# shows up in the first handful of frames.
_PROJECTION_SAMPLE = 8

LIDAR_MOTION_COMPENSATION = "rigid_scan_timestamp"


class GenericRosbagAdapter:
    """Compile one ROS 2 bag into a replayable canonical frame sequence."""

    def __init__(self, spec: RosbagInputSpec) -> None:
        if not isinstance(spec, RosbagInputSpec):
            raise ValueError("GenericRosbagAdapter requires a RosbagInputSpec")
        self._spec = spec

    def preflight(self) -> ResolvedInput:
        spec = self._spec
        report = PreflightReport(ADAPTER_TYPE)
        calibration = load_calibration(spec.calibration_path)
        rectifier = ImageRectifier(calibration)
        reader = RosbagReader(
            spec.rosbag_path,
            topics=spec.topic_roles(),
            time_offsets_ns=dict(calibration.time_offsets_to_image_ns),
        )
        try:
            self._check_frames(reader, calibration, report)
            plan = synchronize(reader, spec, calibration, report)
            report.raise_for_errors()
            contract = self._contract(reader, calibration, rectifier, plan)
            self._check_projection(reader, calibration, plan, rectifier, report)
            report.raise_for_errors()
            source_identity = self._source_identity(reader, calibration)
        finally:
            reader.close()

        def frames() -> Iterator[FrameInputs]:
            replay = RosbagReader(
                spec.rosbag_path,
                topics=spec.topic_roles(),
                time_offsets_ns=dict(calibration.time_offsets_to_image_ns),
            )
            try:
                yield from self._emit(replay, calibration, rectifier, plan)
            finally:
                replay.close()

        return ResolvedInput(
            contract=contract, report=report, source_identity=source_identity, _frames=frames,
        )

    # -- preflight checks ---------------------------------------------------

    def _check_frames(
        self, reader: RosbagReader, calibration, report: PreflightReport,
    ) -> None:
        frames = calibration.frames
        if reader.odometry_frame_ids != {frames["reference"]}:
            report.add_error(
                f"Odometry header.frame_id {sorted(reader.odometry_frame_ids)} does not match "
                f"calibration reference frame {frames['reference']!r}"
            )
        if reader.odometry_child_frame_ids != {frames["pose"]}:
            report.add_error(
                f"Odometry child_frame_id {sorted(reader.odometry_child_frame_ids)} does not match "
                f"calibration pose frame {frames['pose']!r}"
            )
        expected_cloud_frame = (
            frames["reference"] if self._spec.points_frame == "reference_frame" else frames["lidar"]
        )
        if reader.lidar_frame_ids != {expected_cloud_frame}:
            report.add_error(
                f"PointCloud2 header.frame_id {sorted(reader.lidar_frame_ids)} does not match the "
                f"declared points_frame={self._spec.points_frame} ({expected_cloud_frame!r})"
            )
        if any(not frame_id for frame_id in reader.lidar_frame_ids):
            report.add_error("PointCloud2 header.frame_id must not be empty")
        image_span = reader.topic_identities["image"]["effective_timestamp_range_ns"]
        lidar_span = reader.topic_identities["lidar"]["effective_timestamp_range_ns"]
        pose_span = reader.topic_identities["odometry"]["effective_timestamp_range_ns"]
        for name, span in (("lidar", lidar_span), ("odometry", pose_span)):
            if span[1] < image_span[0] or span[0] > image_span[1]:
                report.add_error(f"{name} and image timestamp ranges do not overlap")

    def _check_projection(
        self, reader: RosbagReader, calibration, plan: FramePlan,
        rectifier: ImageRectifier, report: PreflightReport,
    ) -> None:
        positive: list[int] = []
        in_image: list[int] = []
        center_coverage: list[float] = []
        fused_coverage: list[float] = []
        step = max(1, len(plan.accepted) // _PROJECTION_SAMPLE)
        for position in range(0, len(plan.accepted), step)[:_PROJECTION_SAMPLE]:
            observations = build_observations(
                reader, self._spec, calibration, plan.accepted[position],
                plan.window_by_frame[position], camera=rectifier.output_camera,
            )
            positive.append(int(observations.statistics["positive_depth_points"]))
            in_image.append(int(observations.statistics["in_image_points"]))
            center_coverage.append(observations.center.coverage)
            fused_coverage.append(
                observations.fused.coverage if observations.fused is not None else 0.0
            )
        report.record_statistics("projection", {
            "sampled_frames": len(positive),
            "median_positive_depth_points": int(statistics_module.median(positive)),
            "median_in_image_points": int(statistics_module.median(in_image)),
        })
        report.record_statistics("coverage", {
            "lidar_center": float(statistics_module.median(center_coverage)),
            "lidar_fused": float(statistics_module.median(fused_coverage)),
        })
        if statistics_module.median(in_image) <= 0:
            report.add_error(
                "No sampled frame projects any LiDAR point into the image; check the extrinsics "
                "direction, the declared points_frame and the calibration units"
            )
        if statistics_module.median(center_coverage) <= 0:
            report.add_error("Sampled LIDAR_CENTER depth coverage is zero")

    # -- contract and identity ---------------------------------------------

    def _adapter_details(self, reader: RosbagReader, calibration, rectifier: ImageRectifier) -> dict[str, object]:
        return {
            "adapter_type": ADAPTER_TYPE,
            "topics": {
                role: dict(identity) for role, identity in sorted(reader.topic_identities.items())
            },
            "points_frame": self._spec.points_frame,
            "synchronization": self._spec.synchronization.payload(),
            "fusion": self._spec.fusion.payload(),
            "depth": self._spec.depth.payload(),
            "lidar_motion_compensation": LIDAR_MOTION_COMPENSATION,
            "calibration": calibration.payload(),
            "image_processing": rectifier.identity,
        }

    def _contract(
        self, reader: RosbagReader, calibration, rectifier: ImageRectifier, plan: FramePlan,
    ) -> ResolvedInputContract:
        canonical = CanonicalInputContract(
            frame_count=len(plan.accepted),
            reference_frame=calibration.frames["reference"],
            camera_frame=calibration.frames["camera"],
            image_size=(calibration.output_camera.width, calibration.output_camera.height),
            sources=self._spec.sources,
            fusion=self._spec.fusion.payload(),
        )
        return ResolvedInputContract(
            adapter_type=ADAPTER_TYPE,
            canonical=canonical,
            adapter_details=self._adapter_details(reader, calibration, rectifier),
        )

    def _source_identity(self, reader: RosbagReader, calibration) -> str:
        return canonical_json_sha256({
            "input_files_sha256": {
                label: file_sha256(path) for label, path in sorted(reader.input_files.items())
            },
            "calibration_sha256": file_sha256(calibration.path),
            "topics": self._spec.topic_roles(),
            "frame_limit": self._spec.frame_limit if self._spec.frame_limit is not None else -1,
        })

    # -- replay -------------------------------------------------------------

    def _emit(
        self, reader: RosbagReader, calibration, rectifier: ImageRectifier, plan: FramePlan,
    ) -> Iterator[FrameInputs]:
        spec = self._spec
        intrinsics = rectifier.output_camera.matrix()
        image_type = reader.message_types["image"]
        for frame_index, (frame, window) in enumerate(zip(plan.accepted, plan.window_by_frame)):
            payload = reader.payload(frame.image)
            rgb = rectifier.rectify_bgr(decode_image_message(image_type, payload))
            observations = build_observations(
                reader, spec, calibration, frame, window, camera=rectifier.output_camera,
            )
            diagnostics: dict[str, float | int | str] = {
                "candidate_index": frame.candidate_index,
                "lidar_skew_ns": frame.lidar_skew_ns,
                "pose_bracket_span_ns": frame.pose_bracket_ns[1] - frame.pose_bracket_ns[0],
                **{key: int(value) for key, value in observations.statistics.items()},
            }
            if image_type == "sensor_msgs/msg/CompressedImage":
                diagnostics["image_codec"] = compressed_image_codec(payload)
            yield FrameInputs(
                frame_index=frame_index,
                timestamp_ns=int(frame.timestamp_ns),
                image_rgb=np.asarray(rgb, dtype=np.float32) / 255.0,
                intrinsics=intrinsics,
                reference_from_camera=frame.reference_from_camera,
                lidar_center=observations.center,
                lidar_fused=observations.fused,
                metadata=FrameMetadata(
                    source_timestamp_ns={
                        "image": int(frame.image.header_timestamp_ns),
                        "lidar": int(frame.lidar.header_timestamp_ns),
                    },
                    frame_names={
                        "reference": calibration.frames["reference"],
                        "camera": calibration.frames["camera"],
                        "lidar": calibration.frames["lidar"],
                        "pose": calibration.frames["pose"],
                    },
                    diagnostics=diagnostics,
                ),
            )


__all__ = ["GenericRosbagAdapter"]
