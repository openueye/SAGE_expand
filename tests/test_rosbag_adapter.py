"""ROSBAG adapter: decoding, association, transforms, projection and fusion."""

from __future__ import annotations

import numpy as np
import pytest

from sage.input.rosbag import GenericRosbagAdapter
from sage.input.rosbag.calibration import ImageRectifier, load_calibration
from sage.input.rosbag.decoder import (
    decode_image_message,
    decode_points_xyz,
    parse_pointcloud2,
)
from sage.input.rosbag.projection import project_to_depth
from sage.input.rosbag.reader import MessageLocator, RosbagReader
from sage.input.rosbag.spec import FusionSpec, SynchronizationSpec
from sage.input.rosbag.synchronizer import _associate_lidar
from sage.input.rosbag.transforms import (
    PoseSample,
    interpolate_pose,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
    slerp,
    transform_points,
)

import fixtures
from fixtures import synthetic_input, write_calibration


# -- decoding ---------------------------------------------------------------


def test_pointcloud_fields_resolve_by_name_not_offset_order() -> None:
    payload = fixtures.encode_pointcloud2_xyz(
        stamp_sec=1, stamp_nsec=0, frame_id="lidar",
        xyz=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    message = parse_pointcloud2(payload)
    assert [field["name"] for field in message["fields"]] == ["x", "y", "z"]
    np.testing.assert_array_equal(
        decode_points_xyz(message), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_pointcloud_without_xyz_is_rejected() -> None:
    message = parse_pointcloud2(fixtures.encode_pointcloud2_xyz(
        stamp_sec=1, stamp_nsec=0, frame_id="lidar",
        xyz=np.zeros((1, 3), dtype=np.float32),
    ))
    message["fields"] = [field for field in message["fields"] if field["name"] != "z"]
    with pytest.raises(ValueError, match="missing required fields"):
        decode_points_xyz(message)


def test_raw_and_compressed_image_decoders_agree(tmp_path) -> None:
    image = fixtures._image(3)
    raw = decode_image_message("sensor_msgs/msg/Image", fixtures.encode_image(
        stamp_sec=1, stamp_nsec=0, frame_id="camera", image_bgr=image,
    ))
    np.testing.assert_array_equal(raw, image)


def test_unsupported_image_encoding_is_rejected() -> None:
    payload = fixtures.encode_image(
        stamp_sec=1, stamp_nsec=0, frame_id="camera",
        image_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
    ).replace(b"bgr8\x00", b"yuv4\x00")
    with pytest.raises(ValueError, match="Unsupported Image encoding"):
        decode_image_message("sensor_msgs/msg/Image", payload)


# -- association ------------------------------------------------------------


def _events(timestamps: list[int]) -> tuple[MessageLocator, ...]:
    return tuple(
        MessageLocator(value, value, 0, 1, index) for index, value in enumerate(timestamps)
    )


def test_nearest_association_may_select_a_later_scan() -> None:
    events = _events([90, 130])
    selected = _associate_lidar(
        events, [event.effective_timestamp_ns for event in events], 100,
        policy="nearest_to_anchor", max_skew_ns=50,
    )
    assert selected.effective_timestamp_ns == 90


def test_latest_not_after_anchor_never_selects_a_future_scan() -> None:
    events = _events([90, 101])
    timestamps = [event.effective_timestamp_ns for event in events]
    causal = _associate_lidar(
        events, timestamps, 100, policy="latest_not_after_anchor", max_skew_ns=50,
    )
    assert causal.effective_timestamp_ns == 90
    assert _associate_lidar(
        _events([101]), [101], 100, policy="latest_not_after_anchor", max_skew_ns=50,
    ) is None


def test_association_outside_the_skew_budget_is_rejected() -> None:
    events = _events([10])
    assert _associate_lidar(
        events, [10], 100, policy="nearest_to_anchor", max_skew_ns=50,
    ) is None


# -- transforms -------------------------------------------------------------


def test_translation_is_linear_and_rotation_uses_slerp() -> None:
    left = PoseSample(0, np.eye(4))
    right_matrix = np.eye(4)
    right_matrix[:3, :3] = quaternion_xyzw_to_matrix(
        np.asarray([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    )
    right_matrix[:3, 3] = [2.0, 0.0, 0.0]
    right = PoseSample(100, right_matrix)

    middle = interpolate_pose(left, right, 50, max_gap_ns=1000)
    np.testing.assert_allclose(middle[:3, 3], [1.0, 0.0, 0.0])
    expected = quaternion_xyzw_to_matrix(slerp(
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        matrix_to_quaternion_xyzw(right_matrix[:3, :3]),
        0.5,
    ))
    np.testing.assert_allclose(middle[:3, :3], expected, atol=1e-12)


def test_pose_interpolation_never_extrapolates_or_spans_a_gap() -> None:
    left = PoseSample(0, np.eye(4))
    right = PoseSample(100, np.eye(4))
    with pytest.raises(ValueError, match="cannot extrapolate"):
        interpolate_pose(left, right, 150, max_gap_ns=1000)
    with pytest.raises(ValueError, match="exceeding max_pose_gap"):
        interpolate_pose(left, right, 50, max_gap_ns=10)


def test_transform_chain_composes_reference_from_camera(tmp_path) -> None:
    calibration = load_calibration(write_calibration(tmp_path))
    reference_from_pose = np.eye(4)
    reference_from_pose[:3, 3] = [1.0, 2.0, 3.0]
    reference_from_camera = (
        reference_from_pose @ calibration.pose_from_lidar @ calibration.lidar_from_camera
    )
    # A point on the camera's optical axis at 4 m sits 4 m along lidar +x.
    point_camera = np.asarray([[0.0, 0.0, 4.0]])
    point_reference = transform_points(reference_from_camera, point_camera)
    np.testing.assert_allclose(point_reference[0], [5.0, 2.0, 3.0], atol=1e-9)


# -- projection and fusion --------------------------------------------------


def _camera(tmp_path):
    return load_calibration(write_calibration(tmp_path)).output_camera


def test_projection_keeps_camera_z_and_drops_points_behind_the_camera(tmp_path) -> None:
    camera = _camera(tmp_path)
    depth, statistics = project_to_depth(
        np.asarray([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]]),
        camera_from_reference=np.eye(4),
        camera=camera, min_depth_m=0.1, max_depth_m=200.0,
    )
    assert statistics["points"] == 2
    assert statistics["positive_depth_points"] == 1
    assert depth[int(camera.cy), int(camera.cx)] == pytest.approx(5.0)


def test_zbuffer_keeps_the_nearest_point_per_pixel(tmp_path) -> None:
    camera = _camera(tmp_path)
    depth, _ = project_to_depth(
        np.asarray([[0.0, 0.0, 9.0], [0.0, 0.0, 3.0]]),
        camera_from_reference=np.eye(4),
        camera=camera, min_depth_m=0.1, max_depth_m=200.0,
    )
    assert depth[int(camera.cy), int(camera.cx)] == pytest.approx(3.0)


def test_fused_never_overlaps_center_and_records_support_and_age(tmp_path) -> None:
    resolved = GenericRosbagAdapter(synthetic_input(tmp_path)).preflight()
    frames = list(resolved.frames())
    assert frames
    for frame in frames:
        center, fused = frame.lidar_center, frame.lidar_fused
        assert not np.any(center.valid_mask & fused.valid_mask)
        assert center.valid_pixels > 0
        assert np.all(fused.support_count[~fused.valid_mask] == 0)
        assert np.all(fused.temporal_age_s[~fused.valid_mask] == 0)
        if fused.valid_pixels:
            assert fused.support_count[fused.valid_mask].max() >= 1
            assert fused.temporal_age_s[fused.valid_mask].max() <= 0.5


def test_conflicting_history_is_rejected_rather_than_averaged(tmp_path) -> None:
    tight = GenericRosbagAdapter(synthetic_input(
        tmp_path / "tight", fusion=FusionSpec(conflict_threshold_m=1e-6),
    )).preflight()
    loose = GenericRosbagAdapter(synthetic_input(
        tmp_path / "loose", fusion=FusionSpec(conflict_threshold_m=10.0),
    )).preflight()
    tight_fused = sum(frame.lidar_fused.valid_pixels for frame in tight.frames())
    loose_fused = sum(frame.lidar_fused.valid_pixels for frame in loose.frames())
    assert tight_fused < loose_fused


# -- calibration ------------------------------------------------------------


def test_calibration_rejects_a_non_rigid_extrinsic(tmp_path) -> None:
    import yaml

    path = write_calibration(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["extrinsics"]["camera_from_lidar"][0][0] = 2.0
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation must be orthonormal"):
        load_calibration(path)


def test_calibration_rejects_an_unknown_camera_model(tmp_path) -> None:
    import yaml

    path = write_calibration(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["camera"]["model"] = "equidistant"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported camera model"):
        load_calibration(path)


def test_rectifier_output_matches_the_declared_canonical_grid(tmp_path) -> None:
    calibration = load_calibration(write_calibration(tmp_path))
    rectifier = ImageRectifier(calibration)
    image = fixtures._image(0)
    rectified = rectifier.rectify_bgr(image)
    assert rectified.shape == (calibration.output_camera.height, calibration.output_camera.width, 3)
    # Zero distortion and an identical output K make rectification the identity,
    # so the only change is the BGR to RGB channel order.
    np.testing.assert_array_equal(rectified, image[:, :, ::-1])


# -- preflight --------------------------------------------------------------


def test_preflight_freezes_accepted_frames_and_reports_rejections(tmp_path) -> None:
    resolved = GenericRosbagAdapter(synthetic_input(tmp_path)).preflight()
    canonical = resolved.contract.canonical
    # A centered window of five drops two frames at each end.
    assert canonical.frame_count == fixtures.FRAME_COUNT - 4
    assert resolved.report.accepted_frames == canonical.frame_count
    assert {frame.reason for frame in resolved.report.rejected_frames} == {
        "incomplete-fusion-window",
    }
    assert canonical.fusion["causal"] is False
    assert canonical.sources == ("LIDAR_CENTER", "LIDAR_FUSED")


def test_preflight_rejects_a_topic_whose_frame_contradicts_the_calibration(tmp_path) -> None:
    spec = synthetic_input(tmp_path)
    from dataclasses import replace

    with pytest.raises(ValueError, match="does not match the declared points_frame"):
        GenericRosbagAdapter(replace(spec, points_frame="reference_frame")).preflight()


def test_reader_rejects_an_unsupported_message_type(tmp_path) -> None:
    spec = synthetic_input(tmp_path)
    from dataclasses import replace

    with pytest.raises(ValueError, match="Unsupported lidar message type"):
        RosbagReader(
            spec.rosbag_path,
            topics={
                "lidar": spec.image_topic,
                "image": spec.image_topic,
                "odometry": spec.odometry_topic,
            },
            time_offsets_ns={"lidar": 0, "odometry": 0},
        )


def test_missing_topic_is_named_explicitly(tmp_path) -> None:
    from dataclasses import replace

    spec = synthetic_input(tmp_path)
    with pytest.raises(ValueError, match="/does-not-exist"):
        GenericRosbagAdapter(replace(spec, lidar_topic="/does-not-exist")).preflight()


def test_raw_image_topic_produces_the_same_canonical_frames(tmp_path) -> None:
    compressed = GenericRosbagAdapter(
        synthetic_input(tmp_path / "compressed", image_encoding="compressed")
    ).preflight()
    raw = GenericRosbagAdapter(
        synthetic_input(tmp_path / "raw", image_encoding="raw")
    ).preflight()
    assert (
        compressed.identities.canonical_sequence_identity
        == raw.identities.canonical_sequence_identity
    )


def test_synchronization_spec_rejects_pose_extrapolation() -> None:
    with pytest.raises(ValueError, match="extrapolation is not supported"):
        SynchronizationSpec(allow_pose_extrapolation=True)
