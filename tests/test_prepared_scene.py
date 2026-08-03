"""Prepared Scene is the persisted form of a ROSBAG's canonical output.

The point of these tests is the round trip: what a Prepared Scene replays must
be the same canonical frame sequence, element for element, not an approximation
of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sage.input.prepared_scene import PreparedSceneAdapter, PreparedSceneWriter
from sage.input.prepared_scene.manifest import MANIFEST_NAME
from sage.input.rosbag import GenericRosbagAdapter

from fixtures import synthetic_input


_OBSERVATION_FIELDS = ("depth_m", "valid_mask", "confidence", "support_count", "temporal_age_s")


def _export(tmp_path: Path, **spec_overrides):
    resolved = GenericRosbagAdapter(synthetic_input(tmp_path, **spec_overrides)).preflight()
    scene = tmp_path / "scene"
    PreparedSceneWriter(
        scene,
        canonical=resolved.contract.canonical,
        provenance={
            "created_from": resolved.contract.adapter_type,
            "source_identity": resolved.source_identity,
        },
    ).write(resolved.frames())
    return resolved, scene


def _assert_frames_equal(first, second) -> None:
    assert first.frame_index == second.frame_index
    assert first.timestamp_ns == second.timestamp_ns
    np.testing.assert_array_equal(first.image_rgb, second.image_rgb)
    np.testing.assert_array_equal(first.intrinsics, second.intrinsics)
    np.testing.assert_array_equal(first.reference_from_camera, second.reference_from_camera)
    for name in ("lidar_center", "lidar_fused"):
        left, right = getattr(first, name), getattr(second, name)
        assert (left is None) == (right is None), name
        if left is None:
            continue
        for field in _OBSERVATION_FIELDS:
            left_value, right_value = getattr(left, field), getattr(right, field)
            assert (left_value is None) == (right_value is None), f"{name}.{field}"
            if left_value is not None:
                np.testing.assert_array_equal(left_value, right_value, err_msg=f"{name}.{field}")


def test_prepared_scene_replays_the_rosbag_output_element_for_element(tmp_path) -> None:
    resolved, scene = _export(tmp_path)
    replayed = PreparedSceneAdapter(scene).preflight()

    bag_frames = list(resolved.frames())
    scene_frames = list(replayed.frames())
    assert len(bag_frames) == len(scene_frames) == resolved.contract.canonical.frame_count
    for first, second in zip(bag_frames, scene_frames):
        _assert_frames_equal(first, second)


def test_both_paths_share_the_canonical_identities(tmp_path) -> None:
    resolved, scene = _export(tmp_path)
    replayed = PreparedSceneAdapter(scene).preflight()

    assert (
        replayed.identities.canonical_sequence_identity
        == resolved.identities.canonical_sequence_identity
    )
    assert (
        replayed.identities.canonical_contract_identity
        == resolved.identities.canonical_contract_identity
    )
    # Provenance must differ: they were produced by different adapters.
    assert (
        replayed.identities.adapter_provenance_identity
        != resolved.identities.adapter_provenance_identity
    )


def test_a_tampered_asset_is_detected(tmp_path) -> None:
    _, scene = _export(tmp_path)
    depth_path = next((scene / "lidar_center").glob("*.depth.npy"))
    values = np.load(depth_path)
    values[0, 0] += 1.0
    with depth_path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    with pytest.raises(ValueError, match="does not match its recorded hash"):
        PreparedSceneAdapter(scene).preflight()


def test_an_old_prepared_scene_is_rejected_without_migration(tmp_path) -> None:
    _, scene = _export(tmp_path)
    manifest_path = scene / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_name"] = "sage-prepared-scene-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported prepared scene schema"):
        PreparedSceneAdapter(scene).preflight()


def test_asset_paths_may_not_escape_the_scene(tmp_path) -> None:
    _, scene = _export(tmp_path)
    frames_path = scene / "frames.jsonl"
    lines = frames_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["image"] = "../outside.png"
    lines[0] = json.dumps(record)
    frames_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the scene"):
        PreparedSceneAdapter(scene).preflight()


def test_writer_refuses_to_publish_over_an_existing_scene(tmp_path) -> None:
    resolved, scene = _export(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        PreparedSceneWriter(
            scene, canonical=resolved.contract.canonical, provenance={},
        )


def test_prepared_scene_needs_no_ros_topics_or_calibration(tmp_path) -> None:
    _, scene = _export(tmp_path)
    manifest = json.loads((scene / MANIFEST_NAME).read_text(encoding="utf-8"))
    canonical = json.dumps(manifest["canonical"])
    for forbidden in ("topic", "PointCloud2", "tf_static", "rosbag"):
        assert forbidden not in canonical
    details = PreparedSceneAdapter(scene).preflight().contract.adapter_details
    assert set(details) == {
        "adapter_type", "schema_name", "schema_revision",
        "storage_layout", "image_storage", "depth_storage",
    }
