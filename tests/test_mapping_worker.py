"""Mapping progress output coverage."""

from __future__ import annotations

from types import SimpleNamespace

from sage.mapping.mapping_worker import _report_frames


def test_mapping_progress_prints_rosbag_frame_not_canonical_frame(capsys) -> None:
    frame = SimpleNamespace(
        stem="000000",
        canonical=SimpleNamespace(
            metadata=SimpleNamespace(diagnostics={"candidate_index": 139}),
        ),
    )

    assert list(_report_frames(
        [frame], expected_total=1, mapping_started_at=0.0,
    )) == [frame]

    output = capsys.readouterr().out
    assert "SAGE mapping frame 1; ROSBAG frame 139" in output
    assert "canonical frame" not in output
