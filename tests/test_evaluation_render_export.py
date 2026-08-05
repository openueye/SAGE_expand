from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sage.engine.evaluation import EvaluationDepthPolicy, evaluate_frame, evaluate_frames
from sage.engine.render_export import RENDER_KINDS, EvaluationRenderExporter
from sage.engine.rendering import RenderOutput

from helpers import intrinsics_matrix, mapping_frame

_HEIGHT, _WIDTH = 4, 5


def _frame(frame_index: int) -> object:
    rgb = np.full((_HEIGHT, _WIDTH, 3), 0.5, dtype=np.float32)
    return mapping_frame(
        frame_index=frame_index,
        rgb=rgb,
        intrinsics=intrinsics_matrix(2.0, 2.0, _WIDTH / 2, _HEIGHT / 2),
    )


def _render(_model: object, _frame: object) -> RenderOutput:
    return RenderOutput(
        rgb=torch.full((_HEIGHT, _WIDTH, 3), 0.4),
        depth=torch.full((_HEIGHT, _WIDTH), 1.5),
        alpha=torch.full((_HEIGHT, _WIDTH), 0.9),
    )


def _image_metrics(_rendered: torch.Tensor, _target: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(psnr=30.0, ssim=0.9, lpips=0.1)


def _policy() -> EvaluationDepthPolicy:
    return EvaluationDepthPolicy()


def test_render_export_creates_full_tree_on_construction(tmp_path: Path) -> None:
    EvaluationRenderExporter(tmp_path, keyframe_every=5)

    for kind in RENDER_KINDS:
        for bucket in ("keyframe", "non_keyframe"):
            assert (tmp_path / "render" / kind / bucket).is_dir()


def test_evaluate_frame_writes_exactly_four_matching_files(tmp_path: Path) -> None:
    exporter = EvaluationRenderExporter(tmp_path, keyframe_every=5)
    frame = _frame(frame_index=4)  # (4 + 1) % 5 == 0 -> keyframe

    evaluate_frame(
        model=None,
        frame=frame,
        renderer=_render,
        image_metrics=_image_metrics,
        policy=_policy(),
        render_export=exporter,
    )

    for kind in RENDER_KINDS:
        path = tmp_path / "render" / kind / "keyframe" / f"{frame.stem}.png"
        assert path.is_file(), f"missing {kind} export for keyframe frame"
        other = tmp_path / "render" / kind / "non_keyframe" / f"{frame.stem}.png"
        assert not other.exists()


def test_evaluate_frame_routes_non_keyframe(tmp_path: Path) -> None:
    exporter = EvaluationRenderExporter(tmp_path, keyframe_every=5)
    frame = _frame(frame_index=1)  # (1 + 1) % 5 != 0, index != 0 -> non-keyframe

    evaluate_frame(
        model=None,
        frame=frame,
        renderer=_render,
        image_metrics=_image_metrics,
        policy=_policy(),
        render_export=exporter,
    )

    for kind in RENDER_KINDS:
        assert (tmp_path / "render" / kind / "non_keyframe" / f"{frame.stem}.png").is_file()
        assert not (tmp_path / "render" / kind / "keyframe" / f"{frame.stem}.png").exists()


def test_evaluate_frames_exports_every_frame_and_leaves_metrics_unchanged(tmp_path: Path) -> None:
    with_export = tmp_path / "with_export"
    frames = [_frame(frame_index=index) for index in range(3)]

    result = evaluate_frames(
        model=None,
        frames=frames,
        renderer=_render,
        image_metrics=_image_metrics,
        policy=_policy(),
        map_every=1,
        render_export=EvaluationRenderExporter(with_export, keyframe_every=5),
    )

    assert result["frame_count"] == 3
    for frame in frames:
        for kind in RENDER_KINDS:
            bucket = "keyframe" if frame.index == 0 else "non_keyframe"
            assert (with_export / "render" / kind / bucket / f"{frame.stem}.png").is_file()

    # Metrics must be identical whether or not export ran - compute again into a
    # second, separate export root and compare the aggregate report byte-for-byte.
    without_dir = tmp_path / "reference"
    reference = evaluate_frames(
        model=None,
        frames=[_frame(frame_index=index) for index in range(3)],
        renderer=_render,
        image_metrics=_image_metrics,
        policy=_policy(),
        map_every=1,
        render_export=EvaluationRenderExporter(without_dir, keyframe_every=5),
    )
    assert result["aggregate"] == reference["aggregate"]
    assert result["frames"] == reference["frames"]


def test_render_export_overwrites_existing_files(tmp_path: Path) -> None:
    exporter = EvaluationRenderExporter(tmp_path, keyframe_every=5)
    frame = _frame(frame_index=0)

    evaluate_frame(
        model=None, frame=frame, renderer=_render, image_metrics=_image_metrics,
        policy=_policy(), render_export=exporter,
    )
    path = tmp_path / "render" / "GT" / "keyframe" / f"{frame.stem}.png"
    first_mtime = path.stat().st_mtime_ns

    evaluate_frame(
        model=None, frame=frame, renderer=_render, image_metrics=_image_metrics,
        policy=_policy(), render_export=exporter,
    )
    assert path.is_file()
    assert path.stat().st_mtime_ns >= first_mtime


def test_render_export_failure_is_explicit(tmp_path: Path, monkeypatch) -> None:
    exporter = EvaluationRenderExporter(tmp_path, keyframe_every=5)
    frame = _frame(frame_index=0)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("sage.engine.render_export.Image.fromarray", lambda *a, **k: SimpleNamespace(save=_boom))

    with pytest.raises(RuntimeError) as excinfo:
        exporter.export(frame, _render(None, frame))

    message = str(excinfo.value)
    assert frame.stem in message
    assert "GT" in message
    assert str(tmp_path) in message
