from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sage.engine.model import TrainableGaussians
from sage.engine.rendering import (
    RenderOutput,
    RenderStaticFields,
    _camera_matrices,
    alpha_normalize_render_output_depth,
    prepare_render_static,
    render,
)
from sage.foundation.contracts import (
    CameraIntrinsics,
    FrameInputs,
    GrowthInputs,
    INVALID_SOURCE_TYPE,
    MappingObservation,
    Pose,
)
from sage.foundation.identity_schema import _DEPENDENCY_FIELDS


def _frame() -> FrameInputs:
    height = width = 32
    return FrameInputs(
        index=0,
        stem="renderer-test",
        timestamp_ns=0,
        intrinsics=CameraIntrinsics(
            width,
            height,
            32.0,
            32.0,
            width / 2,
            height / 2,
        ),
        pose=Pose(0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 1.0),
        rgb=np.zeros((height, width, 3), dtype=np.float32),
        mapping=MappingObservation(
            np.zeros((height, width), dtype=np.float32),
            np.full((height, width), INVALID_SOURCE_TYPE, dtype=np.uint8),
            np.zeros((height, width), dtype=np.float32),
        ),
        growth=GrowthInputs(()),
    )


def _model(device: torch.device) -> TrainableGaussians:
    return TrainableGaussians(
        torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.05, 1.2]], device=device),
        torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]], device=device),
        torch.zeros((2, 1), device=device),
        torch.full((2, 3), -2.0, device=device),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device),
        torch.arange(2, dtype=torch.int64, device=device),
        torch.zeros(2, dtype=torch.int64, device=device),
        torch.zeros(2, dtype=torch.uint8, device=device),
        torch.ones(2, device=device),
    )


def test_render_static_fields_preserve_public_contract() -> None:
    assert [field.name for field in fields(RenderStaticFields)] == [
        "backend",
        "means3d",
        "scales",
        "rotations",
        "background",
    ]


def test_depth_normalization_compatibility_reexport() -> None:
    output = RenderOutput(
        rgb=torch.zeros((1, 2, 3)),
        depth=torch.tensor([[1.0, 0.0]]),
        alpha=torch.tensor([[0.5, 0.0]]),
    )

    normalized = alpha_normalize_render_output_depth(output, min_alpha=0.1)

    torch.testing.assert_close(normalized.depth, torch.tensor([[2.0, 0.0]]))


def test_legacy_repository_renderer_identity_is_rejected() -> None:
    assert set(_DEPENDENCY_FIELDS["renderer"]) == {"unavailable", "pip-package"}


def test_camera_matrices_use_world_to_camera_and_pinhole_intrinsics() -> None:
    view, intrinsics = _camera_matrices(_frame(), torch.device("cpu"))

    torch.testing.assert_close(
        view,
        torch.tensor([
            [1.0, 0.0, 0.0, -0.1],
            [0.0, 1.0, 0.0, 0.05],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )
    torch.testing.assert_close(
        intrinsics,
        torch.tensor([
            [32.0, 0.0, 16.0],
            [0.0, 32.0, 16.0],
            [0.0, 0.0, 1.0],
        ]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat requires CUDA")
def test_render_and_static_render_gradient_partition() -> None:
    model = _model(torch.device("cuda"))
    frame = _frame()

    for static, expect_geometry_gradient in (
        (None, True),
        (prepare_render_static(model), False),
    ):
        model.zero_grad(set_to_none=True)
        output = render(model, frame, static=static)
        assert output.rgb.shape == (32, 32, 3)
        assert output.depth.shape == (32, 32)
        assert output.alpha.shape == (32, 32)
        assert all(
            bool(torch.isfinite(value).all())
            for value in (output.rgb, output.depth, output.alpha)
        )

        (output.rgb.sum() + output.depth.sum() + output.alpha.sum()).backward()

        assert model.colors.grad is not None
        assert model.opacity_logits.grad is not None
        for name in ("means3d", "log_scales", "rotations"):
            assert (getattr(model, name).grad is not None) is expect_geometry_gradient


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat requires CUDA")
def test_render_rejects_invalid_fused_output_shapes() -> None:
    model = _model(torch.device("cuda"))
    frame = _frame()
    static = prepare_render_static(model)

    def invalid_rasterization(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return (
            torch.zeros((1, 32, 32, 3), device=model.means3d.device),
            torch.zeros((1, 32, 32, 1), device=model.means3d.device),
            {},
        )

    invalid_backend = SimpleNamespace(rasterization=invalid_rasterization)

    with pytest.raises(RuntimeError, match="invalid RGB\\+D/alpha shapes"):
        render(model, frame, static=replace(static, backend=invalid_backend))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat requires CUDA")
def test_static_background_preserves_compositing_contract() -> None:
    model = _model(torch.device("cuda"))
    frame = _frame()
    static = prepare_render_static(model)
    background = torch.tensor([0.1, 0.2, 0.3], device=model.means3d.device)

    def empty_rasterization(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return (
            torch.full((1, 32, 32, 4), 0.25, device=model.means3d.device),
            torch.full((1, 32, 32, 1), 0.5, device=model.means3d.device),
            {},
        )

    output = render(
        model,
        frame,
        static=replace(
            static,
            backend=SimpleNamespace(rasterization=empty_rasterization),
            background=background,
        ),
    )

    expected_rgb = 0.25 + 0.5 * background
    torch.testing.assert_close(output.rgb, expected_rgb.expand(32, 32, 3))
    torch.testing.assert_close(output.depth, torch.full((32, 32), 0.25, device="cuda"))
