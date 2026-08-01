from dataclasses import replace
from pathlib import Path

import torch

from sage.engine.model import TrainableGaussians
from sage.method_config import SageMethodConfig
from sage.refinement.appearance_refinement import (
    AppearanceExposureNuisance,
    AppearanceObjective,
    AppearanceRefiner,
)


def _model() -> TrainableGaussians:
    return TrainableGaussians(
        torch.tensor([[0.2, -0.1, 1.0], [0.1, 0.2, 1.2]]),
        torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]]),
        torch.tensor([[0.1], [-0.2]]),
        torch.tensor([[-2.0, -2.1, -1.9], [-1.8, -2.0, -2.2]]),
        torch.tensor([[1.0, 0.1, 0.0, 0.0], [1.0, 0.0, 0.1, 0.0]]),
        torch.arange(2, dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.uint8),
        torch.tensor([1.0, 0.5]),
    )


def _default_refinement_config():
    root = Path(__file__).resolve().parents[1]
    return SageMethodConfig.load(root / "configs" / "sage.yaml").refinement_config()


def test_default_config_enables_conservative_geometry_refinement() -> None:
    method = SageMethodConfig.load(
        Path(__file__).resolve().parents[1] / "configs" / "sage.yaml"
    )
    refinement = method.refinement_config()

    assert method._structure_parts()["growth"].coverage_alpha_threshold == 0.85
    assert refinement.optimized_parameters == TrainableGaussians.PARAMETER_NAMES
    assert all(
        refinement.learning_rates[name] > 0
        for name in ("means3d", "log_scales", "rotations")
    )


def test_refinement_updates_geometry_without_changing_topology() -> None:
    config = replace(
        _default_refinement_config(),
        iterations=1,
        milestone_steps=(1,),
    )
    model = _model()
    baseline_parameters = {
        name: getattr(model, name).detach().clone()
        for name in model.PARAMETER_NAMES
    }
    baseline_topology = {
        name: getattr(model, name).detach().clone()
        for name in (
            "gaussian_ids",
            "created_at",
            "source_types",
            "source_confidences",
        )
    }
    nuisance = AppearanceExposureNuisance((0, 1), config, device="cpu")

    def objective(
        item: TrainableGaussians,
        _frame_index: int,
        _step: int,
    ) -> AppearanceObjective:
        photo = sum(
            getattr(item, name).square().mean()
            for name in item.PARAMETER_NAMES
        )
        one = photo.detach().new_ones(())
        return AppearanceObjective(
            total=photo,
            terms={"photo": photo},
            diagnostics={
                "dense_valid_normal_pixels": one,
                "dense_normal_weight_mass": one,
            },
        )

    result = AppearanceRefiner(config).optimize(
        model,
        (0, 1),
        objective,
        exposure_nuisance=nuisance,
    )

    assert model.count == 2
    for name, baseline in baseline_parameters.items():
        assert not torch.equal(getattr(model, name).detach(), baseline), name
    for name, baseline in baseline_topology.items():
        assert torch.equal(getattr(model, name), baseline), name
    assert result.milestones[0]["gaussian_count"] == 2
