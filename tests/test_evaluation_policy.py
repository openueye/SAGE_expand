"""Checkpoint-stage selection for streaming evaluation."""

from __future__ import annotations

import pytest

from sage.evaluation.evaluation_run import (
    DEFAULT_CHECKPOINT_STAGES,
    STAGE1_MAPPING,
    STAGE2_REFINEMENT,
    _checkpoint_stage,
    _validate_checkpoint_policy,
)
from sage.foundation.artifact_versions import (
    APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
    CHECKPOINT_VERSION,
)


def test_evaluation_defaults_to_the_refined_checkpoint_only() -> None:
    assert DEFAULT_CHECKPOINT_STAGES == (STAGE2_REFINEMENT,)
    assert _validate_checkpoint_policy((STAGE1_MAPPING, STAGE2_REFINEMENT)) == (
        STAGE1_MAPPING,
        STAGE2_REFINEMENT,
    )


def test_checkpoint_versions_have_explicit_stage_names() -> None:
    assert _checkpoint_stage({"checkpoint_version": CHECKPOINT_VERSION}) == STAGE1_MAPPING
    assert _checkpoint_stage({"checkpoint_version": APPEARANCE_REFINEMENT_CHECKPOINT_VERSION}) == STAGE2_REFINEMENT
    with pytest.raises(ValueError, match="Unsupported"):
        _checkpoint_stage({"checkpoint_version": "unknown"})
    with pytest.raises(ValueError, match="unique subset"):
        _validate_checkpoint_policy((STAGE1_MAPPING, STAGE1_MAPPING))
