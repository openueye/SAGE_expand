import numpy as np
import pytest

from sage.foundation.contracts import (
    DepthEvidence,
    GrowthInputs,
    InputSourceFamily,
    MappingObservation,
    SourceType,
)
from sage.foundation.source_policy import SOURCE_NAMES


def _evidence(source_type: SourceType, family: InputSourceFamily) -> DepthEvidence:
    depth = np.ones((1, 1), dtype=np.float32)
    valid = np.ones((1, 1), dtype=np.bool_)
    return DepthEvidence(source_type, depth, valid, valid.astype(np.float32), family)


def test_stage_i_exposes_only_canonical_gaussian_roles() -> None:
    assert tuple(SourceType) == (
        SourceType.LIDAR_CENTER,
        SourceType.LIDAR_FUSED,
        SourceType.SPNET_COMPLETED,
    )
    assert SOURCE_NAMES == ("LIDAR_CENTER", "LIDAR_FUSED", "SPNET_COMPLETED")


def test_input_family_changes_identity_without_changing_gaussian_role() -> None:
    depth = np.ones((1, 1), dtype=np.float32)
    source_types = np.full((1, 1), int(SourceType.LIDAR_CENTER), dtype=np.uint8)
    confidence = np.ones((1, 1), dtype=np.float32)
    raw = MappingObservation(
        depth, source_types, confidence, InputSourceFamily.LIDAR_RAW,
    )
    slam = MappingObservation(
        depth, source_types, confidence, InputSourceFamily.SLAM_WORLD,
    )

    np.testing.assert_array_equal(raw.source_types, slam.source_types)
    assert raw.source_family == "LIDAR_RAW"
    assert slam.source_family == "SLAM_WORLD"


def test_growth_rejects_mixed_input_families() -> None:
    with pytest.raises(ValueError, match="input source families"):
        GrowthInputs((
            _evidence(SourceType.LIDAR_CENTER, InputSourceFamily.LIDAR_RAW),
            _evidence(SourceType.SPNET_COMPLETED, InputSourceFamily.SLAM_WORLD),
        ))
