import numpy as np

from sage.foundation.contracts import (
    InputSourceFamily,
    MappingObservation,
    SourceType,
)
from sage.foundation.source_policy import SOURCE_NAMES


def test_stage_i_exposes_only_canonical_gaussian_roles() -> None:
    assert tuple(SourceType) == (
        SourceType.LIDAR_CENTER,
        SourceType.LIDAR_FUSED,
        SourceType.SPNET_COMPLETED,
    )
    assert SOURCE_NAMES == ("LIDAR_CENTER", "LIDAR_FUSED", "SPNET_COMPLETED")


def test_input_family_identity_matches_declared_family() -> None:
    depth = np.ones((1, 1), dtype=np.float32)
    source_types = np.full((1, 1), int(SourceType.LIDAR_CENTER), dtype=np.uint8)
    confidence = np.ones((1, 1), dtype=np.float32)
    observation = MappingObservation(
        depth, source_types, confidence, InputSourceFamily.LIDAR_WORLD,
    )
    assert observation.source_family == "LIDAR_WORLD"
