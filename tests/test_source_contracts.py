import numpy as np

from sage.foundation.contracts import MappingObservation, SourceType
from sage.foundation.source_policy import SOURCE_NAMES


def test_stage_i_exposes_only_canonical_gaussian_roles() -> None:
    assert tuple(SourceType) == (
        SourceType.LIDAR_CENTER,
        SourceType.LIDAR_FUSED,
        SourceType.SPNET_COMPLETED,
    )
    assert SOURCE_NAMES == ("LIDAR_CENTER", "LIDAR_FUSED", "SPNET_COMPLETED")


def test_mapping_observation_keeps_per_pixel_source_provenance() -> None:
    depth = np.ones((1, 2), dtype=np.float32)
    source_types = np.array(
        [[int(SourceType.LIDAR_CENTER), int(SourceType.LIDAR_FUSED)]], dtype=np.uint8,
    )
    confidence = np.ones((1, 2), dtype=np.float32)
    observation = MappingObservation(depth, source_types, confidence)
    assert observation.source_types.tolist() == [
        [int(SourceType.LIDAR_CENTER), int(SourceType.LIDAR_FUSED)]
    ]
