from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import SourceType


SOURCE_POLICY_VERSION = "sage-source-policy-v3"


@dataclass(frozen=True)
class SourceDescriptor:
    name: str
    source_type: SourceType
    default_confidence: float
    min_candidate_confidence: float
    default_opacity_threshold: float
    priority: int


SOURCE_DESCRIPTORS = (
    SourceDescriptor("LIDAR_CENTER", SourceType.LIDAR_CENTER, 1.0, 0.0, 0.03, 0),
    SourceDescriptor("LIDAR_FUSED", SourceType.LIDAR_FUSED, 0.7, 0.5, 0.05, 1),
    SourceDescriptor("SPNET_COMPLETED", SourceType.SPNET_COMPLETED, 0.4, 0.25, 0.10, 2),
)
SOURCE_NAMES = tuple(descriptor.name for descriptor in SOURCE_DESCRIPTORS)
ALL_SOURCE_DESCRIPTORS = SOURCE_DESCRIPTORS


def source_policy_value(values: dict[str, object], source_name: str):
    if source_name in values:
        return values[source_name]
    raise ValueError(f"Source policy has no value for {source_name}")


def materialize_source_policy(values: dict[str, object], descriptors: tuple[SourceDescriptor, ...]) -> dict[str, object]:
    return {descriptor.name: source_policy_value(values, descriptor.name) for descriptor in descriptors}


def descriptors_for_types(source_types: object) -> tuple[SourceDescriptor, ...]:
    values = source_types.detach().reshape(-1) if isinstance(source_types, torch.Tensor) else source_types
    supported = torch.tensor(
        [int(descriptor.source_type) for descriptor in SOURCE_DESCRIPTORS],
        dtype=torch.uint8,
        device=values.device if isinstance(values, torch.Tensor) else None,
    )
    if isinstance(values, torch.Tensor):
        torch._assert_async(torch.isin(values.to(torch.uint8), supported).all(), "Unsupported source type")
    else:
        if any(int(value) not in {int(item.source_type) for item in SOURCE_DESCRIPTORS} for value in values):
            raise ValueError("Unsupported source type")
    return SOURCE_DESCRIPTORS


def descriptor_for_type(source_type: int | SourceType) -> SourceDescriptor:
    try:
        return next(descriptor for descriptor in SOURCE_DESCRIPTORS if int(descriptor.source_type) == int(source_type))
    except StopIteration as exc:
        raise ValueError(f"Unsupported source type: {source_type}") from exc


def source_counts(
    source_types: torch.Tensor, *, descriptors: tuple[SourceDescriptor, ...] | None = None,
) -> dict[str, int]:
    return {
        descriptor.name: int((source_types == int(descriptor.source_type)).sum())
        for descriptor in (descriptors if descriptors is not None else descriptors_for_types(source_types))
    }


def opacity_keep_mask(opacities: torch.Tensor, source_types: torch.Tensor, thresholds: dict[str, float]) -> torch.Tensor:
    values = torch.zeros((max(int(item.source_type) for item in ALL_SOURCE_DESCRIPTORS) + 1,), dtype=opacities.dtype, device=opacities.device)
    for descriptor in ALL_SOURCE_DESCRIPTORS:
        values[int(descriptor.source_type)] = source_policy_value(thresholds, descriptor.name)
    return opacities.detach().reshape(-1) >= values[source_types.detach().long()]
