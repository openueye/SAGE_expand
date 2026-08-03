"""Identity-bound reuse of dense SPNet predictions between SAGE stages."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ...foundation.contracts import DepthEvidence, SourceType
from ...core_input import MappingFrame
from .spnet import (
    DenseSPNetProvider,
    SPNetFrameGrid,
    SPNetIdentity,
    derive_spnet_grid,
    validate_spnet_identity_payload,
)


DENSE_SPNET_CACHE_SCHEMA = "sage-dense-spnet-cache-v1"


@dataclass(frozen=True)
class DenseSPNetFrame:
    frame_index: int
    frame_stem: str
    depth_m: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        if (
            type(self.frame_index) is not int
            or self.frame_index < 0
            or not isinstance(self.frame_stem, str)
            or not self.frame_stem
            or depth.dtype != np.float32
            or depth.ndim != 2
            or not np.isfinite(depth).all()
        ):
            raise ValueError("Dense SPNet cache frame is invalid")


@dataclass(frozen=True)
class DenseSPNetCache:
    frames: tuple[DenseSPNetFrame, ...]
    provider_identity: dict[str, object]

    def __post_init__(self) -> None:
        indices = [frame.frame_index for frame in self.frames]
        if not self.frames or indices != sorted(set(indices)):
            raise ValueError("Dense SPNet cache frames must be unique and sorted")
        provider_frames = validate_spnet_identity_payload(self.provider_identity)
        if [frame.frame_index for frame in provider_frames] != indices:
            raise ValueError("Dense SPNet cache provider frames do not match data")


def write_dense_spnet_cache(
    path: Path,
    frames: tuple[DenseSPNetFrame, ...],
    provider_identity: SPNetIdentity,
) -> None:
    cache = DenseSPNetCache(frames, provider_identity.payload())
    destination = Path(path)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite dense SPNet cache: {destination}")
    torch.save({
        "schema_version": DENSE_SPNET_CACHE_SCHEMA,
        "provider_identity": cache.provider_identity,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "frame_stem": frame.frame_stem,
                "depth_m": torch.from_numpy(frame.depth_m.copy()),
            }
            for frame in cache.frames
        ],
    }, destination)


def load_dense_spnet_cache(path: Path) -> DenseSPNetCache:
    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot load dense SPNet cache: {source}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "provider_identity", "frames"}
        or payload["schema_version"] != DENSE_SPNET_CACHE_SCHEMA
        or not isinstance(payload["provider_identity"], dict)
        or not isinstance(payload["frames"], list)
    ):
        raise ValueError("Dense SPNet cache contract is invalid")
    frames: list[DenseSPNetFrame] = []
    for item in payload["frames"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"frame_index", "frame_stem", "depth_m"}
            or not isinstance(item["depth_m"], torch.Tensor)
            or item["depth_m"].device.type != "cpu"
            or item["depth_m"].dtype != torch.float32
        ):
            raise ValueError("Dense SPNet cache frame contract is invalid")
        frames.append(DenseSPNetFrame(
            item["frame_index"],
            item["frame_stem"],
            item["depth_m"].numpy(),
        ))
    return DenseSPNetCache(tuple(frames), payload["provider_identity"])


def _identity_without_frames(identity: SPNetIdentity | dict[str, object]) -> dict[str, object]:
    payload = identity.payload() if isinstance(identity, SPNetIdentity) else dict(identity)
    payload.pop("frame_grids", None)
    return payload


class ReusingDenseSPNetProvider:
    """Serve cached predictions and use the online provider only for cache misses."""

    def __init__(
        self,
        cache: DenseSPNetCache,
        fallback: DenseSPNetProvider,
    ) -> None:
        if _identity_without_frames(cache.provider_identity) != _identity_without_frames(
            fallback.identity
        ):
            raise ValueError("Dense SPNet cache identity does not match the provider")
        self._frames = {frame.frame_index: frame for frame in cache.frames}
        self._fallback = fallback
        self._identity = replace(fallback.identity, frame_grids=())
        self.reused_frames = 0
        self.live_frames = 0

    @property
    def identity(self) -> SPNetIdentity:
        return self._identity

    def dense_evidence_for(self, frame: MappingFrame) -> DepthEvidence:
        cached = self._frames.get(frame.index)
        if cached is None:
            evidence = self._fallback.dense_evidence_for(frame)
            self.live_frames += 1
        else:
            if cached.frame_stem != frame.stem or cached.depth_m.shape != frame.rgb.shape[:2]:
                raise ValueError("Dense SPNet cache frame does not match scene input")
            valid = np.isfinite(cached.depth_m) & (cached.depth_m > 0)
            confidence = float(self._identity.confidence)
            evidence = DepthEvidence(
                SourceType.SPNET_COMPLETED,
                cached.depth_m,
                valid,
                np.where(valid, confidence, 0.0).astype(np.float32),
            )
            self.reused_frames += 1
        self._record_frame(frame)
        return evidence

    def _record_frame(self, frame: MappingFrame) -> None:
        records = self._identity.frame_grids
        if records and frame.index <= records[-1].frame_index:
            raise ValueError("Dense SPNet frames must be consumed in increasing order")
        self._identity = replace(
            self._identity,
            frame_grids=(*records, SPNetFrameGrid(
                frame.index,
                frame.stem,
                derive_spnet_grid(frame.rgb.shape[:2]),
            )),
        )
