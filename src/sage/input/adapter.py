"""The one seam SAGE Core is allowed to know about.

An adapter resolves its input exactly once. `preflight()` compiles the input:
it indexes events, applies effective timestamps, freezes the accepted frame
order, resolves transforms and computes identities. `ResolvedInput.frames()`
then replays that frozen result and nothing else — training may not re-guess a
topic, re-select frames or change synchronization after preflight.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contract import ResolvedInputContract
from .frame import FrameInputs
from .identity import CanonicalSequenceDigest, InputIdentities
from .preflight import PreflightReport


@dataclass(eq=False)
class ResolvedInput:
    """A frozen, replayable input. Only produced by a successful preflight."""

    contract: ResolvedInputContract
    report: PreflightReport
    source_identity: str
    _frames: Callable[[], Iterator[FrameInputs]]
    _sequence_identity: str | None = None

    def frames(self) -> Iterator[FrameInputs]:
        """Yield the accepted canonical frames in frozen order.

        A completed pass also settles the canonical sequence identity, so
        training never has to decode the input twice just to identify it.
        """
        expected = self.contract.canonical.frame_count
        digest = CanonicalSequenceDigest(self.contract.canonical_contract_identity)
        emitted = 0
        for frame in self._frames():
            if frame.frame_index != emitted:
                raise ValueError(
                    f"Resolved input emitted frame_index={frame.frame_index}, expected {emitted}"
                )
            emitted += 1
            if emitted > expected:
                raise ValueError("Resolved input emitted more frames than the frozen contract")
            digest.update(frame)
            yield frame
        if emitted != expected:
            raise ValueError(
                f"Resolved input emitted {emitted} frames, frozen contract declares {expected}"
            )
        settled = digest.hexdigest()
        if self._sequence_identity is not None and settled != self._sequence_identity:
            raise ValueError("Canonical frame sequence changed between replays of one resolved input")
        self._sequence_identity = settled

    @property
    def canonical_sequence_identity(self) -> str:
        """Digest of the frames themselves; computed by replaying once if needed."""
        if self._sequence_identity is None:
            for _ in self.frames():
                pass
        assert self._sequence_identity is not None
        return self._sequence_identity

    @property
    def identities(self) -> InputIdentities:
        return InputIdentities(
            source_identity=self.source_identity,
            adapter_provenance_identity=self.contract.adapter_provenance_identity,
            canonical_contract_identity=self.contract.canonical_contract_identity,
            canonical_sequence_identity=self.canonical_sequence_identity,
        )

    def summary_lines(self) -> list[str]:
        """The uniform run header both adapters print."""
        canonical = self.contract.canonical
        width, height = canonical.image_size
        fusion = canonical.fusion
        lines = [
            f"Input adapter: {self.contract.adapter_type}",
            f"Canonical sequence identity: {self.identities.canonical_sequence_identity}",
            f"Canonical contract identity: {self.identities.canonical_contract_identity}",
            f"Adapter provenance identity: {self.identities.adapter_provenance_identity}",
            "",
            "Frames:",
            f"  accepted: {self.report.accepted_frames}",
            f"  rejected: {len(self.report.rejected_frames)}",
            "",
            "Camera:",
            f"  model: {canonical.camera_model}",
            f"  resolution: {width} x {height}",
            f"  pose convention: {canonical.pose_convention}",
            "",
            "Observations:",
        ]
        for name in ("LIDAR_CENTER", "LIDAR_FUSED"):
            state = "enabled" if name in canonical.sources else "disabled"
            lines.append(f"  {name}: {state}")
        lines.append("  SPNET_COMPLETED: generated during mapping")
        lines.extend([
            "",
            "Fusion:",
            f"  policy: {fusion['policy']}",
            f"  causal: {str(bool(fusion['causal'])).lower()}",
        ])
        for key in ("max_scans", "max_age_ms"):
            if key in fusion:
                lines.append(f"  {key}: {fusion[key]}")
        return lines


@runtime_checkable
class InputAdapter(Protocol):
    """Every input source SAGE supports implements exactly this."""

    def preflight(self) -> ResolvedInput:
        """Resolve and freeze the input once; failures stay in the report."""
        ...


__all__ = ["InputAdapter", "ResolvedInput"]
