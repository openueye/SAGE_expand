"""What preflight actually observed, as opposed to what it froze.

The contract carries semantics; this report carries evidence: how many
candidate frames existed, which ones were rejected and why, and the
distributions a reviewer needs to judge whether the association thresholds were
sane. It is written next to the run artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Mapping, Sequence


PREFLIGHT_REPORT_SCHEMA = "sage_preflight_report"
PREFLIGHT_REPORT_REVISION = 1


@dataclass(frozen=True)
class RejectedFrame:
    candidate_index: int
    timestamp_ns: int
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise ValueError("Rejected frame candidate_index must be a non-negative integer")
        if type(self.timestamp_ns) is not int:
            raise ValueError("Rejected frame timestamp_ns must be an integer")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Rejected frame reason must be non-empty")
        details = dict(self.details)
        json.dumps(details, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "details", details)

    def payload(self) -> dict[str, object]:
        return {
            "candidate_index": self.candidate_index,
            "timestamp_ns": self.timestamp_ns,
            "reason": self.reason,
            "details": dict(self.details),
        }


class PreflightReport:
    """Mutable during preflight, frozen once the adapter hands it over."""

    def __init__(self, adapter_type: str) -> None:
        if not isinstance(adapter_type, str) or not adapter_type:
            raise ValueError("Preflight report adapter_type must be a non-empty name")
        self.adapter_type = adapter_type
        self._errors: list[str] = []
        self._rejected: list[RejectedFrame] = []
        self._statistics: dict[str, object] = {}
        self._candidate_frames = 0
        self._accepted_frames = 0

    def add_error(self, message: str) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("Preflight error must be a non-empty message")
        self._errors.append(message)

    def reject(self, frame: RejectedFrame) -> None:
        if not isinstance(frame, RejectedFrame):
            raise ValueError("Preflight rejection must be a RejectedFrame")
        self._rejected.append(frame)

    def record_counts(self, *, candidate_frames: int, accepted_frames: int) -> None:
        if candidate_frames < accepted_frames or accepted_frames < 0:
            raise ValueError("Preflight accepted frames cannot exceed candidate frames")
        self._candidate_frames = int(candidate_frames)
        self._accepted_frames = int(accepted_frames)

    def record_statistics(self, name: str, values: Mapping[str, object]) -> None:
        payload = dict(values)
        json.dumps(payload, sort_keys=True, allow_nan=False)
        self._statistics[name] = payload

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    @property
    def rejected_frames(self) -> tuple[RejectedFrame, ...]:
        return tuple(self._rejected)

    @property
    def accepted_frames(self) -> int:
        return self._accepted_frames

    def raise_for_errors(self) -> None:
        if self._errors:
            raise ValueError(
                f"{self.adapter_type} preflight failed:\n  - " + "\n  - ".join(self._errors)
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema_name": PREFLIGHT_REPORT_SCHEMA,
            "schema_revision": PREFLIGHT_REPORT_REVISION,
            "adapter_type": self.adapter_type,
            "candidate_frames": self._candidate_frames,
            "accepted_frames": self._accepted_frames,
            "rejected_frames": len(self._rejected),
            "rejections": [frame.payload() for frame in self._rejected],
            "errors": list(self._errors),
            "statistics": dict(self._statistics),
        }


def distribution(values: Sequence[float]) -> dict[str, float]:
    """Mean/max summary used for skew and gap statistics; empty stays explicit."""
    if not values:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


__all__ = ["PreflightReport", "RejectedFrame", "distribution"]
