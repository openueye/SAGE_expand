"""Canonical runtime identity used to bind production evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


RUNTIME_IDENTITY_FIELDS = (
    "python", "python_executable", "torch", "torch_cuda", "cuda_available",
    "cuda_device_count", "cuda_devices", "user_site_path", "user_site_enabled",
    "torchvision_path", "torchmetrics_path", "spnet_identity", "metric_calibration_path",
    "model_paths", "model_identities", "conda_lock_path", "conda_lock_sha256",
)


def runtime_identity_from_report(report: Mapping[str, object]) -> dict[str, object]:
    missing = [name for name in RUNTIME_IDENTITY_FIELDS if name not in report]
    if missing:
        raise ValueError(f"Runtime identity report is missing fields: {missing}")
    return {name: deepcopy(report[name]) for name in RUNTIME_IDENTITY_FIELDS}


__all__ = ["RUNTIME_IDENTITY_FIELDS", "runtime_identity_from_report"]
