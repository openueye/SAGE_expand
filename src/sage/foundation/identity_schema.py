"""Dependency and dataset identity schemas shared across SAGE stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json


_DEPENDENCY_FIELDS = {
    "renderer": {
        "pip-package": {
            "kind", "package", "package_version", "extension_sha256",
            "adapter_schema", "python_soabi", "torch_version", "cuda_version",
            "compute_capability", "torch_cuda_arch_list",
        },
    },
    "spnet": {
        "repository-online": {
            "kind", "source_id", "model_id", "source_commit", "weights_sha256", "source_tree_sha256",
            "depth_scale_m", "confidence", "sample_stride", "adapter", "actual_invocations",
        },
    },
    "metric": {
        "repository-offline": {
            "kind", "model_id", "weights_sha256", "evaluator_schema",
            "torchmetrics_version", "torchvision_version", "calibration_sha256",
            "calibration_package", "device", "dtype",
        },
    },
}

_NATIVE_SPNET_ADAPTER_FIELDS = {
    "policy", "alignment", "rgb_padding", "raw_depth_padding", "input_mask_padding",
    "crop_policy", "raw_anchor_restoration", "sample_grid_policy",
}


def _validate_spnet_adapter_payload(payload: object, *, execution: bool) -> None:
    if not isinstance(payload, dict):
        raise ValueError("SPNet adapter identity must be an object")
    if payload.get("policy") != "native-full-frame-pad-crop-v1":
        raise ValueError("SPNet adapter identity policy is invalid")
    expected = _NATIVE_SPNET_ADAPTER_FIELDS | ({"frame_grids"} if execution else set())
    if set(payload) != expected:
        raise ValueError("Native SPNet adapter identity fields are invalid")
    if (
        payload["alignment"] != 32
        or payload["rgb_padding"] != "replicate-bottom-right"
        or payload["raw_depth_padding"] != "zero-bottom-right"
        or payload["input_mask_padding"] != "zero-bottom-right"
        or payload["crop_policy"] != "top-left-native-crop-no-resampling"
        or payload["raw_anchor_restoration"] != "bit-exact-raw-valid-mask"
        or payload["sample_grid_policy"] != "native-origin-stride"
    ):
        raise ValueError("Native SPNet adapter identity values are invalid")
    if not execution:
        return
    grids = payload["frame_grids"]
    if not isinstance(grids, list):
        raise ValueError("Native SPNet adapter frame grids must be a list")
    seen_frames: set[tuple[int, str]] = set()
    for grid in grids:
        if not isinstance(grid, dict) or set(grid) != {
            "frame_index", "frame_stem", "native_grid", "network_grid", "padding",
        }:
            raise ValueError("Native SPNet adapter frame grid fields are invalid")
        frame = grid["frame_index"], grid["frame_stem"]
        native = grid["native_grid"]
        network = grid["network_grid"]
        padding = grid["padding"]
        if (
            type(frame[0]) is not int
            or frame[0] < 0
            or not isinstance(frame[1], str)
            or not frame[1]
            or frame in seen_frames
            or not isinstance(native, dict)
            or not isinstance(network, dict)
            or not isinstance(padding, dict)
            or set(native) != {"width", "height"}
            or set(network) != {"width", "height"}
            or set(padding) != {"bottom", "right"}
        ):
            raise ValueError("Native SPNet adapter frame grid is invalid")
        values = (*native.values(), *network.values(), *padding.values())
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Native SPNet adapter frame grid dimensions are invalid")
        if (
            native["width"] < 1
            or native["height"] < 1
            or network["width"] != ((native["width"] + 31) // 32) * 32
            or network["height"] != ((native["height"] + 31) // 32) * 32
            or padding["right"] != network["width"] - native["width"]
            or padding["bottom"] != network["height"] - native["height"]
        ):
            raise ValueError("Native SPNet adapter frame grid alignment is invalid")
        seen_frames.add(frame)


@dataclass(frozen=True)
class DependencyIdentity:
    renderer: dict[str, object]
    spnet: dict[str, object]
    metric: dict[str, object]

    def __post_init__(self) -> None:
        for name in ("renderer", "spnet", "metric"):
            payload = getattr(self, name)
            if not isinstance(payload, dict) or not isinstance(payload.get("kind"), str):
                raise ValueError(f"Dependency identity {name} must be a discriminated object")
            allowed = _DEPENDENCY_FIELDS[name].get(payload["kind"])
            if allowed is None or set(payload) != allowed:
                raise ValueError(f"Unknown or incomplete {name} dependency identity branch")
            if name == "spnet" and payload["kind"] == "repository-online":
                _validate_spnet_adapter_payload(payload["adapter"], execution=False)

    @classmethod
    def for_renderer_with_dependencies(
        cls,
        renderer: dict[str, object],
        spnet_identity: object,
        *,
        actual_invocations: int,
        metric_identity: object,
    ) -> "DependencyIdentity":
        dependency_payload = getattr(spnet_identity, "dependency_payload", None)
        if not callable(dependency_payload):
            raise ValueError("Online SPNet identity lacks dependency_payload()")
        return cls(
            renderer=deepcopy(renderer),
            spnet=dependency_payload(actual_invocations=actual_invocations),
            metric=_metric_dependency_payload(metric_identity),
        )

    def payload(self) -> dict[str, object]:
        return {"renderer": deepcopy(self.renderer), "spnet": deepcopy(self.spnet), "metric": deepcopy(self.metric)}


def _metric_dependency_payload(metric_identity: object) -> dict[str, object]:
    dependency_payload = getattr(metric_identity, "dependency_payload", None)
    if not callable(dependency_payload):
        raise ValueError("Metric identity lacks dependency_payload()")
    payload = dependency_payload()
    if not isinstance(payload, dict) or payload.get("kind") != "repository-offline":
        raise ValueError("Metric identity must provide the repository-offline dependency branch")
    return deepcopy(payload)


def _valid_environment_locks(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"environment.yml", "conda-lock.yml"}
        and all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in value.values()
        )
    )


def validate_dataset_identity(checkpoint_snapshot: object, current: object) -> None:
    """Reject reuse of an artifact produced from a different canonical input.

    `current` is the run's `InputIdentities`; only the canonical sequence and
    contract matter, so a checkpoint stays usable across a ROSBAG and the
    Prepared Scene exported from it.
    """
    from ..input.identity import InputIdentities, validate_core_identity_match

    if not isinstance(current, InputIdentities):
        raise ValueError("Current dataset identity must be an InputIdentities")
    if not isinstance(checkpoint_snapshot, dict):
        raise ValueError("Dataset identity snapshot must be an object")
    recorded = checkpoint_snapshot.get("input", checkpoint_snapshot)
    validate_core_identity_match(recorded, current)


def _canonical_identity_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "DependencyIdentity",
    "_valid_environment_locks",
    "_validate_spnet_adapter_payload",
    "validate_dataset_identity",
]
