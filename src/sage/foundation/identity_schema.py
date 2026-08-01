"""Dependency and dataset identity schemas shared across SAGE stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json


_DEPENDENCY_FIELDS = {
    "renderer": {
        "unavailable": {"kind", "purpose"},
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


def validate_dataset_identity(
    checkpoint_snapshot: object,
    current_identity: object,
) -> None:
    if not isinstance(checkpoint_snapshot, dict) or not isinstance(current_identity, dict):
        raise ValueError("Dataset identity must be represented by objects")
    fields = {
        "prepared_manifest_sha256": "prepared_manifest_sha256",
        "transform_contract_sha256": "transform_contract_sha256",
        "scene_content_sha256": "content_sha256",
        "source_mode": "source_mode",
        "source_policy_version": "source_policy_version",
    }
    mismatched = [
        snapshot_name
        for snapshot_name, current_name in fields.items()
        if checkpoint_snapshot.get(snapshot_name) != current_identity.get(current_name)
    ]
    if mismatched:
        raise ValueError(
            "Evaluation dataset identity does not match the checkpoint: "
            f"{', '.join(mismatched)}"
        )


def _canonical_identity_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_dataset_identity(frame_source_identity: object) -> dict[str, object]:
    """Project either FrameSource identity onto the checkpoint comparison contract."""
    if not isinstance(frame_source_identity, dict):
        raise ValueError("Frame source identity must be represented by an object")
    adapter = frame_source_identity.get("adapter")
    if adapter == "prepared-scene":
        return {
            name: frame_source_identity.get(name)
            for name in (
                "prepared_manifest_sha256",
                "transform_contract_sha256",
                "content_sha256",
                "source_mode",
                "source_policy_version",
            )
        }
    if adapter != "rosbag-fixed-lag-v1":
        raise ValueError(f"Unsupported FrameSource identity adapter: {adapter}")
    source = frame_source_identity.get("source")
    input_hashes = frame_source_identity.get("input_files_sha256")
    image_selection = frame_source_identity.get("image_selection")
    if (
        not isinstance(source, dict)
        or not isinstance(input_hashes, dict)
        or not isinstance(image_selection, dict)
        or not isinstance(frame_source_identity.get("preparation_profile"), str)
        or type(frame_source_identity.get("emitted_limit")) is not int
    ):
        raise ValueError("ROSBAG FrameSource identity lacks its dataset contract")
    dataset_contract = {
        name: frame_source_identity[name]
        for name in (
            "adapter",
            "preparation_profile",
            "source",
            "source_policy_version",
            "input_files_sha256",
            "emitted_limit",
            "image_selection",
        )
    }
    return {
        "prepared_manifest_sha256": _canonical_identity_sha256(dataset_contract),
        "transform_contract_sha256": _canonical_identity_sha256(source),
        "content_sha256": (
            str(frame_source_identity["content_sha256"])
            if frame_source_identity.get("content_sha256") is not None
            else _canonical_identity_sha256(input_hashes)
        ),
        "source_mode": str(source["mode"]),
        "source_policy_version": str(frame_source_identity["source_policy_version"]),
    }


__all__ = [
    "DependencyIdentity",
    "_valid_environment_locks",
    "_validate_spnet_adapter_payload",
    "normalize_dataset_identity",
    "validate_dataset_identity",
]
