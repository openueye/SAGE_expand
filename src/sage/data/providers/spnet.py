from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import types
import warnings
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F

from ...engine.model_registry import ModelRegistry, default_registry_path
from ...foundation.config import (
    NATIVE_FULL_FRAME_PAD_CROP_V1,
    SPNetOnlineConfig,
)
from ...foundation.contracts import DepthEvidence, FrameInputs, SourceType


SPNET_SOURCE_IDENTITY_SCHEMA_VERSION = "sage-spnet-source-identity-v1"


def _load_source_identity_lock() -> dict[str, object]:
    path = Path(__file__).with_name("spnet_identity.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read SPNet source identity lock: {path}") from exc
    expected_fields = {
        "schema_version", "source_id", "repository", "commit",
        "tracked_tree_sha256", "tracked_files",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RuntimeError("SPNet source identity lock fields are invalid")
    if payload["schema_version"] != SPNET_SOURCE_IDENTITY_SCHEMA_VERSION:
        raise RuntimeError("SPNet source identity lock schema is unsupported")
    for name in ("source_id", "repository"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise RuntimeError(f"SPNet source identity lock {name} is invalid")
    for name, length in (("commit", 40), ("tracked_tree_sha256", 64)):
        value = payload[name]
        if not isinstance(value, str) or len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuntimeError(f"SPNet source identity lock {name} is invalid")
    tracked_files = payload["tracked_files"]
    if not isinstance(tracked_files, list) or not tracked_files:
        raise RuntimeError("SPNet source identity lock tracked_files is invalid")
    if tracked_files != sorted(set(tracked_files)):
        raise RuntimeError("SPNet source identity lock tracked_files must be unique and sorted")
    for value in tracked_files:
        path_value = PurePosixPath(value) if isinstance(value, str) else None
        if (
            path_value is None
            or path_value.is_absolute()
            or not path_value.parts
            or path_value.parts[0] != "SPNet"
            or ".." in path_value.parts
        ):
            raise RuntimeError("SPNet source identity lock contains an invalid tracked path")
    return payload


_SPNET_SOURCE_IDENTITY = _load_source_identity_lock()
SPNET_SOURCE_PROVENANCE_COMMIT = str(_SPNET_SOURCE_IDENTITY["commit"])
SPNET_SOURCE_TREE_SHA256 = str(_SPNET_SOURCE_IDENTITY["tracked_tree_sha256"])
SPNET_SOURCE_ID = str(_SPNET_SOURCE_IDENTITY["source_id"])
SPNET_SOURCE_REPOSITORY = str(_SPNET_SOURCE_IDENTITY["repository"])
SPNET_TRACKED_FILES = tuple(str(value) for value in _SPNET_SOURCE_IDENTITY["tracked_files"])
SPNET_ROOT_ENVIRONMENT_VARIABLE = "SAGE_SPNET_ROOT"
SPNET_LARGE_DIMS = [192, 384, 768, 1536]
SPNET_LARGE_DEPTHS = [3, 3, 27, 3]
SPNET_ALIGNMENT = 32


@dataclass(frozen=True)
class SPNetGrid:
    native_height: int
    native_width: int
    network_height: int
    network_width: int

    def __post_init__(self) -> None:
        values = (self.native_height, self.native_width, self.network_height, self.network_width)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("SPNet grid dimensions must be positive integers")
        expected_height = ((self.native_height + SPNET_ALIGNMENT - 1) // SPNET_ALIGNMENT) * SPNET_ALIGNMENT
        expected_width = ((self.native_width + SPNET_ALIGNMENT - 1) // SPNET_ALIGNMENT) * SPNET_ALIGNMENT
        if (self.network_height, self.network_width) != (expected_height, expected_width):
            raise ValueError("SPNet network grid must be the aligned native ceiling")

    @property
    def native_size(self) -> tuple[int, int]:
        return self.native_height, self.native_width

    @property
    def network_size(self) -> tuple[int, int]:
        return self.network_height, self.network_width

    @property
    def pad_bottom(self) -> int:
        return self.network_height - self.native_height

    @property
    def pad_right(self) -> int:
        return self.network_width - self.native_width

    def payload(self) -> dict[str, object]:
        return {
            "native_grid": {
                "width": self.native_width,
                "height": self.native_height,
            },
            "network_grid": {
                "width": self.network_width,
                "height": self.network_height,
            },
            "padding": {"bottom": self.pad_bottom, "right": self.pad_right},
        }

@dataclass(frozen=True)
class SPNetFrameGrid:
    frame_index: int
    frame_stem: str
    grid: SPNetGrid

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("SPNet frame grid index must be a non-negative integer")
        if not isinstance(self.frame_stem, str) or not self.frame_stem:
            raise ValueError("SPNet frame grid stem must be non-empty")
        if not isinstance(self.grid, SPNetGrid):
            raise ValueError("SPNet frame grid must contain an SPNetGrid")

    def payload(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_stem": self.frame_stem,
            **self.grid.payload(),
        }

def derive_spnet_grid(native_size: tuple[int, int]) -> SPNetGrid:
    if (
        not isinstance(native_size, tuple)
        or len(native_size) != 2
        or any(type(value) is not int or value < 1 for value in native_size)
    ):
        raise ValueError("SPNet native grid must contain positive height and width")
    height, width = native_size
    return SPNetGrid(
        native_height=height,
        native_width=width,
        network_height=((height + SPNET_ALIGNMENT - 1) // SPNET_ALIGNMENT) * SPNET_ALIGNMENT,
        network_width=((width + SPNET_ALIGNMENT - 1) // SPNET_ALIGNMENT) * SPNET_ALIGNMENT,
    )


@dataclass(frozen=True)
class SPNetIdentity:
    mode: str
    source_id: str
    model_id: str | None = None
    weights_sha256: str | None = None
    source_content_sha256: str | None = None
    source_commit: str | None = None
    adapter_policy: str = NATIVE_FULL_FRAME_PAD_CROP_V1
    depth_scale_m: float | None = None
    confidence: float | None = None
    sample_stride: int | None = None
    frame_grids: tuple[SPNetFrameGrid, ...] = ()

    def __post_init__(self) -> None:
        if self.mode != "online":
            raise ValueError("SPNet provider identity mode must be online")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("SPNet provider identity source_id must be non-empty")
        if self.adapter_policy != NATIVE_FULL_FRAME_PAD_CROP_V1:
            raise ValueError("SPNet provider identity adapter_policy is unsupported")

    def payload(self) -> dict[str, object]:
        def convert(value: object) -> object:
            if hasattr(value, "__dataclass_fields__"):
                return convert(asdict(value))
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(name): convert(item) for name, item in value.items()}
            return value

        return {field.name: convert(getattr(self, field.name)) for field in fields(self)}

    def _adapter_payload(
        self,
        *,
        include_frame_grids: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "policy": NATIVE_FULL_FRAME_PAD_CROP_V1,
            "alignment": SPNET_ALIGNMENT,
            "rgb_padding": "replicate-bottom-right",
            "raw_depth_padding": "zero-bottom-right",
            "input_mask_padding": "zero-bottom-right",
            "crop_policy": "top-left-native-crop-no-resampling",
            "raw_anchor_restoration": "bit-exact-raw-valid-mask",
            "sample_grid_policy": "native-origin-stride",
        }
        if include_frame_grids:
            payload["frame_grids"] = [
                record.payload() for record in self.frame_grids
            ]
        return payload

    def execution_adapter_payload(self) -> dict[str, object]:
        return self._adapter_payload(include_frame_grids=True)

    def dependency_payload(
        self,
        *,
        actual_invocations: int,
    ) -> dict[str, object]:
        required = (
            self.model_id,
            self.source_commit,
            self.weights_sha256,
            self.source_content_sha256,
            self.depth_scale_m,
            self.confidence,
            self.sample_stride,
        )
        if any(value is None for value in required):
            raise ValueError(
                "SPNet identity is incomplete for a repository dependency"
            )
        if type(actual_invocations) is not int or actual_invocations < 0:
            raise ValueError(
                "SPNet invocation count must be a non-negative integer"
            )
        return {
            "kind": "repository-online",
            "source_id": self.source_id,
            "model_id": self.model_id,
            "source_commit": self.source_commit,
            "weights_sha256": self.weights_sha256,
            "source_tree_sha256": self.source_content_sha256,
            "depth_scale_m": self.depth_scale_m,
            "confidence": self.confidence,
            "sample_stride": self.sample_stride,
            "adapter": self._adapter_payload(include_frame_grids=False),
            "actual_invocations": actual_invocations,
        }

def validate_spnet_identity_payload(
    value: object,
) -> tuple[SPNetFrameGrid, ...]:
    """Validate and reconstruct a recorded online SPNet execution identity."""
    expected = {
        "mode",
        "source_id",
        "model_id",
        "weights_sha256",
        "source_content_sha256",
        "source_commit",
        "adapter_policy",
        "depth_scale_m",
        "confidence",
        "sample_stride",
        "frame_grids",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("SPNet provider identity fields are invalid")
    for name in ("source_id", "model_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"SPNet provider identity {name} is invalid")
    for name, length in (
        ("weights_sha256", 64),
        ("source_content_sha256", 64),
        ("source_commit", 40),
    ):
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != length
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"SPNet provider identity {name} is invalid")
    try:
        depth_scale_m = float(value["depth_scale_m"])
        confidence = float(value["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("SPNet provider numeric identity is invalid") from exc
    if (
        value["mode"] != "online"
        or value["adapter_policy"] != NATIVE_FULL_FRAME_PAD_CROP_V1
        or not np.isfinite(depth_scale_m)
        or depth_scale_m <= 0
        or not np.isfinite(confidence)
        or not 0 <= confidence <= 1
        or type(value["sample_stride"]) is not int
        or value["sample_stride"] < 1
        or not isinstance(value["frame_grids"], list)
        or not value["frame_grids"]
    ):
        raise ValueError("SPNet provider execution identity is invalid")
    records = []
    for item in value["frame_grids"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"frame_index", "frame_stem", "grid"}
            or not isinstance(item["grid"], dict)
            or set(item["grid"])
            != {
                "native_height",
                "native_width",
                "network_height",
                "network_width",
            }
        ):
            raise ValueError("SPNet provider frame-grid identity is invalid")
        try:
            grid = SPNetGrid(**item["grid"])
            record = SPNetFrameGrid(
                frame_index=item["frame_index"],
                frame_stem=item["frame_stem"],
                grid=grid,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SPNet provider frame-grid identity is invalid"
            ) from exc
        records.append(record)
    indices = [record.frame_index for record in records]
    if indices != sorted(set(indices)):
        raise ValueError(
            "SPNet provider frame-grid indices must be unique and sorted"
        )
    return tuple(records)


class DenseSPNetProvider(Protocol):
    @property
    def identity(self) -> SPNetIdentity: ...

    def dense_evidence_for(self, frame: FrameInputs) -> DepthEvidence: ...


class SPNetEvidenceProvider(Protocol):
    @property
    def identity(self) -> SPNetIdentity: ...

    def evidence_for(self, frame: FrameInputs) -> DepthEvidence | None: ...


class SPNetPredictor(Protocol):
    def __call__(
        self,
        rgb_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> torch.Tensor: ...


def _sha256_labeled_content(files: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    for label, content in files:
        file_digest = hashlib.sha256(content).hexdigest()
        digest.update(f"{label}\0{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()


def default_spnet_source_root() -> Path:
    configured = os.environ.get(SPNET_ROOT_ENVIRONMENT_VARIABLE)
    root = (
        Path(configured).expanduser().resolve()
        if configured else Path(__file__).resolve().parents[3] / "third_party" / "SPNet"
    )
    if not (root / "src" / "networks.py").is_file():
        raise ValueError(
            "SPNet source is unavailable; clone the locked upstream revision into "
            f"third_party/SPNet or set {SPNET_ROOT_ENVIRONMENT_VARIABLE}: {root / 'src' / 'networks.py'}"
        )
    return root


def _read_spnet_source_tree(source_root: Path | None = None) -> tuple[Path, dict[str, bytes], str]:
    root = (source_root or default_spnet_source_root()).resolve()
    contents: dict[str, bytes] = {}
    for label in SPNET_TRACKED_FILES:
        path = root.joinpath(*PurePosixPath(label).parts[1:])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"SPNet tracked source is unavailable or symbolic: {path}")
        try:
            contents[label] = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Cannot read SPNet tracked source: {path}") from exc
    files = tuple((label, contents[label]) for label in SPNET_TRACKED_FILES)
    return root, contents, _sha256_labeled_content(files)


def verify_spnet_source_tree(source_root: Path | None = None) -> tuple[Path, str]:
    root, _, actual_sha256 = _read_spnet_source_tree(source_root)
    if actual_sha256 != SPNET_SOURCE_TREE_SHA256:
        raise ValueError(
            "SPNet source tree SHA-256 mismatch: "
            f"expected {SPNET_SOURCE_TREE_SHA256}, got {actual_sha256} at {root}"
        )
    return root, actual_sha256


def spnet_source_tree_sha256(source_root: Path | None = None) -> str:
    return _read_spnet_source_tree(source_root)[2]


def verify_spnet_source_identity(
    source_root: Path | None = None,
) -> tuple[Path, str, str]:
    root, source_tree_sha256 = verify_spnet_source_tree(source_root)
    return root, SPNET_SOURCE_PROVENANCE_COMMIT, source_tree_sha256


def capture_spnet_identity(
    config: SPNetOnlineConfig,
    *,
    registry: ModelRegistry | None = None,
    source_root: Path | None = None,
    model_root: Path | None = None,
) -> SPNetIdentity:
    registry = registry or ModelRegistry.load(default_registry_path())
    registry.resolve(config.model_id, model_root=model_root)
    _, source_tree_sha256 = verify_spnet_source_tree(source_root)
    return SPNetIdentity(
        mode="online",
        source_id=SPNET_SOURCE_ID,
        model_id=config.model_id,
        weights_sha256=registry.require(config.model_id).sha256,
        source_content_sha256=source_tree_sha256,
        source_commit=SPNET_SOURCE_PROVENANCE_COMMIT,
        adapter_policy=config.adapter_policy,
        depth_scale_m=config.depth_scale_m,
        confidence=config.confidence,
        sample_stride=config.sample_stride,
    )


def _load_network_module(source_root: Path):
    source_root, source_contents, source_tree_sha256 = _read_spnet_source_tree(source_root)
    if source_tree_sha256 != SPNET_SOURCE_TREE_SHA256:
        raise ValueError(
            "SPNet source tree SHA-256 mismatch: "
            f"expected {SPNET_SOURCE_TREE_SHA256}, got {source_tree_sha256} at {source_root}"
        )
    source_dir = source_root / "src"
    package_name = f"_sage_spnet_{source_tree_sha256[:16]}"
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            sys.modules.pop(name, None)
    package = types.ModuleType(package_name)
    package.__path__ = []
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        loaded_modules: dict[str, types.ModuleType] = {}
        for short_name in ("custom_blocks", "modules", "networks"):
            label = f"SPNet/src/{short_name}.py"
            path = source_dir / f"{short_name}.py"
            module_name = f"{package_name}.{short_name}"
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            module.__package__ = package_name
            sys.modules[module_name] = module
            code = compile(source_contents[label], str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
            loaded_modules[short_name] = module
    except Exception as exc:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise RuntimeError(f"Cannot import user-provided SPNet module: {source_dir / 'networks.py'}") from exc
    network_module = loaded_modules["networks"]
    loaded_path = Path(network_module.__file__).resolve()
    network_path = (source_dir / "networks.py").resolve()
    if loaded_path != network_path:
        raise RuntimeError(f"Loaded SPNet source does not match the verified source: {loaded_path}")
    return network_module


def _load_spnet_network(
    config: SPNetOnlineConfig,
    *,
    registry: ModelRegistry,
    source_root: Path,
    model_root: Path | None,
    device: torch.device,
) -> torch.nn.Module:
    weights_path = registry.resolve(config.model_id, model_root=model_root)
    network_module = _load_network_module(source_root)
    network = network_module.V2Net(SPNET_LARGE_DIMS, SPNET_LARGE_DEPTHS, 0.2, "CNX")
    try:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot load SPNet checkpoint for {config.model_id}: {weights_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"network"}:
        raise ValueError("SPNet checkpoint must contain exactly one network state")
    try:
        network.load_state_dict(payload["network"])
    except Exception as exc:
        raise RuntimeError(f"SPNet checkpoint state is incompatible: {weights_path}") from exc
    return network.to(device=device, dtype=torch.float32).eval()


class OnlineSPNetProvider:
    def __init__(
        self,
        config: SPNetOnlineConfig,
        *,
        device: str | torch.device = "cuda",
        registry: ModelRegistry | None = None,
        source_root: Path | None = None,
        model_root: Path | None = None,
    ) -> None:
        self._config = config
        self._device = torch.device(device)
        self._registry = registry or ModelRegistry.load(default_registry_path())
        self._source_root = (source_root or default_spnet_source_root()).resolve()
        self._model_root = model_root
        self._identity = capture_spnet_identity(
            config,
            registry=self._registry,
            source_root=self._source_root,
            model_root=model_root,
        )
        self._network = _load_spnet_network(
            config,
            registry=self._registry,
            source_root=self._source_root,
            model_root=model_root,
            device=self._device,
        )

    @property
    def identity(self) -> SPNetIdentity:
        return self._identity

    def _predict(
        self,
        rgb_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> torch.Tensor:
        try:
            with torch.inference_mode():
                return self._network(
                    rgb_tensor.to(self._device, dtype=torch.float32),
                    depth_tensor.to(self._device, dtype=torch.float32),
                    mask_tensor.to(self._device, dtype=torch.float32),
                )
        except Exception as exc:
            raise RuntimeError("SPNet online inference failed") from exc

    def dense_evidence_for(self, frame: FrameInputs) -> DepthEvidence:
        evidence = predict_spnet_dense_evidence(
            frame,
            self._predict,
            depth_scale_m=self._config.depth_scale_m,
            confidence=self._config.confidence,
        )
        self._record_frame(frame)
        return evidence

    def evidence_for(self, frame: FrameInputs) -> DepthEvidence:
        evidence = predict_spnet_evidence(
            frame,
            self._predict,
            depth_scale_m=self._config.depth_scale_m,
            confidence=self._config.confidence,
            sample_stride=self._config.sample_stride,
        )
        self._record_frame(frame)
        return evidence

    def _record_frame(self, frame: FrameInputs) -> None:
        self._identity = replace(
            self._identity,
            frame_grids=(*self._identity.frame_grids, SPNetFrameGrid(
                frame.index, frame.stem, derive_spnet_grid(frame.rgb.shape[:2]),
            )),
        )


def _positive_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def prepare_spnet_inputs(
    rgb: np.ndarray,
    raw: DepthEvidence,
    *,
    depth_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    color = np.asarray(rgb)
    if color.dtype != np.float32 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("SPNet RGB input must be float32 HxWx3")
    if not np.isfinite(color).all() or ((color < 0) | (color > 1)).any():
        raise ValueError("SPNet RGB input must be finite and within [0, 1]")
    if raw.source_type not in {SourceType.LIDAR_RAW, SourceType.LIDAR_SLAM_CENTER}:
        raise ValueError("SPNet input preparation requires profile center-anchor evidence")
    if raw.depth_m.shape != color.shape[:2]:
        raise ValueError("SPNet RGB and raw evidence must share image shape")
    scale = _positive_float(depth_scale_m, "SPNet depth_scale_m")
    clean_depth = np.where(raw.valid_mask, raw.depth_m, 0.0).astype(np.float32, copy=False)
    rgb_tensor = torch.from_numpy(color.copy()).permute(2, 0, 1).unsqueeze(0)
    depth_tensor = torch.from_numpy(clean_depth.copy()).unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.from_numpy(raw.valid_mask.copy()).unsqueeze(0).unsqueeze(0).to(torch.float32)
    grid = derive_spnet_grid(color.shape[:2])
    pad = (0, grid.pad_right, 0, grid.pad_bottom)
    rgb_tensor = F.pad(rgb_tensor, pad, mode="replicate")
    depth_tensor = F.pad(depth_tensor, pad, mode="constant", value=0.0)
    mask_tensor = F.pad(mask_tensor, pad, mode="constant", value=0.0)
    depth_tensor = depth_tensor / scale
    return rgb_tensor, depth_tensor, mask_tensor


def _validated_prediction(
    prediction: object, *, expected_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor):
        raise ValueError("SPNet prediction must be a torch.Tensor")
    expected_shape = (1, 1, *expected_size) if expected_size is not None else None
    if prediction.ndim != 4 or tuple(prediction.shape[:2]) != (1, 1):
        raise ValueError("SPNet prediction must have shape 1x1xHxW")
    if expected_shape is not None and tuple(prediction.shape) != expected_shape:
        raise ValueError(f"SPNet prediction shape must be {expected_shape}")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("SPNet prediction contains non-finite values")
    return prediction.detach().to(device="cpu", dtype=torch.float32)


def _spnet_prediction_quality_check(
    prediction: torch.Tensor,
    raw: DepthEvidence,
    *,
    depth_scale_m: float,
) -> float:
    scale = _positive_float(depth_scale_m, "SPNet depth_scale_m")
    if tuple(prediction.shape[2:]) != raw.depth_m.shape:
        raise ValueError("SPNet quality prediction must be cropped to the raw native grid")
    unanchored = prediction[0, 0].numpy() * scale
    if not raw.valid_mask.any():
        return 0.0
    return float((unanchored[raw.valid_mask] - raw.depth_m[raw.valid_mask]).mean())


def predict_spnet_dense_evidence(
    frame: FrameInputs,
    predictor: SPNetPredictor | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    depth_scale_m: float,
    confidence: float,
) -> DepthEvidence:
    """Return the unanchored native SPNet field for LiDAR scale-grid fitting."""
    by_source = {
        evidence.source_type: evidence
        for evidence in frame.growth.evidences
    }
    raw = by_source.get(SourceType.LIDAR_RAW)
    if raw is None:
        raw = by_source.get(SourceType.LIDAR_SLAM_CENTER)
    if raw is None:
        raise ValueError("SPNet prediction requires the profile center-anchor evidence")
    if (
        SourceType.LIDAR_RAW in by_source
        and SourceType.LIDAR_SLAM_CENTER in by_source
    ):
        raise ValueError("SPNet prediction cannot mix raw and SLAM center anchors")
    inputs = prepare_spnet_inputs(
        frame.rgb,
        raw,
        depth_scale_m=depth_scale_m,
    )
    grid = derive_spnet_grid(raw.depth_m.shape)
    prediction = _validated_prediction(
        predictor(*inputs),
        expected_size=grid.network_size,
    )
    native = prediction[:, :, :grid.native_height, :grid.native_width]
    quality_bias_m = _spnet_prediction_quality_check(
        native,
        raw,
        depth_scale_m=depth_scale_m,
    )
    if abs(quality_bias_m) >= 0.1:
        warnings.warn(
            f"SPNet prediction quality poor on frame {frame.stem}: "
            f"absolute mean depth bias {abs(quality_bias_m):.3f}m >= 0.1m "
            "on known LiDAR points",
            RuntimeWarning,
            stacklevel=2,
        )
    depth = (native[0, 0].numpy() * _positive_float(
        depth_scale_m,
        "SPNet depth_scale_m",
    )).astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("SPNet confidence must be numeric") from exc
    if not np.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
        raise ValueError("SPNet confidence must be within [0, 1]")
    confidence_map = np.where(valid, confidence_value, 0).astype(np.float32)
    return DepthEvidence(
        SourceType.SPNET_BLIND,
        depth,
        valid,
        confidence_map,
    )


def restore_spnet_prediction(
    prediction: torch.Tensor,
    center: DepthEvidence,
    *,
    depth_scale_m: float,
) -> np.ndarray:
    """Restore exact LiDAR anchors into a native-grid SPNet prediction."""
    if center.source_type not in {
        SourceType.LIDAR_RAW,
        SourceType.LIDAR_SLAM_CENTER,
    }:
        raise ValueError(
            "SPNet anchor restoration requires center-anchor evidence"
        )
    grid = derive_spnet_grid(center.depth_m.shape)
    prediction = _validated_prediction(
        prediction,
        expected_size=grid.network_size,
    )
    prediction = prediction[:, :, :grid.native_height, :grid.native_width]
    depth = (
        prediction[0, 0].numpy()
        * _positive_float(depth_scale_m, "SPNet depth_scale_m")
    ).astype(np.float32, copy=False)
    depth[center.valid_mask] = center.depth_m[center.valid_mask]
    if not np.isfinite(depth).all():
        raise FloatingPointError(
            "Restored SPNet prediction contains non-finite values"
        )
    return depth


def build_spnet_blind_evidence(
    restored_depth_m: np.ndarray,
    center: DepthEvidence,
    fused: DepthEvidence | None,
    *,
    confidence: float,
    sample_stride: int,
) -> DepthEvidence:
    """Keep only unsupported, regularly sampled SPNet growth candidates."""
    depth = np.asarray(restored_depth_m)
    if depth.dtype != np.float32 or depth.shape != center.depth_m.shape:
        raise ValueError(
            "Restored SPNet depth must be float32 and match center evidence"
        )
    pairs = {
        SourceType.LIDAR_RAW: SourceType.LIDAR_FUSED5,
        SourceType.LIDAR_SLAM_CENTER: SourceType.LIDAR_SLAM_FUSED5,
    }
    if center.source_type not in pairs:
        raise ValueError(
            "SPNet blind exclusion requires center-anchor evidence"
        )
    if fused is not None and (
        fused.source_type != pairs[center.source_type]
        or fused.depth_m.shape != depth.shape
    ):
        raise ValueError("SPNet fused evidence is incompatible")
    confidence_value = float(confidence)
    if (
        not np.isfinite(confidence_value)
        or not 0 <= confidence_value <= 1
    ):
        raise ValueError("SPNet confidence must be within [0, 1]")
    if type(sample_stride) is not int or sample_stride < 1:
        raise ValueError("SPNet sample_stride must be positive")
    supported = center.valid_mask.copy()
    if fused is not None:
        supported |= fused.valid_mask
    sample_grid = np.zeros(depth.shape, dtype=bool)
    sample_grid[0::sample_stride, 0::sample_stride] = True
    valid = np.isfinite(depth) & (depth > 0) & ~supported & sample_grid
    confidence_map = np.where(
        valid,
        confidence_value,
        0.0,
    ).astype(np.float32)
    return DepthEvidence(
        SourceType.SPNET_BLIND,
        depth,
        valid,
        confidence_map,
    )


def predict_spnet_evidence(
    frame: FrameInputs,
    predictor: (
        SPNetPredictor
        | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ),
    *,
    depth_scale_m: float,
    confidence: float,
    sample_stride: int,
) -> DepthEvidence:
    """Produce sparse blind-region evidence for structure growth."""
    by_source = {
        evidence.source_type: evidence
        for evidence in frame.growth.evidences
    }
    center = by_source.get(SourceType.LIDAR_RAW)
    fused_type = SourceType.LIDAR_FUSED5
    if center is None:
        center = by_source.get(SourceType.LIDAR_SLAM_CENTER)
        fused_type = SourceType.LIDAR_SLAM_FUSED5
    if center is None:
        raise ValueError("SPNet prediction requires center-anchor evidence")
    if (
        SourceType.LIDAR_RAW in by_source
        and SourceType.LIDAR_SLAM_CENTER in by_source
    ):
        raise ValueError("SPNet prediction cannot mix LiDAR source families")
    inputs = prepare_spnet_inputs(
        frame.rgb,
        center,
        depth_scale_m=depth_scale_m,
    )
    grid = derive_spnet_grid(center.depth_m.shape)
    prediction = _validated_prediction(
        predictor(*inputs),
        expected_size=grid.network_size,
    )
    native = prediction[:, :, :grid.native_height, :grid.native_width]
    quality_bias_m = _spnet_prediction_quality_check(
        native,
        center,
        depth_scale_m=depth_scale_m,
    )
    if abs(quality_bias_m) >= 0.1:
        warnings.warn(
            f"SPNet prediction quality poor on frame {frame.stem}: "
            f"absolute mean depth bias {abs(quality_bias_m):.3f}m >= 0.1m",
            RuntimeWarning,
            stacklevel=2,
        )
    restored = restore_spnet_prediction(
        prediction,
        center,
        depth_scale_m=depth_scale_m,
    )
    return build_spnet_blind_evidence(
        restored,
        center,
        by_source.get(fused_type),
        confidence=confidence,
        sample_stride=sample_stride,
    )
