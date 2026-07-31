from __future__ import annotations

import torch
from torch import nn

from .artifact_versions import (
    APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
    CHECKPOINT_VERSION,
)
from .config import GaussianInitializationConfig
from .contracts import FrameInputs, GaussianAppendBatch
from .geometry import backproject_depth
from .source_policy import SOURCE_POLICY_VERSION


class TrainableGaussians(nn.Module):
    """Trainable Gaussian rows plus stable provenance buffers."""

    PARAMETER_NAMES = ("means3d", "colors", "opacity_logits", "log_scales", "rotations")

    def __init__(
        self,
        means3d: torch.Tensor,
        colors: torch.Tensor,
        opacity_logits: torch.Tensor,
        log_scales: torch.Tensor,
        rotations: torch.Tensor,
        gaussian_ids: torch.Tensor,
        created_at: torch.Tensor,
        source_types: torch.Tensor,
        source_confidences: torch.Tensor,
    ) -> None:
        super().__init__()
        log_scales = self._normalize_log_scales(log_scales)
        self.means3d = nn.Parameter(means3d)
        self.colors = nn.Parameter(colors)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.log_scales = nn.Parameter(log_scales)
        self.rotations = nn.Parameter(rotations)
        self.register_buffer("gaussian_ids", gaussian_ids.to(dtype=torch.int64, device=means3d.device))
        self.register_buffer("created_at", created_at.to(dtype=torch.int64, device=means3d.device))
        self.register_buffer("source_types", source_types.to(dtype=torch.uint8, device=means3d.device))
        self.register_buffer("source_confidences", source_confidences.to(dtype=torch.float32, device=means3d.device))
        self._validate()

    @property
    def count(self) -> int:
        return int(self.means3d.shape[0])

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    @classmethod
    def from_frame(
        cls,
        frame: FrameInputs,
        *,
        device: str | torch.device,
        gaussian_initialization: GaussianInitializationConfig,
    ) -> "TrainableGaussians":
        depth = torch.as_tensor(frame.mapping.depth_m, dtype=torch.float32)
        points, pixels, z = backproject_depth(depth, frame.intrinsics, frame.pose)
        if points.shape[0] == 0:
            raise ValueError(f"Initial frame has no valid center/fused depth: {frame.stem}")
        rows = pixels[:, 0].cpu().numpy()
        columns = pixels[:, 1].cpu().numpy()
        colors = torch.as_tensor(frame.rgb[rows, columns], dtype=torch.float32)
        source_types = torch.as_tensor(frame.mapping.source_types[rows, columns], dtype=torch.uint8)
        source_confidences = torch.as_tensor(frame.mapping.confidences[rows, columns], dtype=torch.float32)
        focal = (frame.intrinsics.fx + frame.intrinsics.fy) * 0.5
        base_scales = (z / focal).clamp_min(gaussian_initialization.scale_clamp_min)
        anisotropy = torch.as_tensor(
            gaussian_initialization.initial_scale_anisotropy,
            dtype=base_scales.dtype,
            device=base_scales.device,
        )
        log_scales = torch.log((base_scales.unsqueeze(1) * anisotropy).clamp_min(gaussian_initialization.scale_clamp_min))
        count = points.shape[0]
        target = torch.device(device)
        opacity_logit = gaussian_initialization.opacity_logit
        return cls(
            points.to(target), colors.to(target), torch.full((count, 1), opacity_logit, dtype=torch.float32, device=target),
            log_scales.to(target), torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=target).repeat(count, 1),
            torch.arange(count, dtype=torch.int64, device=target),
            torch.full((count,), frame.index, dtype=torch.int64, device=target),
            source_types.to(target), source_confidences.to(target),
        )

    @classmethod
    def from_checkpoint(cls, payload: dict[str, object], *, device: str | torch.device) -> "TrainableGaussians":
        if payload.get("checkpoint_version") not in {
            CHECKPOINT_VERSION,
            APPEARANCE_REFINEMENT_CHECKPOINT_VERSION,
        }:
            raise ValueError("Unsupported SAGE checkpoint version")
        if payload.get("source_policy_version") != SOURCE_POLICY_VERSION:
            raise ValueError("Unsupported SAGE source policy version")
        required = ("means3d", "colors", "opacity_logits", "log_scales", "rotations", "gaussian_ids", "created_at", "source_types", "source_confidences")
        if any(name not in payload or not isinstance(payload[name], torch.Tensor) for name in required):
            raise ValueError("Checkpoint is missing a Gaussian tensor")
        target = torch.device(device)
        return cls(
            *(payload[name].to(target).clone() for name in required[:5]),
            *(payload[name].to(target).clone() for name in required[5:]),
        )

    def parameter_groups(self, learning_rates: dict[str, float]) -> list[dict[str, object]]:
        if set(learning_rates) != set(self.PARAMETER_NAMES):
            raise ValueError(f"Learning rates must define exactly: {', '.join(self.PARAMETER_NAMES)}")
        return [{"name": name, "params": [getattr(self, name)], "lr": learning_rates[name]} for name in self.PARAMETER_NAMES]

    def append_batch(
        self,
        batch: GaussianAppendBatch,
        *,
        created_at: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        self._validate_append_batch(batch)
        count = int(batch.means3d.shape[0])
        if count == 0:
            return 0
        old_count = self.count
        old_parameters = {name: getattr(self, name) for name in self.PARAMETER_NAMES}
        prepared_states = (
            self._prepare_optimizer_states(
                optimizer,
                old_parameters,
                old_count=old_count,
                append_count=count,
            )
            if optimizer is not None else None
        )
        first_id = int(self.gaussian_ids[-1]) + 1 if self.count else 0
        new_ids = torch.arange(first_id, first_id + count, dtype=torch.int64, device=self.means3d.device)
        created = torch.full((count,), created_at, dtype=torch.int64, device=self.means3d.device)
        self._replace(
            **{name: torch.cat((getattr(self, name).detach(), getattr(batch, name))) for name in self.PARAMETER_NAMES},
            gaussian_ids=torch.cat((self.gaussian_ids, new_ids)),
            created_at=torch.cat((self.created_at, created)),
            source_types=torch.cat((self.source_types, batch.source_types)),
            source_confidences=torch.cat((self.source_confidences, batch.source_confidences)),
        )
        if optimizer is not None and prepared_states is not None:
            self._replace_optimizer_parameters(optimizer, old_parameters, prepared_states)
        return count

    def prune_rows(self, keep_mask: torch.Tensor, *, optimizer: torch.optim.Optimizer | None = None) -> int:
        if keep_mask.device != self.means3d.device or keep_mask.dtype != torch.bool or keep_mask.ndim != 1 or keep_mask.shape[0] != self.count:
            raise ValueError("prune keep_mask must be a bool vector on the model device")
        if self.count == 0:
            raise ValueError("Cannot prune an empty Gaussian map")
        keep = keep_mask.clone()
        if not bool(keep.any()):
            keep[torch.argmax(self.opacities.detach().reshape(-1))] = True
        removed = self.count - int(keep.sum())
        if removed == 0:
            return 0
        old_count = self.count
        old_parameters = {name: getattr(self, name) for name in self.PARAMETER_NAMES}
        prepared_states = (
            self._prepare_optimizer_states(
                optimizer,
                old_parameters,
                old_count=old_count,
                keep=keep,
            )
            if optimizer is not None else None
        )
        self._replace(
            **{name: getattr(self, name).detach()[keep] for name in self.PARAMETER_NAMES},
            gaussian_ids=self.gaussian_ids[keep], created_at=self.created_at[keep],
            source_types=self.source_types[keep], source_confidences=self.source_confidences[keep],
        )
        if optimizer is not None and prepared_states is not None:
            self._replace_optimizer_parameters(optimizer, old_parameters, prepared_states)
        self._validate()
        return removed

    @staticmethod
    def _validate_optimizer(
        optimizer: torch.optim.Optimizer,
        old_parameters: dict[str, nn.Parameter],
    ) -> None:
        if len(optimizer.param_groups) != len(old_parameters):
            raise ValueError("Optimizer parameter groups must name each Gaussian parameter")
        names: set[str] = set()
        for group in optimizer.param_groups:
            name = group.get("name")
            params = group.get("params")
            if (not isinstance(name, str) or name not in old_parameters or name in names
                    or not isinstance(params, list) or len(params) != 1
                    or params[0] is not old_parameters[name]):
                raise ValueError("Optimizer parameter groups must name each Gaussian parameter")
            names.add(name)
        if names != set(old_parameters):
            raise ValueError("Optimizer parameter groups must name each Gaussian parameter")

    @classmethod
    def _prepare_optimizer_states(
        cls,
        optimizer: torch.optim.Optimizer,
        old_parameters: dict[str, nn.Parameter],
        *,
        old_count: int,
        append_count: int = 0,
        keep: torch.Tensor | None = None,
    ) -> dict[str, dict[str, object] | None]:
        cls._validate_optimizer(optimizer, old_parameters)
        if append_count < 0 or (keep is not None and keep.shape != (old_count,)):
            raise ValueError("Invalid optimizer state migration shape")
        prepared: dict[str, dict[str, object] | None] = {}
        for name, old_parameter in old_parameters.items():
            state = optimizer.state.get(old_parameter)
            if state is None:
                prepared[name] = None
                continue
            if not isinstance(state, dict):
                raise ValueError("Optimizer state must be a mapping")
            migrated: dict[str, object] = {}
            for key, value in state.items():
                if not torch.is_tensor(value):
                    migrated[key] = value
                    continue
                if value.ndim == 0:
                    migrated[key] = value
                    continue
                if value.shape[0] != old_count:
                    raise ValueError(
                        f"Optimizer state {name}.{key} must be scalar or row-shaped"
                    )
                if keep is not None:
                    migrated[key] = value[keep]
                else:
                    zeros = torch.zeros(
                        (append_count, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    migrated[key] = torch.cat((value, zeros), dim=0)
            prepared[name] = migrated
        return prepared

    def _replace_optimizer_parameters(
        self,
        optimizer: torch.optim.Optimizer,
        old_parameters: dict[str, nn.Parameter],
        prepared_states: dict[str, dict[str, object] | None],
    ) -> None:
        for group in optimizer.param_groups:
            name = group["name"]
            old_parameter = old_parameters[name]
            new_parameter = getattr(self, name)
            optimizer.state.pop(old_parameter, None)
            group["params"] = [new_parameter]
            state = prepared_states[name]
            if state is not None:
                optimizer.state[new_parameter] = state

    def _validate_append_batch(self, batch: GaussianAppendBatch) -> None:
        tensors = (batch.means3d, batch.colors, batch.opacity_logits, batch.log_scales, batch.rotations, batch.source_types, batch.source_confidences)
        if any(tensor.device != self.means3d.device for tensor in tensors):
            raise ValueError("Append tensors must be on the model device")
        count = batch.means3d.shape[0]
        if (
            batch.means3d.ndim != 2 or batch.means3d.shape[1] != 3
            or batch.colors.shape != (count, 3) or batch.opacity_logits.shape != (count, 1)
            or batch.log_scales.ndim != 2 or batch.log_scales.shape[0] != count
            or batch.log_scales.shape[1] != 3 or batch.rotations.shape != (count, 4)
            or batch.source_types.shape != (count,) or batch.source_confidences.shape != (count,)
        ):
            raise ValueError("Append tensors have incompatible row shapes")
        if batch.source_types.dtype != torch.uint8 or batch.source_confidences.dtype != torch.float32:
            raise ValueError("Append provenance tensors have invalid dtypes")
        if not bool(torch.isin(batch.source_types, torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint8, device=batch.source_types.device)).all()):
            raise ValueError("Append source_types contains an unsupported source ID")
        combined = torch.cat((self.source_types, batch.source_types))
        if self._mixes_lidar_families(combined):
            raise ValueError("Append source_types would mix raw and SLAM LiDAR source families")
        if not bool(torch.isfinite(batch.source_confidences).all()) or bool(((batch.source_confidences < 0) | (batch.source_confidences > 1)).any()):
            raise ValueError("Append source_confidences must be finite and within [0, 1]")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors[:5]):
            raise ValueError("Append parameter tensors must be finite")

    def _replace(self, **values: torch.Tensor) -> None:
        for name in self.PARAMETER_NAMES:
            setattr(self, name, nn.Parameter(values[name]))
        for name in ("gaussian_ids", "created_at", "source_types", "source_confidences"):
            setattr(self, name, values[name])
        self._validate()

    def _validate(self) -> None:
        count = self.count
        rows = tuple(getattr(self, name) for name in self.PARAMETER_NAMES) + (self.gaussian_ids, self.created_at, self.source_types, self.source_confidences)
        if any(tensor.ndim == 0 or tensor.shape[0] != count for tensor in rows):
            raise ValueError("Gaussian parameters and provenance buffers must stay row aligned")
        if any(tensor.device != self.means3d.device for tensor in rows):
            raise ValueError("Gaussian parameters and provenance buffers must share a device")
        if self.gaussian_ids.dtype != torch.int64 or self.created_at.dtype != torch.int64 or self.source_types.dtype != torch.uint8 or self.source_confidences.dtype != torch.float32:
            raise ValueError("Gaussian provenance buffers have invalid dtypes")
        if count > 1 and not bool((self.gaussian_ids[1:] > self.gaussian_ids[:-1]).all()):
            raise ValueError("Gaussian IDs must be strictly increasing")
        if not bool(torch.isin(self.source_types, torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint8, device=self.source_types.device)).all()):
            raise ValueError("Gaussian source_types contains an unsupported source ID")
        if self._mixes_lidar_families(self.source_types):
            raise ValueError("Gaussian source_types mix raw and SLAM LiDAR source families")
        if not bool(torch.isfinite(self.source_confidences).all()) or bool(((self.source_confidences < 0) | (self.source_confidences > 1)).any()):
            raise ValueError("Gaussian source_confidences must be finite and within [0, 1]")
        if self.log_scales.ndim != 2 or self.log_scales.shape[1] != 3:
            raise ValueError("Gaussian log_scales must be Nx3")
        if not all(bool(torch.isfinite(parameter).all()) for parameter in self.parameters()):
            raise ValueError("Gaussian parameters must be finite")

    @staticmethod
    def _mixes_lidar_families(source_types: torch.Tensor) -> bool:
        raw = torch.isin(source_types, torch.tensor([0, 1], dtype=torch.uint8, device=source_types.device)).any()
        slam = torch.isin(source_types, torch.tensor([3, 4], dtype=torch.uint8, device=source_types.device)).any()
        return bool(raw and slam)

    @staticmethod
    def _normalize_log_scales(log_scales: torch.Tensor) -> torch.Tensor:
        if log_scales.ndim != 2:
            raise ValueError("log_scales must be 2-D")
        columns = log_scales.shape[1]
        if columns == 1:
            return log_scales.repeat(1, 3)
        if columns != 3:
            raise ValueError("log_scales must have either 1 or 3 columns")
        return log_scales
