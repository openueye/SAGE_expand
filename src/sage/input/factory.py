"""Select the one input adapter a run uses, from the config `input:` section."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .adapter import InputAdapter
from .prepared_scene.adapter import ADAPTER_TYPE as PREPARED_SCENE_TYPE, PreparedSceneAdapter
from .rosbag.adapter import GenericRosbagAdapter
from .rosbag.online_adapter import OnlineRosbagAdapter
from .rosbag.spec import (
    ADAPTER_TYPE as ROSBAG_TYPE,
    EXECUTION_ONLINE_WINDOW_V2,
    RosbagInputSpec,
)


ADAPTER_TYPES = (ROSBAG_TYPE, PREPARED_SCENE_TYPE)


def create_input_adapter(payload: Mapping[str, Any], *, base_dir: Path) -> InputAdapter:
    """Build the adapter named by `input.type`; paths resolve against the config."""
    if not isinstance(payload, Mapping) or "type" not in payload:
        raise ValueError("SAGE input section must declare a type")
    adapter_type = str(payload["type"])
    if adapter_type == ROSBAG_TYPE:
        spec = RosbagInputSpec.from_payload(payload, base_dir=base_dir)
        if spec.execution == EXECUTION_ONLINE_WINDOW_V2:
            return OnlineRosbagAdapter(spec)
        return GenericRosbagAdapter(spec)
    if adapter_type == PREPARED_SCENE_TYPE:
        expected = {"type", "scene"}
        if set(payload) != expected:
            raise ValueError("input (prepared_scene) must define exactly: type, scene")
        scene = Path(str(payload["scene"])).expanduser()
        return PreparedSceneAdapter(scene if scene.is_absolute() else base_dir / scene)
    raise ValueError(f"Unsupported input.type {adapter_type!r}; supported: {ADAPTER_TYPES}")


__all__ = ["ADAPTER_TYPES", "create_input_adapter"]
