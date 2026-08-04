"""`sage prepare`: the one explicit path that persists an input.

Training never writes a Prepared Scene as a side effect. This command resolves
the configured input exactly as training would and hands the same canonical
frames to the writer, so the exported scene replays bit for bit.
"""

from __future__ import annotations

from pathlib import Path

from .input.prepared_scene import PreparedSceneWriter
from .method_config import SageMethodConfig


def prepare_scene(method: SageMethodConfig, output: Path) -> Path:
    config = method.resolve(Path(output).resolve())
    resolved = config.input.create_adapter().preflight()
    print("\n".join(resolved.summary_lines()), flush=True)
    if resolved.contract.canonical.frame_count is None:
        raise ValueError(
            "sage prepare does not support online-window-v2 because a Prepared Scene "
            "requires a frame count before it starts writing. Use input.execution: "
            "batch-v1 for an explicit offline export."
        )
    writer = PreparedSceneWriter(
        Path(output).resolve(),
        canonical=resolved.contract.canonical,
        provenance={
            "created_from": resolved.contract.adapter_type,
            "source_identity": resolved.source_identity,
            "adapter_provenance_identity": resolved.contract.adapter_provenance_identity,
            "adapter_details": dict(resolved.contract.adapter_details),
            "preflight_report": resolved.report.payload(),
        },
    )
    return writer.write(resolved.frames())


__all__ = ["prepare_scene"]
