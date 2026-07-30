from __future__ import annotations

import sys

from sage.execution import run_with_execution_receipt


def test_execution_receipt_relays_child_streams_and_keeps_tails(
    tmp_path,
    capsys,
) -> None:
    receipt = run_with_execution_receipt(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('mapping-progress', flush=True); "
                "print('mapping-warning', file=sys.stderr, flush=True)"
            ),
        ],
        output=tmp_path / "receipt.json",
    )

    captured = capsys.readouterr()
    assert "mapping-progress" in captured.out
    assert "mapping-warning" in captured.err
    assert "mapping-progress" in receipt["stdout_tail"]
    assert "mapping-warning" in receipt["stderr_tail"]
