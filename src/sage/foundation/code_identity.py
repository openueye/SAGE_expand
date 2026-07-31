"""Reproducible source-state identity for published run evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


_SOURCE_PATHS = (
    "src",
    "configs",
    "tools",
    "pyproject.toml",
)


def source_state_sha256(repository: Path | None = None) -> str:
    root = (repository or Path(__file__).resolve().parents[3]).resolve()
    files: list[Path] = []
    for relative in _SOURCE_PATHS:
        path = root / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate for candidate in path.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def repository_code_identity(repository: Path | None = None) -> dict[str, object]:
    root = (repository or Path(__file__).resolve().parents[3]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout)
    return {
        "repository": str(root),
        "commit": commit,
        "worktree_dirty": dirty,
        "source_state_sha256": source_state_sha256(root),
    }


__all__ = ["repository_code_identity", "source_state_sha256"]
