"""Everything a mapping run must be able to prove about its inputs.

Input identity comes from the resolved adapter, not from paths: the canonical
sequence and contract identities decide checkpoint compatibility, while source
and adapter provenance are recorded for audit only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import subprocess

from ..foundation.config import SageConfig
from ..foundation.hashing import sha256_file
from ..foundation.identity_schema import DependencyIdentity
from ..foundation.source_policy import SOURCE_POLICY_VERSION
from ..input.adapter import ResolvedInput
from ..input.identity import InputIdentities


def _producer_code_identity() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"], check=True,
            capture_output=True, text=True,
        ).stdout)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Cannot capture SAGE producer identity") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Invalid SAGE Git commit identity")
    return commit, dirty


def _environment_lock_identity() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    locks = {name: repository / name for name in ("environment.yml", "conda-lock.yml")}
    if any(not path.is_file() for path in locks.values()):
        raise ValueError("SAGE environment lock files are unavailable")
    return {name: sha256_file(path) for name, path in locks.items()}


@dataclass(frozen=True)
class RunInputIdentity:
    config_sha256: str
    input_identities: InputIdentities
    input_contract: dict[str, object]
    preflight_report: dict[str, object]
    source_policy_version: str
    producer_code_commit: str
    producer_code_worktree_dirty: bool
    environment_locks: dict[str, str]
    dependencies: DependencyIdentity

    @classmethod
    def capture(
        cls,
        config: SageConfig,
        *,
        dependencies: DependencyIdentity,
        resolved: ResolvedInput,
        require_clean: bool = False,
    ) -> "RunInputIdentity":
        commit, dirty = _producer_code_identity()
        if require_clean and dirty:
            raise ValueError("SAGE producer worktree must be clean before mapping")
        return cls(
            config_sha256=sha256_file(config.config_path),
            input_identities=resolved.identities,
            input_contract=resolved.contract.payload(),
            preflight_report=resolved.report.payload(),
            source_policy_version=SOURCE_POLICY_VERSION,
            producer_code_commit=commit,
            producer_code_worktree_dirty=dirty,
            environment_locks=_environment_lock_identity(),
            dependencies=dependencies,
        )

    def validate_unchanged(self, config: SageConfig) -> None:
        try:
            if sha256_file(config.config_path) != self.config_sha256:
                raise ValueError("Configuration changed")
            commit, dirty = _producer_code_identity()
            if (commit, dirty) != (self.producer_code_commit, self.producer_code_worktree_dirty):
                raise ValueError("SAGE producer code identity changed")
            if _environment_lock_identity() != self.environment_locks:
                raise ValueError("SAGE environment lock identity changed")
        except ValueError as exc:
            raise ValueError("Run inputs changed during mapping") from exc

    def producer_code_payload(self) -> dict[str, object]:
        return {
            "repository": "sage", "commit": self.producer_code_commit,
            "worktree_dirty": self.producer_code_worktree_dirty,
        }

    def identity_snapshot(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "input": self.input_identities.payload(),
            "input_contract": deepcopy(self.input_contract),
            "source_policy_version": self.source_policy_version,
            "producer_code": self.producer_code_payload(),
            "environment_locks": dict(self.environment_locks),
            "dependencies": self.dependencies.payload(),
        }


__all__ = ["RunInputIdentity"]
