"""Capture and re-validation of a mapping run's input identity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import subprocess

from ..data.frame_source import FrameSource, frame_source_for_config
from ..data.scene import PreparedScene, scene_content_sha256, sha256_file
from ..foundation.config import ALL_ACCEPTED_FRAME_LIMIT, SageConfig
from ..foundation.identity_schema import (
    DependencyIdentity,
    _canonical_identity_sha256,
    normalize_dataset_identity,
)
from ..foundation.source_policy import SOURCE_POLICY_VERSION


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


def _stream_input_hashes(config: SageConfig) -> dict[str, str]:
    scene = config.scene
    if scene.rosbag_dir is None or scene.calibration_path is None:
        raise ValueError("Streaming scene is missing bag or calibration path")
    files = {
        "metadata.yaml": scene.rosbag_dir / "metadata.yaml",
        **{f"rosbag/{path.name}": path for path in sorted(scene.rosbag_dir.glob("*.db3"))},
        "cam_in_ex.txt": scene.calibration_path,
    }
    return {label: sha256_file(path) for label, path in sorted(files.items())}


@dataclass(frozen=True)
class RunInputIdentity:
    config_sha256: str
    prepared_manifest_sha256: str
    source_manifest_sha256: str
    transform_contract_sha256: str
    scene_content_sha256: str
    source_mode: str
    source_policy_version: str
    producer_code_commit: str
    producer_code_worktree_dirty: bool
    environment_locks: dict[str, str]
    dependencies: DependencyIdentity
    frame_source_identity: dict[str, object]

    @classmethod
    def capture(
        cls, config: SageConfig, *, dependencies: DependencyIdentity,
        require_clean: bool = False, frame_source: FrameSource | None = None,
    ) -> "RunInputIdentity":
        scene = PreparedScene(config.scene) if config.scene.input_adapter == "prepared-scene" else None
        owns_source = frame_source is None
        source_adapter = frame_source or frame_source_for_config(
            config.scene, frame_limit=ALL_ACCEPTED_FRAME_LIMIT,
        )
        try:
            frame_source_identity = source_adapter.start_identity()
        finally:
            if owns_source:
                source_adapter.abort("identity capture does not consume frames")
        prepared = scene._validate_contract() if config.scene.input_adapter == "prepared-scene" else None
        commit, dirty = _producer_code_identity()
        if require_clean and dirty:
            raise ValueError("SAGE producer worktree must be clean before mapping")
        normalized_identity = normalize_dataset_identity(frame_source_identity)
        if prepared is not None:
            source = prepared["source"]
            assert isinstance(source, dict)
            source_manifest_sha256 = sha256_file(scene.prepared_manifest_path)
            transform_sha256 = str(normalized_identity["transform_contract_sha256"])
            source_mode = str(source["mode"])
            source_policy_version = SOURCE_POLICY_VERSION
        else:
            source_manifest_sha256 = _canonical_identity_sha256(
                frame_source_identity["input_files_sha256"]
            )
            prepared_manifest_sha256 = str(normalized_identity["prepared_manifest_sha256"])
            transform_sha256 = str(normalized_identity["transform_contract_sha256"])
            source_mode = str(normalized_identity["source_mode"])
            source_policy_version = str(normalized_identity["source_policy_version"])
        return cls(
            config_sha256=sha256_file(config.config_path),
            prepared_manifest_sha256=prepared_manifest_sha256 if prepared is None else sha256_file(scene.prepared_manifest_path),
            source_manifest_sha256=source_manifest_sha256,
            transform_contract_sha256=transform_sha256,
            scene_content_sha256=(
                str(normalized_identity["content_sha256"])
                if prepared is None
                else scene_content_sha256(scene.input_files(limit=ALL_ACCEPTED_FRAME_LIMIT))
            ),
            source_mode=source_mode,
            source_policy_version=source_policy_version,
            producer_code_commit=commit,
            producer_code_worktree_dirty=dirty,
            environment_locks=_environment_lock_identity(),
            dependencies=dependencies,
            frame_source_identity=frame_source_identity,
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
            if config.scene.input_adapter == "prepared-scene":
                scene = PreparedScene(config.scene)
                manifest = scene._validate_contract()
                current_manifest_hash = sha256_file(scene.prepared_manifest_path)
                current_content_hash = scene_content_sha256(
                    scene.input_files(limit=ALL_ACCEPTED_FRAME_LIMIT)
                )
                if current_manifest_hash != self.prepared_manifest_sha256 or current_content_hash != self.scene_content_sha256:
                    raise ValueError("Prepared Scene content changed")
                if str(manifest.get("source", {}).get("mode", manifest.get("source_mode"))) != self.source_mode:
                    raise ValueError("Prepared Scene source mode changed")
            else:
                identity = self.frame_source_identity
                if _stream_input_hashes(config) != identity.get("input_files_sha256"):
                    raise ValueError("Streaming input files changed")
                if identity.get("preparation_profile") != config.scene.preparation_profile:
                    raise ValueError("Streaming preparation profile changed")
                source = identity.get("source")
                if not isinstance(source, dict) or source.get("mode") != config.scene.source_mode:
                    raise ValueError("Streaming source mode changed")
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
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "transform_contract_sha256": self.transform_contract_sha256,
            "scene_content_sha256": self.scene_content_sha256,
            "source_mode": self.source_mode,
            "source_policy_version": self.source_policy_version,
            "producer_code": self.producer_code_payload(),
            "environment_locks": dict(self.environment_locks),
            "dependencies": self.dependencies.payload(),
            "frame_source": deepcopy(self.frame_source_identity),
        }


__all__ = ["RunInputIdentity"]
