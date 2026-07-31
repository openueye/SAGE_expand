from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from ..foundation.hashing import sha256_file as _sha256_file


REGISTRY_SCHEMA_VERSION = "sage-model-registry-v1"
SPNET_MODEL_ID = "spnet-large-300"
ALEXNET_MODEL_ID = "alexnet-imagenet"
MODEL_ROOT_ENVIRONMENT_VARIABLE = "SAGE_MODEL_ROOT"


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    filename: str
    sha256: str
    purpose: str
    source: str
    license_status: str
    acquisition_policy: str


class ModelRegistry:
    def __init__(self, records: tuple[ModelRecord, ...]) -> None:
        if len({record.model_id for record in records}) != len(records):
            raise ValueError("Model registry contains duplicate model_id values")
        self._records = {record.model_id: record for record in records}

    @classmethod
    def load(cls, path: Path) -> "ModelRegistry":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read model registry: {path}") from exc
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "models"}:
            raise ValueError("Model registry must contain exactly schema_version and models")
        if payload["schema_version"] != REGISTRY_SCHEMA_VERSION or not isinstance(payload["models"], list):
            raise ValueError("Unsupported model registry schema")
        fields = set(ModelRecord.__dataclass_fields__)
        records = []
        for raw in payload["models"]:
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("Model record fields do not match the strict registry schema")
            if any(not isinstance(raw[name], str) or not raw[name] for name in fields):
                raise ValueError("Model record values must be non-empty strings")
            filename = Path(raw["filename"])
            if filename.is_absolute() or ".." in filename.parts:
                raise ValueError(f"Model filename must be relative and confined to the model root: {raw['model_id']}")
            record = ModelRecord(**raw)
            if len(record.sha256) != 64 or any(character not in "0123456789abcdef" for character in record.sha256):
                raise ValueError(f"Invalid SHA-256 for model {record.model_id}")
            if record.acquisition_policy != "user-source-required":
                raise ValueError("SAGE only supports the user-source-required acquisition policy")
            records.append(record)
        return cls(tuple(records))

    def require(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as exc:
            raise ValueError(f"Undeclared model_id: {model_id}") from exc

    def resolve(self, model_id: str, *, model_root: Path | None = None) -> Path:
        record = self.require(model_id)
        root = _model_root(model_root)
        path = _model_path(record, root)
        if not path.is_file():
            raise ValueError(
                f"Model {model_id} weights are unavailable at {path}; expected SHA-256 "
                f"{record.sha256}. Import them with tools/download_models.py "
                f"{model_id} --source /path/to/{record.filename}"
            )
        actual = sha256_file(path)
        if actual != record.sha256:
            raise ValueError(
                f"Model {model_id} weights SHA-256 mismatch at {path}: "
                f"expected {record.sha256}, got {actual}"
            )
        return path


def default_registry_path() -> Path:
    source_path = Path(__file__).resolve().parents[2] / "models" / "manifest.json"
    if source_path.is_file():
        return source_path
    installed_path = Path(sys.prefix) / "sage_gs" / "models" / "manifest.json"
    if not installed_path.is_file():
        raise ValueError("Installed SAGE distribution lacks models/manifest.json")
    return installed_path


def sha256_file(path: Path) -> str:
    try:
        return _sha256_file(path)
    except OSError as exc:
        raise ValueError(f"Cannot read model file: {path}") from exc


def default_model_root() -> Path:
    configured = os.environ.get(MODEL_ROOT_ENVIRONMENT_VARIABLE)
    if not configured:
        raise ValueError(
            f"{MODEL_ROOT_ENVIRONMENT_VARIABLE} must be set; implicit user cache paths are disabled"
        )
    return Path(configured).expanduser().resolve()


def _model_root(model_root: Path | None) -> Path:
    return Path(model_root).expanduser().resolve() if model_root is not None else default_model_root()


def _model_path(record: ModelRecord, model_root: Path) -> Path:
    path = (model_root / record.filename).resolve()
    if not path.is_relative_to(model_root):
        raise ValueError(f"Model filename escapes the model root: {record.model_id}")
    return path


def bootstrap_model(
    registry: ModelRegistry,
    model_id: str,
    source_path: Path,
    *,
    model_root: Path | None = None,
) -> Path:
    record = registry.require(model_id)
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Model source file does not exist: {source}")
    source_sha256 = sha256_file(source)
    if source_sha256 != record.sha256:
        raise ValueError(
            f"Model {model_id} source SHA-256 mismatch: expected {record.sha256}, got {source_sha256}"
        )

    root = _model_root(model_root)
    target = _model_path(record, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != record.sha256:
            raise ValueError(f"Refusing to overwrite existing model with a different SHA-256: {target}")
        return target
    if source == target:
        return target

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    if sha256_file(target) != record.sha256:
        raise ValueError(f"Imported model failed post-copy SHA-256 verification: {target}")
    return target
