from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from threading import Thread
from time import perf_counter
from typing import Any

from .hashing import sha256_file


EXECUTION_RECEIPT_SCHEMA_VERSION = "sage-execution-receipt-v1"
EXECUTION_CHILD_ENV = "SAGE_EXECUTION_CHILD"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def publish_json_atomic(
    destination: Path,
    payload: dict[str, object],
) -> None:
    """Durably replace one JSON document without exposing partial content."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        replaced = True
        try:
            directory_fd = os.open(
                destination.parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if replaced:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def formal_train_command(
    *,
    config_path: Path,
    output_dir: Path,
    device: str,
    input_format: str = "prepared-scene",
    data_root: Path | None = None,
    calibration: Path | None = None,
    write_through: Path | None = None,
) -> list[str]:
    if input_format not in {"prepared-scene", "odin-rosbag"}:
        raise ValueError("input_format must be prepared-scene or odin-rosbag")
    if input_format == "prepared-scene" and (
        calibration is not None or write_through is not None
    ):
        raise ValueError("Prepared Scene commands cannot declare ROSBAG-only paths")
    if input_format == "odin-rosbag" and (
        data_root is None or calibration is None
    ):
        raise ValueError("Odin ROSBAG commands require data and calibration paths")
    command = [
        sys.executable,
        "-m",
        "sage.mapping_worker",
        "train",
        "--config",
        str(Path(config_path).resolve()),
        "--output",
        str(Path(output_dir).resolve()),
        "--device",
        str(device),
        "--execution-child",
        "--input-format",
        input_format,
    ]
    if data_root is not None:
        command.extend(["--data-root", str(Path(data_root).resolve())])
    if calibration is not None:
        command.extend(["--calibration", str(Path(calibration).resolve())])
    if write_through is not None:
        command.extend(["--write-through", str(Path(write_through).resolve())])
    return command


def _rss_bytes(value: int) -> int:
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def _forward_child_output(
    stream: Any,
    destination: Any,
    captured: list[str],
) -> None:
    """Relay one child stream while retaining the receipt tail source."""
    try:
        for line in iter(stream.readline, ""):
            captured.append(line)
            destination.write(line)
            destination.flush()
    finally:
        stream.close()


def run_with_execution_receipt(
    command: list[str],
    *,
    output: Path,
    manifest: Path | None = None,
    formal_config: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    if not command:
        raise ValueError("Execution receipt requires a non-empty command")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite execution receipt: {output}")
    manifest = Path(manifest).resolve() if manifest is not None else None
    formal_config = Path(formal_config).resolve() if formal_config is not None else None
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    child_usage = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(Path(cwd).resolve()) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Child process output pipes are unavailable")
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stdout_thread = Thread(
            target=_forward_child_output,
            args=(process.stdout, sys.stdout, stdout_lines),
        )
        stderr_thread = Thread(
            target=_forward_child_output,
            args=(process.stderr, sys.stderr, stderr_lines),
        )
        stdout_thread.start()
        stderr_thread.start()
        if hasattr(os, "wait4"):
            _, status, child_usage = os.wait4(process.pid, 0)
            exit_code = int(os.waitstatus_to_exitcode(status))
        else:
            exit_code = int(process.wait())
        stdout_thread.join()
        stderr_thread.join()
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
    except BaseException as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        exit_code = 127
    wall_clock_seconds = perf_counter() - started
    rss_method = "os.wait4(child_pid)" if child_usage is not None else "unavailable-per-child-rss"
    rss_status = "measured" if child_usage is not None else "unmeasured"
    artifacts: dict[str, object] = {}
    if manifest is not None:
        artifacts["run_manifest"] = {
            "path": str(manifest),
            "exists": manifest.is_file(),
            "sha256": sha256_file(manifest) if manifest.is_file() else None,
        }
        checkpoint = manifest.with_name("checkpoint.pt")
        artifacts["checkpoint"] = {
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
            "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        }
    receipt = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "command": command,
        "cwd": str(Path(cwd).resolve()) if cwd is not None else str(Path.cwd().resolve()),
        "started_at_utc": started_at,
        "wall_clock_seconds": wall_clock_seconds,
        "exit_code": exit_code,
        "peak_process_rss_bytes": (
            _rss_bytes(int(child_usage.ru_maxrss)) if child_usage is not None else "unmeasured"
        ),
        "rss_status": rss_status,
        "rss_scope": "fresh child process",
        "rss_method": rss_method,
        "platform": platform.platform(),
        "formal_config": (
            {
                "path": str(formal_config),
                "exists": formal_config.is_file(),
                "sha256": sha256_file(formal_config) if formal_config.is_file() else None,
            }
            if formal_config is not None else None
        ),
        "artifacts": artifacts,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    publish_json_atomic(output, receipt)
    return receipt


def _valid_formal_train_command(
    command: object,
    *,
    config_path: Path,
    output_dir: Path,
) -> bool:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return False
    if len(command) < 13 or (len(command) - 11) % 2:
        return False
    tail = command[11:]
    names = tail[::2]
    values = tail[1::2]
    if len(set(names)) != len(names) or any(not name.startswith("--") for name in names):
        return False
    options = dict(zip(names, values))
    input_format = options.get("--input-format")
    data_root = Path(options["--data-root"]) if "--data-root" in options else None
    calibration = Path(options["--calibration"]) if "--calibration" in options else None
    write_through = Path(options["--write-through"]) if "--write-through" in options else None
    if set(options) - {
        "--input-format", "--data-root", "--calibration", "--write-through",
    }:
        return False
    try:
        expected = formal_train_command(
            config_path=config_path,
            output_dir=output_dir,
            device=command[9],
            input_format=str(input_format),
            data_root=data_root,
            calibration=calibration,
            write_through=write_through,
        )
    except ValueError:
        return False
    normalized = [str(Path(command[0]).resolve()), *command[1:]]
    expected[0] = str(Path(expected[0]).resolve())
    return bool(command[9]) and normalized == expected


def validate_execution_receipt(
    receipt_path: Path,
    *,
    manifest_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    receipt_path = Path(receipt_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    config_path = Path(config_path).resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read execution receipt: {receipt_path}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION
        or receipt.get("exit_code") != 0
    ):
        raise ValueError("Execution receipt does not describe a successful formal run")
    formal_config = receipt.get("formal_config")
    if (
        not isinstance(formal_config, dict)
        or formal_config.get("exists") is not True
        or Path(str(formal_config.get("path", ""))).resolve() != config_path
        or not config_path.is_file()
        or formal_config.get("sha256") != sha256_file(config_path)
    ):
        raise ValueError("Execution receipt does not bind the current formal config")
    command = receipt.get("command")
    if (
        not _valid_formal_train_command(
            command,
            config_path=config_path,
            output_dir=manifest_path.parent,
        )
        or Path(str(receipt.get("cwd", ""))).resolve() != REPOSITORY_ROOT
    ):
        raise ValueError("Execution receipt command is not the formal SAGE entrypoint")
    wall_clock = receipt.get("wall_clock_seconds")
    if (
        type(wall_clock) not in {int, float}
        or not math.isfinite(wall_clock)
        or wall_clock < 0
    ):
        raise ValueError("Execution receipt wall-clock evidence is invalid")
    rss_status = receipt.get("rss_status")
    peak_rss = receipt.get("peak_process_rss_bytes")
    if (
        rss_status == "measured"
        and (type(peak_rss) is not int or peak_rss <= 0)
        or rss_status == "unmeasured"
        and peak_rss != "unmeasured"
        or rss_status not in {"measured", "unmeasured"}
    ):
        raise ValueError("Execution receipt RSS evidence is invalid")
    artifacts = receipt.get("artifacts")
    expected = {
        "run_manifest": manifest_path,
        "checkpoint": manifest_path.with_name("checkpoint.pt"),
    }
    if not isinstance(artifacts, dict):
        raise ValueError("Execution receipt lacks artifact identities")
    for name, path in expected.items():
        artifact = artifacts.get(name)
        if (
            not isinstance(artifact, dict)
            or artifact.get("exists") is not True
            or Path(str(artifact.get("path", ""))).resolve() != path
            or not path.is_file()
            or artifact.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Execution receipt does not bind current {name}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Execution receipt manifest is unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("config_sha256") != formal_config["sha256"]
    ):
        raise ValueError("Run manifest does not bind the execution receipt config")
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "wall_clock_seconds": wall_clock,
        "peak_process_rss_bytes": peak_rss,
        "rss_status": rss_status,
        "config_sha256": formal_config["sha256"],
        "artifact_hashes": {
            name: artifacts[name]["sha256"] for name in expected
        },
    }
