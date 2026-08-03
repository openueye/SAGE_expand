"""Internal commands and typed entry points for SAGE structure mapping."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from ..data.frame_source import frame_source_for_config
from ..data.providers.spnet import OnlineSPNetProvider, SPNetEvidenceProvider
from ..engine.metrics import ImageMetricEvaluator
from ..engine.model import TrainableGaussians
from ..engine.rendering import CachedRenderer, capture_renderer_identity
from ..execution import (
    EXECUTION_CHILD_ENV,
    formal_train_command,
    run_with_execution_receipt,
)
from ..foundation.config import (
    ALL_ACCEPTED_FRAME_LIMIT,
    SageConfig,
)
from ..foundation.hashing import sha256_file
from ..foundation.identity_schema import DependencyIdentity
from .mapper import MappingEngine
from .mapping_artifacts import write_run_artifacts
from .run_identity import RunInputIdentity


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "sage.yaml"
_MAPPING_PROGRESS_EVERY = 50


def _format_elapsed(elapsed: float) -> str:
    ms = int(elapsed * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _report_frames(
    frames,
    *,
    expected_total: int,
    mapping_started_at: float,
):
    for completed, frame in enumerate(frames, start=1):
        if (
            completed == 1
            or completed % _MAPPING_PROGRESS_EVERY == 0
            or (expected_total > 0 and completed == expected_total)
        ):
            elapsed = perf_counter() - mapping_started_at
            rosbag_frame: str = (
                f"{int(frame.stem):06d}"
                if frame.stem.isdigit()
                else frame.stem
            )
            print(
                f"[{_format_elapsed(elapsed)}] SAGE mapping frame {completed}; "
                f"rosbag frame {rosbag_frame}",
                flush=True,
            )
        yield frame


def _build_spnet_provider(config: SageConfig, *, device: str) -> SPNetEvidenceProvider:
    return OnlineSPNetProvider(
        config.growth_sources.spnet,
        device=device,
        model_root=config.model_root,
    )


def _dependency_identity(
    renderer_identity: dict[str, object],
    metric_identity: object,
    spnet_provider: SPNetEvidenceProvider,
    *,
    actual_spnet_invocations: int,
) -> DependencyIdentity:
    return DependencyIdentity.for_renderer_with_dependencies(
        renderer_identity,
        spnet_provider.identity,
        actual_invocations=actual_spnet_invocations,
        metric_identity=metric_identity,
    )


def _resolved_config(
    config_path: Path,
    *,
    output: Path | None,
    input_format: str,
    data_root: Path | None,
    calibration: Path | None,
    write_through: Path | None,
    preparation_profile: str = "odin1-lidar-world-native-v1",
) -> SageConfig:
    from ..method_config import SageInput, SageMethodConfig

    if data_root is None:
        raise ValueError("SAGE input root is required")
    if input_format == "odin-rosbag":
        source = SageInput(
            kind="rosbag",
            root=data_root,
            calibration=calibration,
            write_through=write_through,
            preparation_profile=preparation_profile,
        )
    elif input_format == "prepared-scene":
        source = SageInput(kind="prepared_scene", root=data_root)
    else:
        raise ValueError(
            "SAGE input format must be odin-rosbag or prepared-scene"
        )
    destination = Path(output).resolve() if output is not None else Path.cwd()
    return SageMethodConfig.load(config_path).resolve(
        source,
        destination,
    )


def train(
    config: SageConfig,
    *,
    device: str,
    require_clean_code: bool | None = None,
) -> Path:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("SAGE training requires an available CUDA device")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    require_clean = (
        config.scene.require_clean_worktree
        if require_clean_code is None
        else require_clean_code
    )
    if type(require_clean) is not bool:
        raise ValueError("require_clean_code must be a boolean when provided")

    renderer_identity = capture_renderer_identity()
    metric_evaluator = ImageMetricEvaluator(device, model_root=config.model_root)
    spnet_provider = _build_spnet_provider(config, device=device)
    dependencies = _dependency_identity(
        renderer_identity,
        metric_evaluator.identity,
        spnet_provider,
        actual_spnet_invocations=0,
    )
    source = frame_source_for_config(config.scene, frame_limit=ALL_ACCEPTED_FRAME_LIMIT)
    input_identity = RunInputIdentity.capture(
        config,
        dependencies=dependencies,
        require_clean=require_clean,
        frame_source=source,
    )
    mapping_started_at = perf_counter()
    source.start_identity()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        result = MappingEngine(
            config.mapping, config.pruning, config.growth,
            device=device,
            renderer=CachedRenderer(),
            spnet_provider=spnet_provider, metric_evaluator=metric_evaluator,
            gaussian_initialization=config.gaussian_initialization, loss_policy=config.loss,
            seed=config.seed,
        ).run(
            _report_frames(
                source.frames(),
                expected_total=ALL_ACCEPTED_FRAME_LIMIT,
                mapping_started_at=mapping_started_at,
            )
        )
        torch.cuda.synchronize(device)
        if result.spnet_actual_invocations != result.spnet_expected_invocations:
            raise RuntimeError("SPNet invocation count changed during SAGE training")
        input_identity = replace(
            input_identity,
            dependencies=_dependency_identity(
                renderer_identity,
                metric_evaluator.identity,
                spnet_provider,
                actual_spnet_invocations=result.spnet_actual_invocations,
            ),
        )
        if hasattr(source, "set_runtime_metrics"):
            source.set_runtime_metrics({"peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))})
        artifacts = write_run_artifacts(
            config, result, input_identity=input_identity, completion_receipt=source.prepare_close(),
            commit_source=source.commit, rollback_source=source.rollback_commit,
        )
        return artifacts.run_dir
    except BaseException as exc:
        source.abort(exc)
        raise


def _frame_source_identity(config: SageConfig) -> dict[str, object]:
    source = frame_source_for_config(config.scene, frame_limit=ALL_ACCEPTED_FRAME_LIMIT)
    try:
        return source.start_identity()
    finally:
        source.abort("identity capture complete")


def _verify(
    config_path: Path,
    *,
    input_format: str,
    data_root: Path | None,
    calibration: Path | None,
    write_through: Path | None,
    require_models: bool,
    preparation_profile: str = "odin1-lidar-world-native-v1",
) -> int:
    from ..verify import execution_preflight, verify

    config = _resolved_config(
        config_path,
        output=None,
        input_format=input_format,
        data_root=data_root,
        calibration=calibration,
        write_through=write_through,
        preparation_profile=preparation_profile,
    )
    report = verify(
        require_models=require_models,
        model_root=config.model_root,
        require_clean_worktree=config.scene.require_clean_worktree,
    )
    report["config_sha256"] = sha256_file(config.config_path)
    report["config"] = config.manifest_dict()
    report["dataset_identity"] = _frame_source_identity(config)
    if require_models:
        report["execution_preflight"] = execution_preflight(config)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _execution_receipt_path(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.execution.json")


def run_formal_training(
    *,
    config_path: Path,
    output: Path,
    input_format: str,
    data_root: Path | None,
    calibration: Path | None,
    write_through: Path | None,
    device: str,
    preparation_profile: str = "odin1-lidar-world-native-v1",
) -> int:
    """Run structure mapping through the fresh-process evidence boundary."""
    config = _resolved_config(
        config_path,
        output=output,
        input_format=input_format,
        data_root=data_root,
        calibration=calibration,
        write_through=write_through,
        preparation_profile=preparation_profile,
    )
    receipt_path = _execution_receipt_path(config.output_dir)
    command = formal_train_command(
        config_path=config_path,
        output_dir=config.output_dir,
        device=device,
        input_format=input_format,
        data_root=data_root,
        calibration=calibration,
        write_through=write_through,
        preparation_profile=preparation_profile,
    )
    receipt = run_with_execution_receipt(
        command,
        output=receipt_path,
        manifest=config.output_dir / "run_manifest.json",
        formal_config=config.config_path,
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, EXECUTION_CHILD_ENV: "1"},
    )
    print(
        f"SAGE mapping execution receipt: {receipt_path} "
        f"(exit_code={receipt['exit_code']})",
        flush=True,
    )
    return int(receipt["exit_code"])


def run_training_preflight(
    *,
    config_path: Path,
    output: Path,
    input_format: str,
    data_root: Path | None,
    calibration: Path | None,
    write_through: Path | None,
    device: str,
    preparation_profile: str = "odin1-lidar-world-native-v1",
) -> int:
    """Validate the exact structure-training invocation without writing output."""
    config = _resolved_config(
        config_path,
        output=output,
        input_format=input_format,
        data_root=data_root,
        calibration=calibration,
        write_through=write_through,
        preparation_profile=preparation_profile,
    )
    return _train_preflight(
        config,
        config_path=config_path,
        input_format=input_format,
        data_root=data_root,
        calibration=calibration,
        write_through=write_through,
        device=device,
        preparation_profile=preparation_profile,
    )


def _train_preflight(
    config: SageConfig,
    *,
    config_path: Path,
    input_format: str,
    data_root: Path | None,
    calibration: Path | None,
    write_through: Path | None,
    device: str,
    preparation_profile: str = "odin1-lidar-world-native-v1",
) -> int:
    from ..verify import execution_preflight, verify

    if config.output_dir.exists():
        raise ValueError(f"Refusing an existing training output path: {config.output_dir}")
    receipt_path = _execution_receipt_path(config.output_dir)
    if receipt_path.exists():
        raise ValueError(f"Refusing an existing training execution receipt: {receipt_path}")
    report = verify(
        require_models=True,
        model_root=config.model_root,
        require_clean_worktree=config.scene.require_clean_worktree,
    )
    report["config_sha256"] = sha256_file(config.config_path)
    report["config"] = config.manifest_dict()
    report["dataset_identity"] = _frame_source_identity(config)
    report["execution_preflight"] = execution_preflight(config, device=device)
    report["formal_train_command"] = formal_train_command(
        config_path=config_path,
        output_dir=config.output_dir,
        device=device,
        input_format=input_format,
        data_root=data_root,
        calibration=calibration,
        write_through=write_through,
        preparation_profile=preparation_profile,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-format",
        choices=("prepared-scene", "odin-rosbag"),
        default="prepared-scene",
        help="select the explicit SAGE input adapter",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Prepared Scene root or finite Odin1 ROSBAG directory",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Odin1 cam_in_ex.txt; valid only with --input-format odin-rosbag",
    )
    parser.add_argument(
        "--write-through",
        type=Path,
        help="optional Prepared Scene output for odin-rosbag; omit for stream-only training",
    )
    parser.add_argument(
        "--preparation-profile",
        default="odin1-lidar-world-native-v1",
        help="ROSBAG preparation_profile to use with --input-format odin-rosbag",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sage.mapping.mapping_worker",
        description="Internal SAGE structure-mapping implementation",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="validate runtime, models, renderer, and configuration")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    _add_input_arguments(verify)
    verify.add_argument("--require-models", action="store_true")
    train_parser = commands.add_parser(
        "train",
        help="train the SAGE structure map from scratch",
    )
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--output", required=True, type=Path)
    _add_input_arguments(train_parser)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate the final training inputs and CUDA execution path without training",
    )
    train_parser.add_argument("--execution-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        return _verify(
            args.config,
            input_format=args.input_format,
            data_root=args.data_root,
            calibration=args.calibration,
            write_through=args.write_through,
            require_models=args.require_models,
            preparation_profile=args.preparation_profile,
        )
    config = _resolved_config(
        args.config,
        output=args.output,
        input_format=args.input_format,
        data_root=args.data_root,
        calibration=args.calibration,
        write_through=args.write_through,
        preparation_profile=args.preparation_profile,
    )
    if args.preflight:
        if args.execution_child:
            raise ValueError("training preflight cannot be an execution child")
        return run_training_preflight(
            config_path=args.config,
            output=args.output,
            input_format=args.input_format,
            data_root=args.data_root,
            calibration=args.calibration,
            write_through=args.write_through,
            device=args.device,
            preparation_profile=args.preparation_profile,
        )
    if args.execution_child:
        if os.environ.get(EXECUTION_CHILD_ENV) != "1":
            raise ValueError("internal fresh-process boundary cannot be invoked directly")
        print(f"SAGE complete: {train(config, device=args.device)}", flush=True)
        return 0
    return run_formal_training(
        config_path=args.config,
        output=args.output,
        input_format=args.input_format,
        data_root=args.data_root,
        calibration=args.calibration,
        write_through=args.write_through,
        device=args.device,
        preparation_profile=args.preparation_profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
