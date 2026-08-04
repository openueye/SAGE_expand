"""Internal commands and typed entry points for SAGE structure mapping."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from ..core_input import CoreObservationAssembler
from ..data.providers.spnet import OnlineSPNetProvider, SPNetEvidenceProvider
from ..engine.metrics import ImageMetricEvaluator
from ..engine.model import TrainableGaussians
from ..engine.rendering import CachedRenderer, capture_renderer_identity
from ..execution import (
    EXECUTION_CHILD_ENV,
    formal_train_command,
    run_with_execution_receipt,
)
from ..foundation.config import SageConfig
from ..input.prefetch import BoundedResultStream
from ..input.frame import source_frame_label
from ..refinement.stage2_cache import (
    Stage2InputCacheWriter,
    default_stage2_cache_path,
    remove_stage2_cache,
)
from ..foundation.hashing import sha256_file
from ..foundation.identity_schema import DependencyIdentity
from .mapper import MappingEngine, should_invoke_spnet
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
    expected_total: int | None,
    mapping_started_at: float,
):
    for completed, frame in enumerate(frames, start=1):
        if (
            completed == 1
            or completed % _MAPPING_PROGRESS_EVERY == 0
            or (expected_total is not None and completed == expected_total)
        ):
            elapsed = perf_counter() - mapping_started_at
            print(
                f"[{_format_elapsed(elapsed)}] SAGE mapping frame {completed}; "
                f"{source_frame_label(frame.canonical)}",
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


def _resolved_config(config_path: Path, *, output: Path | None) -> SageConfig:
    from ..method_config import SageMethodConfig

    destination = Path(output).resolve() if output is not None else Path.cwd()
    return SageMethodConfig.load(config_path).resolve(destination)


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
        config.input.require_clean_worktree
        if require_clean_code is None
        else require_clean_code
    )
    if type(require_clean) is not bool:
        raise ValueError("require_clean_code must be a boolean when provided")

    resolved = config.input.create_adapter().preflight()
    assembler = CoreObservationAssembler(resolved.contract.canonical.sources)
    renderer_identity = capture_renderer_identity()
    metric_evaluator = ImageMetricEvaluator(device, model_root=config.model_root)
    spnet_provider = _build_spnet_provider(config, device=device)
    dependencies = _dependency_identity(
        renderer_identity,
        metric_evaluator.identity,
        spnet_provider,
        actual_spnet_invocations=0,
    )
    expected_frames = resolved.contract.canonical.frame_count
    stage2_cache_path = default_stage2_cache_path(config.output_dir)
    stage2_cache = Stage2InputCacheWriter(stage2_cache_path)
    stage2_cache_published = False
    mapping_started_at = perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    # Frame assembly is CPU/IO work (PNG or CDR decode, projection, NumPy
    # merge) that ran serially inside the optimization loop; one bounded
    # producer keeps it off the critical path without unbounded buffering.
    stream = BoundedResultStream(
        lambda: assembler.frames(resolved.frames()),
        identity={"adapter": resolved.contract.adapter_type},
        queue_capacity=config.input.prefetch_depth,
    )
    runtime_metrics: dict[str, object] = {}

    def cache_mapping_frames(frames):
        for frame in frames:
            if should_invoke_spnet(frame.index, config.mapping):
                stage2_cache.write(frame)
            yield frame

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
                cache_mapping_frames(stream.frames()),
                expected_total=expected_frames,
                mapping_started_at=mapping_started_at,
            )
        )
        stream.close()
        torch.cuda.synchronize(device)
        if result.spnet_actual_invocations != result.spnet_expected_invocations:
            raise RuntimeError("SPNet invocation count changed during SAGE training")
        input_identity = RunInputIdentity.capture(
            config,
            dependencies=_dependency_identity(
                renderer_identity,
                metric_evaluator.identity,
                spnet_provider,
                actual_spnet_invocations=result.spnet_actual_invocations,
            ),
            resolved=resolved,
            require_clean=require_clean,
        )
        stage2_cache.finalize(input_identity.input_identities)
        stage2_cache_published = True
        runtime_metrics["stage2_input_cache"] = {
            "path": str(stage2_cache_path),
            "mapping_frame_count": stage2_cache.mapping_frame_count,
            "lifecycle": "transient-until-stage2-publication",
        }
        runtime_metrics["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        artifacts = write_run_artifacts(
            config,
            result,
            input_identity=input_identity,
            resolved=resolved,
            runtime_metrics=runtime_metrics,
        )
        return artifacts.run_dir
    except BaseException:
        stream.abort("SAGE mapping failed")
        stage2_cache.abort()
        if stage2_cache_published:
            remove_stage2_cache(stage2_cache_path)
        raise


def _input_identity_payload(config: SageConfig) -> dict[str, object]:
    resolved = config.input.create_adapter().preflight()
    # Explicit training preflight is allowed to consume the configured input.
    # In online-window-v2 this is the only way to settle its EOF identity; the
    # normal mapping path instead settles it while it trains.
    if resolved.contract.canonical.frame_count is None:
        for _ in resolved.frames():
            pass
    return {
        "contract": resolved.contract.payload(),
        "preflight": resolved.report.payload(),
        "identities": resolved.identities.payload(),
    }


def _verify(config_path: Path, *, require_models: bool) -> int:
    from ..verify import execution_preflight, verify

    config = _resolved_config(config_path, output=None)
    report = verify(
        require_models=require_models,
        model_root=config.model_root,
        require_clean_worktree=config.input.require_clean_worktree,
    )
    report["config_sha256"] = sha256_file(config.config_path)
    report["config"] = config.manifest_dict()
    report["input"] = _input_identity_payload(config)
    if require_models:
        report["execution_preflight"] = execution_preflight(config)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _execution_receipt_path(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}.execution.json")


def run_formal_training(*, config_path: Path, output: Path, device: str) -> int:
    """Run structure mapping through the fresh-process evidence boundary."""
    config = _resolved_config(config_path, output=output)
    receipt_path = _execution_receipt_path(config.output_dir)
    command = formal_train_command(
        config_path=config_path, output_dir=config.output_dir, device=device,
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


def run_training_preflight(*, config_path: Path, output: Path, device: str) -> int:
    """Validate the exact structure-training invocation without writing output."""
    config = _resolved_config(config_path, output=output)
    from ..verify import execution_preflight, verify

    if config.output_dir.exists():
        raise ValueError(f"Refusing an existing training output path: {config.output_dir}")
    receipt_path = _execution_receipt_path(config.output_dir)
    if receipt_path.exists():
        raise ValueError(f"Refusing an existing training execution receipt: {receipt_path}")
    report = verify(
        require_models=True,
        model_root=config.model_root,
        require_clean_worktree=config.input.require_clean_worktree,
    )
    report["config_sha256"] = sha256_file(config.config_path)
    report["config"] = config.manifest_dict()
    report["input"] = _input_identity_payload(config)
    report["execution_preflight"] = execution_preflight(config, device=device)
    report["formal_train_command"] = formal_train_command(
        config_path=config_path, output_dir=config.output_dir, device=device,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sage.mapping.mapping_worker",
        description="Internal SAGE structure-mapping implementation",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="validate runtime, models, renderer, and configuration")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    verify.add_argument("--require-models", action="store_true")
    train_parser = commands.add_parser("train", help="train the SAGE structure map from scratch")
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--output", required=True, type=Path)
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
        return _verify(args.config, require_models=args.require_models)
    if args.preflight:
        if args.execution_child:
            raise ValueError("training preflight cannot be an execution child")
        return run_training_preflight(
            config_path=args.config, output=args.output, device=args.device,
        )
    if args.execution_child:
        if os.environ.get(EXECUTION_CHILD_ENV) != "1":
            raise ValueError("internal fresh-process boundary cannot be invoked directly")
        config = _resolved_config(args.config, output=args.output)
        print(f"SAGE complete: {train(config, device=args.device)}", flush=True)
        return 0
    return run_formal_training(
        config_path=args.config, output=args.output, device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
