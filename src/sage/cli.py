"""Public command-line entry point for the complete SAGE workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation.evaluation_run import run_evaluation
from .method_config import SageInput, SageMethodConfig
from .pipeline import preflight_training, run_training


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sage.yaml"


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--rosbag",
        type=Path,
        help="Odin ROSBAG directory",
    )
    inputs.add_argument(
        "--prepared-scene",
        type=Path,
        help="prepared scene directory",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="camera/LiDAR calibration required by --rosbag",
    )
    parser.add_argument(
        "--write-through",
        type=Path,
        help="optional Prepared Scene recording for --rosbag",
    )


def _source(args: argparse.Namespace, parser: argparse.ArgumentParser) -> SageInput:
    if args.rosbag is not None:
        if args.calibration is None:
            parser.error("--calibration is required with --rosbag")
        return SageInput(
            kind="rosbag",
            root=args.rosbag,
            calibration=args.calibration,
            write_through=args.write_through,
        )
    if args.calibration is not None or args.write_through is not None:
        parser.error(
            "--calibration and --write-through are valid only with --rosbag"
        )
    return SageInput(kind="prepared_scene", root=args.prepared_scene)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sage",
        description=(
            "Run complete SAGE structure, appearance, and evaluation"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and evaluate SAGE")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--output", required=True, type=Path)
    _add_input_arguments(train)
    train.add_argument("--device", default="cuda")
    train.add_argument("--preflight", action="store_true")

    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate a final SAGE checkpoint over all accepted frames",
    )
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument("--output", required=True, type=Path)
    _add_input_arguments(evaluate)
    evaluate.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    method = SageMethodConfig.load(args.config)
    source = _source(args, parser)
    if args.command == "train":
        if args.preflight:
            return preflight_training(
                method=method,
                source=source,
                output=args.output,
                device=args.device,
            )
        output = run_training(
            method=method,
            source=source,
            output=args.output,
            device=args.device,
        )
        print(f"SAGE complete: {output}", flush=True)
        return 0
    output = run_evaluation(
        method.resolve(source, args.output),
        args.checkpoint,
        args.output,
        device=args.device,
    )
    print(f"SAGE evaluation complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
