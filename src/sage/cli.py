"""Public command-line entry point for the complete SAGE workflow.

The input is defined entirely by the config's `input:` section. The CLI carries
run control only — no topics, no calibration, no adapter selection — so a run
cannot silently disagree with the configuration it records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation.evaluation_run import run_evaluation
from .method_config import SageMethodConfig
from .pipeline import preflight_training, run_training
from .prepare import prepare_scene


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "sage.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sage",
        description="Run complete SAGE structure, appearance, and evaluation",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and evaluate SAGE")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--device", default="cuda")
    train.add_argument("--preflight", action="store_true")

    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate a final SAGE checkpoint over all accepted frames",
    )
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--device", default="cuda")

    prepare = commands.add_parser(
        "prepare",
        help="export the configured input as a Prepared Scene (optional; training never needs it)",
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    method = SageMethodConfig.load(args.config)
    if args.command == "train":
        if args.preflight:
            return preflight_training(method=method, output=args.output, device=args.device)
        output = run_training(method=method, output=args.output, device=args.device)
        print(f"SAGE complete: {output}", flush=True)
        return 0
    if args.command == "prepare":
        scene = prepare_scene(method, args.output)
        print(f"SAGE prepared scene: {scene}", flush=True)
        return 0
    output = run_evaluation(
        method.resolve(args.output),
        args.checkpoint,
        args.output,
        device=args.device,
    )
    print(f"SAGE evaluation complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
