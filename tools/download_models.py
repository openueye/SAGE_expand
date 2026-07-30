from __future__ import annotations

import argparse
from pathlib import Path

from sage.model_registry import (
    ModelRegistry,
    bootstrap_model,
    default_registry_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import one user-supplied SAGE model after SHA-256 verification."
    )
    parser.add_argument("model_id", help="Logical model ID declared in models/manifest.json")
    parser.add_argument("--source", required=True, type=Path, help="User-supplied source weights file")
    parser.add_argument("--model-root", type=Path, help="Offline model cache root")
    parser.add_argument("--registry", type=Path, help="Model registry JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = ModelRegistry.load(args.registry or default_registry_path())
    destination = bootstrap_model(
        registry,
        args.model_id,
        args.source,
        model_root=args.model_root,
    )
    print(f"Imported {args.model_id} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
