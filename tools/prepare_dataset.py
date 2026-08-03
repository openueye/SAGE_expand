"""Materialize a complete Odin1 ROSBAG or validate a SAGE Prepared Scene."""

from __future__ import annotations

import argparse
from pathlib import Path

from sage.data.rosbag.streaming import OdinBagFixedLagResultStream
from sage.data.scene import PreparedScene


MANIFEST_NAME = "sage_prepared_scene_manifest.json"


def validate_prepared_dataset(root: Path) -> Path:
    root = Path(root).resolve()
    PreparedScene.from_root(root)._validate_contract()
    return root


def materialize_rosbag_dataset(
    rosbag_dir: Path,
    output_dir: Path,
    *,
    calibration_path: Path,
    preparation_profile: str = "odin1-lidar-world-native-v1",
) -> Path:
    """Stream one complete finite bag into an atomically published Prepared Scene."""
    output = Path(output_dir).resolve()
    source = OdinBagFixedLagResultStream(
        Path(rosbag_dir).resolve(),
        preparation_profile=preparation_profile,
        calibration_path=Path(calibration_path).resolve(),
        write_through_dir=output,
    )
    try:
        for _ in source.frames():
            pass
        receipt = source.prepare_close()
        if receipt.get("bag_exhausted") is not True or receipt.get("truncated") is not False:
            raise RuntimeError("Prepared Scene production did not exhaust the ROSBAG")
        source.commit()
        return output / MANIFEST_NAME
    except BaseException as exc:
        source.abort(exc)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Prepared Scene root to validate")
    inputs.add_argument("--rosbag", type=Path, help="finite Odin1 ROSBAG directory to materialize")
    parser.add_argument("--calibration", type=Path, help="camera/LiDAR cam_in_ex.txt")
    parser.add_argument("--output", type=Path, help="new complete Prepared Scene destination")
    parser.add_argument(
        "--profile", default="odin1-lidar-world-native-v1",
        help="preparation_profile name (default: odin1-lidar-world-native-v1)",
    )
    args = parser.parse_args(argv)
    if args.input is not None:
        if args.calibration is not None or args.output is not None:
            parser.error("--calibration and --output are only valid with --rosbag")
        print(validate_prepared_dataset(args.input))
        return 0
    if args.calibration is None or args.output is None:
        parser.error("--rosbag requires --calibration and --output")
    print(materialize_rosbag_dataset(
        args.rosbag,
        args.output,
        calibration_path=args.calibration,
        preparation_profile=args.profile,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
