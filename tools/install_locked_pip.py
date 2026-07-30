"""Install conda-lock's pip artifacts into the target Conda environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def locked_pip_requirements(lock_path: Path, *, platform: str = "linux-64") -> tuple[str, ...]:
    try:
        payload = yaml.safe_load(Path(lock_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read Conda lock: {lock_path}") from exc
    packages = payload.get("package") if isinstance(payload, dict) else None
    if not isinstance(packages, list):
        raise ValueError("Conda lock has no package list")
    selected: list[tuple[str, str]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Conda lock package entry must be an object")
        if package.get("manager") != "pip" or package.get("platform") != platform:
            continue
        name = package.get("name")
        url = package.get("url")
        digest = package.get("hash", {}).get("sha256")
        if not isinstance(name, str) or not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Locked pip package must have a name and HTTPS URL")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Locked pip package lacks a SHA-256 identity: {name}")
        selected.append((name, f"{url}#sha256={digest}"))
    if not selected:
        raise ValueError(f"Conda lock contains no pip artifacts for {platform}")
    return tuple(requirement for _, requirement in sorted(selected))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="sage")
    parser.add_argument("--lock", type=Path, default=ROOT / "conda-lock.yml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    requirements = locked_pip_requirements(args.lock)
    commands = [
        [
            "conda",
            "env",
            "config",
            "vars",
            "set",
            "-n",
            args.environment,
            "PYTHONNOUSERSITE=1",
        ],
        [
            "conda",
            "run",
            "-n",
            args.environment,
            "python",
            "-m",
            "pip",
            "install",
            "--no-user",
            "--no-deps",
            *requirements,
        ],
        [
            "conda",
            "run",
            "-n",
            args.environment,
            "python",
            "-m",
            "pip",
            "install",
            "--no-user",
            "--no-deps",
            "--editable",
            str(ROOT),
        ],
    ]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
