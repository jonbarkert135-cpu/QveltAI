"""Expand a sweep YAML into validated `qvelt pretrain` runs; optionally execute."""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict
from pathlib import Path

from qvelt.train.sweep import execute, expand, load_sweep, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="run every point")
    action.add_argument("--summarize", action="store_true", help="rank results")
    args = parser.parse_args()
    spec = load_sweep(args.config)
    runs = expand(spec)
    output = Path(spec.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sweep-plan.json").write_text(
        json.dumps([asdict(run) for run in runs], indent=2) + "\n"
    )
    results = execute(spec, runs) if args.execute else []
    if args.execute:
        print(json.dumps(results, indent=2))
    if args.execute or args.summarize:
        rows = summarize(runs)
        (output / "sweep-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows, indent=2))
    else:
        for run in runs:
            print(shlex.join(run.command))
    if any(result["returncode"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
