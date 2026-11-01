"""Prepare original BPE data, optimize a tiny decoder and verify atomic training replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.train.pretraining_smoke import run_pretraining_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            run_pretraining_smoke(args.directory, args.repository, steps=args.steps),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
