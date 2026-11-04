"""Run the original exact-math SFT optimization and interruption/replay check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.posttrain.smoke import run_sft_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    print(
        json.dumps(
            run_sft_smoke(args.directory, args.repository, steps=args.steps),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
