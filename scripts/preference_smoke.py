"""Run original exact-answer SFT to DPO learning and strict interruption replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.posttrain.preference_smoke import run_preference_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    result = run_preference_smoke(args.directory, args.repository, steps=args.steps)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
