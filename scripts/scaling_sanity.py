"""Fit a descriptive four-size original-fixture curve; no large training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.train.diagnostics import scaling_sanity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            scaling_sanity(
                args.data, args.directory, args.repository, steps=args.steps
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
