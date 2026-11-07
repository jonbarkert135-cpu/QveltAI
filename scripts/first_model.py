"""Execute the documented first-hour RFC tiny pipeline with real resume/SFT/eval/export/HTTP."""

import argparse
import json
from pathlib import Path

from qvelt.train.first_model import first_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    print(
        json.dumps(
            first_model(args.directory, args.repository, steps=args.steps),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
