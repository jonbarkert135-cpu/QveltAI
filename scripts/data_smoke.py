"""Run original offline corpus curation and exact prefetched batch replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.data.smoke import run_data_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--documents", type=int, default=96)
    args = parser.parse_args()
    print(
        json.dumps(
            run_data_smoke(args.directory, args.repository, args.documents),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
