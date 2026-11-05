"""Print a run summary with heuristic red flags; optionally pack logs to send."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.train.report import archive_run, summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path, help="write logs (no weights) .tar.gz")
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.directory), indent=2, sort_keys=True))
    if args.archive:
        names = archive_run(args.directory, args.archive)
        print(f"wrote {args.archive} with {len(names)} files (no tensors)")


if __name__ == "__main__":
    main()
