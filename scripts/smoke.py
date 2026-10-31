"""Thin CLI for the real synthetic-training/replay acceptance check."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from qvelt.train.smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    directory = args.directory
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="qvelt-smoke-"))
    else:
        directory.mkdir(parents=False, exist_ok=False)
    result = run_smoke(directory, args.repository.resolve())
    print(json.dumps({"directory": str(directory), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
