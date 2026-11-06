"""Execute native CPU export/reload and true batched streaming generation checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.inference.smoke import run_inference_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    print(
        json.dumps(run_inference_smoke(args.directory, args.repository), sort_keys=True)
    )


if __name__ == "__main__":
    main()
