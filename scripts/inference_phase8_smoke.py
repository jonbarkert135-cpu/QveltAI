"""Run Phase8 actual offline exports,quantization,needles and loopback HTTP acceptance."""

import argparse
import json
from pathlib import Path

from qvelt.inference.phase8_smoke import run_phase8_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run_phase8_smoke(args.directory, args.repository), sort_keys=True))


if __name__ == "__main__":
    main()
