"""Run the offline mixed Phase7 suite; optionally compare the installed standard harness."""

import argparse
import json
from pathlib import Path

from qvelt.eval.suite_smoke import run_suite_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--with-harness", action="store_true")
    parser.add_argument("--external-sources", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_suite_smoke(
                args.directory,
                args.repository,
                with_harness=args.with_harness,
                external_sources=args.external_sources,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
