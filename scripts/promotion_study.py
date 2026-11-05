"""Run the preregistered original CPU matched-initialization comparison/LR ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.eval.study_smoke import run_study_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20261105, 20261106, 20261107]
    )
    args = parser.parse_args()
    result = run_study_smoke(args.directory, args.repository, seeds=tuple(args.seeds))
    print(
        json.dumps(
            {
                "study_sha256": result["study_sha256"],
                "source_sha256": result["result"]["source_sha256"],
                "seeds": result["result"]["initialization_seeds"],
                "comparisons": result["result"]["comparisons_to_sft"],
                "automatic_promotion": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
