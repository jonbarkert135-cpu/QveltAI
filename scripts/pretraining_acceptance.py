"""Full RFC tiny: original BPE data,200 updates and actual process SIGKILL replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qvelt.train.diagnostics import (
    acceptance_recipe,
    create_acceptance_data,
    kill_resume_check,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=False)
    path = create_acceptance_data(args.directory / "data", args.repository)
    result = kill_resume_check(
        acceptance_recipe(path, args.repository),
        args.directory / "training",
        args.repository,
        kill_after=89,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "loss_curve"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
