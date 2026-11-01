"""Compare all four execution policies for one small resolved CPU training job."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from qvelt.config.execution import ModelExecution
from qvelt.config.training import load_training_job
from qvelt.model.profiling import profile_cpu
from qvelt.utils.provenance import capture_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=Path("configs/train/smoke.yaml"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    base = load_training_job(args.job, args.overrides)
    _, code = capture_code(args.repository)
    profiles = []
    for backend in ("reference", "sdpa"):
        for recompute in (False, True):
            job = replace(
                base,
                training=replace(
                    base.training, execution=ModelExecution(backend, recompute)
                ),
            )
            profiles.append(profile_cpu(job, repeats=args.repeats))
    print(json.dumps({"code": asdict(code), "profiles": profiles}, sort_keys=True))


if __name__ == "__main__":
    main()
