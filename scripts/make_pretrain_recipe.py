"""Write a runnable pretraining recipe from a preset and an actual dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from qvelt.config.pretraining import dump_pretraining_job
from qvelt.train.recipes import build_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", type=Path, default=Path("configs/pretrain/small.yaml")
    )
    parser.add_argument("--data", type=Path, required=True, help="prepared manifest")
    parser.add_argument("--world-size", type=int, required=True, help="total GPUs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    recipe = build_recipe(
        args.preset, args.data, world_size=args.world_size, overrides=args.overrides
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(dump_pretraining_job(recipe))
    print(
        f"wrote {args.output}: steps={recipe.job.training.steps} "
        f"accumulation={recipe.accumulation_steps} "
        f"strategy={recipe.job.training.strategy} data={recipe.data_sha256[:12]}"
    )


if __name__ == "__main__":
    main()
