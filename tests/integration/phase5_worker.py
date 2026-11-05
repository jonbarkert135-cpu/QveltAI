"""Two-rank recoverable nonfinite window: complete reducers and continue training."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor

from qvelt.model.decoder import Decoder
from qvelt.train.pretraining import train_pretraining
from qvelt.train.pretraining_smoke import demo_recipe

if __name__ == "__main__":
    path, output = map(Path, sys.argv[1:])
    root = Path(__file__).resolve().parents[2]
    recipe = demo_recipe(path, root, steps=4, accumulation=2, world_size=2)
    recipe = replace(
        recipe,
        nonfinite_policy="rollback",
        validation_batches=1,
        validation_every=4,
        job=replace(
            recipe.job,
            training=replace(recipe.job.training, strategy="ddp", checkpoint_every=1),
        ),
    )
    original = Decoder.forward
    counter = {"value": 0}

    def forward(self: Decoder, *args: Any, **kwargs: Any) -> Tensor:
        logits = original(self, *args, **kwargs)
        if self.training and torch.is_grad_enabled():
            counter["value"] += 1
            if counter["value"] == 4 and os.environ["RANK"] == "0":
                return logits * float("nan")
        return logits

    with patch.object(Decoder, "forward", forward):
        result = train_pretraining(recipe, output, root)
    assert result["rollbacks"] == 1 and result["optimizer_updates"] == 3
    if result["rank"] == 0:
        (output / "recovery-result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n"
        )
