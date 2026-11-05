"""Compare base/SFT/DPO greedy answers on the registered16-problem demo control."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from qvelt.config.training import parse_training_job
from qvelt.data.math import generate_problems
from qvelt.model.decoder import Decoder
from qvelt.posttrain.dpo import load_dpo
from qvelt.posttrain.preferences import family_split
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, hash_file
from qvelt.utils.checkpoints import latest_checkpoint, source_digest
from qvelt.utils.manifest_types import CheckpointInfo
from qvelt.utils.manifests import manifest_digest
from qvelt.utils.reproducibility import seed_everything


def evaluate(run: Path) -> dict[str, Any]:
    """Use exactly the same prefix,budget,verifier and problems for three models."""
    recipe = load_dpo(run / "dpo.yaml")
    torch.set_num_threads(1)
    seed_everything(recipe.job.experiment.runtime)
    token_path = run / "pairs/tokenizer/manifest.json"
    token_manifest = ArtifactStore(token_path.parent).read_manifest()
    tokenizer, _ = load_tokenizer_bundle(token_path, manifest_digest(token_manifest))
    # shortcut: fixed authored fixture,generalize only with licensed benchmark contracts.
    problems = [
        p
        for p in generate_problems(32, 20261105, "en")
        if family_split(p.family) == "validation"
    ]
    assert len(problems) == 16
    assert all(p.reward(p.answer) == 1 and p.reward("1+1") == 0 for p in problems)
    records: list[dict[str, Any]] = []
    for name, directory in (
        ("base", run / "sft/base"),
        ("sft", run / "sft/continuous"),
        ("dpo", run / "continuous"),
    ):
        checkpoint = latest_checkpoint(directory)
        store = ArtifactStore(checkpoint)
        manifest = store.read_manifest()
        if not isinstance(
            manifest.metadata, CheckpointInfo
        ) or manifest.metadata.tokenizer_sha256 != manifest_digest(token_manifest):
            raise ValueError("Generation checkpoint/tokenizer identity mismatch")
        with store.open_file("job.yaml") as stream:
            job = parse_training_job(stream.read(128 * 1024 + 1).decode())
        if job.experiment.model != recipe.job.experiment.model:
            raise ValueError("Generation model configurations disagree")
        if name == "sft" and manifest_digest(manifest) != recipe.reference_sha256:
            raise ValueError("SFT checkpoint is not the pinned DPO reference")
        model = Decoder(recipe.job.experiment.model).eval()
        model.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
        rows: list[dict[str, Any]] = []
        for problem in problems:
            tokens = [1, 4, *tokenizer.encode(problem.question), 6, 5]
            generated = []
            stop = None
            if len(tokens) + 16 > model.config.context:
                raise ValueError("Control budget exceeds model context")
            with torch.no_grad():
                for _ in range(16):
                    logits = model(torch.tensor(tokens).unsqueeze(0))[0, -1]
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Nonfinite generation logits")
                    token = int(logits.argmax())
                    tokens.append(token)
                    if token in (6, 2):
                        stop = token
                        break
                    generated.append(token)
            text = tokenizer.decode(generated)
            valid = bool(
                re.fullmatch(r"-?[0-9]{1,60}(?:/[1-9][0-9]{0,59})?", text.strip())
            )
            complete = stop == 6
            score = problem.reward(text) if complete else 0.0
            assert score == 0 or valid and complete
            rows.append(
                {
                    "id": problem.identifier,
                    "family": problem.family,
                    "answer": problem.answer,
                    "completion": text,
                    "new_tokens": len(generated) + (stop is not None),
                    "stop_token": stop,
                    "complete": complete,
                    "numeric_format": valid,
                    "reward": score,
                }
            )
        records.append(
            {
                "model": name,
                "checkpoint_sha256": manifest_digest(manifest),
                "weights_sha256": hash_file(checkpoint / "model.safetensors").sha256,
                "optimizer_step": manifest.metadata.optimizer_step,
                "correct": sum(row["reward"] == 1 for row in rows),
                "problems": len(rows),
                "complete": sum(row["complete"] for row in rows),
                "numeric_format": sum(row["numeric_format"] for row in rows),
                "truncated": sum(row["stop_token"] is None for row in rows),
                "rows": rows,
            }
        )
    return {
        "scope": "Original validation-only necessary model-promotion control; not public/final evaluation",
        "date": "2026-11-05",
        "source_sha256": source_digest(),
        "tokenizer_sha256": manifest_digest(token_manifest),
        "torch": str(torch.__version__),
        "device": "cpu",
        "precision": "fp32",
        "batch": 1,
        "max_new_tokens": 16,
        "sampling": "greedy full-prefix",
        "required_complete_stop": 6,
        "models": records,
        "necessary_gate_passed": any(
            row["correct"] > 0 for row in records if row["model"] in ("sft", "dpo")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="Completed preference_smoke directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON receipt path"
    )
    args = parser.parse_args()
    # Reserve before doing work; never overwrite an existing receipt.
    with args.output.open("x", encoding="utf-8") as stream:
        result = evaluate(args.run)
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                **result,
                "models": [
                    {k: v for k, v in row.items() if k != "rows"}
                    for row in result["models"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
