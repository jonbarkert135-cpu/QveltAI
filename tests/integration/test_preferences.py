"""Licensed original pairs,independent frame scoring,frozen reference and replay."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from qvelt.cli import main
from qvelt.config import ConfigError
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.records import write_raw_bundle
from qvelt.model.decoder import Decoder
from qvelt.posttrain.dpo import dump_dpo, load_dpo, pair_logps, parse_dpo, train_dpo
from qvelt.posttrain.preference_smoke import math_preferences, run_preference_smoke
from qvelt.posttrain.preferences import (
    EncodedPair,
    PreferencePair,
    encode_pair,
    load_preferences,
    prepare_preferences,
)
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore, hash_file
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def preference_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("preferences") / "smoke"
    run_preference_smoke(root, ROOT, steps=12)
    return root


def test_actual_optimization_exact_curve_and_split_contract(
    preference_run: Path,
) -> None:
    result = json.loads((preference_run / "result.json").read_text())
    assert all(
        result[k]
        for k in (
            "weights_bit_exact",
            "optimizer_bit_exact",
            "rng_bit_exact",
            "consumer_state_exact",
            "all_losses_exact",
            "reference_frozen",
        )
    )
    assert result["final_train"]["dpo_loss"] < result["initial_train"]["dpo_loss"]
    assert result["initial_train"]["ties"] == 1
    recipe = load_dpo(preference_run / "dpo.yaml")
    _, _, pairs = load_preferences(
        Path(recipe.data_manifest),
        recipe.data_sha256,
        recipe.job.experiment.model.vocab,
        recipe.job.training.sequence_length,
    )
    families = {
        name: {p.family for p in pairs if p.split == name}
        for name in ("train", "validation", "test")
    }
    assert families["train"] and families["validation"]
    assert not (
        families["train"] & families["validation"]
        or families["train"] & families["test"]
        or families["validation"] & families["test"]
    )
    assert parse_dpo(dump_dpo(recipe)) == recipe
    assert sum(result["split_pairs"].values()) == 32


def test_batched_scores_match_naive_unpadded_response_only(
    preference_run: Path,
) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    _, _, pairs = load_preferences(
        Path(recipe.data_manifest),
        recipe.data_sha256,
        recipe.job.experiment.model.vocab,
        recipe.job.training.sequence_length,
    )
    selected = [
        min(pairs, key=lambda p: len(p.chosen)),
        max(pairs, key=lambda p: len(p.chosen)),
    ]
    torch.manual_seed(187)
    model = Decoder(recipe.job.experiment.model)
    batched, _ = pair_logps(model, selected)
    for i, pair in enumerate(selected):
        for j, frame in enumerate((pair.chosen, pair.rejected)):
            inputs = torch.tensor(frame[:-1]).unsqueeze(0)
            with torch.no_grad():
                logits = model(inputs)[0]
            scores = torch.log_softmax(logits, dim=-1)[
                torch.arange(len(frame) - 1), torch.tensor(frame[1:])
            ]
            expected = scores[frame.index(5) : -1].sum()
            torch.testing.assert_close(batched[i, j], expected, atol=2e-5, rtol=1e-6)


def test_reference_parameters_stay_unchanged_and_receive_no_gradients(
    preference_run: Path, tmp_path: Path
) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    recipe = replace(
        recipe,
        job=replace(
            recipe.job,
            training=replace(
                recipe.job.training,
                steps=2,
                checkpoint_every=1,
                warmup_steps=1,
                decay_steps=1,
            ),
        ),
    )
    snapshots: dict[int, tuple[Decoder, dict[str, torch.Tensor]]] = {}
    original = pair_logps

    def observe(model: Decoder, pairs: list[EncodedPair]) -> tuple[torch.Tensor, int]:
        if (
            not any(p.requires_grad for p in model.parameters())
            and id(model) not in snapshots
        ):
            snapshots[id(model)] = (
                model,
                {k: v.clone() for k, v in model.state_dict().items()},
            )
        return original(model, pairs)

    with patch("qvelt.posttrain.dpo.pair_logps", side_effect=observe):
        result = train_dpo(recipe, tmp_path / "ref-check", ROOT)
    assert result["reference_frozen"] and len(snapshots) == 1
    for reference, before in snapshots.values():
        assert all(p.grad is None for p in reference.parameters())
        for name, value in before.items():
            assert torch.equal(value, reference.state_dict()[name])


def test_changed_beta_and_unimplemented_modes_fail_closed(preference_run: Path) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    with pytest.raises(ValueError, match="recipe"):
        train_dpo(
            replace(recipe, beta=0.2), preference_run / "replayed", ROOT, resume=True
        )
    with pytest.raises(ConfigError, match="CPU"):
        replace(
            recipe,
            job=replace(
                recipe.job, training=replace(recipe.job.training, compile=True)
            ),
        )
    with pytest.raises(ConfigError):
        parse_dpo(dump_dpo(recipe) + "unknown: 1\n")
    with pytest.raises(ConfigError):
        replace(recipe, beta=0)


def test_pretraining_checkpoint_cannot_masquerade_as_sft_reference(
    preference_run: Path, tmp_path: Path
) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    base = preference_run / "sft/base/checkpoints/step-00000003"
    manifest = ArtifactStore(base).read_manifest()
    recipe = replace(
        recipe,
        reference_manifest=str(base / "manifest.json"),
        reference_sha256=manifest_digest(manifest),
    )
    with pytest.raises(ValueError, match="SFT"):
        train_dpo(recipe, tmp_path / "not-sft", ROOT)
    assert not (tmp_path / "not-sft").exists()


def test_cli_executes_real_update_and_rejects_invalid_recipe(
    preference_run: Path, tmp_path: Path
) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    path = tmp_path / "job.yaml"
    path.write_text(dump_dpo(recipe))
    assert (
        main(
            [
                "dpo",
                str(path),
                "--directory",
                str(tmp_path / "run"),
                "--repository",
                str(ROOT),
                "--stop-after",
                "1",
            ]
        )
        == 0
    )
    assert (tmp_path / "run/checkpoints/step-00000001/manifest.json").exists()
    path.write_text("beta: 0\n")
    assert main(["dpo", str(path), "--directory", str(tmp_path / "invalid")]) == 1


@pytest.mark.parametrize(
    "mode", ["different_prompt", "wrong_split", "partial_response"]
)
def test_rechecks_encoded_pair_semantics(
    preference_run: Path, tmp_path: Path, mode: str
) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    root = tmp_path / mode
    shutil.copytree(preference_run / "pairs", root)
    store = ArtifactStore(root)
    manifest = store.read_manifest()
    path = root / "pairs.json"
    body = json.loads(path.read_text())
    row = body["pairs"][0]
    if mode == "different_prompt":
        row["chosen"][2] = 32 if row["chosen"][2] != 32 else 33
    elif mode == "wrong_split":
        row["split"] = "invalid"
    else:
        row["chosen"][-2] = 32
    path.write_text(json.dumps(body))
    record = hash_file(path)
    files = tuple(
        replace(f, size_bytes=record.size_bytes, sha256=record.sha256)
        if f.path == "pairs.json"
        else f
        for f in manifest.files
    )
    modified = replace(manifest, files=files)
    (root / "manifest.json").unlink()
    store.publish(modified)
    with pytest.raises(ValueError):
        load_preferences(
            root / "manifest.json",
            manifest_digest(modified),
            recipe.job.experiment.model.vocab,
            recipe.job.training.sequence_length,
        )


def test_rights_approval_duplicates_and_truncation_are_rejected(
    preference_run: Path, tmp_path: Path
) -> None:
    recipe = load_dpo(preference_run / "dpo.yaml")
    token_path = preference_run / "pairs/tokenizer/manifest.json"
    tm = ArtifactStore(token_path.parent).read_manifest()
    token_sha = manifest_digest(tm)
    documents = math_preferences()
    source = write_raw_bundle(
        documents,
        tmp_path / "eval-only",
        authored_source("preference-rights-test"),
        PERMISSION,
        allowed_uses=("evaluation",),
    )
    with pytest.raises(ValueError, match="permit"):
        prepare_preferences(
            tmp_path / "eval-only/manifest.json",
            manifest_digest(source),
            token_path,
            token_sha,
            tmp_path / "blocked",
            approved_for_training=True,
        )
    with pytest.raises(ValueError, match="approval"):
        prepare_preferences(
            tmp_path / "eval-only/manifest.json",
            manifest_digest(source),
            token_path,
            token_sha,
            tmp_path / "not-approved",
        )
    duplicate = write_raw_bundle(
        [*documents, documents[0]],
        tmp_path / "duplicate",
        authored_source("preference-duplicate-test"),
        PERMISSION,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_preferences(
            tmp_path / "duplicate/manifest.json",
            manifest_digest(duplicate),
            token_path,
            token_sha,
            tmp_path / "duplicate-out",
            approved_for_training=True,
        )
    tokenizer, _ = load_tokenizer_bundle(token_path, token_sha)
    with pytest.raises(ValueError):
        encode_pair(
            PreferencePair("What is two plus two?", "4", "5", "arithmetic", "one"),
            tokenizer,
            4,
        )
    assert not (tmp_path / "blocked").exists()
    assert recipe.beta == 0.1
