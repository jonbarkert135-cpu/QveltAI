"""Licensed SFT packing, initialization, replay and fail-closed objective contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from qvelt.cli import main
from qvelt.config.pretraining import load_pretraining_job
from qvelt.data.demo import PERMISSION, authored_source
from qvelt.data.loader import LoaderConfig, PackedDataset, ResumableLoader
from qvelt.data.records import write_raw_bundle
from qvelt.model.decoder import Decoder
from qvelt.posttrain.data import prepare_sft
from qvelt.posttrain.smoke import math_supervision, run_sft_smoke
from qvelt.train.pretraining import initialize_weights, prepare_inputs
from qvelt.train.pretraining_smoke import demo_recipe
from qvelt.utils.artifacts import ArtifactIntegrityError, ArtifactStore
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_sft_overfit_replay_and_contracts(tmp_path: Path) -> None:
    result = run_sft_smoke(tmp_path / "smoke", ROOT, steps=30)
    assert result["weights_bit_exact"] and result["optimizer_bit_exact"]
    assert result["rng_bit_exact"] and result["consumer_state_exact"]
    assert result["last_training_loss"] < result["first_training_loss"]
    root = tmp_path / "smoke"
    recipe = load_pretraining_job(root / "sft.yaml")
    inputs = prepare_inputs(recipe)
    report = json.loads((root / "data/report.json").read_text())
    families = [
        set(report["counts"][split]["families"])
        for split in ("train", "validation", "test")
    ]
    assert not (
        families[0] & families[1]
        or families[0] & families[2]
        or families[1] & families[2]
    )
    for dataset in (inputs.training, inputs.validation):
        for index in range(len(dataset)):
            sample = dataset[index]
            assert 0 < sample["loss_mask"].sum() < sample["valid_mask"].sum()
            for doc in set(sample["document_ids"].tolist()):
                if doc < 0:
                    continue
                selected = sample["document_ids"] == doc
                assert sample["input_ids"][selected][0] == 1
                assert sample["labels"][selected][-1] == 2
                assert sample["positions"][selected][0] == 0
    loader = ResumableLoader(
        inputs.training, LoaderConfig(max_batches=2), inputs.stages
    )
    next(loader)
    state = loader.state_dict()
    expected = next(loader)
    loader.close()
    with ResumableLoader(
        inputs.training, LoaderConfig(max_batches=2, workers=2), inputs.stages, state
    ) as restarted:
        actual = next(restarted)
        assert all(torch.equal(actual[k], expected[k]) for k in expected)
    model = Decoder(recipe.job.experiment.model)
    initialize_weights(model, recipe, result["continuous"]["tokenizer_sha256"])
    original = load_file(
        str(Path(recipe.initial_checkpoint or "").parent / "model.safetensors")
    )
    assert all(
        torch.equal(value, original[name]) for name, value in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="tokenizer"):
        initialize_weights(model, recipe, "0" * 64)
    with pytest.raises(ValueError, match="architecture"):
        initialize_weights(
            model,
            replace(
                recipe,
                job=replace(
                    recipe.job,
                    experiment=replace(
                        recipe.job.experiment,
                        model=replace(recipe.job.experiment.model, context=256),
                    ),
                ),
            ),
            result["continuous"]["tokenizer_sha256"],
        )
    with pytest.raises(ValueError, match="objective"):
        prepare_inputs(replace(recipe, objective="pretraining"))
    ordinary = demo_recipe(root / "base-data/prepared/manifest.json", ROOT)
    with pytest.raises(ValueError, match="objective"):
        prepare_inputs(
            replace(
                ordinary,
                objective="sft",
                initial_checkpoint=recipe.initial_checkpoint,
                initial_sha256=recipe.initial_sha256,
            )
        )
    assert (
        main(["pretrain", str(root / "sft.yaml"), "--directory", str(root / "wrong")])
        == 1
    )
    assert (
        main(
            [
                "sft",
                str(root / "sft.yaml"),
                "--directory",
                str(root / "replayed"),
                "--repository",
                str(ROOT),
                "--resume",
                "latest",
            ]
        )
        == 0
    )
    # A corrupted initial checkpoint cannot be used for a new SFT run.
    checkpoint = Path(recipe.initial_checkpoint or "").parent
    with (checkpoint / "model.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ArtifactIntegrityError):
        initialize_weights(model, recipe, result["continuous"]["tokenizer_sha256"])


def test_permissions_and_invalid_supervision(
    prepared_demo: Path, tmp_path: Path
) -> None:
    token_path = prepared_demo / "prepared/tokenizer/manifest.json"
    token_sha = manifest_digest(ArtifactStore(token_path.parent).read_manifest())
    source = write_raw_bundle(
        math_supervision(8),
        tmp_path / "source",
        authored_source("eval-only"),
        PERMISSION,
        allowed_uses=("evaluation",),
    )
    with pytest.raises(ValueError, match="permit training"):
        prepare_sft(
            tmp_path / "source/manifest.json",
            manifest_digest(source),
            token_path,
            token_sha,
            tmp_path / "blocked",
            approved_for_training=True,
        )
    assert not (tmp_path / "blocked").exists()
    from qvelt.data.records import RawDocument

    invalid = write_raw_bundle(
        [RawDocument('{"schema_version":1,"messages":[]}', family="any")],
        tmp_path / "invalid",
        authored_source("invalid-chat"),
        PERMISSION,
    )
    with pytest.raises(ValueError):
        prepare_sft(
            tmp_path / "invalid/manifest.json",
            manifest_digest(invalid),
            token_path,
            token_sha,
            tmp_path / "bad",
            approved_for_training=True,
        )
    assert not (tmp_path / "bad/manifest.json").exists()


@pytest.mark.parametrize("long_response", [False, True])
def test_packed_chat_labels_and_truncation(
    prepared_demo: Path, tmp_path: Path, long_response: bool
) -> None:
    from qvelt.data.records import RawDocument

    token_path = prepared_demo / "prepared/tokenizer/manifest.json"
    token_sha = manifest_digest(ArtifactStore(token_path.parent).read_manifest())
    documents = [
        RawDocument(
            json.dumps(
                {
                    "schema_version": 1,
                    "messages": [
                        {"role": "user", "content": "x"},
                        {
                            "role": "assistant",
                            "content": "Z" * (1000 if long_response else 1),
                        },
                    ],
                }
            ),
            d.identifier,
            d.family,
            d.language,
        )
        for d in math_supervision(8)
    ]
    raw = write_raw_bundle(
        documents, tmp_path / "raw", authored_source("packing-test"), PERMISSION
    )
    with pytest.raises(ValueError, match="explicit training approval"):
        prepare_sft(
            tmp_path / "raw/manifest.json",
            manifest_digest(raw),
            token_path,
            token_sha,
            tmp_path / "unapproved",
        )
    assert not (tmp_path / "unapproved").exists()
    prepare_sft(
        tmp_path / "raw/manifest.json",
        manifest_digest(raw),
        token_path,
        token_sha,
        tmp_path / "packed",
        context=64,
        approved_for_training=True,
    )
    data = PackedDataset(tmp_path / "packed/manifest.json")
    saw_multiple = False
    for index in range(len(data)):
        block = data[index]
        doc_ids = {v for v in block["document_ids"].tolist() if v >= 0}
        saw_multiple |= len(doc_ids) > 1
        for doc in doc_ids:
            selected = block["document_ids"] == doc
            inputs = block["input_ids"][selected].tolist()
            labels = block["labels"][selected].tolist()
            mask = block["loss_mask"][selected].tolist()
            assert inputs[0] == 1 and labels[-1] == 2
            # User 'x' is one ordinary byte token. Independent role/target oracle.
            assert inputs[:2] == [1, 4] and inputs[3:5] == [6, 5]
            assert not any(mask[:4]) and all(mask[4:-1]) and mask[-1] is False
            if long_response:
                assert 6 not in labels[4:-1] and len(labels) == 64
            else:
                assert labels[-2] == 6 and len(labels) == 7
    assert saw_multiple == (not long_response)
    shard = data.root / data.sources["supervision"]["shards"][0]["path"]
    with shard.open("ab") as stream:
        stream.write(b"X")
    with pytest.raises(ArtifactIntegrityError):
        PackedDataset(tmp_path / "packed/manifest.json")[0]
