"""Small safety fixture uses genuine family-isolated assistant-only SFT targets."""

import json
from collections import defaultdict
from pathlib import Path

from qvelt.data.loader import PackedDataset
from qvelt.eval.smoke import create_evaluation_fixture
from qvelt.posttrain.safety import build_safety_data, safety_examples
from qvelt.tokenizer.workflow import load_tokenizer_bundle
from qvelt.utils.artifacts import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]


def test_safety_fixture_exact_targets_and_family_isolation(tmp_path: Path) -> None:
    fixture = create_evaluation_fixture(tmp_path / "fixture", ROOT)
    tokenizer, _ = load_tokenizer_bundle(
        Path(fixture.tokenizer.path), fixture.tokenizer.sha256
    )
    build_safety_data(fixture.tokenizer, tmp_path / "safety")
    rows = safety_examples()
    assert len(rows) == 24 and {row.language for row in rows} == {"en", "ru"}
    assert sum(row.behavior == "refuse" for row in rows) == sum(
        row.behavior == "comply" for row in rows
    )
    expected = {i: tokenizer.encode(row.response) + [6] for i, row in enumerate(rows)}
    families = []
    seen: set[int] = set()
    report = json.loads((tmp_path / "safety/prepared/report.json").read_text())
    for split in ("train", "validation", "test"):
        if not report["counts"][split]["documents"]:
            continue
        data = PackedDataset(tmp_path / "safety/prepared/manifest.json", split=split)
        targets: dict[int, list[int]] = defaultdict(list)
        for i in range(len(data)):
            batch = data[i]
            for j, flag in enumerate(batch["loss_mask"].tolist()):
                if flag:
                    targets[int(batch["document_ids"][j])].append(
                        int(batch["labels"][j])
                    )
        assert all(tokens == expected[doc] for doc, tokens in targets.items())
        assert not (set(targets) & seen)
        seen.update(targets)
        families.append(set(report["counts"][split]["families"]))
        assert report["counts"][split]["truncated"] == 0
    assert seen == set(expected)
    assert all(not a & b for i, a in enumerate(families) for b in families[i + 1 :])
    ArtifactStore(tmp_path / "safety/prepared").read_manifest()
