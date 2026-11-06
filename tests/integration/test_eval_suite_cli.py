"""Original mixed task contract and executable evaluation command dispatch."""

from pathlib import Path

import pytest

from qvelt.cli import main
from qvelt.data.decontamination import normalized_tokens
from qvelt.eval.suite import SuiteSettings, load_suite_settings, original_cases

ROOT = Path(__file__).resolve().parents[2]


def test_original_suite_known_answers_and_settings(tmp_path: Path) -> None:
    settings = load_suite_settings(ROOT / "configs/eval/quick.yaml")
    cases = original_cases(settings)
    assert len(cases) == 2 * settings.math_per_language + 6
    assert {c.language for c in cases} == {"en", "ru"}
    assert {c.kind for c in cases} == {"math", "choice", "lm", "code"}
    assert len({c.identifier for c in cases}) == len(cases)
    assert all(c.partition == "validation" for c in cases)
    with pytest.raises(ValueError):
        SuiteSettings(samples=0)
    path = tmp_path / "invalid.yaml"
    path.write_text("unknown: true\n")
    with pytest.raises(ValueError):
        load_suite_settings(path)
    assert normalized_tokens("A TWO") == ["a", "two"]


def test_registry_results_cli_and_stale_checks(tmp_path: Path) -> None:
    inventory = tmp_path / "registry.yaml"
    assert main(["eval-registry", "--output", str(inventory)]) == 0
    assert main(["eval-registry", "--output", str(inventory), "--check"]) == 0
    inventory.write_text("changed")
    assert main(["eval-registry", "--output", str(inventory), "--check"]) != 0
    results = tmp_path / "RESULTS.md"
    assert main(["results", "--output", str(results)]) == 0
    assert main(["results", "--output", str(results), "--check"]) == 0
    assert main(["results", "--output", str(results), "--release"]) != 0
