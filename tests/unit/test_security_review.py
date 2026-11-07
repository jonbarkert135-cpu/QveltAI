"""Source inventory recognizes aliases and refuses missing/dynamic deserialization flags."""

from pathlib import Path

import pytest

from qvelt.utils.security_review import (
    deserialization_inventory,
    installed_requirements,
    review_sources,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "source,safe",
    [
        ("import torch;torch.load('state',weights_only=True)", True),
        ("from torch import load as restore;restore('state',weights_only=True)", True),
        ("import torch as t;t.load('state')", False),
        ("import torch;torch.load('state',weights_only=False)", False),
        ("import torch;torch.load('state',weights_only=trusted)", False),
        ("import pickle;pickle.loads(data)", False),
        ("import numpy as np;np.load(path,allow_pickle=False)", True),
        ("import yaml;yaml.load(data)", False),
        ("import yaml;yaml.load(data,Loader=yaml.SafeLoader)", True),
        ("import yaml;yaml.load(data,Loader=yaml.UnsafeLoader)", False),
        (
            "import yaml\nclass Unique(yaml.SafeLoader):pass\nyaml.load(data,Loader=Unique)",
            True,
        ),
        ("eval(data)", False),
    ],
)
def test_deserialization_contract(source: str, safe: bool) -> None:
    rows = deserialization_inventory(source)
    assert len(rows) == 1 and rows[0]["safe"] is safe


def test_first_party_has_no_unrestricted_direct_loads() -> None:
    review = review_sources(ROOT)
    assert review["calls"] and review["unsafe_calls"] == []
    assert all(
        "+cpu" not in row and "+cu" not in row for row in installed_requirements()
    )
    assert not any(
        row.lower().startswith("qvelt==") for row in installed_requirements()
    )
