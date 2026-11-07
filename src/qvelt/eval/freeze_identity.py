"""Validate the historical holdout byte identity across producer-version upgrades."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from qvelt import __version__
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest


def verify_frozen_identity(
    current: dict[str, Any], registered: dict[str, Any], directory: Path
) -> None:
    """Permit only the producer envelope to differ from the original0.1.0.dev0 freeze.

    Reconstruct and hash the entire original envelope,including every payload,
    rights record and timestamp. Never rewrite the committed freeze or generator.
    """
    normalized = copy.deepcopy(current)
    for partition in ("validation", "final"):
        entry = normalized["partitions"][partition]
        if entry["manifest"] != f"{partition}/manifest.json":
            raise ValueError("Unexpected frozen-corpus manifest path")
        path = directory / entry["manifest"]
        manifest = ArtifactStore(path.parent).read_manifest(
            path.name, expected_sha256=entry["manifest_sha256"]
        )
        if manifest.producer_version != __version__:
            raise ValueError("Unexpected frozen-corpus producer version")
        entry["manifest_sha256"] = manifest_digest(
            replace(manifest, producer_version="0.1.0.dev0")
        )
    if normalized != registered:
        raise ValueError("Protected corpus differs from its preregistered freeze")
