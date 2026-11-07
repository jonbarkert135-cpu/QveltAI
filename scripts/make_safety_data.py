"""Prepare the original approved safety/helpfulness design fixture;not safety certification."""

import argparse
import json
from pathlib import Path

from qvelt.eval.config import ArtifactRef
from qvelt.posttrain.safety import build_safety_data
from qvelt.utils.manifests import manifest_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_safety_data(
        ArtifactRef(str(args.tokenizer_manifest), args.tokenizer_sha256), args.directory
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest_digest(manifest),
                "safety_certification": False,
            }
        )
    )


if __name__ == "__main__":
    main()
