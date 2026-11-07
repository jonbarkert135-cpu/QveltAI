"""Validate a framework candidate and its archives;never create a tag or upload weights."""

import argparse
import json
from pathlib import Path

from qvelt.utils.release import validate_distribution, validate_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    version = validate_version(args.repository, args.tag)
    files = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
    if len(files) != 2:
        raise ValueError("Expected exactly one wheel and one sdist")
    hashes = {path.name: validate_distribution(path) for path in files}
    (args.dist / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    print(
        json.dumps(
            {"version": version, "framework_only": True, "sha256": hashes},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
