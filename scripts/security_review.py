"""Inventory first-party loaders and query advisories without ignoring known findings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from qvelt.utils.security_review import installed_requirements, review_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--dependencies", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    review = review_sources(args.repository)
    (args.directory / "deserialization.json").write_text(
        json.dumps(review, sort_keys=True, indent=2) + "\n"
    )
    dependency_code = 0
    if args.dependencies:
        requirements = args.directory / "requirements.txt"
        requirements.write_text("\n".join(installed_requirements()) + "\n")
        dependency_code = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--disable-pip",
                "--no-deps",
                "-r",
                str(requirements),
                "--cache-dir",
                str(args.directory / "cache"),
                "--format",
                "json",
                "--output",
                str(args.directory / "dependencies.json"),
            ],
            check=False,
        ).returncode
    print(
        json.dumps(
            {
                "unsafe_calls": len(review["unsafe_calls"]),
                "dependency_audit_exit": dependency_code,
                "scope": review["scope"],
            },
            sort_keys=True,
        )
    )
    raise SystemExit(1 if review["unsafe_calls"] else dependency_code)


if __name__ == "__main__":
    main()
