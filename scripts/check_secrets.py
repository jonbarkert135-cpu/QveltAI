"""Offline tracked-file credential scan;only reviewed exact findings may match baseline."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def identities(report: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (filename, row["type"], row["hashed_secret"])
        for filename, rows in report["results"].items()
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    repository = args.repository.resolve()
    baseline = json.loads((repository / ".secrets.baseline").read_text())
    if any(
        row.get("is_secret") is not False
        for rows in baseline["results"].values()
        for row in rows
    ):
        raise ValueError(
            "Every allowed finding requires explicit false-positive review"
        )
    command = [
        sys.executable,
        "-m",
        "detect_secrets",
        "scan",
        "--no-verify",
        "--exclude-files",
        r"^\.secrets\.baseline$",
        "--disable-plugin",
        "HexHighEntropyString",
    ]
    process = subprocess.run(
        command, cwd=repository, capture_output=True, text=True, check=True
    )
    report = json.loads(process.stdout)
    unexpected = identities(report) - identities(baseline)
    # Never echo matched values,credential material or authorization headers.
    print(
        json.dumps(
            {
                "unexpected": [
                    {"path": path, "type": kind} for path, kind, _ in sorted(unexpected)
                ],
                "tracked_finding_count": sum(map(len, report["results"].values())),
                "scope": "tracked files,offline provider/keyword/base64 detectors;bare hex entropy excluded due to SHA provenance",
            },
            sort_keys=True,
        )
    )
    raise SystemExit(bool(unexpected))


if __name__ == "__main__":
    main()
