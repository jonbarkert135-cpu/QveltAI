"""Implemented configuration and runtime diagnostics; no training entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from qvelt import __version__
from qvelt.artifacts import ArtifactError, ArtifactStore
from qvelt.config import ConfigError, Device, config_digest, dump_config, load_config
from qvelt.logging import configure_logging
from qvelt.manifest_types import ManifestError
from qvelt.manifests import manifest_digest
from qvelt.provenance import ProvenanceError, initialize_run
from qvelt.reproducibility import RuntimeUnavailableError, inspect_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qvelt", description="Qvelt configuration and runtime tools"
    )
    parser.add_argument("--version", action="version", version=f"qvelt {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="Validate or resolve experiment YAML")
    actions = config.add_subparsers(dest="action", required=True)
    for action in ("validate", "resolve"):
        sub = actions.add_parser(action)
        sub.add_argument("path", help="UTF-8 experiment YAML path")
    doctor = commands.add_parser(
        "doctor", help="Execute a tensor diagnostic on one device"
    )
    doctor.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run = commands.add_parser(
        "run", help="Create provenance bundles; does not train a model"
    )
    run_actions = run.add_subparsers(dest="action", required=True)
    initialize = run_actions.add_parser("init")
    initialize.add_argument("path", help="Experiment YAML")
    initialize.add_argument(
        "--directory",
        required=True,
        help="New output directory under an existing parent",
    )
    initialize.add_argument(
        "--repository", default=".", help="Existing code checkout with uv.lock"
    )
    manifest = commands.add_parser(
        "manifest", help="Inspect metadata or verify registered bytes"
    )
    manifest_actions = manifest.add_subparsers(dest="action", required=True)
    for action in ("inspect", "verify"):
        sub = manifest_actions.add_parser(action)
        sub.add_argument("path", help="Top-level bundle manifest JSON")
        sub.add_argument(
            "--expected-sha256", help="Trusted canonical manifest identity"
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 on verified command success, 1 on config/runtime/IO failure."""
    args = _parser().parse_args(argv)
    logger = configure_logging()
    try:
        if args.command == "config":
            config = load_config(args.path)
            logger = configure_logging(config.runtime.log_level)
            if args.action == "resolve":
                print(dump_config(config), end="")
            else:
                print(
                    json.dumps(
                        {
                            "valid": True,
                            "schema_version": config.schema_version,
                            "sha256": config_digest(config),
                            "analytical_parameters": config.model.parameter_count,
                        },
                        sort_keys=True,
                    )
                )
        elif args.command == "doctor":
            print(
                json.dumps(
                    asdict(inspect_runtime(cast(Device, args.device))), sort_keys=True
                )
            )
        elif args.command == "run":
            manifest = initialize_run(
                load_config(args.path), args.directory, args.repository
            )
            print(
                json.dumps(
                    {
                        "kind": "run",
                        "sha256": manifest_digest(manifest),
                        "owner": manifest.owner,
                    },
                    sort_keys=True,
                )
            )
        else:
            path = Path(args.path)
            store = ArtifactStore(path.parent)
            manifest = store.read_manifest(
                path.name,
                expected_sha256=args.expected_sha256,
                verify=args.action == "verify",
            )
            if args.action == "inspect":
                print(
                    json.dumps(
                        {
                            "manifest": asdict(manifest),
                            "sha256": manifest_digest(manifest),
                            "files_verified": False,
                        },
                        sort_keys=True,
                    )
                )
            else:
                print(
                    json.dumps(
                        {
                            "verified": True,
                            "sha256": manifest_digest(manifest),
                            "files": len(manifest.files),
                            "bytes": sum(file.size_bytes for file in manifest.files),
                        },
                        sort_keys=True,
                    )
                )
    except (
        ConfigError,
        OSError,
        RuntimeUnavailableError,
        ManifestError,
        ArtifactError,
        ProvenanceError,
    ) as error:
        logger.error("%s", error)
        return 1
    return 0
