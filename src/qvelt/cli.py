"""Implemented configuration and runtime diagnostics; no training entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import cast

from qvelt import __version__
from qvelt.config import ConfigError, Device, config_digest, dump_config, load_config
from qvelt.logging import configure_logging
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
        else:
            print(
                json.dumps(
                    asdict(inspect_runtime(cast(Device, args.device))), sort_keys=True
                )
            )
    except (ConfigError, OSError, RuntimeUnavailableError) as error:
        logger.error("%s", error)
        return 1
    return 0
