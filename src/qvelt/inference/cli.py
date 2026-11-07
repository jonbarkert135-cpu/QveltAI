"""Thin native snapshot conversion and local authenticated server commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

COMMANDS = {"export-hf", "quantize", "serve"}


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name in ("export-hf", "quantize"):
        command = commands.add_parser(
            name, help="Convert a verified native inference snapshot"
        )
        command.add_argument("path", type=Path)
        command.add_argument("--sha256", required=True)
        command.add_argument("--directory", type=Path, required=True)
        command.add_argument("--repository", type=Path, default=Path("."))
        if name == "quantize":
            command.add_argument("--bits", type=int, choices=(4, 8), required=True)
        else:
            command.add_argument("--tokenizer-path", type=Path)
            command.add_argument("--tokenizer-sha256")
            command.add_argument(
                "--context-extension", type=int, choices=(1, 2, 4), default=1
            )
    command = commands.add_parser("serve", help="Run the bounded local OpenAI chat API")
    command.add_argument("path", type=Path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from qvelt.eval.config import ArtifactRef
    from qvelt.utils.manifests import manifest_digest

    if args.command == "export-hf":
        from qvelt.inference.huggingface import export_huggingface

        if bool(args.tokenizer_path) != bool(args.tokenizer_sha256):
            raise ValueError(
                "Checkpoint export requires both tokenizer path and SHA256"
            )
        if args.tokenizer_path:
            from qvelt.inference.hf_checkpoint import export_checkpoint_huggingface

            result = export_checkpoint_huggingface(
                ArtifactRef(str(args.path), args.sha256),
                ArtifactRef(str(args.tokenizer_path), args.tokenizer_sha256),
                args.directory,
                args.repository,
                context_extension=args.context_extension,
            )
        else:
            result = export_huggingface(
                ArtifactRef(str(args.path), args.sha256),
                args.directory,
                args.repository,
                context_extension=args.context_extension,
            )
    elif args.command == "quantize":
        from qvelt.inference.quantization import export_quantized

        result = export_quantized(
            ArtifactRef(str(args.path), args.sha256),
            args.directory,
            args.repository,
            bits=args.bits,
        )
    elif args.command == "serve":
        import os

        import torch

        try:
            import uvicorn
        except ImportError as error:
            raise RuntimeError("Install the frozen serving extra") from error
        from qvelt.inference.api import create_app
        from qvelt.inference.artifacts import load_snapshot
        from qvelt.inference.config import EngineLimits
        from qvelt.inference.generation import GenerationEngine
        from qvelt.inference.serving_config import load_serving

        settings = load_serving(args.path)
        key = os.environ.get(settings.api_key_env)
        if not key and not settings.allow_unauthenticated:
            raise ValueError("Serving auth environment binding is missing")
        if settings.snapshot_format == "hqq":
            from qvelt.inference.quantization import load_quantized

            snapshot = load_quantized(settings.snapshot)
        else:
            snapshot = load_snapshot(settings.snapshot)
        torch.set_num_threads(settings.num_threads)
        engine = GenerationEngine(
            snapshot.model,
            snapshot.tokenizer,
            EngineLimits(max_context=min(2048, snapshot.model.config.context)),
        )
        app = create_app(engine, settings, api_key=key)
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            workers=1,
            access_log=False,
            log_level="warning",
        )
        return dict(app.state.shutdown_summary)
    else:
        raise ValueError("Unknown inference command")
    return {"manifest_sha256": manifest_digest(result)}
