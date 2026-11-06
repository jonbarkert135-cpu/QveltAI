"""Configuration, tokenizer and bounded synthetic decoder-training workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from qvelt import __version__
from qvelt.config import ConfigError, Device, config_digest, dump_config, load_config
from qvelt.tokenizer.core import TokenizerError
from qvelt.utils.artifacts import ArtifactError, ArtifactStore
from qvelt.utils.logging import configure_logging
from qvelt.utils.manifest_types import ManifestError
from qvelt.utils.manifests import manifest_digest
from qvelt.utils.provenance import ProvenanceError, initialize_run
from qvelt.utils.reproducibility import RuntimeUnavailableError, inspect_runtime


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
        sub.add_argument("--set", dest="overrides", action="append", default=[])
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
    initialize.add_argument("--set", dest="overrides", action="append", default=[])
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
    tokenizer = commands.add_parser(
        "tokenizer", help="Train, verify or evaluate byte-BPE"
    )
    tokenizer_actions = tokenizer.add_subparsers(dest="action", required=True)
    resolve = tokenizer_actions.add_parser(
        "resolve", help="Resolve a strict YAML recipe"
    )
    resolve.add_argument("path", type=Path)
    train = tokenizer_actions.add_parser(
        "train", help="Train on registered raw JSONL data"
    )
    train.add_argument("path", type=Path, help="Tokenizer YAML recipe")
    train.add_argument("--data-manifest", type=Path, required=True)
    train.add_argument("--directory", type=Path, required=True)
    train.add_argument("--repository", type=Path, default=Path("."))
    for action in ("verify", "evaluate"):
        sub = tokenizer_actions.add_parser(action)
        sub.add_argument("path", type=Path, help="Tokenizer bundle manifest")
        sub.add_argument("--expected-sha256")
        if action == "evaluate":
            sub.add_argument("--data-manifest", type=Path, required=True)
            sub.add_argument("--data-sha256", required=True)
    train = commands.add_parser(
        "train", help="Train the RFC decoder on synthetic engineering fixtures"
    )
    train.add_argument("path", type=Path)
    train.add_argument("--directory", type=Path, required=True)
    train.add_argument("--repository", type=Path, default=Path("."))
    train.add_argument("--resume", choices=("latest",))
    train.add_argument("--stop-after", type=int)
    train.add_argument("--set", dest="overrides", action="append", default=[])
    from qvelt.data.cli import add_parser

    add_parser(commands)
    for command, help_text in (
        ("pretrain", "Train on pinned packed BPE data"),
        ("sft", "Fine-tune pinned Qvelt weights on assistant targets"),
    ):
        pretrain = commands.add_parser(command, help=help_text)
        pretrain.add_argument("path", type=Path)
        pretrain.add_argument("--directory", type=Path, required=True)
        pretrain.add_argument("--repository", type=Path, default=Path("."))
        pretrain.add_argument("--resume", choices=("latest", "auto"))
        pretrain.add_argument("--stop-after", type=int)
        pretrain.add_argument("--set", dest="overrides", action="append", default=[])
    for command, help_text in (
        ("dpo", "Optimize pinned demo pairs with a frozen SFT reference"),
        ("grpo", "Bounded original-demo CPU outcome-GRPO"),
    ):
        preference = commands.add_parser(command, help=help_text)
        preference.add_argument("path", type=Path)
        preference.add_argument("--directory", type=Path, required=True)
        preference.add_argument("--repository", type=Path, default=Path("."))
        preference.add_argument("--resume", choices=("latest",))
        preference.add_argument("--stop-after", type=int)
    gsm8k = commands.add_parser(
        "import-gsm8k", help="Import the pinned reviewed original train-question subset"
    )
    gsm8k.add_argument("path", type=Path)
    gsm8k.add_argument("--sha256", required=True)
    gsm8k.add_argument("--license-file", type=Path, required=True)
    gsm8k.add_argument("--license-sha256", required=True)
    gsm8k.add_argument("--directory", type=Path, required=True)
    gsm8k.add_argument("--count", type=int, default=16)
    reasoning = commands.add_parser(
        "reasoning-data",
        help="Reject-sample answer-verified traces from an own checkpoint",
    )
    reasoning.add_argument("path", type=Path)
    reasoning.add_argument("--directory", type=Path, required=True)
    external = commands.add_parser(
        "ingest-traces",
        help="Explicit reviewed external synthetic trace ingress; off by default",
    )
    external.add_argument("path", type=Path)
    external.add_argument("--sha256", required=True)
    external.add_argument("--directory", type=Path, required=True)
    external.add_argument("--external-synthetic", action="store_true")
    external.add_argument("--approved-for-training", action="store_true")
    external.add_argument("--terms", type=Path, required=True)
    external.add_argument("--terms-sha256", required=True)
    from qvelt.eval.cli import add_parser as add_eval_parser

    add_eval_parser(commands)
    evaluation = commands.add_parser("evaluate", help="Read-only pinned CPU evaluation")
    evaluation.add_argument("path", type=Path)
    evaluation.add_argument("--directory", type=Path, required=True)
    evaluation.add_argument("--repository", type=Path, default=Path("."))
    study = commands.add_parser(
        "compare", help="Run matched-lineage CPU evaluation gates"
    )
    study.add_argument("path", type=Path)
    study.add_argument("--directory", type=Path, required=True)
    study.add_argument("--repository", type=Path, default=Path("."))
    export = commands.add_parser(
        "export", help="Export a verified native CPU inference snapshot"
    )
    export.add_argument("path", type=Path, help="Checkpoint manifest")
    export.add_argument("--checkpoint-sha256", required=True)
    export.add_argument("--tokenizer-manifest", type=Path, required=True)
    export.add_argument("--tokenizer-sha256", required=True)
    export.add_argument("--directory", type=Path, required=True)
    export.add_argument("--repository", type=Path, default=Path("."))
    generate = commands.add_parser(
        "generate", help="Generate from a native snapshot with bounded CPU requests"
    )
    generate.add_argument("path", type=Path, help="Generation recipe")
    generate.add_argument(
        "--stream", action="store_true", help="Emit cumulative-text JSONL events"
    )
    collective = commands.add_parser(
        "distributed-check", help="Run an exact integer all-reduce"
    )
    collective.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 on verified command success, 1 on config/runtime/IO failure."""
    args = _parser().parse_args(argv)
    logger = configure_logging()
    try:
        if args.command in {
            "eval-import",
            "eval-audit",
            "eval-receipt",
            "eval-registry",
            "eval-repair",
            "eval-harness",
            "results",
        }:
            from qvelt.eval.cli import run as run_eval_command

            print(json.dumps(run_eval_command(args), sort_keys=True))
        elif args.command == "import-gsm8k":
            from qvelt.eval.config import ArtifactRef
            from qvelt.posttrain.gsm8k import import_gsm8k_training

            math_manifest = import_gsm8k_training(
                ArtifactRef(str(args.path), args.sha256),
                ArtifactRef(str(args.license_file), args.license_sha256),
                args.directory,
                count=args.count,
            )
            print(json.dumps({"manifest_sha256": manifest_digest(math_manifest)}))
        elif args.command == "reasoning-data":
            from qvelt.posttrain.reasoning import build_reasoning_data, load_reasoning

            reasoning_manifest = build_reasoning_data(
                load_reasoning(args.path), args.directory
            )
            print(json.dumps({"manifest_sha256": manifest_digest(reasoning_manifest)}))
        elif args.command == "ingest-traces":
            from qvelt.eval.config import ArtifactRef
            from qvelt.posttrain.reasoning import ingest_external_traces

            trace_manifest = ingest_external_traces(
                ArtifactRef(str(args.path), args.sha256),
                args.directory,
                approved_for_training=args.approved_for_training,
                external_synthetic=args.external_synthetic,
                terms=ArtifactRef(str(args.terms), args.terms_sha256),
            )
            print(json.dumps({"manifest_sha256": manifest_digest(trace_manifest)}))
        elif args.command == "export":
            from qvelt.eval.config import ArtifactRef
            from qvelt.inference.artifacts import export_snapshot

            snapshot_manifest = export_snapshot(
                ArtifactRef(str(args.path), args.checkpoint_sha256),
                ArtifactRef(str(args.tokenizer_manifest), args.tokenizer_sha256),
                args.directory,
                args.repository,
            )
            print(
                json.dumps(
                    {
                        "format": "qvelt-inference-v1",
                        "sha256": manifest_digest(snapshot_manifest),
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "generate":
            from contextlib import closing

            from qvelt.inference.artifacts import load_snapshot
            from qvelt.inference.config import load_generation
            from qvelt.inference.generation import GenerationEngine

            generation_recipe = load_generation(args.path)
            snapshot = load_snapshot(generation_recipe.snapshot)
            engine = GenerationEngine(
                snapshot.model, snapshot.tokenizer, generation_recipe.limits
            )
            requests = engine.encode_requests(
                generation_recipe.requests, generation_recipe.prompt_format
            )
            if args.stream:
                try:
                    with closing(
                        engine.stream(requests, use_cache=generation_recipe.use_cache)
                    ) as events:
                        for event in events:
                            print(
                                json.dumps(
                                    asdict(event), sort_keys=True, ensure_ascii=False
                                ),
                                flush=True,
                            )
                except KeyboardInterrupt:
                    return 130
            else:
                print(
                    json.dumps(
                        [
                            asdict(completion)
                            for completion in engine.generate(
                                requests, use_cache=generation_recipe.use_cache
                            )
                        ],
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
        elif args.command == "compare":
            from qvelt.eval.study import run_study
            from qvelt.eval.study_config import load_study

            result = run_study(load_study(args.path), args.directory, args.repository)
            print(
                json.dumps(
                    {
                        "study_sha256": result["study_sha256"],
                        "comparisons": result["result"]["comparisons_to_sft"],
                        "automatic_promotion": False,
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "evaluate":
            from qvelt.eval.config import load_eval
            from qvelt.eval.runner import run_evaluation

            print(
                json.dumps(
                    run_evaluation(
                        load_eval(args.path),
                        args.directory,
                        args.repository,
                        command=(
                            "qvelt",
                            "evaluate",
                            str(args.path),
                            "--directory",
                            str(args.directory),
                            "--repository",
                            str(args.repository),
                        ),
                    ),
                    sort_keys=True,
                )
            )
        elif args.command == "config":
            config = load_config(args.path, args.overrides)
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
                load_config(args.path, args.overrides), args.directory, args.repository
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
        elif args.command == "tokenizer":
            from qvelt.tokenizer.recipe import (
                dump_tokenizer_recipe,
                load_tokenizer_recipe,
            )
            from qvelt.tokenizer.workflow import (
                evaluate_tokenizer,
                load_tokenizer_bundle,
                train_tokenizer_bundle,
            )

            if args.action == "resolve":
                print(dump_tokenizer_recipe(load_tokenizer_recipe(args.path)), end="")
            elif args.action == "train":
                manifest = train_tokenizer_bundle(
                    load_tokenizer_recipe(args.path),
                    args.data_manifest,
                    args.directory,
                    args.repository,
                )
                print(
                    json.dumps(
                        {
                            "kind": "tokenizer",
                            "sha256": manifest_digest(manifest),
                            "metadata": asdict(manifest.metadata),
                        },
                        sort_keys=True,
                    )
                )
            else:
                tokenizer, manifest = load_tokenizer_bundle(
                    args.path, args.expected_sha256
                )
                if args.action == "verify":
                    print(
                        json.dumps(
                            {
                                "verified": True,
                                "sha256": manifest_digest(manifest),
                                "vocab_size": tokenizer.vocab_size,
                            },
                            sort_keys=True,
                        )
                    )
                else:
                    metrics = evaluate_tokenizer(
                        tokenizer, args.data_manifest, args.data_sha256
                    )
                    print(
                        json.dumps(
                            {
                                **asdict(metrics),
                                "bytes_per_token": metrics.bytes_per_token,
                                "roundtrips_verified": True,
                                "tokenizer_sha256": manifest_digest(manifest),
                                "data_sha256": args.data_sha256,
                            },
                            sort_keys=True,
                        )
                    )
        elif args.command == "data":
            from qvelt.data.cli import run as run_data

            print(json.dumps(run_data(args), sort_keys=True))
        elif args.command in ("pretrain", "sft"):
            from qvelt.config.pretraining import load_pretraining_job
            from qvelt.train.pretraining import train_pretraining

            recipe = load_pretraining_job(args.path, args.overrides)
            if recipe.objective != ("sft" if args.command == "sft" else "pretraining"):
                raise ValueError("CLI command and recipe objective disagree")
            result = train_pretraining(
                recipe,
                args.directory,
                args.repository,
                base=args.path.parent,
                resume=args.resume or False,
                stop_after=args.stop_after,
            )
            if result["rank"] == 0:
                print(json.dumps(result, sort_keys=True))
            if result["interrupted"]:
                return 3
        elif args.command == "dpo":
            from qvelt.posttrain.dpo import load_dpo, train_dpo

            result = train_dpo(
                load_dpo(args.path),
                args.directory,
                args.repository,
                resume=args.resume == "latest",
                stop_after=args.stop_after,
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "grpo":
            from qvelt.posttrain.grpo_training import load_grpo, train_grpo

            result = train_grpo(
                load_grpo(args.path),
                args.directory,
                args.repository,
                resume=args.resume == "latest",
                stop_after=args.stop_after,
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "train":
            from qvelt.config.training import load_training_job
            from qvelt.train.runner import train_synthetic

            result = train_synthetic(
                load_training_job(args.path, args.overrides),
                args.directory,
                args.repository,
                resume=args.resume == "latest",
                stop_after=args.stop_after,
            )
            if result["rank"] == 0:
                print(json.dumps(result, sort_keys=True))
        elif args.command == "distributed-check":
            from qvelt.utils.distributed import all_reduce_probe

            result = all_reduce_probe(args.device)
            if result["rank"] == 0:
                print(json.dumps(result, sort_keys=True))
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
        TokenizerError,
        RuntimeError,
        ValueError,
    ) as error:
        logger.error("%s", error)
        return 1
    return 0
