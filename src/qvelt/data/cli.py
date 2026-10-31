"""Thin data CLI dispatch; heavy libraries are loaded only for the requested action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def add_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "data", help="Curate licensed text, pack shards and measure delivery"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    demo = actions.add_parser(
        "demo", help="Build original offline EN/RU fixture and prepared data"
    )
    demo.add_argument("--directory", type=Path, required=True)
    demo.add_argument("--repository", type=Path, default=Path("."))
    demo.add_argument("--documents", type=int, default=192)
    prepare = actions.add_parser("prepare", help="Run strict approved-source recipe")
    prepare.add_argument("path", type=Path)
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--repository", type=Path, default=Path("."))
    download = actions.add_parser(
        "download", help="Fetch one approved checksum-pinned HTTPS source"
    )
    download.add_argument("path", type=Path)
    download.add_argument("--directory", type=Path, required=True)
    verify = actions.add_parser(
        "verify", help="Verify every byte, tokenizer linkage and packed block"
    )
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-sha256")
    benchmark = actions.add_parser("benchmark", help="Measure CPU host batch delivery")
    benchmark.add_argument("path", type=Path)
    benchmark.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    benchmark.add_argument("--batches", type=int, default=64)
    benchmark.add_argument("--warmup", type=int, default=8)
    benchmark.add_argument("--batch-size", type=int, default=8)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from qvelt.utils.manifests import manifest_digest

    if args.action == "demo":
        from qvelt.data.demo import create_demo

        manifest = create_demo(args.directory, args.repository, args.documents)
    elif args.action == "prepare":
        from qvelt.data.pipeline import prepare_data

        manifest = prepare_data(args.path, args.directory, args.repository)
    elif args.action == "download":
        from qvelt.data.download import acquire_source

        manifest = acquire_source(args.path, args.directory)
    elif args.action == "benchmark":
        from qvelt.data.benchmark import benchmark_loader

        return {
            "measurements": [
                benchmark_loader(
                    args.path,
                    workers=w,
                    batches=args.batches,
                    warmup=args.warmup,
                    batch_size=args.batch_size,
                )
                for w in args.workers
            ]
        }
    else:
        from qvelt.data.loader import PackedDataset
        from qvelt.tokenizer.workflow import load_tokenizer_bundle
        from qvelt.utils.artifacts import ArtifactStore
        from qvelt.utils.manifest_types import DataInfo

        manifest = ArtifactStore(args.path.parent).read_manifest(
            args.path.name, expected_sha256=args.expected_sha256
        )
        if (
            not isinstance(manifest.metadata, DataInfo)
            or manifest.metadata.representation != "token_ids"
        ):
            raise ValueError("Data verification requires a packed tokenized bundle")
        load_tokenizer_bundle(
            args.path.parent / "tokenizer/manifest.json",
            manifest.metadata.tokenizer_sha256,
        )
        # The layout is hash-checked by PackedDataset before selecting split names.
        train = PackedDataset(args.path)
        checked = 0
        for split in ("train", "validation", "test"):
            layout = json.loads((args.path.parent / "layout.json").read_bytes())
            if not any(split in parts for parts in layout["sources"].values()):
                continue
            dataset = train if split == "train" else PackedDataset(args.path, split)
            for i in range(len(dataset)):
                dataset[i]
                checked += 1
        return {
            "verified": True,
            "sha256": manifest_digest(manifest),
            "blocks_checked": checked,
        }
    return {"kind": manifest.kind, "sha256": manifest_digest(manifest)}
