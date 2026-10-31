# Data inputs and approval

`gsm8k_test.yaml` is the working, pinned **evaluation/indexing-only** acquisition
spec. It downloads749738 bytes/1319 question-answer rows, verifies the original
publisher MIT license and payload checksums, exports only questions and records
an explicit no-training usage policy. Keep it separate from all tokenizer/training
sources. Data is not stored in Git; approval and caveats are in DATA_CARD.md.

`candidates.yaml` is a **blocked research catalog**, not an executable recipe.
Its immutable card metadata do not approve corpus rights or supply nonexistent
Parquet/WARC converters. D01–D13 remain conditional/blocked as individually
explained in docs/research/data_mix.md. Once cleared, create an actual DownloadSpec
and a format-specific tested adapter. Do not pass the catalog to `data download`.

Generate a complete working offline preparation recipe (including actual source
and authored-holdout hashes) with `qvelt data demo`; see the exact commands in
runbooks/01_data_preparation.md. It uses original Apache-2.0 EN/RU math/code text,
not a public benchmark stand-in for production. A production recipe must supply
approved raw manifests and the complete planned benchmark registry. A generic
text-jsonl spec is supported only after filling real immutable URLs/checksums,
license evidence, max_bytes and explicit approval; no placeholder spec is shipped.
