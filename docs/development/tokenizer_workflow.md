# M3.1 workflow design

Status: complete for the M3.1 fixture workflow. Owner: Lead agent. Date: 2026-10-31.

Use the existing strict YAML parser, byte-BPE trainer, typed manifests, local
atomic store and observed provenance. No new dependency or model code is needed.

- `TokenizerRecipe`: schema 1, portable name, training-data canonical manifest
  SHA-256, immutable TokenizerSettings. CLI paths are supplied separately so a
  resolved recipe has no machine-specific paths. Unknown/duplicate YAML keys,
  implicit coercion, anchors and future versions are rejected.
- `train_tokenizer_bundle(recipe, data_manifest, directory, repository)` consumes
  raw-data manifest files with role data as strict UTF-8 JSONL records containing
  only text. File order is the schema's lexicographic path order; record order is
  physical line order. Bound each line/document, hash bytes actually read, reject
  changed files/count mismatches before committing any tokenizer manifest.
- Store tokenizer JSON, resolved recipe, canonical raw-data manifest, copied
  dependency lock and observed local code/package provenance. The manifest is
  published last. An interrupted/failed operation may leave an uncommitted
  directory; never reuse or automatically delete unknown outputs.
- `load_tokenizer_bundle` verifies bytes, supported tokenizer semantics, recipe
  and raw-manifest identity linkage. Only a matching supplied expected metadata
  ID provides an external integrity anchor. Hashes are not signatures, legal
  approvals or proof that training on declared data actually occurred.
- CLI `tokenizer train`, `verify`, `evaluate` expose implemented operations only.
  Evaluation on a separate raw-data manifest measures byte/character/token counts
  and whole-string round trips; it does not evaluate reasoning or data quality.
- Registered authored train/heldout fixture JSONL files and manifests enable a
  complete offline smoke recipe. They are software fixtures with explicit limited
  permission for repository development/tests, not a model-training corpus.

Acceptance: malformed input/corruption/lineage mismatch/partial writes fail;
independent processes with the pinned release produce matching tokenizer bytes
on the fixture; save/load/token IDs, heldout Unicode/LaTeX/numerals pass. Run
first-party pytest/Ruff/mypy and supported CPU Python environments, and record
actual results. Cross-hardware/version equality is measured, never universal.

The source directory is trusted local storage, as in M2.2; no hostile filesystem
sandbox or network-store guarantee. BPE pair-statistic memory is not bounded by
streaming. Production corpus selection, chat serialization and M3.2 preparation
remain separate gates. All affected surfaces: config parsing, tokenizer API,
artifact reads, CLI, fixture/data cards, tests, CI/runbooks and project state.

## Offline CLI runbook

From a Qvelt Git checkout with the frozen environment. Outputs require a new
root under an existing parent; do not reuse or delete previous artifacts blindly.

```bash
uv sync --frozen --extra cpu
qvelt_demo_root="$(mktemp -d)"
uv run --frozen --extra cpu qvelt tokenizer resolve configs/tokenizer_smoke.yaml
uv run --frozen --extra cpu qvelt tokenizer train configs/tokenizer_smoke.yaml --data-manifest tests/fixtures/tokenizer/train.manifest.json --directory "$qvelt_demo_root/tokenizer" --repository .
uv run --frozen --extra cpu qvelt tokenizer verify "$qvelt_demo_root/tokenizer/manifest.json"
uv run --frozen --extra cpu qvelt tokenizer evaluate "$qvelt_demo_root/tokenizer/manifest.json" --data-manifest tests/fixtures/tokenizer/heldout.manifest.json --data-sha256 0f84b8ef256117e11ca0dd6f94bef4d006fb78692a257c11bca10964f2258be4
```

Train prints the canonical tokenizer-bundle SHA-256; pass it using
`verify/evaluate --expected-sha256` when enforcing that creator-provided identity.
The bare verify command checks internal integrity/semantics only. No torch is
required by tokenizer operations; the base no-dev profile was independently tested.
The demo consumes only registered authored fixtures. External data still requires
the DATA_CARD approval gate; there is no downloader or automatic legal approval.

Recipe keys: schema_version1, portable name, training_data_sha256 and tokenizer
mapping (vocab_size, min_frequency, isolate_digits). Paths stay outside the YAML;
resolved defaults affect its canonical recipe digest. Vocabulary is a ceiling.

Raw data must have representation raw and registered .jsonl files with role data.
Each physical line contains exactly {"text": "..."}; escaped multiline text and
empty strings are valid, physical blank lines are errors. A final newline is
optional. Limits: decoded UTF-8 text 1 MiB, raw JSONL record 6 MiB+128 bytes.
All records, including empty strings, count toward manifest documents. At least
one nonempty training document is required. Files are checked before iteration
and their actual consumed bytes are checked again before successful exhaustion.

The new bundle contains tokenizer.json, recipe.yaml, training-data.manifest.json,
provenance.json and uv.lock; top-level manifest.json is the final commit marker.
The copied data manifest retains file/source identities; raw corpus files are not
copied into the tokenizer output. Loading verifies its lineage metadata without
requiring the original corpus to remain available. Dataset payload validity is
checked during training/evaluation, not reconstructed from a stored declaration.

Provenance records full local HEAD/dirty/lock and installed package observations,
plus canonical recipe hash. It does not assert environment/lock equivalence,
checkpoint replay, or a signed proof of how an arbitrary supplied tokenizer was
trained. Generic `manifest verify` checks bytes; `tokenizer verify` additionally
checks the supported BPE pipeline, controls, vocabulary, recipe/data/provenance
links. A metadata-only M2.2 tokenizer fixture is not a complete M3.1 workflow bundle.

Fixture segmentation and actual three-version results:
[validation report](validation_m3_1.md). Corpus-scale preparation, dedup/splits,
shards and packing follow in M3.2; production tokenizer training needs approved
representative data and measured memory, not this tiny smoke vocabulary.
