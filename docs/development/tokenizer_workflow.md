# M3.1 workflow design

Status: implementation in progress. Owner: Lead agent. Date: 2026-10-31.

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
