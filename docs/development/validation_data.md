# Phase3 data validation

Date:2026-10-31. Scope: offline original text curation/packing/loader correctness,
not production-corpus training or model reasoning quality. Design and limitations:
[data_design.md](data_design.md); exact steps:
[runbook01](../../runbooks/01_data_preparation.md).

## Executed local correctness checks

- `make setup` installs the frozen CPU lock, including tokenizers0.22.2,
  datasketch1.6.5, langdetect1.0.9, SymPy1.14.0 and PyTorch2.9.1+cpu.
- `make lint` passes Ruff lint/format; `make typecheck` passes82 first-party/script/
  test/research files. `make test`:421 passed,2 absent optional TensorBoard/W&B
  SDK tests skipped,2 GPU tests deselected in65.10s. No skipped test is a pass.
-67 focused Phase3 tests passed in19.29s: raw/metadata hashes/alignment, strict
  recipes, control V1/V2 compatibility, Unicode/code/LaTeX/RU round trips, exact
  math references/bounded rewards, PII/heuristics/LID, SQLite exact/near candidates,
  numeric-atom protection, all-registry n-grams/production refusal, storage dtype/
  pair oracles, masks, ranks, stages, RNG isolation and cursor checksums.
- Real EN/RU workflow builds actual BPE/shards. Rebuild has identical tokenizer
  JSON and binary payload hashes. Descriptive timestamps/provenance/timings can
  change whole-bundle identities and are not represented as deterministic bytes.
- Prefetched consumer cursor replay is tensor-exact for workers2->0 and0->2,
  including labels/positions/masks/IDs, across a stage transition. Source counts,
  topology/batch/data/attention/seed/schedule identity are checked; topology
  migration is rejected. A packed BPE batch executes the real decoder backward.
- `pre-commit run --all-files`: lint/format and strict type hooks pass.

## External source evidence

GSM8K test publisher snapshot and MIT license were read/recorded before acquisition;
see DATA_CARD.md. Actual pinned CLI download succeeds,749738 bytes/1319 original
rows, question-only raw manifest canonical SHA256
`5f5fb9b48439db1928555aa5e19eeff6e4b4d0efe96f9df78b578ac5b892ba84`.
The explicit raw usage policy prevents training. No benchmark contents are tracked.
This verifies acquisition/rights scope, not a public benchmark score or complete
public-data decontamination. Production also requires MATH/AIME/MMLU and the frozen
project-final heldout, whose approvals/snapshots are incomplete.

## Remaining acceptance observations at this checkpoint

The final clean-code experiment receipts and GitHub Actions results will be
recorded below after execution. No numeric digit-policy/fertility/throughput result
is asserted at this checkpoint. GPU/NCCL, actual i5 laptop performance, corpus-scale
I/O/LSH/LID/PII audits, a trained quality classifier and packed-training optimizer
resume are outside this validation. The implementation is ready for those measured
checks but is not represented as a production-approved training corpus.
