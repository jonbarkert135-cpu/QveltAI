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

## Scope at the initial implementation checkpoint

The initial752a6ca correctness checkpoint preceded the final lifecycle fix and
measurements below. Its remote3.13 resource failure is recorded in R007; it is not
the final accepted implementation. GPU/NCCL, actual i5 laptop performance, corpus-scale
I/O/LSH/LID/PII audits, a trained quality classifier and packed-training optimizer
resume are outside this validation. The implementation is ready for those measured
checks but is not represented as a production-approved training corpus.

## Clean-code measured evidence

Final implementation/measurement revision:
`9f6cef941ceccf3c5eb929256e408a7644b65896` (clean at measurement).
Lock SHA256 `0da2d2e3aa43780f7a4d01cdf5f125704a1e329ece8287882a5ccd3150e75314`.
Machine-readable original receipt:
[evidence/phase3-data-9f6cef9.json](evidence/phase3-data-9f6cef9.json).
Exact hypotheses/configuration/results/conclusions: EXPERIMENTS R008–R010.

The digit study fits768 original math texts and measures384 new-seed EN/RU texts:
12700 individual-digit versus9454 grouped-digit tokens (-25.559055%),0 round-trip
failures. Actual vocab509/1024 at a1024 ceiling. No reasoning gain follows from
compression. Default individual digits remains a controlled baseline; both policies
are supported. Measured host batch delivery:51315/64540 valid targets/s for0/2
workers, with startup0.0151/3.8188s, B8/T64,8 warmup/64 measured batches and a
cached83-block corpus. No GPU/laptop/corpus-scale performance claim.

Preparation measured2.886784s,404 input documents/374 retained,18098 stored tokens,
17724 targets,483224 prepared bytes/599036 total demo file bytes. The566396-byte
scratch snapshot excludes anonymous buffers and is not peak storage. Runbook disk/
time allowances for the i5 laptop are explicitly planning assumptions.

Lifecycle regression evidence: contextlib.closing releases SQLite on both success
and interruption; Python3.13 focused data suite72 passed in20.73s. Shared-Tensor
pending-prefetch shutdown abort was reproduced under a stronger test; workers now
collate plain NumPy IPC and the consumer converts tensors. The test abandons the
live prefetched iterator before draining it and compares replay with a separate
uninterrupted oracle. Warnings remain errors; none were suppressed.

Actual approved GSM8K questions:1319 indexed into55284 unique13-grams. Protected
copy detected, unrelated control clean; receipt
[evidence/phase3-gsm8k-index.json](evidence/phase3-gsm8k-index.json).
The adapter indexes questions only, not gold solutions. Other planned public and
frozen project-final coverage remains missing; production is blocked.


## Final clean-clone acceptance

Fresh validation checkout from the9f6cef9 commit, separate virtual environment:
`make setup lint typecheck test`, `uv lock --check`, `uv pip check` and `make build`.
Actual full suite:426 passed,2 optional SDK skips,2 GPU deselections in64.72s.
Ruff passes and mypy checks82 files; dependency validation reports50 installed
packages compatible. Build result and remote CI completion are recorded below.
No tracked source is changed by the checks. A prior clean752a6ca checkout passed
421 CPU tests on3.12, but its3.13 CI leak was corrected before final acceptance.

Final data smoke receipt:
[evidence/phase3-smoke-9f6cef9.json](evidence/phase3-smoke-9f6cef9.json).
212 inputs/194 retained,153 packed blocks verified; cursor3 restores7 tensor-exact
batches to step10, with discarded_prefetched_iterator=true. It remains explicitly
production_ready=false. The existing numeric-fixture100-step trainer is separate.

Build succeeded for wheel and sdist, and the fresh checkout remained clean.
Actual [GitHub run41247846988](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/41247846988)
completed successfully for all6 jobs at9f6cef9. CPU3.11/3.12/3.13 each report426
passed/2 optional skips/2 GPU deselections (38.88/47.77/58.86s). Optional offline
metrics3 passed in5.11s. Data and numeric-training smoke artifacts are uploaded;
IDs13059896976 (129914 bytes) and13059827229 (6749990 bytes), respectively.
Machine-readable job/step evidence:
[evidence/phase3-ci-9f6cef9.json](evidence/phase3-ci-9f6cef9.json).

The direct training-job log confirms100 actual updates/6400 numeric targets,
117056 parameters, first/last loss5.6883039474487305/5.023181915283203 and exact
stop37/resume100 weights/Adam replay. The GitHub CLI merged-log display associated
a data-smoke payload with a training-step label; the direct job endpoint and
separate named artifacts resolve the discrepancy. No merged label alone is treated
as proof that a training run executed. Remaining production/hardware limitations
are unchanged and explicit in STATE/RISKS; no final model or public score exists.
