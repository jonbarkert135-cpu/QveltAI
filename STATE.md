# Qvelt current state

Updated:2026-11-01. Owner:Lead agent.

## Current task

M5.1 packed-data trainer and CPU-DDP extensions are implemented locally. Real BPE
inputs use valid-token gradient accumulation/schedule, heldout probe and atomic
actual-manifest/model/Adam/RNG/per-rank-consumer checkpoints. Local full suite521
passed/2 optional SDK skips/4 GPU deselections in133.66s before the last added
FP32-factory regression and supplied GPU test. Latest targeted33 passed/1 GPU
deselected; two-rank SDPA+checkpoint replay/oracle passed. Ruff/mypy97 files pass.
Next:publish code, run clean20-step actual-BPE smoke/fresh clone and observe7-job CI.
M4.1/available-CPU M4.2 are complete; CUDA/FSDP/shared-store and production input
approvals remain blocked/unvalidated. No paid/final training. Publication active.

## Implemented and verified

- Byte-BPE default vocab ceiling32768; V2 reasoning/tool controls, old V1 loading,
  exact Unicode/code/LaTeX/RU round trips. Original exact EN/RU math with bounded
  rational rewards and independent SymPy/stdlib checks; no teacher/public training data.
- Strict source recipes, license/usage gates and actual pinned GSM8K eval-only
  downloader. Seeded LID/heuristics/PII patterns/phrase policy, disk exact+math-aware
  MinHash-LSH, family splits, train-only BPE and all-registry n-gram exclusion.
- Compact uint16/uint32 shards, document-pair packing/masks and optional cross-doc
  attention, weighted pretrain/anneal batch schedules, readonly memmaps and exact
  consumer cursor replay across0/2 workers. Pending prefetch is actually abandoned;
  SQLite closes on success/error; worker NumPy IPC avoids the reproduced shutdown issue.
- Fresh clone:Make setup/lint/types/test/build, lock/dependency checks pass;
  426 passed/2 optional SDK skips/2 GPU deselections in64.72s; mypy82 files.
  All3 remote CPU versions also pass426; optional metrics job passes3 tests.
- Offline smoke:212 inputs/194 retained,153 blocks verified, cursor3->10 with7
  tensor-exact replay batches. New-seed digit study:12700 individual/9454 grouped
  tokens (-25.559055%),0 round-trip failures. No trained-math result.
- Cloud host delivery B8/T64:51315/64540 valid targets/s with0/2 workers;
  startup0.015/3.819s, small cached/reused fixture. Laptop/GPU not measured.
- Earlier engineering foundation remains:actual RFC decoder/counts/references,
 100-update numeric-fixture trainer with exact Adam/weight replay, two-rank CPU
  DDP and DCP/DTensor tests, atomic checkpoints/provenance and optional metrics.

Evidence:docs/development/validation_data.md, committed evidence JSON, EXPERIMENTS
R007–R010, runbooks/01_data_preparation.md. Foundation evidence remains separate.

## Open limits

No approved external training corpus. GSM8K questions were acquired/indexed only;
MATH/AIME/MMLU and the frozen project-final evaluation snapshots/rights are missing.
Production preparation rejects incomplete coverage. Gold GSM8K solutions are not
indexed by its question adapter. Lexical exclusion/LSH/PII/toxicity/LID are incomplete
and need corpus-specific audits; no trained quality classifier is provided.

HF BPE pair-statistics RAM can grow with corpus diversity. Affine epoch shuffles
are bijections, not uniform N! permutations. Loader replay is tested separately;
the existing trainer still uses numeric fixtures and does not yet atomically
checkpoint the real-data loader. GPU/NCCL/FSDP2/Slurm/multi-node/shared-store/native
Windows, actual i5/16GB laptop performance, SFT/RL and serving remain unvalidated.

## Next three tasks

1. M5.1:connect real BPE packed batches, valid-token accumulation/schedule and
   per-rank consumer/RNG/model/optimizer state in one atomic checkpoint.
2. M3.3:clear production sources/benchmarks, freeze the distinct project final set,
   add rights-approved format adapters and corpus-scale labeled filtering audits.
3. Verify short-window/uneven-mask normalization, heldout validation and exact
   optimization replay, then extend actual-data DDP with independent rank oracles.

## Reproduce

```bash
make setup
make lint
make typecheck
make test
uv run --frozen --extra cpu python scripts/data_smoke.py --directory /tmp/qvelt-data-check-01 --repository .
uv run --frozen --extra cpu qvelt data benchmark /tmp/qvelt-data-check-01/prepared/manifest.json --workers 0 2
```

Choose a new output path. uv0.12.23; Linux/Python3.11–3.13, CPU default. The exact
preparation commands/disk and explicitly assumed laptop times are in runbook01.
