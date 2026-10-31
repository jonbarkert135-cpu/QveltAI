# Qvelt experiment register

Last updated: 2026-10-31. Owner: Lead agent.

**Completed ML experiments: 0. Training runs: 0. Benchmark results: 0.**
The initial session created documentation only. Phase 1 adds source review and
analytical checks, not ML experiments. Their evidence is recorded below and in
`docs/research/validation.md`; no learning or GPU smoke result is claimed.

## R001 — Analytical size and resource validation (not an ML experiment)

- Date/owner/status: 2026-10-31; Lead agent; completed within analytical scope.
- Hypothesis: the selected four dense specifications have consistent unique
  parameter counts and dimensionally consistent FLOP/memory/cost calculations.
- Config: dataclasses in `docs/research/estimates.py`; baseline `db35482` plus
  Phase 1 local changes. No dataset/tokenizer/seed/accelerator required for pure
  integer enumeration and deterministic report generation.
- Control: explicit tensor-shape enumeration, a hand-counted 172-parameter case,
  dimensional unit cases and exact RFC totals, not a second model implementation.
- Commands: calculator, 30 unit/reference-arithmetic/CLI/manifest/mix checks,
  Ruff, formatting, strict mypy, byte comparison; full commands/outcomes and
  artifact hashes in `docs/research/validation.md`. All final checks exited 0.
- Result: 19,338,560 / 125,854,464 / 493,553,920 / 1,510,082,560 parameters;
  exact report regeneration. Initial test-arithmetic/type failures were corrected
  and retained in the validation account, not hidden as training failures.
- Conclusion: analytical contracts are internally consistent. They do not prove
  model correctness, learning, measured peak memory, throughput, or math quality.
  Next: implement the pinned config/package scaffold, then actual module oracles.
- Artifacts: committed research source, tests, JSON reports, source manifest and
  validation record. Paper caches in `/tmp` are not durable release evidence.

## R002 — Pinned package/config/runtime verification (not ML training)

- Date/owner/status: 2026-10-31; Lead agent; completed for M2.1 infrastructure.
- Hypothesis: the typed scaffold rejects ambiguous configs, reproduces CPU RNG
  streams, installs from a frozen lock and operates from a built wheel across
  the declared Python range without requiring CUDA.
- Code/config: `9348442`; five `configs/*.yaml` presets, exact lock/wheel hashes
  and commands in `docs/development/validation_m2_1.md`. No corpus/tokenizer/model
  artifacts or learning objective involved; RNG reference tests use fixed seeds.
- Control: hand-counted config parameters, independent research counts, standard
  Python/NumPy/torch RNG calls and a known 2x2 matmul checksum. Fresh no-torch
  environment validates config but correctly refuses a tensor diagnostic.
- Results: 121 tests passed on each of Linux/Python 3.11.14, 3.12.14, 3.13.5;
  strict mypy on 12 files and Ruff/format passed; frozen dependency checks, wheel/
  source build, installed-wheel CLI and conflicting-extra rejection verified.
- Initial schema typing/frozen-test issues were corrected without ignores; see
  the validation record for diagnoses. Mocked failures do not validate GPUs.
- Conclusion: M2.1 package/config/CPU runtime contract is operational. No model,
  dataset, GPU, benchmark, training or checkpoint-replay success is implied.
- Next: M2.2 typed provenance manifests and atomic integrity-checked artifacts.

## R003 — Manifest/integrity/interruption validation (not ML training)

- Date/owner/status: 2026-10-31; Lead agent; completed for M2.2 metadata/local IO.
- Hypothesis: strict manifest IDs and no-overwrite/manifest-last publication
  distinguish complete registered bytes from partial, corrupted or mismatched
  artifacts without deserializing models or requiring torch.
- Code/config: `9027efe`, `f4fd2cc`; unchanged lock. Typed synthetic metadata/
  opaque byte fixtures plus real temporary Git repositories; no approved corpus
  or neural checkpoint. Commands/hashes in `docs/development/validation_m2_2.md`.
- Controls: known SHA-256 vector, fixed artifact bytes, same-size corruption,
  wrong trusted ID, real competing writers and SIGKILL, injected IO failures,
  growing-file mutation and observed committed/dirty Git/package identities.
- Result: 220 tests passed on each of Linux/Python 3.11.14, 3.12.14, 3.13.5;
  strict mypy on 21 files, Ruff/format, dependency/build checks passed. Installed
  no-torch CLI created/inspected/verified a clean-code bundle; corrupted clone
  and wrong expected ID both failed with exit 1. Final run code was `f4fd2cc`.
- Diagnosis: an early growing-file test exposed an unbounded EOF-following loop;
  it was interrupted and the snapshot read was bounded. Parent-entry syncing
  was strengthened for concurrent creators. Final checks used corrected code;
  no assertions or verification were disabled.
- Conclusion: artifact metadata and raw-byte contracts pass within the documented
  Linux local-filesystem scope. No power-loss, GPU, distributed checkpoint,
  model replay, training or reasoning-quality result is established.
- Next: M3.1 byte-BPE workflow/tests on approved authored fixtures, then M3.2
  actual data-shard/packing/order semantics. Semantic training replay stays M5.

## Required record for an actual experiment

Create an entry when a command is actually run. Before launch, specify its
hypothesis, control, measurement, and affordable acceptance gate. Record:

- ID, date, owner, status (`planned`, `running`, `completed`, `failed`, or
  `inconclusive`), and linked milestone/decision.
- Hypothesis and what outcome would refute it; baseline/control where relevant.
- Exact code revision plus dirty-tree information, resolved config and hash,
  dependency versions, seed(s), device/precision/world size, and data/tokenizer
  revisions, licenses, and hashes.
- Working directory, complete command, expected steps/tokens/sample counts,
  resource limits, and log/artifact paths with checksums. Keep secrets out.
- Actual exit code, steps/tokens completed, test counts, failures/skips,
  measured metrics, uncertainty, and resource use when measured.
- Conclusion, limitations, reproducibility command, and next action. Retain
  negative results and interruptions; an interrupted run is not a completed run.

Do not use illustrative numbers as results or describe a runbook as an executed
experiment. Logs stored outside Git must have a persistent, documented location
before they are cited as release evidence.

## Queued experiments — not executed

| Queue ID | Hypothesis / purpose | Configuration needed | Acceptance | Dependency | Status |
| --- | --- | --- | --- | --- | --- |
| Q01 | A tiny portable decoder matches an independent numerical reference | M1.1 design, tiny FP32 CPU tensors, fixed seed, no external data | Forward/gradient errors within justified tolerances and causal tests pass | M4.1 | planned; no executable config yet |
| Q02 | The tiny training loop learns its deterministic fixture and resumes correctly | Tiny CPU training config, authored fixture, complete checkpoint state | Real updates, predeclared loss reduction criterion, matching next samples and supported update replay | M5.1 | planned; no executable config yet |
| Q03 | Two CPU ranks produce the reference global-token-weighted update | Fixed effective batch, unequal valid-token masks, Gloo, two processes | Distributed and single-process reference updates agree within justified tolerances | M5.2 | planned; no executable config yet |
| Q04 | The SFT loop learns only the intended assistant target tokens | Tiny CPU SFT config, authored chat/math fixture | Loss-mask oracle and real tiny update/resume checks pass | M6.1 | planned; no executable config yet |
| Q05 | The reasoning-RL loop can perform a correct bounded-reward update | Reviewed objective, tiny rollout config, authored math fixture, bounded verifier | Objective/gradient tests and actual rollout/update pass, including degenerate rewards | M6.3 | planned; no executable config yet |

Numerical thresholds and actual step counts must be chosen from the implemented
reference contract before these experiments run. None of these checks, even if
successful, would establish general mathematical reasoning quality.

## R004 — Byte-BPE fixture segmentation (engineering only)

2026-10-31; M3.1a, CPU, Python 3.12.14, tokenizers 0.22.2. Hypothesis:
byte-level training preserves authored Unicode and digit isolation changes
compression. Config: target384, min_frequency 2, three authored strings repeated
four times. Exact recipe/results: docs/development/tokenizer.md. Plain BPE actual
V381 and token counts9/9/16; isolated digits V371 and9/9/26, both full round trips.
252 tests passed in 4.59s (32 new), Ruff/format/mypy green on23 files. This is not
model training or evidence of reasoning quality. First attempt found untyped
third-party imports and a callback return annotation; both were addressed without
weakening tests. Further YAML/manifest/corpus validation remains M3.1.

## R005 — Manifest-linked tokenizer workflow and independent fixture replay

Date: 2026-10-31; code `2d1a05a`; engineering/tokenizer experiment, no LLM training.
Hypothesis: pinned ordered inputs/settings reproduce learned byte-BPE JSON across
independent supported Python processes; held-out whole-text byte coverage survives.
Config: configs/tokenizer_smoke.yaml,48 authored records, V=512, min_frequency 2;
plain versus ASCII-digit isolation,12 separate held-out records/390 bytes.
Result:315 tests pass on Python 3.11.14/3.12.14/3.13.5 with lint/types green.
Six installed-wheel CLI runs match JSON SHA-256 per policy despite varied seeds/
Rayon threads. All round trips pass. Plain316 tokens, isolated 321; both V=512.
Conclusion: M3.1 fixture workflow gate passed; no reasoning/production-quality
claim, no external dataset approval. Full commands, hashes, cases, limitations
and an early oversized test-ID/exec-environment harness failure: docs/development/validation_m3_1.md.
Reproduce using its workflow runbook and checked-in structured fixture results.

## R006 — Expanded foundation numerical and training acceptance (complete)

2026-10-31. User-requested real100-update CPU smoke of the RFC decoder with
explicit117056-parameter override, B2/T32,6400 synthetic targets. Initial actual
loss5.6883039474487305 to5.023181438446045; uninterrupted versus stop37/resume100
weights AND Adam states match exactly.356 tests passed with optional logging
extras,2 GPU cases deselected,45.25s. CPU DDP and actual DTensor/DCP sharding,
RNG restore and offline SDK outputs are exercised. These are engineering checks,
not pretraining/benchmark evidence. DCP stale-gradient restore-template and
interleaved rank stdout issues were fixed with actual regression tests.
Clean-clone/current-main CI gates pending; final evidence will be added to
validation_foundation.md. Early dirty-source smoke remains temporary evidence,
not the final reproducible release state. Original code uses Apache-2.0.

R006 final evidence: code 9ee9443; clean clone: 354 passed/2 optional skips/2 GPU
cases deselected,55.04s, Make/hooks/build green, clean tree. Actual GitHub
run 41242952575 succeeded in all five jobs; each Python CPU job354 passed, optional
SDK job 3 passed; uploaded smoke artifact 13058222452 exists. Actual clean-provenance
100-step/replay metrics match the early smoke. See validation_foundation.md for
commands, wheel/lock/manifest hashes and scope. Phase 3 data work now begins.
