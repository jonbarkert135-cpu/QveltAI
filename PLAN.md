# Qvelt master roadmap

Last updated: 2026-11-01. Owner of every milestone: **Lead agent**.

## Mission and boundaries

Build an original, reproducible LLM framework with particular emphasis on
mathematical reasoning: tokenizer → data → model → distributed pretraining →
SFT/preferences/reasoning RL → evaluation → inference and export.

Target model tiers are approximately 100M parameters for development, followed
by 500M–1.5B. Larger sizes are a configuration and systems goal, not a committed
training run. Quality must be established experimentally; no performance targets
are presented as achieved. No training servers or budget are currently supplied.

Required execution tiers are tiny CPU correctness, single-GPU development,
single-node multi-GPU training, and an extensible multi-node design. Real GPU
and multi-node validation require actual hardware. Tiny CPU execution does not
establish that full-sized models are practical on weak devices.

## Status and acceptance conventions

- `complete`: acceptance criteria have evidence in `STATE.md` or `EXPERIMENTS.md`.
- `in progress`: work has begun but the gate has not passed.
- `planned`: no implementation or verification claimed.
- `blocked`: a named external prerequisite prevents completion.
- All tasks use checkboxes. A milestone requires every task and its acceptance
  gate to pass; a skipped required check is not a pass.
- Every new module includes unit tests; numerical modules include independent
  reference tests. After M2.1, required checks include Ruff and mypy.
- Each milestone ends with updated state/decisions, evidence, and a reviewable
  commit. Gates apply only to their implemented scope, not the whole roadmap.

## Baseline: documentation bootstrap

### M0 — Establish project memory

Owner: Lead agent. Status: complete. Dependencies: repository inspection.

- [x] Inspect the checkout, Git state, instructions, and existing project files.
- [x] Create the roadmap, state, decision, experiment, risk, and data registers.
- [x] Record hardware limits, research uncertainty, and the no-model-code boundary.
- [x] Verify document structure and local links; preserve a clean committed baseline.

Acceptance: the documents agree on current capabilities and next steps; the
initial delivery contains no model or training code and no invented results.
Evidence: `STATE.md`, documentation checks recorded on 2026-10-31.

## Phase 1 — Research and architecture decisions

### M1.1 — Establish an evidence-backed architecture proposal

Owner: Lead agent. Status: complete. Dependencies: M0.

- [x] Read and version primary sources for a dense causal-decoder baseline,
  normalization, positional encoding, MLP activation, attention, and initialization.
- [x] Compare architecture candidates in `DECISIONS.md`; specify tensor shapes,
  residual paths, masking, loss, special tokens, and supported context lengths.
- [x] Derive parameter counts for tiny, approximately 100M, and 500M–1.5B tiers;
  distinguish embedding, non-embedding, and tied parameters.
- [x] Choose numerical reference paths and optional optimized attention paths;
  document prerequisites, licensing, precision, and fallback behavior.

Acceptance: an approved-by-evidence design and interface document exists with
source status for each substantive choice, explicit equations, and no unresolved
choice that would make the first model implementation ambiguous. No architecture
is selected merely because a candidate is listed in `DECISIONS.md`.

### M1.2 — Define resource, reproducibility, and math-evaluation contracts

Owner: Lead agent. Status: complete. Dependencies: M1.1.

- [x] Specify tokens, optimizer updates, global/per-device batch semantics,
  sequence lengths, precision, and a reproducible run-manifest schema.
- [x] Estimate weights, gradients, optimizer state, activation, and checkpoint
  memory with stated assumptions; provide CPU/tiny and GPU capability tiers.
- [x] Define a fixed mathematical validation protocol, held-out split policy,
  answer extraction rules, contamination checks, and evidence required for claims.
- [x] Separate exact same-topology resume from cross-topology migration;
  define supported deterministic mode and explicit failure conditions.

Acceptance: a documented resource matrix and testable training/evaluation
contracts exist. Estimates are labeled estimates; no paid or large run is
scheduled without the user's hardware details and budget approval.

## Phase 2 — Repository scaffold

### M2.1 — Establish a pinned, typed, tested Python package

Owner: Lead agent. Status: complete. Dependencies: M1.2.

- [x] Create package layout, `pyproject.toml`, compatible dependency pins/lock,
  Python support policy, Ruff, mypy, and pytest configuration.
- [x] Implement strict YAML/dataclass configuration loading, validation, resolved
  configuration output, seed utilities, logging, and command entry points.
- [x] Add unit tests for config errors, round trips, seeds, and public interfaces;
  establish lightweight CPU CI using the chosen dependency versions.

Acceptance: installation from the lock works in a fresh environment; configured
tests execute nonzero tests, Ruff and mypy pass, and invalid configs fail clearly.
No CLI advertises commands that have no working implementation.

### M2.2 — Define artifacts and run provenance

Owner: Lead agent. Status: complete. Dependencies: M2.1.

- [x] Define versioned run, data, tokenizer, and checkpoint manifests with hashes,
  code revision, dirty-tree state, environment versions, and artifact ownership.
- [x] Implement atomic local writes, schema checks, and corruption/error tests.
- [x] Keep generated artifacts outside tracked source and exclude credentials.

Acceptance: manifest round trips, incompatible versions, interrupted writes,
and corrupted artifacts are tested; another process can inspect a run's identity.

Evidence: `docs/development/validation_m2_2.md`; schema/identity and real local
IO tests, clean-commit CLI bundle verification/corruption detection, CPU Python
matrix. Completion covers metadata and byte integrity, not model state replay.

## Phase 3 — Tokenizer and data

### M3.1 — Train and validate a mathematical-text tokenizer

Owner: Lead agent. Status: complete. Dependencies: M2.2.

- [x] Select a tokenizer algorithm/library after source and license review;
  train it from scratch on a small approved corpus before scaling.
- [x] Specify Unicode handling, byte fallback, special/chat tokens, vocabulary
  size, and number/LaTeX handling; measure candidate segmentation rather than
  assuming that digit splitting improves reasoning.
- [x] Test round trips on English, Cyrillic, numbers, equations, and malformed or
  unusual Unicode; verify stable IDs and save/load compatibility.

Acceptance: a versioned tokenizer artifact and training recipe are reproducible;
coverage and tokens-per-text measurements are recorded with no math-quality claim.

Evidence: M3.1a library API plus M3.1 YAML/CLI bundle workflow; code `2d1a05a`,
`docs/development/validation_m3_1.md` and authored fixture results. 315 tests pass
on Linux/Python 3.11/3.12/3.13; six installed-wheel training processes produce
matching tokenizer bytes per policy. No production 32768-token artifact or
external-corpus quality claim. Corpus-scale selection/training remains gated
by DATA_CARD and measured resources; M3.2 is next.

### M3.2 — Build deterministic, auditable data preparation

Owner: Lead agent. Status: complete (offline framework acceptance). Dependencies: M3.1.

- [x] Approve each consumed dataset revision/license in `DATA_CARD.md`; retain provenance,
  source IDs, transformation rules, and redistribution restrictions.
- [x] Implement bounded-memory ingestion, filtering, exact/near deduplication,
  stable document-level splits, decontamination, and configured data mixtures.
- [x] Implement checksummed token shards, causal sequence packing with explicit
  document-boundary policy, loss masks, deterministic shuffling, and resumable
  rank/worker allocation without accidental loss or duplication.
- [x] Test ordering, restart positions, shard corruption, label shifting,
  packing boundaries, and split isolation with local synthetic fixtures.

Acceptance: rebuilding fixture shards produces matching hashes; interrupted
iteration resumes with the documented sample order; tokenizer/data versions and
licenses are traceable from the run manifest. No unapproved corpus is consumed.
Measured digit/fertility/host-delivery evidence, offline CI and the preparation
runbook close expanded Phase3 acceptance; production source/benchmark approval
remains a separately blocked prerequisite for actual corpus training.

Evidence:code9f6cef9; actual GitHub run41247846988 (6 successful jobs), fresh
clone426 passed/2 optional skips/2 GPU deselections,82 typed files, exact binary
rebuild/pending-prefetch replay, EXPERIMENTS R007–R010 and validation_data.md.
Measured768/384-document digit study and0/2-worker host throughput are recorded;
runbooks/01_data_preparation.md is runnable offline. This is not production-corpus
approval or real-data optimizer replay.

### M3.3 — Clear and validate production-scale inputs

Owner:Lead agent. Status:blocked. Dependencies:M3.2, exact source/benchmark rights,
actual data/storage resources and the frozen project-final evaluation contract.

- [ ] Approve complete MATH/AIME/MMLU and project-final exclusion snapshots;
  retain exact questions/solutions/derivatives where permitted and test coverage.
- [ ] Approve actual training-source revisions (D01–D13 remain blocked), implement
  tested format-specific Parquet/WARC/code-provenance adapters only when needed.
- [ ] Calibrate LSH/math-atom/PII/LID/toxicity heuristics on labeled EN/RU/domain
  slices; develop a licensed learned quality classifier using the documented protocol.
- [ ] Profile representative source/tokenizer samples for16GB RAM/SSD and future
  server I/O; freeze production data/tokenizer identities before final training.

Acceptance:production recipe succeeds against every planned approved exclusion
set, while removal audits/rights/size/resource evidence and residual semantic
leakage risks are explicit. No flag bypasses missing licenses or mandatory sets.
GPU saturation and atomic packed-training replay are M5 hardware/trainer gates.

## Phase 4 — Model implementation

### M4.1 — Implement the portable decoder and numerical oracles

Owner: Lead agent. Status: in progress. Dependencies: M3.2.

- [x] Review the existing M2.3 decoder against RFC-001 and the Phase3 packed-data
  contract; record inspected components, rerun references and remaining gaps.
- [x] Implement the M1.1 architecture in small typed modules: embeddings,
  normalization, position handling, attention, MLP, blocks, output head, and loss.
- [ ] Add independent small-tensor reference comparisons for values and
  gradients; test causality, shapes, initialization, padding, loss masking,
  non-finite inputs, and configuration errors.
- [ ] Verify shifted next-token targets and global valid-token loss accounting;
  enumerate actual parameter counts for each size configuration.

Acceptance: tiny FP32 CPU forward/backward and all reference tests pass with
justified tolerances. Tests demonstrate no future-token leakage; declared model
sizes match the implemented parameter counts.

Review evidence (2026-11-01):docs/development/decoder_rfc_review.md;18 existing
CPU reference/integration tests pass. Dedicated initialization, non-finite policy
and tied-embedding combined-gradient regressions remain open in M4.1; the audit
does not close the entire milestone.

### M4.2 — Add measured accelerator and decoding paths

Owner: Lead agent. Status: planned. Dependencies: M4.1.

- [ ] Integrate supported optimized attention and mixed precision behind runtime
  capability checks; retain a usable portable path.
- [ ] Implement KV-cache prefill/incremental decode and activation checkpointing;
  test cached versus uncached results and recomputation gradients.
- [ ] Profile memory and throughput on available devices with warmup,
  synchronization, shapes, dtypes, and tool versions recorded.

Acceptance: portable tests pass; available accelerator paths pass numerical and
gradient checks. Unsupported hardware paths remain explicitly unverified and
cannot be advertised as tested. Optional compilation requires separate evidence.

## Phase 5 — Training framework

### M5.1 — Make single-process pretraining correct and resumable

Owner: Lead agent. Status: planned. Dependencies: M4.1; M4.2 for GPU fast paths.

- [ ] Implement optimizer parameter groups, token-based schedule, accumulation,
  precision controls, gradient clipping, validation, and finite-value handling.
- [ ] Save/restore model, optimizer, schedule, precision-scaler state when used,
  every relevant RNG stream, per-rank data cursor, resolved config, and counters.
- [ ] Test uneven loss masks and partial accumulation windows against a
  reference global batch; test interruption, invalid resume, and corruption.
- [ ] Run tiny CPU training on a deterministic learnable fixture; record loss
  behavior and compare uninterrupted versus resumed steps under fixed conditions.

Acceptance: real optimizer updates and fixture learning are logged; resume
reproduces the next samples and updates within the declared contract. Unit tests,
Ruff, and mypy pass. This is a smoke test, not evidence of general math ability.

### M5.2 — Validate distributed correctness and add sharding

Owner: Lead agent. Status: planned. Dependencies: M5.1, M4.2.

- [ ] Establish two-process CPU/Gloo correctness tests, then one-node CUDA/DDP;
  use explicit rank, device, world-size, and valid-token normalization contracts.
- [ ] Compare distributed updates with a single-process reference, including
  unequal valid-token counts, accumulation, rank-aware RNG, and shard boundaries.
- [ ] Select a version-supported sharding interface after API verification;
  test checkpoint save/load and document topology-change limitations.
- [ ] Add multi-node launch/rendezvous runbooks, failure/timeout behavior, and
  restart procedures; validate on real hardware when supplied.

Acceptance: CPU multiprocess tests pass; each claimed GPU/sharding/multi-node
tier has real logged tests. Unavailable hardware blocks that tier's acceptance,
not progress on independent portable work. A runbook alone proves no scalability.

### M5.3 — Deliver user-operated pretraining recipes

Owner: Lead agent. Status: planned. Dependencies: M5.2, approved training data.

- [ ] Provide measured dry runs and memory budgets for approximately 100M and
  500M–1.5B configurations, then larger configurable tiers as supported.
- [ ] Write token-budget, data-mixture, instability-recovery, checkpoint-storage,
  monitoring, cost-estimation, and launch/stop runbooks.
- [ ] Define acceptance gates from tiny → debug-scale → larger training; require
  user authorization before costly runs. Do not train final models autonomously.

Acceptance: tested smoke commands and explicit prerequisites accompany every
recipe; estimates and user-run results are separated. No full training run is
marked complete without its actual configuration and logs.

## Phase 6 — Post-training and mathematical reasoning

### M6.1 — Implement SFT and mathematical supervision

Owner: Lead agent. Status: planned. Dependencies: M5.1; trainable checkpoint;
approved supervision data. Distributed SFT additionally depends on M5.2.

- [ ] Implement versioned chat formatting, assistant-only loss, packing/truncation,
  and prompt/response boundary tests; preserve held-out problem families.
- [ ] Support licensed mathematical solutions and verified synthetic examples
  with teacher/model provenance and permitted-use records.
- [ ] Run a tiny SFT overfit and resume test; record token counts and masked loss.

Acceptance: formatting/mask reference tests pass, a real tiny SFT run learns its
fixture, and its held-out evaluation is reported without contamination claims
based solely on an empty exact-match search.

### M6.2 — Implement and verify preference optimization

Owner: Lead agent. Status: planned. Dependencies: M6.1; approved preference data.

- [ ] Compare supported preference objectives and select one with verified
  primary-source equations; DPO is a candidate, not an implementation promise.
- [ ] Implement pair validation, response log-probability accounting, reference
  policy semantics, and any reference-cache identity checks.
- [ ] Test the objective and gradients against hand-computed/reference cases;
  run a tiny training and resume test with fixed pairs.

Acceptance: objective tests and actual optimizer updates pass; length effects,
reference model cost, and data limitations are recorded. No preference-win claim
is made without a measured, held-out evaluation.

### M6.3 — Implement verifiable-reward reasoning RL

Owner: Lead agent. Status: planned. Dependencies: M6.1, M6.2, M1.2 evaluation
contract, tested generation, and a reviewed verifier.

- [ ] Select a reasoning-RL objective from verified sources; evaluate on-policy
  rollout, grouping, KL/regularization, reward normalization, and memory costs.
- [ ] Build a bounded mathematical answer verifier with rational/numeric cases,
  explicit tolerances, timeouts, and adversarial tests. Never evaluate arbitrary
  generated Python in the trainer process.
- [ ] Implement rollout/policy identity, token masks, truncation handling,
  reward diagnostics, deterministic fixtures, and resumable optimization.
- [ ] Run a tiny end-to-end RL update; test degenerate/zero-variance groups,
  malformed answers, non-finite rewards, and reward-hacking examples.

Acceptance: verifier and objective reference tests pass; a logged tiny rollout
and update completes. Improvement requires fixed-budget held-out comparisons;
longer reasoning traces alone are not evidence of better reasoning.

## Phase 7 — Evaluation

### M7.1 — Deliver a reproducible, contamination-aware evaluation harness

Owner: Lead agent. Status: planned. Dependencies: M6.3 for full post-training
comparison; basic fixture evaluation is required earlier by M5.1 and M6.1.

- [ ] Version benchmark datasets, licenses, prompts, answer extraction, context
  limits, seeds, generation budgets, and metric implementations.
- [ ] Implement exact/numeric/verifier-backed math scoring with independent
  fixtures; add language-model loss/perplexity and general capability controls.
- [ ] Audit training/evaluation overlap with exact and approximate methods;
  record contamination uncertainty and split/problem-family limitations.
- [ ] Report problem counts, failed/invalid generations, pass@1 and any sampled
  metrics with their precise estimator, sample budget, and uncertainty.

Acceptance: fixture metrics match known answers, reruns are reproducible under
declared conditions, and every score resolves to artifacts, prompts, and a
specific checkpoint. No hidden-test or unsupported benchmark claim is published.

### M7.2 — Establish evidence-based promotion and ablation gates

Owner: Lead agent. Status: planned. Dependencies: M7.1; actual user-run checkpoints.

- [ ] Compare base, SFT, preference, and reasoning-RL checkpoints with fixed
  budgets and comparable prompts; include data/compute/seed controls.
- [ ] Run affordable ablations for architecture/data/math choices; distinguish
  exploratory observations from repeated improvements.
- [ ] Require regression checks for reasoning, general text quality, robustness,
  verifier exploitation, and inference resource use before promotion.

Acceptance: model promotion is justified by reproducible measured results, with
negative findings and uncertainty retained in `EXPERIMENTS.md`.

## Phase 8 — Inference and export

### M8.1 — Build correct generation and stable artifact export

Owner: Lead agent. Status: planned. Dependencies: M7.1, M4.2; early generation
used by M6.3 must already have reference tests.

- [ ] Provide greedy and seeded sampling, stopping rules, context checks,
  batching, streaming, KV-cache lifecycle, and per-request resource limits.
- [ ] Export weights, tokenizer, config, and metadata in a reviewed versioned
  format; test reload/logit parity and reject incompatible or corrupt artifacts.
- [ ] Add quantization or external-runtime compatibility only after capability,
  license, accuracy, and memory validation; do not promise universal support.

Acceptance: cached/uncached and export/reload reference comparisons pass;
generation reproducibility has explicit device/precision/batch conditions.

### M8.2 — Deliver a measured serving interface

Owner: Lead agent. Status: planned. Dependencies: M8.1.

- [ ] Implement a documented request schema, streaming/cancellation, admission
  limits, backpressure, health/readiness, error handling, and safe logging.
- [ ] Test concurrent requests, cache isolation, malformed inputs, capacity
  limits, shutdown, and timeouts; document deployment/authentication boundaries.
- [ ] Measure latency, tokens/s, and peak memory on named hardware/workloads;
  keep correctness tests and load tests separate.

Acceptance: functional and failure-path integration tests pass. Any throughput
or production-readiness claim names its tested deployment and limitations.

## Phase 9 — Documentation, CI, and release

### M9.1 — Reproduce supported workflows in clean environments

Owner: Lead agent. Status: planned. Dependencies: M8.2 and earlier accepted gates.

- [ ] Expand early CPU CI into unit/reference, tiny training/resume, lint/type,
  packaging, and available accelerator/distributed checks with explicit skips.
- [ ] Verify install/train/evaluate/export/serve runbooks from clean environments;
  retain logs and publish supported version/hardware matrices.
- [ ] Document APIs, equations, configuration schemas, contribution rules,
  failure recovery, and artifact compatibility/migration.

Acceptance: documented commands execute successfully in the declared matrix;
CI cannot turn missing required tests into a green release gate.

### M9.2 — Prepare an honest, licensed release

Owner: Lead agent. Status: planned. Dependencies: M9.1, M7.2 for model releases;
user decision on project license.

- [ ] Select a project license with the user before publication; audit third-party
  code, dependency, dataset, tokenizer, and model-artifact redistribution rights.
- [ ] Provide model/data cards, evaluation limitations, provenance, reproducible
  manifests, checksums, release notes, and a version/compatibility policy.
- [ ] Prepare release artifacts and regression evidence; publish only with
  explicit authorization. Framework-only releases make no trained-model claim.

Acceptance: each released artifact has a clear license and provenance, supported
workflows pass, and claims are bounded by actual evidence. Publication and final
model training remain user-controlled actions.

## Phase 10 — Continuous improvement

### M10.1 — Operate a measured research backlog

Owner: Lead agent. Status: planned. Dependencies: M9.2.

- [ ] Prioritize math/generalization failures, data quality, training efficiency,
  numerical stability, and serving regressions using retained evidence.
- [ ] Define hypothesis, control, budget, acceptance metric, and rollback for
  every experiment; log failures and negative results as well as successes.
- [ ] Revisit candidate architecture, tokenizer, objectives, and scaling choices
  through controlled ablations rather than unsupported complexity.

Acceptance: changes are promoted only after applicable tests and predeclared
measurement gates pass; the roadmap and decisions reflect what was learned.

### M10.2 — Maintain reproducibility and supported operations

Owner: Lead agent. Status: planned. Dependencies: M10.1; ongoing after release.

- [ ] Regularly verify dependency updates, checkpoint migration, clean restores,
  license changes, data-source availability, and security fixes.
- [ ] Revalidate version/device support before changing the published matrix;
  retain old run manifests and compatible artifact readers as policy requires.
- [ ] Update documentation, risks, and next tasks at every session boundary.

Acceptance: upgrades have replay/regression evidence and a documented migration
or rollback; unsupported combinations are never silently treated as verified.

## Environment and publication follow-up (2026-10-31)

- [x] Install isolated bootstrap tools with hash-locked transitive dependencies.
- [x] Verify repeatable installation and tool startup; preserve the uploaded brief.
- [x] Publish a feature branch without rewriting main or user history.
- [x] Save install/start instructions and the GitHub API domain in the environment draft.
- [x] Follow the latest user instruction: publish directly to main; PR creation
  is optional and its unavailable API is not a delivery blocker.

Historical bootstrap scope: this follow-up preceded M2.1. Package/runtime/CI
were implemented in later milestones. See `SETUP.md` for bootstrap context.

## Research acceptance evidence (2026-10-31)

`docs/research/` contains the verified survey/source manifest, architecture and
training RFCs, conditional data proposal, risks/exclusions, and reproducible
parameter/FLOP/memory calculations. Actual user hardware is RTX 3060 Laptop,
Core i5, 16 GB RAM; VRAM is an explicit unverified planning assumption.
Analytical/reference arithmetic tests, Ruff and strict mypy passed; no model,
GPU benchmark, dataset ingestion or training ran. M1.1/M1.2 completion refers to
the design contracts, not implementation or measured resource feasibility.
No corpus is approved; proposed budgets/LRs/data weights require later trials.

## Immediate next session

M3.2 expanded offline acceptance is complete; M3.3 production approvals remain
blocked. Proceed with the user's next Phase4 scope. The portable RFC decoder was
implemented/tested early in M2.3; review/integrate it rather than duplicate it.
M4.2 cache/checkpointing and M5 actual packed-data trainer are still planned.
Preserve all rights/benchmark gates and distinguish loader replay from atomic
model/optimizer/loader training replay. Publish verified work and reports to main;
no paid/final-scale runs or old environment configuration changes.

## Supporting task — Pinned agent skills

Owner: Lead agent. Status: complete. Dependencies: user request, source/license inspection.

- [x] Vendor all redistributable requested collections and math subtree; retain licenses.
- [x] Record commit/file identities, catalog, exclusions and task-specific usage rules.
- [x] Add README/agent entry points; verify hashes, first-party tests and lint/types.

Acceptance: 408 vendor skill entries have intact source files and license notices;
6 entries with restrictive or unspecified licenses are explicitly excluded.
The attached ponytail skill adds one required project-wide coding workflow. Source authenticity is Git provenance,
not a claim of comprehensive security or third-party runtime validation.

## Expanded Phase 2 request — M2.3 engineering acceptance

Owner: Lead agent. Status: complete. Dependencies: M2.1/M2.2/M3.1 and the
user's explicit expanded foundation request. The new request advances the RFC
model/synthetic-training slice before M3.2; full M4/M5 acceptance remains separate.

- [x] Requested primary layout, old import/config compatibility, scope-only future
  directories and unit/integration tests; shared dataclass/CLI override rules.
- [x] Pinned packaging, Make targets, local pre-commit, markers and expanded CI.
- [x] Real RFC decoder with numerical/gradient references and all four meta counts.
- [x] Real100-step synthetic trainer, JSON/optional offline metrics and token LR.
- [x] Committed checkpoints, safetensors, rank RNG/Adam/sampler replay and CPU DCP/
  DTensor shards. Actual CUDA/NCCL/FSDP2 tests provided but hardware unavailable.
- [x] Apache-2.0/NOTICE, CONTRIBUTING and assumed-platform server/all-reduce runbook.
- [x] Pass the requested Make chain from an isolated clean clone, hooks/build/smoke.
- [x] Observe green GitHub Actions for the final published main revision.

Acceptance: no fake training/checkpoint/backend commands; CPU, numerical,
restart and distributed evidence recorded, optional/GPU skips distinguished.
Future real-data/post-training/evaluation/serving work remains explicitly planned.


Evidence for M2.3: code 9ee9443, actual GitHub run 41242952575 (all five jobs
successful), clean-clone Make/hooks/build and100-step/replay logs recorded in
validation_foundation.md. New user request starts expanded Phase 3 below.

## Expanded Phase 3 — Tokenizer and data pipeline

Owner: Lead agent. Status: in progress. Dependencies: M3.1 and M2.3.

- [ ] Extend versioned controls for reasoning/tool delimiters without silently
  reinterpreting old tokenizer artifacts; measure EN/RU fertility and digit policy.
- [ ] Register original exact-answer math/demo data and optional source acquisition
  with pinned checksums, rights records and fail-closed approval rules.
- [ ] Streaming cleaning/LID/quality/PII/toxic policy, exact and MinHash-LSH dedup,
  with auditable counts and no blanket privacy/quality guarantee.
- [ ] Mandatory benchmark registry/n-gram decontamination; reject production mode
  unless all planned evaluation snapshots and their rights are approved/present.
- [ ] Checksummed uint16/uint32 memmap shards, document-aware packing/attention/loss
  masks, deterministic rank/worker order, stage mixtures and resumable loaders.
- [ ] Program-verifiable arithmetic/algebra/number-theory/combinatorics generators,
  documented difficulty and bounded answer verifier for later RL environments.
- [ ] Offline end-to-end/crash/decontamination tests, measured host throughput,
  runbooks/01_data_preparation.md with measured demo disk and labeled i5 estimates.

Acceptance: genuine raw-to-batch pipeline and exact committed cursor replay,
including worker prefetch; real benchmark/data availability is a mandatory
production gate, never faked with toy fixtures. Observe current-head CI.
