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

## R007 — Phase3 curation, packing and consumer replay (complete for offline scope)

Hypothesis: approved original raw text can be curated, excluded against every
registered authored holdout, split before BPE, packed compactly and replayed from
the consumer cursor despite worker prefetch. Design: data_design.md and ADR29–31.
Initial full local suite:422 passed/2 GPU deselected in70.19s before the final
math-atom preservation regression; targeted revised acceptance and measured
compression/throughput evidence will be recorded below. No quality result yet.

R007 lifecycle follow-up: the first actual GitHub run41247260902 had success on
CPU3.11/3.12, synthetic100-step smoke, data smoke and optional metrics, but3.13
reported unclosed SQLite ResourceWarnings (418 passed/3 failed/2 optional skips/
2 GPU deselections). Closing contexts and success/error resource tests correct the
cause. A stricter pending-prefetch abandonment test also reproduced a Tensor IPC
worker abort; NumPy worker collation/consumer tensor conversion passes that
reproducer. These failed checks are recorded rather than described as green CI.


## R008 — Math digit segmentation and EN/RU fertility

- Hypothesis: grouping digits reduces token cost; single digits may provide a
  simpler numeric representation, but compression cannot establish reasoning gain.
- Executed clean code:9f6cef941ceccf3c5eb929256e408a7644b65896, dirty=false,
  Python3.12.14/tokenizers0.22.2. Command:
  `uv run --frozen --extra cpu python scripts/data_experiments.py --directory /tmp/qvelt-data-measured-9f6cef9 --repository .`.
-768 original verified EN/RU math train texts (seed113),384 different-seed texts
  (997,192 per language), shared eight template families, requested vocab1024,
  minimum frequency2. Corpus SHA256
  `b80d8d3baf828cb1943fe32de8d8091da6b84df1dcd96d06a93cae1b4cdbb4fb`.
  No public benchmark inputs or gold answers were used.

| Policy | Actual vocab | EN tokens / whitespace words | EN fertility | RU tokens / words | RU fertility | Round-trip failures |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| Individual digits |509 |6422/2448 |2.623366 |6278/2256 |2.782801 |0/384 |
| Grouped digits |1024 |4799/2448 |1.960376 |4655/2256 |2.063387 |0/384 |

Total12700 vs9454 tokens: grouping reduces25.559055% on this sample. Learned JSON
hashes and exact example digit IDs are in the committed receipt. Vocabulary is a
ceiling, so the individual-digit grammar learns fewer merges. Conclusion: retain
both explicit policies and the RFC individual-digit controlled baseline; this study
supports grouping for compression only. Revisit the default using approved corpus
fertility and fixed-budget trained-model math evaluations, not this toy result.
Whitespace-word fertility is explicitly defined and is not morphological coverage.

## R009 — Compact data preparation and CPU host delivery

- Hypothesis: compact shards and ordered worker prefetch support useful host batch
  delivery while preserving target counts/order. No GPU/model work is measured.
- Same clean9f6cef9 code/command/receipt as R008. Cloud CPU: Intel Xeon Platinum8370C,
  container cpu.max400000/100000 (4 CPU quota), memory.max34359738368 (32GiB).
  Linux/Python3.12.14/PyTorch2.9.1+cpu; one main-process Torch thread. The user's
  i5/16GB/RTX3060 Laptop was not accessed or benchmarked.
- Demo192 math documents/language +20 canaries =404 inputs. Retained374
  (train146/validation82/test146); removed24 exact,2 near,2 protected overlaps,
  2 phrase quarantines;2 emails/2 phone patterns redacted. LID mislabeled some
  short Russian templates as Bulgarian, so annotation remains non-filtering.
- Preparation including raw generation:2.886784s;599036 bytes total demo files,
  483224 prepared bytes. SQLite/snapshot observed566396 bytes at the recorded
  point; this is not peak disk/RAM. Stored18098 tokens including boundaries,
  17724 valid targets.83 available train blocks; repeated sources are intentional.

| Workers | First-batch startup s |64-batch steady s | Valid targets | Valid targets/s | Capacity tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
|0 |0.015052 |0.625429 |32094 |51315.16 |52392.82 |
|2 |3.818806 |0.497274 |32094 |64539.87 |65895.26 |

B=8,T=64,prefetch2,8 warmup then64 measured batches. Both runs have identical
final cursors EN120/RU456 after72 batches. NumPy IPC then consumer tensor conversion
is included. Small reused/cached input, short single measurements; no uncertainty
interval, cold-storage, laptop, GPU-saturation or training-throughput claim. Initial
pre-refinement run752a6ca measured51374/45604 targets/s for0/2 workers; this is
historical transport/timing evidence, not averaged into the final receipt. Measure
on the intended machine/source size before choosing worker count.

## R010 — Actual permitted GSM8K exclusion-index audit

- Hypothesis: the implementation can index every approved question and detect a
  copied question without training on the evaluation source.
- Acquired publisher revision3101c7d5072418e28b9008a6636bde82a006892c via the
  pinned eval-only spec;749738 bytes/1319 original rows. MIT license and payload
  hashes are recorded in DATA_CARD.md. No raw content is committed.
- Actual SQLite index of1319 question texts, ngram13:55284 unique grams. One
  protected-question canary with surrounding original prose matched gsm8k;
  unrelated authored control matched nothing. Elapsed0.352094s on this host.
- Conclusion: this checks real question acquisition/indexing/matching only.
  Gold solutions are not indexed by the question adapter. MATH/AIME/MMLU and the
  frozen project-final suite remain missing; production_ready=false. No evaluation
  score, semantic/translated leakage guarantee or full public coverage is claimed.
  Full numeric audit receipt: docs/development/evidence/phase3-gsm8k-index.json.


Final R007 acceptance:9f6cef9, separate fresh-clone frozen Make setup/lint/types/
test/build pass;426 passed/2 optional SDK skips/2 GPU deselections in64.72s.
Actual GitHub run41247846988:6 jobs successful; CPU3.11/3.12/3.13 each426 passed,
optional metrics3 passed, independent data smoke and numeric100-step training
smoke successful. Direct per-job training log confirms117056 parameters/6400
numeric targets, loss5.6883039474487305->5.023181915283203, bit-exact Adam/weights
for stop37/resume100. No real-corpus optimizer training is claimed. Data smoke
212->194 documents,153 verified blocks,7 resumed batches from cursor3 to10,
pending-prefetch iterator discarded. Committed receipts and validation_data.md
record exact identities and limitations; production data approval remains blocked.

## R011 — Bounded portable decoder/RFC review

Date:2026-11-01 (Europe/Kiev). Hypothesis:the decoder already built in M2.3 matches
the selected RFC equations and accepts Phase3 packed batches without modification.
Reviewed code:91e9dc85956815e04fd7d5032a084060e59e51bb. Command:
`uv run --frozen --extra cpu pytest tests/unit/test_model.py tests/integration/test_data_pipeline.py::test_packed_batch_runs_real_decoder_backward -q`.
Result:18 existing tests passed in6.56s on CPU, including independent component
values/gradients, four meta counts, actual tiny backward, causality/document
isolation and a real BPE packed-batch backward. No inspected architecture mismatch.
Conclusion:reuse the implementation. Record missing dedicated initialization,
non-finite policy and tied-embedding combined-gradient checks before full M4.1
acceptance; KV-cache/activation checkpointing remain M4.2. No new test/model code,
GPU run or trained-model quality result. Report:decoder_rfc_review.md.

## R012 — Initialization, tied gradients and non-finite fail-stop

Date:2026-11-01. Hypothesis:the remaining RFC invariants hold for the existing
portable model without changing architecture. Tests add seeded state equality,
changed-seed inequality, L3/d128/H8/KV2/f256 initialization mean/variance checks
(at least4096 matrix elements, variance rtol.08), independently split embedding
input/head gradient sum (rtol2e-5/atol2e-6), and NaN/+Inf/-Inf loss/norm propagation.
An actual numeric trainer run with injected non-finite logits must stop before
AdamW.step and publish no completed checkpoint. No invalid values are sanitized.
Commands: `uv run --frozen --extra cpu pytest tests/unit/test_model.py tests/integration/test_training.py::test_nonfinite_loss_stops_before_optimizer_update -q`
and `make lint typecheck test`. Observed24 targeted passed in4.08s;433 full-suite
passed/2 optional SDK skips/2 GPU deselections in69.69s; Ruff and mypy82 files pass.
Conclusion:M4.1 portable gate closes, GPU/cache/recomputation remain M4.2.

## R013 — Cached decoding, native SDPA and block recomputation

Date:2026-11-01. Hypothesis:explicit cache metadata/compact KV heads reproduce
full causal/document logits; native SDPA and non-reentrant checkpointing preserve
the reference values/gradients and resumable optimizer state. Tests cover MHA/MQA/
GQA, multiple prefix/suffix partitions, packed reset positions/padding, stale/load/
replaced weights, weak owner lifetime, malformed caches and BF16 policy changes.
Reference QK/softmax now explicitly disables autocast for RFC FP32 semantics.
Targeted49 execution/cache tests initially passed; storage/recompute tests2 passed
and actual four-step checkpointed training/replay passes for both backends.
`make lint typecheck test`:487 passed/2 absent optional SDK skips/4 GPU cases
deselected in81.70s; Ruff and strict mypy89 files pass. The additional GPU FP32/
BF16 tests are supplied, not executed. CPU timing/storage profile receipt follows
from clean committed code; none of the tests establishes GPU/quality performance.

R013 clean-code profile:efa6f30,dirty=false,default117056-parameter job,B2/T32,
2 warmup/8 measured repetitions,one Torch CPU thread. Reference saved graph storage
932236 bytes versus157836 with block recomputation; SDPA881036 versus157836.
Forward/backward targets/s9571.36/4494.59 reference without/with recomputation,
9459.94/4975.82 SDPA. Cache25392 bytes in all policies; full/cached logits close.
Cached/uncached timings include16-token prefill+8 appends; exact values and resolved
jobs are in docs/development/evidence/phase4-profile-efa6f30.json. No optimizer update,
peak allocator memory, GPU speedup or model-quality result is implied. M4.2 closes
for available CPU paths; provided accelerator tests await actual hardware.

R013 remote acceptance:actual GitHub run41287953291,code efa6f30,all6 jobs success
(CPU3.11/3.12/3.13,100-update numeric smoke,offline data smoke,optional metrics).
This closes available-CPU M4.2 acceptance, with CUDA tests still explicitly unrun.

## R014 — Actual BPE optimizer/consumer replay and masked accumulation

Date:2026-11-01. Hypothesis:real original-tokenizer/sharded batches can drive
normalized optimization and retain exact weights/Adam/consumer/RNG state after
checkpoint restart, including partial windows, worker changes and uneven ranks.
An early executed four-update demo heldout LM probe decreased6.126704->6.020058;
continuous/resumed weights, Adam and cursor matched exactly. These are original
fixture LM losses, not public math accuracy. Root manifests bind actual data/BPE
hashes, not numeric descriptors. Separate global-batch references cover uneven
masks,0-target local microbatches and a short final window; world2 Gloo gradients
and coordinated one-rank IO failure pass. The actual-data two-rank SDPA+checkpoint
continuous/resume test also passes (33.20s).
Initial `make lint typecheck test`:521 passed/2 optional SDK skips/4 GPU deselections
in133.66s; subsequent actual-data focused33 passed/1 newly supplied GPU case
deselected,97 files typed. A final FP32 initialization factory regression follows
an executed ambient-float64 observation. Final clean-code20-step receipts and
remote7-job CI will be recorded after execution; no final/GPU run is implied.

R014 final clean-code acceptance:557e4da5a4a9ba3d800afc374af1ac5be44ce471,
dirty=false,Python3.12.14/PyTorch2.9.1+cpu. Command:
`uv run --frozen --extra cpu python scripts/pretraining_smoke.py --directory /tmp/qvelt-pretraining-clean-557e4da --repository . --steps 20`.
Actual127104 parameters (saved-weight enumeration agrees with config),20 optimizer
updates,40 microbatches,4959 actual targets versus5120 nominal padded capacity.
Fixed256-target heldout LM probe6.126703500747681->5.756516456604004. Full versus
stop6/resume20 weights/Adam/consumer state exactly equal; direct saved Python/
NumPy/Torch RNG comparisons are also equal. No external/public scoring input.
Receipts:pretraining-smoke-557e4da.json and pretraining-provenance-557e4da.json.

Fresh clone `make setup lint typecheck test`,lock/dependency checks and build pass:
522 passed/2 optional SDK skips/5 GPU deselections in122.79s;97 typed files.
Actual GitHub run41290979913 at557e4da:7 jobs successful,all3 CPU versions522 pass
(3.11 85.51s,3.12 96.48s,3.13 103.55s). Direct packed-job log confirms20 updates/
4959 targets/exact replay;probe6.12670373916626->5.756516695022583 differs slightly
across hosts without contradicting within-run replay. Added explicit partial/DDP
RNG regressions complement the model/cursor checks;see final targeted log/CI.
M5.1 closes for available CPU;CPU-DDP M5.2 is verified while hardware tiers remain
unvalidated. SFT/preferences/RL and production data clearance are separate gates.


## R015 — Managed CPU environment replay revalidation

Date:2026-11-04. Base260b431 with cache/README changes in the working tree;
model/training code and lock unchanged. Frozen Python3.12.14/uv0.12.23/
Torch2.9.1+cpu, optional SDKs installed. Numeric100 updates/6400 targets:
loss5.6883039474487305->5.023181438446045;stop37/resume100 weights/Adam bit-exact.
Actual BPE20 updates/40 microbatches/4959 valid targets:heldout256-target probe
6.126703500747681->5.756516695022583;stop6/resume20 weights/Adam/consumer exact.
These are existing original demo workflows, not a reasoning benchmark or a new
training capability. Full CPU/optional-SDK suite524 passed,5 GPU deselected,
no skips in94.75s with explicit execution network permission. An earlier default-
sandbox attempt was interrupted at16 passed while local distributed sockets waited;
it is not counted as a full pass. No GPU/paid/online-metrics/final-scale run.
Commands, environment and machine-readable receipts:validation_environment.md and
docs/development/evidence/environment-2026-11-04.json.


## R016 — Assistant-only exact-math SFT and atomic replay

Date:2026-11-04. Source:7eed79c with M6.1 working-tree changes; exact source digest,
recipe,initial checkpoint and data/tokenizer manifests retained in
`docs/development/evidence/sft-smoke-2026-11-04.json`. Python3.12.14,uv0.12.23,
Torch2.9.1+cpu,FP32,training seed20261031,source seed20261104,one CPU thread.
64 original EN/RU exact-answer dialogs;24 train/16 validation/24 test with disjoint
SFT families. No external teacher/corpus; licenses and permitted use in DATA_CARD.
Three-step engineering base initializes only weights for the new127104-parameter
SFT run. Actual80 updates/80 microbatches/2520 assistant targets,B2/T128,LR0.003.
Fixed69-target training probe5.897945459338202->0.14195556225983993;distinct16-target
heldout probe5.810949802398682->4.761350393295288. Stop26/resume80 matches weights,
Adam,RNG and committed application/consumer state bit-exactly. This is fixture
memorization/restart evidence,not mathematical answer accuracy or decontamination.

Final full CPU/optional-SDK run543 passed,5 GPU deselected,no skips in97.32s;
Ruff/format,mypy103 files,package/lock/dependency/hook checks pass. Full suite uses
explicit network permission,GLOO_SOCKET_IFNAME=lo,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1
following earlier process-exit timeouts (not counted as passes). Full commands,
failed/interrupted-run boundaries,scope and limitations are in validation_sft.md.
No CUDA/distributed-SFT/final-scale/paid/online-metrics run was performed.


## R017 — Selective supplied Phase4 closure and new CPU operator profile

Date:2026-11-04. Base8f16a9c plus selected/adapted user patch;receipt in
`docs/development/evidence/phase4-current-profile.json` records source/script hashes.
22 focused model/compile/profile tests passed in63.36s (AOT and CPU Inductor ran;
GCC14.2.0),105 typed files,lint/format/build pass. Seed517 pre-import versus current
reference/SDPA FP32/BF16 packed+padded logits/gradients/weights are tensor-exact.
Actual full tiny overfit(<0.05 on32 random targets within100 updates),whole-model
FP64 gradcheck,depth-scale checks and30 random cache-length trials per backend/
scale1 or3 pass. Source is the user's project patch;no vendor model code copied.

Fresh CPU profiler:tiny19338560 parameters,B2/T128,one thread,2 warmups/3 repeats.
SDPA488.39,reference468.05,SDPA+recompute404.89 targets/s;these short shared-host
forward/backward measurements omit optimizer updates and are not a speed ranking.
SDPA operator trace:aten::mm377.5ms self time,fill_16.8ms,add15.0ms. No CUDA device,
peak GPU memory or supplied peak TFLOP rate;MFU remains null. The uploaded old
profile JSON and old ADR0038/R015/state files were not imported as new evidence.

Final local regression562 passed,1 numeric-DDP exit SIGABRT,5 GPU deselected
in104.85s;all model/SFT/profiler and packed two-rank replay cases pass. USE_LIBUV=0
was an executed diagnostic alternative,not a proven fix. Similar distributed
stalls occurred before this import;root cause remains unresolved. Earlier full
run561 passed/2 failed in237.07s included a profiler metadata field omission caught
by its added assertion;the field was fixed and profiler subset3 passed in3.36s.
No tests were relaxed or disabled. Full current-source remote CI remains pending.
See validation_phase4_closure.md for commands,scope and RFC deviations. No GPU,
paid/final-scale training or long-context quality claim.

R017 remote first pass:actual GitHub41908101017 at886ce61 has7 successful jobs,
including CPU3.12/3.13 and dedicated numeric/data/packed/SFT/optional-metrics jobs.
CPU3.11:560 passed,1 Inductor failure,2 optional SDK skips,5 GPU deselected in110.45s.
The failure is a missing setuptools developer dependency,not a numerical mismatch.
Pin existing locked84.2.1 explicitly in dev;all locked package versions/hashes stay
unchanged. Frozen setup,lint/format and mypy105 files pass after the dependency fix;
repeat current-source/current-lock CI before declaring matrix acceptance.


R017 final corrected-source acceptance:fee1e3db497b7ce899216827f71ba721ada1ba3e,
actual GitHub41909315717,all8 jobs successful. CPU3.11/3.12/3.13 each561 passed,
2 optional SDK skips,5 GPU deselected in117.80s/125.46s/126.23s. Optional metrics
independently3 passed in4.70s;numeric/data/packed/SFT jobs all green. A separately
created bare Python3.11.17 CPU developer environment,without TensorBoard/W&B,
installs setuptools84.2.1 and passes actual Inductor execution in44.92s. Final
frozen dependency/lock,lint,mypy105 files and wheel/sdist build pass. Original
local562-pass/1-DDP-exit-failure observation remains valid and unresolved;remote
success is not presented as a local runtime fix. CPU M4.3 closes;GPU remains open.


## R018 — Expanded pretraining,actual200-update hard-kill and measured controls

Date:2026-11-04. Start b82131c;selected/adapted supplied Phase5 WIP. Registered
D-TRAIN-200 authored Apache-2.0 fixture before ingestion. Full RFC tiny19,338,560
parameters,V32768,T16/B2,FP32,seed20261031,one CPU thread:200 updates/6386 valid
targets. Actual SIGKILL at observed89,restore75;all200 loss values match exactly,
weights/Adam/RNG/application bit-exact. Probe10.47552204->9.00020885;authored
code/math/web final PPL3163.41/2402.49/2653.38. No production-domain or reasoning
quality claim. Final-source f14b266 repeat passes bit-exactly,source digestba3bb6d89d5aa74dbea7a80d3a1f43e1fdfab309ac8a0f04d0c635859de45e50.

Compile8 updates/962 targets:cached startup+checkpoint wall0.811s eager/6.386s
compiled;median steady11,589.89/13,509.28 targets/s/device;max weight delta7.445e-6.
No isolated speedup/cold-compile/GPU/MFU claim. First benchmark failed correctly on
setuptools-added platformdirs metadata;exact bundled path exclusion and fresh-process
inventory regression fix it. Four20-update models with same2296 targets and LR0.001:
P13344/38976/76896/127104,heldout5.991178/5.803633/5.660614/5.567677;
log-log slope-0.03270178,R²0.99782621. Descriptive fixture fit,not general scaling.
Small125,854,464-param12-command sweep expanded against realV32768/T2048 demo
shards;no small optimizer/GPU/server job launched. Receipts in evidence/phase5-*.

Pre-final local full586 passed,3 distributed process failures/timeouts,7 GPU
deselected in365.00s. Focused final38 passed,2 GPU deselected in36.42s(includes
actual two-rank NaN rollback,SIGTERM,hard-kill,source-probe,sweep/provenance checks).
No tests relaxed. Current-source clean CI,final type/build and refreshed200-step
receipt pending;see validation_phase5.md. No paid or final-scale run.


R018 final acceptance2026-11-05:source f14b266 and source digestba3bb6d89d5aa74dbea7a80d3a1f43e1fdfab309ac8a0f04d0c635859de45e50.
Final local200-step repeat is bit-exact for all200 losses/weights/Adam/RNG/cursor.
Ruff/format,strict mypy122 files,frozen lock/dependencies,wheel/sdist pass. Local
restarted full590 passed,1 packed-DDP timeout,7 GPU deselected in318.63s;this is
not a local green claim. GitHub42069685890 atf14b266 has all9 jobs successful:
CPU3.11/3.12/3.13 each589 passed,2 optional SDK skips,7 GPU deselected;SDKs
separately green. Dedicated CI200-step recovery also bit-exact;hardware-dependent
probe final9.65952873 versus local9.00020885 is not cross-host equality. Portable
Phase5 accepted;CUDA/NCCL/FSDP/server and production inputs remain unverified.

## R019 — Original verifier-pair DPO and exact optimization replay

Date:2026-11-05. Baseae3dee7 plus original M6.2 source. Registered D-PREF-01
before generation. SymPy verifies d/dm log(1+exp(-beta*m))=-beta/(1+exp(beta*m))
for finite real margin and positive beta. Independent NumPy loss/analytic gradients,
response-sum/padding/mask and naive unpadded decoder references are executed.
Pre-final focused23 tests passed in9.11s;strict mypy128 files and Ruff pass.

First successful40-step CPU run:reduced127104-parameter decoder,V445,T128,B2,
seed20261031,beta0.1,LR0.001. Fresh30-step SFT weights initialize both policy and
frozen reference;32 new original exact-answer pairs split12/16/4 by DPO family.
80 consumed pairs/1582 response targets;stop13/resume40 has exact all40 losses,
weights,Adam,RNG and consumer/application state. Train DPO0.69314718->0.35221379,
validation0.69314718->0.84319099;reference-relative positive-margin fraction
train0.916667,validation0.375,initial all ties. These are offline fixture scores,
not sampled win rates or mathematical accuracy. Negative heldout finding retained;
no checkpoint promoted. No external teacher/production/GPU/paid/final-scale run.

The first preparation attempt correctly failed because the new preference payload
lacked the manifest's required data role;the payload role/loader were corrected.
Three adversarial test helpers initially attempted to replace an exclusive commit
marker;helpers now unlink only their copied fixture marker before republishing.
No runtime trust check or failing test was weakened. Final source-bound40-step
repeat,regression/CI and packaging checks pending;see validation_preferences.md.


R019 final local repeat:source digestc3c18aa16e68110c5cb7e4b9750773a942d822984d537307e87f9b5f666dda08;
preferences-40.json binds actual pair/reference hashes. All40 losses/weights/Adam/
RNG/cursor match exactly;train/heldout values match the pre-final experiment.
Full local614 passed,no skips,7 GPU deselected in158.37s;focused23 passed in9.11s,
Ruff/format,strict mypy128 files,wheel/sdist pass. Clean CI pending. This green
local run does not resolve earlier intermittent Gloo faults;historical failures
remain reported. No paid/GPU/final-scale training or quality promotion.

## R020 — Bounded base/SFT/DPO generation-readiness control

Date:2026-11-05. R019's16 original validation problems only,B1 CPU FP32,greedy
full-prefix,same chat prefix,max16 generated tokens,required complete end-turn6.
Registered protocol in reasoning_rl_readiness.md before execution. Source3b3ef4a,
source digestc3c18aa16e68110c5cb7e4b9750773a942d822984d537307e87f9b5f666dda08.
Base0 correct/0 complete/16 truncated;SFT3 correct/8 complete/8 truncated;
DPO0 correct/0 complete/16 truncated. Original exact-fraction verifier rejects
malformed/partial responses;no model code executes. Per-problem completions,
checkpoint/weight/tokenizer hashes are in reasoning-readiness.json. Repro script
verifies tokenizer/model/SFT identities,known-positive and hostile verifier cases,
context bounds and exclusive output. This is not project-final/public evaluation.
The tiny necessary control passes for SFT only;DPO remains rejected for promotion.
A bounded engineering RL check may start from SFT;general reasoning improvement
still needs independent fixed-budget evaluation. No paid/GPU/large rollout run.


R019 remote acceptance:3b3ef4a087d53f6ed47db683cbadb05a7f6fddae,GitHub42073726045,
all10 jobs successful. CPU3.11/3.12/3.13 each612 passed,2 optional SDK skips,7 GPU
deselections;optional SDKs separately green. DPO40-step exact loss/weight/Adam/RNG/
cursor replay and full RFC tiny200-step SIGKILL replay both pass. Remote DPO
train0.35152350/heldout0.84594192 retains the negative heldout finding;hardware
float differences are not cross-host replay failures. Portable M6.2 closes.
Source-unchanged R020 repro script passes Ruff/format and strict mypy129 files.

## R021 — Original outcome-GRPO on-policy signal and exact8-window replay

Date:2026-11-05. Base716539d plus original GRPO prototype. D-RL-01 registered
before sampling. Primary DeepSeekMath v3 Eq3/4/outcome normalization read;SymPy
KL derivative/curvature/minimum identities and NumPy value/analytic-gradient
references executed. A historical pre-RoPE SFT manifest correctly failed config
integrity;no bypass. Fresh current-schema SFT80 reproduces train probe0.14195556,
heldout4.76135039 and exact SFT state replay. Seeded4x4 training-only signal probe
has3 nonconstant groups;real generated wrong/invalid answers are retained.

Final-source8-window prototype:127104-parameter V445/T128 CPU FP32,seed20261105,
P4/G4,max16 actions,temperature1,LR1e-6,clip0.2,KL0.01,WD0,gradient clip1.
8 actual policy-gradient/Adam updates,745 generated actions,11/32 zero-variance
groups. Stop4/resume8 matches all losses and complete rollout digests plus weights,
Adam,RNG and consumer state. Near-zero surrogate values at fresh ratio1 are expected;
nonzero gradient updates and actual weight changes are independently asserted.
Validation families only:1/8 correct,8/8 complete both before and after. Test
families unused. No improvement/promotion/public benchmark claim. Source-bound
receipt in evidence/grpo-8.json. Full local regression/clean CI pending.

Pre-final combined DPO/GRPO focused31 passed in17.50s;verifier deadline/nonfinite
coverage then gives32 passed in18.57s. Added whole-window all-zero reward test
checks untouched Adam/weights and resume rather than faking a step. Mypy135 files
and Ruff pass. Repro commands and policy/length/reward limits in06_reasoning_rl.md.
No paid/GPU/full-scale run or model-generated Python execution.


R021 final local source4775afbb264baa8c842cf40b67e0372162ee4e690f1c3738e227db42e7136b1b:
self-contained8-window repeat exact for all losses/rollouts/model/Adam/RNG/cursor,
8 actual policy updates/745 actions/11 constant groups. Fixed validation1/8 remains
unchanged. Explicit FP32 reward/behavior tensors and post-call bounded verifier are
included. Source-focused33 passed in22.23s;mypy135 files,Ruff/format pass. Pre-final
full622 passed,1 numeric-DDP subprocess timeout,7 GPU deselected in258.63s;no green
local-full claim or assertion of a Gloo fix. Clean current-source CI pending.
