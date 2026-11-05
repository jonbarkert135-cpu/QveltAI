# Qvelt risk register

Last updated: 2026-10-31. Owner for all risks: Lead agent unless external action
is named. Likelihoods are qualitative judgments, not measured probabilities.

| ID | Risk | Status / impact | Mitigation and acceptance evidence | Review gate |
| --- | --- | --- | --- | --- |
| R01 | Laptop RTX 3060/Core i5/16 GB RAM supplied; server/budget and exact VRAM unknown | Open; blocks real scale validation and final training | Deliver tiny CPU correctness first; estimate memory/compute explicitly; require user hardware details and approval before expensive runs | M1.2, M5.2–M5.3 |
| R02 | Ambition mistaken for demonstrated quality or lab parity | Open; misleading expectations | Publish only artifact-backed capability and quality claims; compare fixed-budget held-out results; never equate engineering scope with model strength | Every delivery, M7.2 |
| R03 | Unstable optimization, bad initialization, or mixed-precision overflow | Open; lost compute or silent damage | Reference tests, finite-value diagnostics, gradient checks/clipping, tiny overfit, checkpoint recovery, and controlled learning-rate/precision studies | M4.1–M5.3 |
| R04 | Causal masking, label shift, normalization, or gradient accumulation is wrong | Open; plausible but invalid learning | Independent forward/gradient oracles; future-token perturbation tests; uneven valid-token and partial-accumulation comparisons | M3.2, M4.1, M5.1–M5.2 |
| R05 | Checkpoints omit state, corrupt writes, or change sample order on restart | Open; irreproducible or unusable runs | Versioned atomic artifacts, integrity checks, rank-aware RNG/data state, interrupted-versus-continuous replay tests; reject unsupported migration | M2.2, M5.1–M5.2 |
| R06 | Distributed deadlocks, invalid token weighting, duplicated data, or bad collectives | Open; correctness failures and wasted resources | Two-process CPU reference tests; real GPU/rank tests before support claims; explicit launch/timeouts/failure runbooks | M5.2 |
| R07 | Dataset or teacher-output rights are unknown or incompatible | Open; blocks lawful intended use and redistribution | Approve exact sources/revisions and uses in `DATA_CARD.md`; retain license evidence; exclude unresolved sources | M3.1–M3.2, M6, M9.2 |
| R08 | Benchmark leakage and near-duplicate mathematical problems inflate results | Open; invalid quality claims | Stable held-out families, provenance, exact/approximate audits, independent checks, and explicit residual uncertainty | M3.2, M6, M7 |
| R09 | Verifier exploitation or unsafe generated-code/symbolic execution | Open; misleading RL reward or host compromise | Bounded parsing, timeouts/resource limits, adversarial examples; no arbitrary generated execution in the trainer; separately reviewed isolation if tools are added | M6.3 |
| R10 | Tokenizer damages numbers, LaTeX, Unicode, or Cyrillic | Open; poor coverage or mathematical representation | Round-trip/coverage tests and measured segmentation comparisons; do not assume a digit policy improves reasoning | M3.1, M7.2 |
| R11 | Hardware-specific kernels, toolchain drift, or precision changes break portability | Open; CPU/single-GPU workflow unavailable | Pin versions; verify APIs and capabilities; keep tested portable alternatives; publish supported combinations only | M2.1, M4.2, M9.1 |
| R12 | Data preparation or optimizer/checkpoint state exceeds RAM/disk | Open; weak-device workflow fails despite small model | Bounded-memory streaming, storage/memory estimates, shard integrity, retention policy, resource profiling before larger runs | M1.2, M3.2, M5.3 |
| R13 | RL or synthetic data worsens general capability while improving a narrow metric | Open; brittle or misleading reasoning gains | Base/SFT/preference/RL controls, fixed inference budgets, general-language regressions, held-out verifier checks, and ablations | M6.3, M7.2 |
| R14 | Model/export/serving paths disagree, leak caches, or mishandle capacity | Open; incorrect outputs or service failures | Logit/cache parity, artifact validation, request isolation, concurrency/cancellation tests, capacity bounds, deployment-specific evidence | M8 |
| R15 | Project license and brand availability are unresolved | Open; release and reuse uncertainty | User chooses license before publication; audit third-party rights; do not claim Qvelt name/domain/trademark uniqueness without separate verification | Before publication, M9.2 |
| R16 | Scope expansion and cross-session drift skip correctness gates | Open; unfinished or unmaintainable stack | Maintain plan/state/ADRs; prefer dependency-ready milestones; no placeholders; stop at the explicit first-session boundary | Every session, M10 |

## Current blockers and non-blockers

- Lack of hardware does not block source review, documentation, or future tiny
  CPU implementation; it blocks claims about actual GPU/multi-node performance.
- No approved corpus blocks real-data tokenizer/pretraining/post-training runs,
  but future project-authored fixtures can validate software correctness.
- The project-license choice and name availability need resolution before public
  release/brand commitments, not before reversible local research and code work.
- There is no evidence of an application bug: no application exists yet.

## Phase 1 update (2026-10-31)

The user has a laptop RTX 3060, Core i5 and 16 GB RAM. Server training remains
unallocated; assumed 6 GiB VRAM is not a measured inventory. See
`docs/research/risks.md` for resource/cost uncertainty, conditional corpus rights,
missing approved code data, math repetition and RL verifier risks. The explicit
exclusion list defers MoE/MLA, FP8, untested long context, costly teacher/final
training, unsafe tool execution and publication during the current pause.

## M2.1 update (2026-10-31)

Pinned CPU environments, unit/RNG checks and strict type/lint checks passed on
Linux/Python 3.11–3.13. Base-only configuration and built-wheel operation were
validated separately. R11 portability risk is partially mitigated for this
matrix; CUDA/native Windows, actual laptop resources, model kernels and training
remain untested. The 2.9.1 PyTorch CPU/cu128 baseline is intentionally pinned,
not a promise of latest dependencies or long-term upstream support; updates need
fresh compatibility/security review and tests before deployment. Configuration
hashes/seeds do not close R05 checkpoint/replay risk; M2.2/M5 remain pending.

## M2.2 update (2026-10-31)

R05 is partially mitigated for typed provenance and raw-byte integrity: bounded
hash/copy, no-overwrite publication, manifest-last commit, process interruption,
concurrent writers and corruption checks passed on Linux local storage. This
does not close model/optimizer/scheduler/RNG/sampler replay risk, which remains
M5 work. R06 distributed transactions/multi-node recovery remain open. Native
Windows, object stores, NFS/shared-storage durability and physical power loss
were not validated. Source license strings do not close R07 ingestion rights;
no corpus or trained checkpoint was created.

## Third-party agent skill instructions

408 vendor skills are pinned instructions/assets, not an audited execution
layer. Some assume different client tools, unpinned libraries, network services
or immediate paid jobs. Read the selected skill and helpers before use; verify
APIs, pin dependencies, preserve root authorization and original Qvelt code.
Six Anthropic entries with restrictive or unspecified licenses are omitted. Inventory hashes detect drift,
not malicious upstream content; upgrades require source/license diff review.

Tokenizer M3.1a: byte coverage applies to whole valid UTF-8 strings; isolated
byte fragments may decode with replacement. Lone surrogates are rejected.
Trainer pair statistics are not globally memory-bounded. YAML/CLI/manifests and Linux/Python 3.11–3.13 fixture replay are now tested;
corpus-scale training/memory, cross-hardware/other-release equality remain unvalidated.
Ponytail simplicity must retain required validation, tests and reproducibility.

M3.1 byte/lineage checks and fixture permission reduce accidental input drift;
they do not establish external rights, a trustworthy historical training claim
or representative compression/math performance. A manifest-less partial output
is uncommitted; recovery must not silently overwrite it. M3.2 still needs data
approval, cross-source dedup/split/decontamination and deterministic shard replay.

## Expanded engineering foundation

Portable model numerical/gradient references and synthetic same-topology CPU
training/replay now have actual tests, including DDP and CPU DTensor/DCP shards.
These do not establish GPU/NCCL/FSDP2/multi-node/shared-store correctness or
large-model memory/throughput. The server recipe assumes Ubuntu24.04 and explicitly
labels Slurm resources; the user's hardware has not been profiled. Optimizer/DCP
inputs are trusted project checkpoints; checksums alone do not authenticate them.
Numeric fixture descriptor hashes are not real BPE/corpus manifests. Optional
backends are tested for local outputs, not full stochastic-model restart parity.
Synthetic training does not close corpus contamination, quality or reasoning risks.

## Expanded Phase3 data scope

- Production public-benchmark clearance is incomplete: GSM8K test was acquired
  under its publisher MIT license for exclusion/evaluation only. MATH/AIME/MMLU
  snapshots/rights remain unresolved; preparation fails closed without all required
  sets. Authored CI fixtures do not establish real benchmark decontamination.
- Sorted/seeded langdetect can mislabel short equations and code; retain declared
  language separately, annotate by default. PII patterns miss obfuscated contacts,
  names/addresses and novel secrets; phrase quarantine is not a toxicity classifier.
  A learned quality filter requires approved labels, slice metrics and calibration.
- MinHash/LSH has probabilistic false negatives; preserving math atoms sacrifices
  some web dedup recall. Literal n-grams miss translated/semantic leakage. Inspect
  labeled pairs and audit source/domain/family leakage before any production claim.
- Disk-backed indices bound page cache, not total storage or index construction
  time; HF BPE pair statistics still scale with the distinct corpus. Large tokenizer
  fitting on16GB RAM requires a separately approved representative train-only sample.
- Compact document fragments reset positions across blocks and cannot attend to
  previous blocks. Optional cross-document attention changes the learning task and
  is included in the loader state identity. Affine epoch permutations are bijections,
  not uniform N! shuffles; measure ordering effects before long training runs.
- CPU host delivery benchmarks include intentional source reuse and warm filesystem
  cache; they do not demonstrate laptop performance, GPU saturation, cold remote
  storage, native Windows spawn behavior or corpus-scale multi-node durability.
- A loader cursor saved separately from model/optimizer is not an atomic training
  checkpoint. The future real-corpus trainer must capture committed cursor and model
  state together and test full optimization replay. Current data replay is exact;
  the existing100-step trainer still uses numeric engineering fixtures.

Phase4 bounded review (2026-11-01):existing reference/packed-batch tests pass,
but dedicated initialization/residual-scale, non-finite-input policy and combined
tied-embedding gradient regressions remain M4.1 acceptance gaps. Do not infer
full Phase4 completion from the18-test review; preserve these checks before cache/
accelerator extensions. The mapping and scope are in decoder_rfc_review.md.

M4.1 review gaps are closed by R012. M4.2 CPU tests/profile do not establish GPU
kernel choice/VRAM or serving efficiency. Request cache trusts non-mutated tensors;
unsafe .data/concurrent mutation bypasses its contract, and Tensor._version is a
pinned internal dependency. Functional cache concatenation can incur transient
copy cost. Saved graph storage bytes are not process/allocator peak memory.

Packed pretraining now has actual-data/cursor/optimizer CPU tests, including two
ranks; this does not validate CUDA/NCCL, packed FSDP, shared filesystem or multi-node
recovery. The heldout loss is a fixed weighted probe with possible intentional
source reuse, not exhaustive benchmark coverage. Token LR boundaries are explicit;
an optimizer-step cap need not land exactly at decay end. Full artifact checksum
preflight costs a complete input read. Resume permits worker/prefetch/mount changes
only after identity checks, while source/software/topology/model changes reject.
CUDA replay tests are supplied but require real hardware and deterministic setup.

Exact packed RNG/optimizer replay is demonstrated with the default JSON sink.
Optional TensorBoard/W&B validation covers offline outputs; full stochastic
trainer/SDK restart parity is not established and must be checked before relying
on extra SDK state in a stochastic rollout/training workflow.

## M6.1 bounded assistant supervision

Whole-dialog packing leaves unused capacity. Right truncation can discard solution
tails and rejects prefixes that cut in a prompt/header or have no assistant target;
no truncation-policy quality optimum is claimed. The generic formatter preserves
licensed mathematical solutions but verifies only structure, not imported proofs.
Original demo exact answers are independently checked by Fraction/math and SymPy;
no teacher model is used. SFT train/validation families are separate across languages,
but the base engineering demo may see other instances of those task families.
Fixed train/heldout probes can reuse blocks, and low training loss does not establish
answer accuracy, instruction-following quality or production decontamination.

Weights-only initialization needs a trusted pinned full Qvelt checkpoint with the
same tokenizer and architecture; it starts a new optimizer and is not exact base-run
resume. Checksums are integrity/identity checks, not external authorship guarantees.
CPU single-process SFT is the acceptance target; CUDA/distributed SFT, stochastic
SDK restart, third-party checkpoints and production datasets remain unverified.


## Supplied Phase4 closure review

RoPE interpolation(scale>1) is an explicit non-baseline extension,not validated
long-context quality. Adding ModelConfig.rope_scale changes config identity;old
exact run resume remains rejected by source/config checks. Default weights and
values are preserved on tested inputs,which is distinct from full run compatibility.
Deliberate weights-only initialization still requires matching validated model/
tokenizer identities. FP64 gradcheck covers logits;loss retains its FP32 contract.
CPU AOT/Inductor and packed-mask correctness do not establish CUDA compilation,
FlashAttention selection or fused-varlen performance. The fresh operator profile
is short,shared-host,forward/backward only;it excludes Adam/communication and CPU
allocator peak. Imported historical profile claims were not promoted to new runs.

Local Gloo/DDP processes intermittently hang or abort on exit after completed
optimization/checkpoints,including before this import. Loopback/one-thread and
USE_LIBUV=0 settings do not establish a general fix. Final local562 passed/1 DDP
failure is not all-green acceptance;observe current-source CI and preserve this
runtime limitation. Do not mask it with os._exit,relaxed tests or fabricated success.


Inductor development on Python3.11 requires the explicitly pinned dev setuptools;
Torch's transitive marker supplied it only on Python>=3.12. Optional SDKs must not
be relied on to supply compiler tooling. Plain eager package use and validated
`make setup` development have distinct dependency scopes.


## Expanded Phase5 runtime limits (2026-11-04)

- Rollback-and-skip intentionally changes learning: restoring an older finite
  checkpoint discards good intervening updates while retaining the forward cursor.
  Use failure receipts and optimizer_updates; a clean hard-kill replay is different.
- Local managed Gloo processes can abort/stall; current-source clean CI is a separate
  required gate. Never hide these failures with forced exit,xfail or skipped tests.
- FSDP2/CUDA/NCCL,shared-store/multi-node and real server MFU/VRAM are unverified;
  compiled FSDP and TP/PP are rejected. CPU DCP cannot close those hardware gates.
- Domain fixtures contain authored nonces/math,not real production web/code/math.
  Source-heldout probes can wrap tiny datasets. Short four-point fits cannot establish
  universal scaling,LR transfer to small or mathematical reasoning quality.
- Exact recovery binds executable source,software,topology,precision and data identity.
  Retention removes old committed snapshots only when explicitly selected; quarantine
  remains on disk. Log archives can contain prompts,paths and generations; review
  before manually attaching. No external messaging/upload is automatic.
