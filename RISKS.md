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

408 installed skills are pinned instructions/assets, not an audited execution
layer. Some assume different client tools, unpinned libraries, network services
or immediate paid jobs. Read the selected skill and helpers before use; verify
APIs, pin dependencies, preserve root authorization and original Qvelt code.
Six restrictive Anthropic entries are omitted. Inventory hashes detect drift,
not malicious upstream content; upgrades require source/license diff review.

Tokenizer M3.1a: byte coverage applies to whole valid UTF-8 strings; isolated
byte fragments may decode with replacement. Lone surrogates are rejected.
Trainer pair statistics are not globally memory-bounded. Corpus-scale training,
YAML/CLI/manifests and cross-environment reproducibility remain unvalidated.
Ponytail simplicity must retain required validation, tests and reproducibility.
