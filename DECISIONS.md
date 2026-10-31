# Qvelt decision log

Last updated: 2026-10-31. Project owner for implementation decisions: Lead agent.

## Evidence rules

`[verified]` means the cited text was read and supports the stated limited claim.
It does not mean a Qvelt implementation exists or a result was reproduced.
`[unverified]` means a source or its applicability has not yet been checked.
User requirements and local inspection are labeled separately; they are not
research evidence. All decisions below date to 2026-10-31.

The PyTorch references below are fixed to **v2.6.0** for a stable source revision.
This is a documentation reference, **not a dependency version selection**.
At research bootstrap PyTorch was absent. M2.1 now pins/tests PyTorch 2.9.1+cpu
and Python 3.11–3.13; CUDA dependencies resolve but are not hardware-validated.
See ADR-0014 onward. Recheck model/kernel APIs when their modules are implemented.

## ADR-0001 — Start with project memory and explicit evidence gates

- **Status:** accepted for the initial documentation scope.
- **Context:** local inspection found an empty checkout, no project files, and no
  existing commits. The user explicitly requires a documentation-only first step.
- **Options:** immediately write a model; scaffold dependencies first; establish
  a source-aware roadmap and persistent state before implementation.
- **Choice:** create the management documents, contributor protocol, README, and
  empty dataset register. Defer model code, dependency installation, and training.
- **Rationale:** maintain continuity across sessions without pretending that
  planned capabilities are implemented. Checkboxes encode acceptance evidence.
- **Sources:** user mission and first-action instructions (2026-10-31; project
  requirement); local Git/file inspection (2026-10-31; observed). No scientific
  or library claim is made by this process decision.
- **Revisit when:** the user changes the mission or later milestones require a
  different document structure; preserve the decision history.

## ADR-0002 — Separate portable correctness from hardware scale

- **Status:** accepted as an execution-tier contract; implementation is pending.
- **Context:** no training servers are available. The user wants weak-device
  debugging and extensibility to powerful single/multi-node hardware.
- **Options:** require CUDA everywhere; attempt full-size training on every
  device; make tiny CPU correctness and progressively validated GPU tiers explicit.
- **Choice:** require a tiny portable path, then single GPU, one-node data
  parallelism, and eventually sharding/multi-node validation. CPU/Gloo and
  CUDA/NCCL are the initial distributed-backend candidates. Choose the concrete
  sharding API only after version-specific inspection in M5.2.
- **Rationale:** PyTorch documents CPU/Gloo, CUDA/NCCL, and multiprocess DDP.
  Backend support does not establish that this project scales or fits in memory.
- **Sources:** [verified] PyTorch v2.6.0 library documentation,
  [Distributed communication package](https://github.com/pytorch/pytorch/blob/v2.6.0/docs/source/distributed.rst),
  read 2026-10-31: DDP introduction, backend capability table, and backend guidance.
  User hardware statement (2026-10-31; requirement, not a resource measurement).
- **Revisit when:** actual devices, memory, interconnect, budget, backend/API
  support, or measured distributed-correctness results are available.

## ADR-0003 — Define reproducibility within explicit execution conditions

- **Status:** accepted as a requirement; no replay has been implemented.
- **Context:** a seed alone cannot justify reproducibility across versions,
  hardware, data workers, dropout, optimizer state, and resumed jobs.
- **Options:** seed-only runs; an unconditional bitwise cross-device guarantee;
  deterministic ordering and complete state with a bounded replay contract.
- **Choice:** pin dependencies, record resolved configs and artifact hashes,
  control RNG and data ordering, and checkpoint all training progress. Test
  exact/order-equivalent replay on a fixed software/device/world-size setup.
  Define tolerance-based comparisons where arithmetic is not bitwise stable.
  Treat topology-changing resumes as a separate migration capability.
- **Rationale:** capture the state our workflow needs and test it, while respecting
  PyTorch's stated reproducibility limits. Full checkpoint contents are a project
  contract to validate, not a promise made by the documentation.
- **Sources:** [verified] PyTorch v2.6.0,
  [Reproducibility](https://github.com/pytorch/pytorch/blob/v2.6.0/docs/source/notes/randomness.rst),
  read 2026-10-31: no guarantee across releases/platforms or CPU/GPU; seeding and
  deterministic-algorithm guidance. [verified] PyTorch v2.6.0,
  [Serialization semantics](https://github.com/pytorch/pytorch/blob/v2.6.0/docs/source/notes/serialization.rst),
  read 2026-10-31: `state_dict` serialization and loading restrictions.
- **Revisit when:** nondeterministic operators, sharded state, changed topology,
  dependency upgrades, or performance costs change the supported replay contract.

## ADR-0004 — Keep correctness oracles independent of optimized paths

- **Status:** accepted as a testing gate; no numerical module exists yet.
- **Context:** attention, masks, losses, gradients, cached decoding, and parallel
  reductions can produce plausible outputs despite incorrect semantics.
- **Options:** smoke-only tests; compare two calls through the same optimized
  path; use independent tiny references plus functional and integration tests.
- **Choice:** require module unit tests and small independent numerical oracles,
  compare forward values and relevant gradients, and choose documented
  dtype-aware tolerances. Test distributed global-token weighting against an
  equivalent single-process batch and cached decoding against full recomputation.
- **Rationale:** closeness requires explicit tolerances and matching semantics;
  passing a numerical comparison alone does not establish training or reasoning
  quality. The roadmap also requires learning, resume, and evaluation checks.
- **Sources:** [verified] PyTorch v2.6.0 library source/documentation,
  [`torch.testing.assert_close`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/testing/_comparison.py),
  read 2026-10-31: signature, tolerance handling, and comparison contract.
  User unit/reference-test requirement (2026-10-31).
- **Revisit when:** a kernel's precision, determinism, supported dtype, or reference
  semantics change. Do not loosen tolerances merely to hide a regression.

## ADR-0005 — Use explicit provenance and approval before ingesting data

- **Status:** accepted as governance; no datasets are approved.
- **Context:** mathematical corpora, solutions, synthetic teacher outputs, and
  benchmarks can have different licenses, restrictions, and overlap risks.
- **Options:** import popular datasets by default; infer rights from public
  availability; approve exact revisions and uses before ingestion.
- **Choice:** gate ingestion on `DATA_CARD.md` records covering authoritative
  license evidence, revision, permitted use, origin, transformations, and split
  isolation. Unknown rights remain blocked. Do not scrape against site terms.
- **Rationale:** public availability is insufficient provenance. Licensing and
  contamination require separate records; near-duplicate checks are not proof of
  the absence of benchmark leakage.
- **Sources:** user licensing/data-ethics and mathematical-reasoning requirements
  (2026-10-31; governance requirements). No dataset license or legal conclusion
  is asserted. Dataset-specific authoritative sources must be added before use.
- **Revisit when:** a specific source, license, intended use, redistribution plan,
  or teacher policy is supplied and verified. Unclear legal rights need user input.

## ADR-0006 — Defer architecture and objective selection to primary-source review

- **Status:** accepted as a research gate; technical candidates remain undecided.
- **Context:** the user requests original core code and a math-focused model family,
  without a hardware budget or an existing baseline to measure.
- **Options:** choose fashionable components immediately; promise a complex
  mixture of methods; compare a small dense baseline and justified alternatives.
- **Choice:** investigate a dense causal decoder first; decide normalization,
  positional encoding, attention layout, tokenizer, activation, initialization,
  optimizer, and post-training objectives through M1.1/M1.2 and later objective
  gates. Do not claim that combining named techniques produces strong reasoning.
- **Rationale:** a small referenceable design is a research starting point, not a
  selected architecture or a proven quality result. Additional complexity needs
  correctness, resource, and held-out ablation evidence.
- **Sources:** user mission and first-action restriction (2026-10-31). Candidate
  papers below are [unverified] and are not used as verified scientific support.
- **Revisit when:** primary sources, resource estimates, actual hardware, and
  affordable ablations justify a concrete architecture or training objective.

## Research reading queue — not selected designs

Historical bootstrap queue: Phase 1 has now verified the corresponding primary
texts in `docs/research/survey.md` and `sources.json`, including the accessed v1
title of the AdamW preprint. The old labels below record what was unknown at
bootstrap; use ADR-0008 onward for current choices. Scientific verification is
claim-scoped, not empirical validation of Qvelt.

Each item is **[unverified]**: the paper text and its applicability have not yet
been read in this project. Titles/years are starting bibliographic entries to
check, not claimed results. Record exact links/revisions and findings during M1.1
or the corresponding post-training milestone. Official library API checks remain
necessary even after a paper has been read.

| Candidate source | Decision to investigate | Gate |
| --- | --- | --- |
| *Attention Is All You Need* (2017) | Causal decoder baseline, residual/attention conventions | M1.1 |
| *Root Mean Square Layer Normalization* (2019) | RMSNorm versus alternative normalization | M1.1 |
| *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) | Rotary positions, numerical implementation, context limits | M1.1 |
| *GLU Variants Improve Transformer* (2020) | Gated MLP variants and parameter budget | M1.1 |
| *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) | MHA versus grouped-query attention and KV memory | M1.1 |
| *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) | Exact attention optimization and hardware constraints | M1.1, M4.2 |
| *Training Compute-Optimal Large Language Models* (2022) | Parameter/token budgeting and applicability limits | M1.2 |
| *Decoupled Weight Decay Regularization* (2019) | Optimizer/weight-decay semantics and parameter groups | M1.2, M5.1 |
| *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023) | Preference objective and reference-policy semantics | M6.2 |
| *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024) | Verifiable-reward reasoning RL and its assumptions | M6.3 |

Dataset, model-weight, tokenizer-algorithm, external-runtime, and project-license
choices are **not made** by this initial delivery. No source above authorizes
copying implementation code or using an associated dataset.

## ADR-0007 — Isolate bootstrap tooling and publish through a feature branch

- **Date:** 2026-10-31. **Status:** accepted for environment setup.
- **Context:** the user authorizes routine pushes and PR creation. Remote main
  now contains an uploaded project brief and GitHub redirects to `QveltAI`.
- **Options:** alter global Python; prematurely select the ML stack; isolate only
  development tools while retaining the architecture/scaffold gates.
- **Choice:** use `/workspace/qvelt-tools/venv`, a fully pinned hash-locked
  requirements file, and a saved installation script. Preserve both Git histories
  via a merge and publish `qvelt/environment-setup`; retain main for PR review.
- **Rationale:** make tool installation repeatable without changing model
  dependencies or losing the user's uploaded file. Git access and API access
  are distinct: a successful push does not imply PR creation permission.
- **Sources:** user authorization (2026-10-31); [verified] uv 0.12.23 installed
  CLI documentation (`uv pip sync --help`, `uv pip compile --help`), hash
  generation/requirements enforcement; observed successful sync and push.
- **Revisit when:** M2.1 selects package dependencies, or GitHub API access is
  applied. Bootstrap versions are not a claim of latest versions or ML support.

### ADR-0007 amendment — direct main workflow

On 2026-10-31 the user explicitly selected `main` for all development and
publication. This supersedes the feature-branch review preference. Fetch and
integrate current main before normal fast-forward pushes; never force-push.
PR API access is optional and does not block Git publication. Source: latest
user instruction; no new library or scientific claim.

### Publication pause (2026-10-31)

After publishing the environment, the user explicitly said not to republish or
reapply old edits. The current research request authorizes local research/files,
not an explicit lifting of that publication pause. Keep this session's commits
local and do not change the environment draft. This supersedes automatic pushes
for this session; future publication can follow the next user instruction.

## ADR-0008 — Select a dense, tied, GQA baseline and exact size family

- **Date/status:** 2026-10-31; accepted for implementation, empirically unvalidated.
- **Context:** actual hardware is a laptop RTX 3060/Core i5/16 GB RAM; future
  server unknown. Need original, referenceable core code and realistic scaling.
- **Options:** dense GQA+tied embeddings; MHA+untied head; eight-expert top-2 MoE;
  MLA/local attention as separate candidates.
- **Choice:** pre-RMSNorm, RoPE theta=10000, SwiGLU, bias-free GQA, tied E,
  full block-diagonal causal attention. Sizes 19,338,560 / 125,854,464 /
  493,553,920 / 1,510,082,560 unique parameters, with exact dimensions and
  equations in `docs/research/architecture_rfc.md`. No QK-norm by default.
- **Rationale:** lower KV/state cost and simpler correctness oracles. Reference
  full-vocabulary loss and document masks precede optional fused paths. Reject
  MoE's total storage/routing overhead as a baseline, not as universally inferior.
- **Sources:** [verified] *Root Mean Square Layer Normalization* (2019),
  *On Layer Normalization in the Transformer Architecture* (2020), *RoFormer*
  (2021), *GLU Variants Improve Transformer* (2020), *GQA* (2023), *Using the
  Output Embedding to Improve Language Models* (2016 v1), *MobileLLM* (2024),
  *Switch Transformers* (2021), official Llama source. Exact titles/links,
  versions and claim limits are in survey S02/S03/S07/S09/S12/S14/S32/S37/S45.
  [verified] analytical enumeration/hand-case tests; **not** an instantiated model.
- **Revisit:** actual memory/throughput, tokenizer freeze, equal-budget MHA/GQA
  ablations or instability justify a changed design. No hidden method additions.

## ADR-0009 — Freeze the tokenizer budget, not untested tokenizer quality

- **Date/status:** 2026-10-31; accepted design baseline, artifact not trained.
- **Options:** large multilingual vocabulary; fixed small vocabulary; byte-only.
- **Choice:** byte-level BPE, 32768 total symbols including 256 byte symbols and
  32 controls, individual ASCII digits, preserved whitespace/Unicode. Train from
  scratch only after approved corpus selection; token IDs/hash freeze in M3.1.
- **Rationale:** common parameter budget across model sizes and laptop memory
  constraints; digit policy has public precedent but needs our own compression/
  math/Russian checks. No claim of mathematical superiority.
- **Sources:** [verified] *Neural Machine Translation of Rare Words with Subword
  Units* (2015 v1), *Language Models are Unsupervised Multitask Learners* (2019),
  *LLaMA* (2023), survey S43/S44/S13. 32768/32 are explicit project choices.
- **Revisit:** poor Russian/LaTeX coverage, excessive token inflation or repeatable
  reasoning/latency evidence; changes invalidate tokenizer/model/data compatibility.

## ADR-0010 — Use AdamW/WSD and staged budgets as tested hypotheses

- **Date/status:** 2026-10-31; selected starting recipe, no executed optimization.
- **Options:** cosine, WSD, muP transfer; AdamW, Lion, Muon; ~20P versus longer
  inference-oriented training. No optimizer is assumed universally best.
- **Choice:** FP32 AdamW state, supported BF16 compute, beta=(0.9,0.95), clip=1,
  token-based WSD warmup/stable/10% linear decay. Tiny 10M debug targets; small
  10B, base 50B, large 150B extended targets, with ~20P research decision gates.
  Exact size-specific LR/batch/mask/replay semantics in `training_rfc.md`.
- **Rationale:** transparent optimizer reference and extendible stable checkpoints;
  match data/horizon/endpoint when comparing schedules. Token budgets and LRs
  are engineering hypotheses, not predictions from an independently fitted law.
- **Sources:** [verified] AdamW's accessed *Fixing Weight Decay Regularization in
  Adam* (2017 v1), *LLaMA* (2023), *Training Compute-Optimal Large Language Models*
  (2022), *MiniCPM* (2024), *SmolLM2* (2025), *Muon is Scalable for LLM Training*
  (2025), *Symbolic Discovery of Optimization Algorithms* (2023), *Tensor Programs
  V* (2022); survey S13/S15–S20/S30. Source-reported gains are not our results.
- **Revisit:** bounded LR/batch/schedule trials, source capacity, measured budget,
  optimizer/sharding correctness and quality-per-compute support a different plan.

## ADR-0011 — Separate the laptop development path from server training

- **Date/status:** 2026-10-31; accepted planning contract, hardware not benchmarked.
- **Options:** promise all sizes on the laptop; offload large state to 16 GB RAM;
  tiny/short-context pilots plus independently justified server scaling.
- **Choice:** CPU FP32 references, laptop tiny and profiled short-context small;
  single GPU where it fits; DDP first on adequate nodes, FSDP for memory pressure.
  Defer tensor/pipeline/expert parallelism. Assume 6 GiB laptop VRAM only until
  inventory confirms it. No paid/final training or automatic resource provisioning.
- **Rationale:** 16P persistent AdamW state alone is ~22.5 GiB for large. Explicit
  FLOP/memory formulas are not profiler peaks. Include vocabulary head, attention,
  activation workspaces, checkpoint/gather margins, and separate inference KV.
- **Sources:** [verified] PyTorch distributed documentation in ADR-0002, SDPA
  v2.6.0 source (S46), *FlashAttention* (2022; S36). Hardware description from
  the user's reply; estimates are project derivations verified by research tests.
- **Revisit:** exact VRAM/OS/driver, throughput, actual server/interconnect,
  measured memory, budget or a future larger model requires another strategy.

## ADR-0012 — Treat data mixtures as conditional proposals, not approvals

- **Date/status:** 2026-10-31; accepted provenance gate, all ingestion unapproved.
- **Options:** trust wrapper badges; reject research on all public data; inspect
  immutable publisher cards and flag unresolved underlying rights precisely.
- **Choice:** catalog 13 candidate sources in `docs/research/data_mix.md` and
  `sources.json`; token-weighted main/anneal/SFT mixtures and prompt-weighted RL
  with explicit denominators. Current code-data slot has no approved corpus.
  Prefer authored fixtures for software validation; no corpus download now.
- **Rationale:** publisher license assertions and metadata access can be verified
  without claiming every underlying right is cleared. ODC-By/CC legal-text
  endpoints were inaccessible; SmolTalk imported components have their own
  licenses; web/code/contest derivatives need separate review.
- **Sources:** [verified] exact publisher README snapshots, Common Crawl terms,
  Apache-2.0 text; URLs/hashes/access failures in `sources.json`. [verified]
  *The FineWeb Datasets* (2024), *Deduplicating Training Data Makes Language
  Models Better* (2021), *SmolLM2* (2025), *2 OLMo 2 Furious* (2025), survey
  S21/S22/S30/S42. Complete rights verification remains **[unverified]**.
- **Revisit:** authoritative license/terms evidence, per-file provenance,
  tokenizer-measured capacity, user language goals and actual data audits.

## ADR-0013 — Build math supervision before verifier-based RL

- **Date/status:** 2026-10-31; selected staged research direction, not implemented.
- **Options:** immediate long-trace RL; expensive process reward/teacher tools;
  concise verified SFT, rejection sampling, defensible preference pairs, then RL.
- **Choice:** bounded exact-answer fixtures and frozen held-out families; SFT
  first, preference objective when valid pairs exist, GRPO-style G=4 pilot only
  after verifier/correctness gates. Explicit zero-variance/truncation handling,
  no style reward, no host execution of generated code. Tools/Lean optional later.
- **Rationale:** correct answers do not certify traces; small models may have
  no useful RL signal. Fixed token budgets and out-of-family controls are needed
  to distinguish real gains from length, contamination and verifier exploitation.
- **Sources:** [verified] *Chain of Thought Prompting Elicits Reasoning in Large
  Language Models* (2022), *Let's Verify Step by Step* (2023), *STaR* (2022),
  *DeepSeekMath* (2024), *Direct Preference Optimization* (2023), *ToRA* (2023),
  *DeepSeek-Prover* (2024); full titles/versions in survey S23–S28/S40.
- **Revisit:** verified math success rates, legal supervision supply, independent
  verifier audits, and actual rollout cost justify process/tool/formal tracks.

## ADR-0014 — Package a lightweight core with locked, exclusive ML profiles

- **Date/status:** 2026-10-31; accepted and CPU-validated for M2.1.
- **Context:** config inspection should work without CUDA/torch, while actual
  tensor/RNG features need an installed and tested PyTorch backend.
- **Options:** one mandatory GPU-heavy dependency set; unpinned optional imports;
  a pinned base plus mutually exclusive explicit CPU/CUDA uv extras.
- **Choice:** source-layout `qvelt` package, Python >=3.11,<3.14, uv 0.12.23;
  PyYAML 6.0.3/NumPy 2.2.6 core, PyTorch 2.9.1 CPU or cu128 profiles, pinned
  development/build dependencies and checked-in universal `uv.lock` with hashes.
  Build using the locked Hatchling environment, not a newly resolved isolated one.
- **Rationale:** the same PyTorch release exists in both selected indexes, and
  NumPy supports the declared Python floor. This conservative GPU-compatibility
  baseline is not the latest PyTorch release or a promise of ongoing upstream
  support. Pip wheel metadata alone does not preserve uv index routing; document
  uv installation. No package license or public release is selected by setup.
- **Sources:** [verified] uv 0.12.23 project configuration and PyTorch integration
  documentation, PyPI version metadata, official CPU/cu128 index artifacts;
  links in `docs/development/scaffold.md`. [verified] actual frozen CPU installs
  on Python 3.11.14/3.12.14/3.13.5, built-wheel/base-without-torch checks and
  installed API behavior. GitHub CI is defined, not remotely executed.
- **Revisit:** actual laptop/server driver support, upstream patches/security
  fixes, a new Python support range or model/kernel requirements. Every upgrade
  must update the lock and revalidate the declared matrix.

## ADR-0015 — Validate a small explicit YAML schema before runtime work

- **Date/status:** 2026-10-31; accepted and unit/reference-validated.
- **Context:** ambiguous YAML coercion, duplicates, unknown fields and unsafe
  constructors can silently change experimental configuration.
- **Options:** permissive dictionaries; a heavyweight config framework; frozen
  dataclasses plus a bounded explicit SafeLoader subclass and schema checks.
- **Choice:** schema 1 only covers model dimensions and runtime policy. Reject
  aliases/anchors/tags/merges/duplicates/non-string keys and unknown fields;
  bound bytes/depth/events; accept decimal signed-32-bit positive dimensions,
  uint32 seeds, finite positive floats and lowercase true/false. Do not coerce
  quoted numeric strings or bool-as-int. Normalize float fields, resolve defaults
  and hash canonical UTF-8 JSON. No includes, interpolation or file overwrites.
- **Rationale:** a narrow inspectable contract catches errors before allocation;
  it does not guarantee hardware fit or implement training. Future training and
  artifact schemas must be versioned explicitly. Equivalent YAML formatting
  resolves to the same config digest, while meaningful settings change it.
- **Sources:** [verified] PyYAML 6.0.3 SafeLoader/constructor/resolver sources;
  Python 3.12 installed dataclasses/json source and docstrings; canonical digest,
  malformed-input, bounds and frozen-constructor tests. Exact source links are
  in `docs/development/scaffold.md`. Resource limits are project policy choices.
- **Revisit:** schema migration, additional implemented components or measured
  config needs. Never broaden parsing by disabling validation to accept a typo.

## ADR-0016 — Make runtime diagnostics and seed guarantees narrow and explicit

- **Date/status:** 2026-10-31; accepted; CPU validated, CUDA unrun.
- **Options:** seed only torch; implicit GPU fallback; explicit Python/NumPy/
  torch seeds, indexed NumPy streams and a small requested-device diagnostic.
- **Choice:** seed utility consumes validated runtime settings, configures strict
  deterministic algorithms when requested, disables cuDNN benchmarking/TF32,
  and leaves PYTHONHASHSEED/complete checkpoint replay out of its guarantees.
  CUDA requires availability, native BF16 capability when requested, and an
  explicit supported cuBLAS workspace value for strict seeding. Doctor executes
  a real FP32 2x2 matmul on the requested device and reports observed versions;
  native operation errors are reported, never replaced by CPU success.
  JSON logging uses a reserved Qvelt handler without changing the root logger.
- **Rationale:** diagnostics distinguish an installed API, an operational CPU
  path and untested GPU capability. A seed is necessary but insufficient for
  resumable training; M5 must checkpoint all relevant progress/state.
- **Sources:** [verified] PyTorch v2.9.1 reproducibility notes and installed
  `torch.cuda.is_bf16_supported` source (`including_emulation=False`), NVIDIA
  cuBLAS reproducibility documentation, Python logging source; links in scaffold
  design. [verified] CPU RNG comparisons, exact tensor probe, logger isolation,
  missing-runtime and mocked failure paths. Mocks are not CUDA validation.
- **Revisit:** actual CUDA execution, distributed rank/worker RNG contracts,
  checkpoint schemas or dtype/backend changes. No cross-device bitwise claim.

## ADR-0017 — Use strict typed manifests and canonical content identities

- **Date/status:** 2026-10-31; accepted and validated for M2.2 metadata/bytes.
- **Context:** future run/data/tokenizer/checkpoint artifacts need inspectable
  identities without importing models or trusting executable serialization.
- **Options:** arbitrary dictionaries/pickle; raw-file hashes only; immutable
  schema-versioned typed metadata plus registered file hashes.
- **Choice:** schema-1 frozen records, exact required fields and role contracts,
  strict bounded JSON, normalized UTC timestamps, explicit source/license
  declarations and lineage IDs. Canonical metadata ID is SHA-256 of sorted
  compact UTF-8 JSON plus newline; raw artifact IDs hash raw bytes. No implicit
  version migration or arbitrary environment dictionary. Expected metadata IDs
  can be supplied from a trusted creator/registry.
- **Rationale:** another process can inspect and verify without torch or unsafe
  object loading. Format-normalized metadata IDs and raw byte IDs are distinct.
  Neither license strings nor opaque checkpoint roles prove rights or replay;
  those semantic gates belong to M3/M5. Hashes provide integrity, not signatures.
- **Sources:** [verified] installed Python 3.12 `json` documentation/hooks,
  dataclass helpers, `hashlib.sha256`, datetime APIs; actual round-trip, duplicate,
  unknown-field/version, range, identity and corruption tests. Design/limits:
  `docs/development/artifacts.md`; limits are project policy, not library defaults.
- **Revisit:** new implemented formats, a justified schema migration, signatures,
  hierarchical shard manifests or larger metadata needs. Never silently accept
  incompatible versions or relabel metadata tests as training-state validation.

## ADR-0018 — Commit write-once local artifacts before the manifest marker

- **Date/status:** 2026-10-31; accepted, Linux local filesystem tests passed.
- **Context:** interrupted writes, concurrent producers and corrupted inputs
  must not publish a successful-looking partial artifact/checkpoint bundle.
- **Options:** direct overwrite; replace-in-place; private same-directory temp
  files and a no-overwrite hard-link commit followed by directory fsync.
- **Choice:** stream in bounded chunks, flush/fsync temp, link to an absent final
  name, fsync its parent, remove only the owned temp. Each writer confirms all
  parent directory entries, even if another writer created them. Verify all
  declared files/config/lock, then publish the manifest last. Root creation
  requires an existing parent; existing run directories are never reused.
- **Rationale:** caught prepublication failures leave no final name; process
  death may leave uncommitted temps. Postpublication directory-fsync errors
  preserve complete bytes and report durability uncertainty rather than false
  rollback. Growing source hashing is bounded to its initial size plus one;
  source copying has an initial-size ceiling and post-copy checks. No automatic
  deletion of unknown partial outputs, distributed transaction or hostile-root
  sandbox is claimed. Network storage/native Windows need separate validation.
- **Sources:** [verified] Python `os.link`, `os.fsync`, `tempfile.mkstemp`, file
  and stat APIs; Linux [link(2)](https://man7.org/linux/man-pages/man2/link.2.html)
  (existing names are not overwritten) and
  [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html)
  (directory entries require separate sync), read 2026-10-31. [verified] real
  competing writers/SIGKILL tests and injected callback/fsync/link failures;
  these are not physical power-loss or multi-node storage tests.
- **Revisit:** object/network stores, async/sharded writers, validated native
  Windows support or a new checkpoint transaction/recovery protocol in M5.

## ADR-0019 — Capture observed provenance without copying credentials or patches

- **Date/status:** 2026-10-31; accepted and exercised from an installed wheel.
- **Context:** a run should identify code/config/dependencies without assuming
  a remote branch, GPU, clean checkout or credential value exists.
- **Options:** infer remote versions/defaults; dump environment/source patches;
  query actual local Git, installed package metadata and the selected lock.
- **Choice:** full local HEAD, dirty flag, lock hash/copy, resolved config/digest,
  interpreter/OS/machine and installed distribution name/version inventory.
  UUID/UTC identify the bundle; init declares seed/world_size=1 but does not
  apply RNG policy or start training. Unborn/unreadable code is an error, never
  an invented revision. CLI inspect is explicitly metadata-only; verify audits
  registered bytes, config/lock consistency and optional trusted identity.
- **Rationale:** no process-environment values, remote URLs, source patches or
  arbitrary credential files are captured. Dirty status is an observation, not
  a code snapshot or execution lease. Package inventory does not prove lock
  agreement or that installed code equals the checkout. M5 binds actual launch,
  ranks, sampler/RNG/optimizer contents and environment compatibility.
- **Sources:** [verified] Python `importlib.metadata`/`subprocess.run` docs and
  installed Git `rev-parse`/`status`; actual temporary committed/unborn/dirty Git
  tests, installed package inventory, and separate-process no-torch CLI workflow.
  `docs/development/validation_m2_2.md` records exact code and artifact identities.
- **Revisit:** distributed launch, dirty-code snapshots, runtime/hardware capture,
  checkpoint semantic validation or stricter release provenance requirements.

## ADR-0020 — Vendor pinned skill instructions with phase-specific use

- **Date/status:** 2026-10-31; accepted at the user's explicit installation request.
- **Context/options:** floating npx installers versus immutable project-local snapshots.
- **Choice:** 408 skill entries, source-relative assets, license notices and SHA-256
  inventory; omit six Anthropic entries with restrictive or unspecified licenses. Use root agent policy and
  phase-specific read-before-use; exclude third-party helpers from first-party Ruff.
- **Rationale:** reproducible instruction access across agents without running
  installers/hooks or replacing original model code. Vendor payment/publication
  directives remain subordinate to current user authorization. Publish verified
  local work to main following the user's explicit resumption instruction.
- **Sources:** [verified] exact six Git source commits, license texts, SymPy,
  Continuous Claude math and HF trainer entry instructions; full inventory in
  docs/development/skills.lock.json and catalog. Remaining instructions are
  installed, not exhaustively reviewed or executed; API claims remain [unverified]
  until checked for each development task. No upstream popularity/safety claim adopted.
- **Revisit:** source/license changes, verified incompatibilities, malicious or
  obsolete content; upgrades require a reviewed diff and refreshed hashes.

## ADR-0021 — Mandatory ponytail skill and initial byte-BPE API

- **Date/status:** 2026-10-31; accepted, tokenizer subtask M3.1a CPU-tested.
- **Context/options:** ignore attachment versus mandatory coding workflow; whole-text BPE versus ASCII-digit-isolated reversible byte BPE.
- **Choice:** project-wide application of user-attached ponytail in full mode: minimal
  complete changes with required validation and tests. Tokenizers
  0.22.2, no normalization/padding/truncation, all byte symbols and 32 fixed
  controls; default ASCII digit isolation, optional comparison baseline. Ordinary
  text cannot inject control IDs; no BOS/EOS insertion implicit in encode.
- **Rationale:** preserve exact valid UTF-8, whitespace and equations while
  keeping role boundaries explicit. Reject lone surrogates and documents over
  1 MiB. Vocabulary is a ceiling, not a guaranteed trained size on small data.
  Pair-statistic memory still grows with corpus diversity. Fixing these IDs now
  is an artifact compatibility contract, not a mathematical quality claim.
- **Sources:** [verified] tokenizers 0.22.2 installed API docstrings and executed
  BPE/ByteLevel/Split/Regex/BpeTrainer/encode_special_tokens/save/load behavior;
  Apache-2.0 text at https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/LICENSE.
  [verified] read vendored Orchestra huggingface-tokenizers SKILL.md/training
  reference and attached ponytail instructions (MIT declared in frontmatter);
  upstream skill performance claims remain [unverified]. Scoped mypy missing-import
  override documents tokenizers' absent py.typed marker; wrapper typing passes,
  third-party internals are not statically checked. No helper scripts executed.
- **Revisit:** corpus-scale memory/segmentation results, tokenizer upgrades,
  chat templates, proven need for digit policy changes; changing IDs invalidates
  tokenizer/data/model compatibility. YAML/manifest integration stays in M3.1.

## ADR-0022 — Bind tokenizer recipes to ordered byte-verified raw manifests

- **Date/status:** 2026-10-31; accepted and fixture-validated, M3.1 complete.
- **Context:** YAML/CLI training needs inspectable source/config/code lineage.
- **Options:** bare input globs/unversioned JSON; a new data framework; reuse
  M2 strict YAML, manifests, atomic storage and observed provenance.
- **Choice:** path-free schema 1 recipe with canonical data-manifest ID and
  TokenizerSettings. Separate CLI paths. Strict text-only JSONL in canonical
  manifest-path/physical-line order, bounded records/documents, consumed-byte
  SHA-256 and exact document count checks. Publish tokenizer, recipe, source
  manifest, provenance and lock before the final commit marker; never overwrite.
  Semantic tokenizer verification adds recipe/data/control/vocabulary/digit and
  provenance-link checks to generic byte verification.
- **Rationale:** reuse existing boundaries without a downloader, plugin service,
  implicit legal approval or full corpus copies. A stopped iterator is not a
  verified corpus. File growth is bounded by declared bytes; BPE statistics still
  grow with corpus diversity. Metadata declarations do not prove rights, actual
  historical training, installed-lock equality or signed authenticity.
- **Sources:** [verified] installed Python 3.12 json object_pairs_hook docstring,
  buffered file reads, hashlib/dataclasses APIs; PyYAML 6.0.3 SafeLoader rules
  already reviewed in ADR-0015, reused strict parser and executed malformed-input
  tests. [verified] tokenizers 0.22.2 installed API behavior and byte-BPE tests,
  M2.2 POSIX protocol (ADR-0018), lineage/corruption/concurrent-edit and real CLI
  tests. Design/contract: docs/development/tokenizer_workflow.md. Limits/order
  are explicit project policies, not upstream performance claims.
- **Revisit:** actual corpus formats/scale, measured memory, larger manifests,
  schema migrations, remote storage or changed preprocessing/ID contracts.

## ADR-0023 — Gate tokenizer workflow on authored fixtures and measured replay

- **Date/status:** 2026-10-31; accepted, scoped CPU evidence recorded.
- **Options:** infer reproducibility from seeds; unapproved web sample; registered
  authored train/held-out fixtures with exact manifests and independent processes.
- **Choice:** 48/12 authored records with development/test permission; compare
  plain/ASCII-digit BPE V=512. Six clean-commit installed-wheel processes across
  Python 3.11–3.13, varied hash seeds/Rayon counts, matching JSON bytes per policy.
  Keep the researched digit-isolated default without claiming a reasoning gain.
- **Rationale:** confirms a concrete portable workflow, not a production tokenizer.
  Held-out token counts 316 versus 321 on 390 bytes are only tiny segmentation probes.
  Corpus-scale quality/memory, model reasoning and cross-hardware equivalence need
  separate evidence. No production 32768-token artifact is frozen.
- **Sources:** [verified] actual commands/configs/IDs and results in
  docs/development/validation_m3_1.md and tokenizer_fixture_results.json; exact
  authored source hashes/permission in DATA_CARD.md and fixture manifests.
  [verified] previous research basis for byte-BPE/digit policy, architecture RFC
  S13/S43/S44; no additional empirical claim imported from a skill.
- **Revisit:** approved representative corpora, frozen model/data vocabulary,
  hardware/library changes or measurements supporting a different digit policy.

## ADR-0024 — Complete the expanded foundation without replaying M3.1

- **Date/status:** 2026-10-31; accepted; clean-checkout and actual CI gates passed (code 9ee9443/run 41242952575).
- **Context/options:** the user explicitly requested the detailed Phase 2 scaffold
  after M3.1; either repeat existing work, provide fake smoke functions, or extend
  the foundation with the real numerical/training path needed for100 updates.
- **Choice:** primary config/tokenizer/data/model/train/utils packages; scope-only
  README files in future posttrain/eval/inference directories. Preserve existing
  public imports with compatibility exports, flat config paths with internal
  symlinks; canonical model presets live in configs/model. Tests split into unit
  and integration. Scripts are thin entry points; Make/CI/hooks share locked tools.
  A training-job dataclass embeds the unchanged experiment schema; shared strict
  leaf overrides revalidate before hashing. No implicit schema1 digest migration.
- **Rationale:** preserve tested interfaces and original manifest identities,
  fulfill the new request and avoid claiming a real-data trainer exists. Advance
  the RFC decoder and synthetic slice of M4/M5 explicitly; M3.2 and other gates
  remain open. Linux symlink behavior is tested, native Windows layout is not.
- **Sources:** [verified] user Phase 2 specification, existing RFC-001/training RFC,
  installed uv0.12.23/Hatchling1.27/PyYAML6.0.3 APIs, Python dataclasses/pathlib and
  actual import/config/parity tests. Designs: foundation_design.md and ADR-0015.
- **Revisit:** supported platform/package layout or schema changes, real-data needs.

## ADR-0025 — Implement the RFC decoder and a bounded real synthetic trainer

- **Date/status:** 2026-10-31; CPU implemented/numerically tested; GPU unvalidated.
- **Options:** mock loss/steps; an unrelated toy network; a small config of the
  actual selected decoder with independent numerical oracles.
- **Choice:** original pre-RMSNorm, interleaved RoPE, full causal GQA, SwiGLU,
  tied E and bias-free projections, FP32 norm/attention/loss accumulation. Explicit
  document/padding masks avoid all-masked attention rows. Meta model counts for
  all four tiers; real tiny CPU forward/backward. Smoke override117056 parameters,
  B2/T32,100 AdamW updates. Stateless cyclic numeric-symbol documents define
  sample IDs/next-token shifts; no text tokenizer or benchmark quality claim.
  DDP uses world/global-target normalization; clip global L2 at1 after reduction.
  Functional RFC WSD uses valid targets with fixed synthetic windows whose warmup/
  tail boundaries align exactly; no variable-mask/accumulation promise yet.
- **Sources:** [verified] *Root Mean Square Layer Normalization* (2019), *RoFormer*
  (2021), *GLU Variants Improve Transformer* (2020), *GQA* (2023), tied embeddings
  (survey S12) and *Language Models are Unsupervised Multitask Learners* (2019),
  verified in the Phase1 survey/RFC. [verified] installed PyTorch2.9.1 Module,
  autograd, cross_entropy/softmax/AdamW/clip_grad_norm APIs and actual independent
  value/gradient/causality/document-mask/global-DDP tests. No copied model code.
- **Revisit:** actual corpus packing, precision/kernel ablations, accumulation,
  performance/memory, data/evaluation evidence; do not extrapolate fixture loss.

## ADR-0026 — Checkpoint completed updates with explicit replay compatibility

- **Date/status:** 2026-10-31; local CPU/DDP/DCP tested, GPU FSDP2 test provided.
- **Options:** weights only; overwrite latest; manifest-committed state with
  typed lineage, software/config/sample/token counters and rank RNG.
- **Choice:** fresh step directories, fsync/manifest-last checksums; safetensors
  weights and weights_only optimizer/RNG loading; FSDP2 bottom-up wrapping then
  optimizer creation, DCP model/optimizer shards. Completed-step saves only.
  Same resolved job, loaded-source digest, installed environment, Python/Torch/
  NumPy/device/threads/world-size/precision required for replay. Scheduler is
  functional, so saved tokens/step plus full job define its exact continuation.
  Ignore uncommitted directories; corrupt latest fails rather than silently
  falling back. Clear gradients before DCP optimizer template initialization.
- **Rationale:** actual state/next-sample replay, not metadata-only M2.2. Synthetic
  data/tokenizer slots hash saved explicit numeric generator/mapping descriptors;
  they are not a claim to contain actual BPE/data manifests. Real-data lineage
  must use those artifacts later. Normalized pip-compatible installed pins avoid
  capturing direct URLs/editable paths/credentials; uv.lock controls installation.
  Dirty source is identified by loaded Python bytes as well as observed Git.
  DCP metadata remains trusted-project pickle; hashes are not signatures or a
  sandbox for foreign checkpoints. GPU/shared-store/multi-node durability untested.
- **Sources:** [verified] PyTorch2.9.1 installed fully_shard, DCP save/load and
  state_dict get/set/StateDictOptions source/docstrings; _init_optim_state source
  shows stale grads prevent state initialization. Public saver/loader module
  functions are used because the top-level reexports are not explicit to mypy.
  [verified] safetensors0.6.2 actual export/load, Python hashlib/IO/fsync protocol
  (ADR-0018), two CPU ranks/real CPU DTensor shards and restart/RNG/Adam tests.
- **Revisit:** topology migrations, new releases/devices/precision, hostile inputs,
  shared/object stores, asynchronous or very-large-model memory constraints.

## ADR-0027 — Local metrics with optional offline adapters and honest MFU

- **Date/status:** 2026-10-31; implemented and actual local outputs tested.
- **Choice:** typed MetricSink, JSON records per attempt, optional TensorBoard2.19.0
  and W&B0.19.11 offline-only adapters behind explicit extras. No silent backend
  fallback or online upload. Valid targets/s uses max step duration across ranks;
  logging/checkpoint overhead is excluded. Gradient norm is preclip; estimated
  MFU uses the RFC operation estimate divided by explicitly supplied peak FLOPs,
  otherwise null. Never invent GPU peak rates/utilization.
- **Options/rationale:** avoid mandatory SDKs and remote telemetry; per-attempt logs
  preserve evidence after restart instead of truncating newer logs. Rank0 prints
  distributed command summaries so concurrent JSON writes cannot interleave.
- **Sources:** [verified] installed TensorBoard event reader/SummaryWriter and W&B
  init/log/finish APIs; actual event file and offline .wandb output checks. The
  SummaryWriter API lacks full type annotations, so its boundary is explicitly
  cast; first-party interfaces are strict. [verified] RFC FLOP formula/reference
  arithmetic; actual Gloo/DDP tests. No GPU utilization benchmark was run.
- **Revisit:** online backend authorization, production logging/throughput profiling,
  SDK updates, measured peak/capability assumptions or multi-node timing contracts.

## ADR-0028 — Apache-2.0 and an assumed-platform server runbook

- **Date/status:** 2026-10-31; user delegated license choice; Apache default selected.
- **Context/options:** MIT versus Apache-2.0 for original project code/docs.
- **Choice/rationale:** Apache-2.0 with explicit patent terms suits the intended
  open/commercially reusable framework. Retain vendor/fixture notices; code
  licensing grants no external dataset rights. Add LICENSE/NOTICE/contribution
  policy and Ubuntu24.04 NVIDIA server instructions without executing driver,
  scheduler, paid or final-scale jobs. Prefer CUDA12.8 GA driver baseline570.26+
  over assuming older minor-compat drivers support every required feature.
- **Sources:** [verified] user response delegating choice, official
  https://www.apache.org/licenses/LICENSE-2.0.txt , SHA256
  cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30.
  [verified] NVIDIA CUDA12.8 GA release notes, driver570.26 / minor-compat525.60.13
  tables, SHA256345ee17c724fedf831cb7a8cd3b84feb9547915ed835059468f3a07a7fc53cfc;
  installed torchrun/Gloo/NCCL API declarations. Actual GPU/Slurm/server validation
  remains [unverified]; runbook resource headers are labeled assumptions.
- **Revisit:** user licensing direction, contributor/vendor obligations, actual
  server OS/GPU/cluster topology or CUDA/PyTorch upgrades.

## ADR-0029 — Versioned controls and exact original math environments

- **Date/status:** 2026-10-31; implementation and local tests, expanded acceptance pending.
- **Context/options:** preserve all old IDs versus silently redefine reserved IDs.
- **Choice:** V2 retains BOS/EOS/PAD/role IDs0–7, allocates think/end-think/tool-call/
  end-tool-call/tool-result IDs8–12 and19 reserves; load old V1 as V1 without
  conversion. The32-control budget and RFC32768 vocabulary ceiling remain fixed.
  Plain text never creates control IDs. Default single-digit policy remains until
  the separately measured compression study; no math-skill claim follows from it.
- **Math choice:** original bilingual templates, stdlib Fraction/math answers
  cross-checked by SymPy1.14.0 exact arithmetic/solvers. Bounded integer/rational
  reward parser; never evaluate untrusted model text with SymPy or Python.
- **Sources:** [verified] tokenizers0.22.2 installed BPE/AddedToken/ByteLevel and
  serialized tokenizer policy, legacy/V2 round-trip tests; [verified] SymPy1.14.0
  installed Rational/solve/gcd/lcm/binomial/factorial APIs and executed independent
  answer checks. SymPy installed license BSD-3-Clause. Public design/claims from
  Phase1 cited in survey.md; no copied generator/model implementation.
- **Revisit:** reasoning ablations, new control protocol/vocabulary migration,
  multilingual corpus fitting and richer reward/environment security requirements.

## ADR-0030 — Fail-closed licensed curation and benchmark completeness

- **Date/status:** 2026-10-31; local offline gates tested; public coverage incomplete.
- **Choice:** raw manifest contract plus aligned optional metadata/explicit usage
  policy; strict approved HTTPS spec with immutable revision/checksums and retained
  license. Evaluation-only acquisition cannot train. Original synthetic fixtures
  support CI; all research-catalog sources remain blocked absent clearance.
- **Curation choice:** seeded offline langdetect1.0.9 annotation, auditable heuristics,
  pattern PII scrubbing, explicit phrase quarantine; no fake learned classifier.
  SQLite SHA256 exact and datasketch1.6.5 MinHash band candidates, verified Jaccard,
  bounded candidate overflow rejects. Ordered numeric/operator atoms must match
  before near deletion, preserving distinct equations despite similar prose.
  Family hash splits precede train-only BPE.
- **Decontamination:** NFC/casefold math-aware token n-grams, all registered sets,
  complete short-phrase rule. Production always requires GSM8K/MATH/AIME/MMLU/project-final plus
  any additional registered required sets. Authored fixtures never satisfy that
  gate. Missing licenses/snapshots stop preparation, not silently disable it.
- **Options/rationale:** in-memory LSH scales RAM with the corpus; disk index fits
  development hardware. Probabilistic LSH and n-grams have stated recall limits;
  neither proves semantic or translated exclusion. Preserve math numerals/signs.
- **Sources:** [verified] installed datasketch1.6.5 MinHash source/API and MIT LICENSE,
  SQLite3 parameterized index/transaction behavior in actual IO tests; [verified]
  langdetect1.0.9 DetectorFactory local seed/profile APIs, actual Apache-2.0
  LICENSE/NOTICE despite inconsistent MIT metadata. [verified] Phase1 survey S21/S22
  MinHash/dedup evidence. [verified] OpenAI GSM8K immutable README/repository MIT
  license, full hashes and narrowly approved evaluation use in DATA_CARD.md.
- **Revisit:** labeled dedup/PII/LID audit, learned classifier approval, benchmark
  rights/coverage, multilingual semantic leakage and corpus-scale index profiling.

## ADR-0031 — Compact packed shards and consumer-committed loading

- **Date/status:** 2026-10-31; local value/replay tests pass, host profiling pending.
- **Choice:** raw little-endian uint16 (V<=65536) / uint32,24-byte segment and16-byte
  block indices, readonly lazy memmaps with checksums and bounded caches. One
  BOS/text/EOS frame per document; one-token overlap between windows retains
  every original adjacent target once; default document-causal attention, explicit
  cross-document causal option. Padding never contributes loss.
- **Sampling/resume:** stage-weighted sources, coprime affine seeded per-source
  epoch permutation (bijection, not uniformly selected from all N! permutations).
  Global source cursors allocate disjoint rank slots before intentional exhaustion
  reuse. Only delivered batches advance saved state; ordered speculative worker
  prefetch never does. Worker/prefetch changes supported; topology/batch/attention/
  data/seed/schedule changes rejected. Loader seeding uses its own Torch generator.
- **Alternatives/rationale:** expanded six-array samples waste disk; saving the
  DataLoader's prefetched sampler cursor loses data on restart; random full-corpus
  permutations consume RAM. Copy only active batch arrays to int64 tensors.
- **Sources:** [verified] installed NumPy2.2.6 memmap/dtype serialization; PyTorch2.9.1
  DataLoader signature/source (in_order,spawn,prefetch_factor,generator) and actual
  0/2-worker bit-exact replay tests. Existing Qvelt decoder/loss numerical references
  validate masks. No GPU throughput or uniform-permutation randomness claim.
- **Revisit:** measured corpus-scale I/O/page-cache limits, scalable alternative
  permutations, cross-topology migration or pinned-memory GPU training evidence.

Phase3 lifecycle refinement (2026-10-31): actual GitHub run41247260902 passed
Python3.11/3.12 and smoke jobs but failed3.13 because SQLite connection context
management commits/rolls back without closing. Use contextlib.closing, with explicit
success commits and tests proving close on both success/error. [verified] installed
Python3.13 sqlite3 Connection/contextlib docstrings and actual3.13 test execution;
no ResourceWarning suppression. A stronger crash test abandons a live iterator
with prefetch pending and compares against a separate uninterrupted oracle. That
reproduced a worker shutdown abort during Tensor IPC. Batch workers now collate
plain NumPy arrays over IPC, converted to tensors in the consumer; this avoids the
shared-Tensor handle lifecycle in that reproducer at the cost of a batch copy.
[verified] installed torch2.9.1 default collate shared-storage implementation and
executed early-stop/replay tests. No private shutdown API is called. Re-profile
GPU/pinned-memory transport before a server-throughput claim. HTTPS spec validation
also explicitly rejects non-text and whitespace/control-byte URLs.
