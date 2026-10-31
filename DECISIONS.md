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
