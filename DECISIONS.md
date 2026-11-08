# QVELT decision log

Last updated: 2026-11-07. Implementation lead: Lead agent; human review gates in ADR-0061.

## Evidence rules

`[verified]` means the cited text was read and supports the stated limited claim.
It does not mean a QVELT implementation exists or a result was reproduced.
`[unverified]` means a source or its applicability has not yet been checked.
User requirements and local inspection are labeled separately; they are not
research evidence. Entries carry their own dates; initial decisions date to
2026-10-31. The owner supplied the complete QVELT strategic specification in the
conversation on 2026-11-07 (Q1-Q19 and principles 1-5). Its requirements are owner
requirements; funding, access, team and competitive statements are not independently
verified facts. Amendments below supersede earlier prospective choices explicitly,
without changing historical artifacts or claiming implementation.

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
claim-scoped, not empirical validation of QVELT.

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
  JSON logging uses a reserved QVELT handler without changing the root logger.
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

- **Date/status:** 2026-10-31; offline acceptance complete; measured receipts and green6-job CI recorded.
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

- **Date/status:** 2026-10-31; local value/replay and host profiling verified; accelerator performance unverified.
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
  0/2-worker bit-exact replay tests. Existing QVELT decoder/loss numerical references
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

The documented M3.3 learned-quality proposal uses fixed-seed hashed character
ngrams and a regularized logistic head before a multilingual encoder, to fit a
small labeled EN/RU baseline within limited RAM. This is a future, untrained
proposal; removal thresholds require independent slice precision/recall/false-drop
measurements. [verified] installed PyTorch2.9.1 BCEWithLogitsLoss documentation/
logit0 reference supports the loss definition only, not quality-prediction accuracy.
No classifier placeholder or claimed learned quality result is shipped.

## ADR-0032 — Review the existing portable decoder before extending Phase4

- **Date/status:**2026-11-01; bounded review complete, M4.1 still in progress.
- **Context/options:**the original Phase4 checklist predates the decoder built in
  M2.3. Rewriting it would duplicate tested code; alternatively inspect and reuse it.
- **Choice/rationale:**reuse the original implementation, map RFC equations to code
  and existing reference tests, verify the Phase3 packed-data interface, and keep
  uncovered initialization/non-finite/tied-gradient checks explicit in the plan.
  No architecture or runtime behavior changes in this task.
- **Sources:**[verified] RFC-001, model/layers.py,model/decoder.py,model/loss.py,
  shared config and18 executed CPU tests at91e9dc8. Full mapping/results are in
  docs/development/decoder_rfc_review.md; public component sources remain in RFC-001.
- **Revisit:**a failing independent oracle, architecture/config changes or the
  implementation of cached/optimized paths. Review is not GPU/quality validation.

## ADR-0033 — Close portable numerical coverage without rank-local sanitization

- **Date/status:**2026-11-01; implementation-reference tests added.
- **Choice:** fixed-seed exact repeatability and sufficiently large-matrix mean/
  variance checks for RFC residual initialization; separate input/output embedding
  gradients must sum to the tied parameter gradient. Preserve IEEE NaN/Inf
  propagation and the existing trainer's collective finite-loss check before any
  optimizer step. Even masked non-finite logits surface as a non-finite loss;
  empty-target zero loss is defined only for finite logits. No silent clipping or
  replacement, and no layer-local error that could strand distributed peers.
- **Options/rationale:** per-layer eager scans/sanitization change performance or
  hide instability; an all-rank training boundary is the established fail-stop
  location. The current numeric fixture rejects globally empty windows as invalid.
- **Sources:**[verified] RFC-001 initialization/loss equations; current original
  model/runner and executed independent gradient/variance/NaN tests; PyTorch2.9.1
  autograd and clip_grad_norm_(error_if_nonfinite=True) installed interfaces.
- **Revisit:** precision/backend changes, stochastic layers, new distributed trainer
  or a deliberately different empty-window policy. GPU behavior remains unverified.

## ADR-0034 — Explicit model execution and request-owned KV state

- **Date/status:**2026-11-01; CPU acceptance and clean-code profile verified; green6-job CI41287953291.
- **Choice:** frozen ModelExecution separate from learned ModelConfig; reference/
  PyTorch SDPA backend and non-reentrant block recomputation recorded in the resolved
  training job/resume identity. Preserve default reference behavior and parameter
  names. Reference QK/softmax runs outside autocast to enforce RFC FP32 semantics.
- **Cache:** original functional forward_cached API returns new compact KV heads
  and shared packed metadata; no global/model-owned request state. Offset-aware
  causal mask supports multi-token suffixes/reset positions/padding. Weak owner
  reference plus parameter identity/data_ptr/_version/config/execution validation
  rejects stale/cross-model state; bounded context, fixed compute policy. Caller
  must not mutate .data/cache tensors or model concurrently. Concatenation is a
  correctness baseline; paged allocation belongs to Phase8, not a speed claim.
- **Alternatives/rationale:** storing expanded query-head K/V wastes memory;
  is_causal=true alone misaligns rectangular suffix attention and cannot isolate
  packed documents. Mutable implicit caches hide lifetime/error semantics. Cache
  returns suffix logits rather than introducing an ambiguous model.decode wrapper
  for the RFC's earlier conceptual decode interface.
- **Sources:**[verified] installed PyTorch2.9.1 scaled_dot_product_attention docstring
  (boolean allow mask,dropout0,enable_gqa,math fallback), CPU GQA probe/value/gradient
  tests; checkpoint docs/source recommend explicit use_reentrant=false and warn
  about state changes. [verified] Tensor.data_ptr/_version mutation probes,
  load_state_dict signature and actual stale-cache regressions; _version is a
  pinned internal dependency, not a cross-version guarantee. Original public RoPE/
  GQA architecture citations remain in RFC-001/survey (2021/2023 component papers).
- **Revisit:** torch upgrades, model mutation/concurrency API, measured cache copy
  cost or availability of real GPU kernel/dtype/VRAM evidence. CPU agreement is not
  GPU validation; do not import skill speed claims as measured results.

## ADR-0035 — Bounded CPU graph-storage and execution profiling

- **Date/status:**2026-11-01; profiler implemented and storage/recompute tests pass.
- **Choice:** a small resolved CPU FP32 job, warmup/repeated timing, saved_tensors_hooks
  unique underlying nonparameter storage and actual cache-storage bytes. Report
  exactly those quantities, not allocator/process peak memory. Cached and uncached
  timings include prefill and compare logits before measuring. No optimizer update.
- **Sources:**[verified] installed PyTorch2.9.1 saved_tensors_hooks documentation,
  untyped_storage data_ptr/nbytes APIs and exact12-float/48-byte duplicate-view oracle;
  checkpoint graph-storage reduction and RNG/thread restoration tests.
- **Revisit:** actual GPU profiler/allocator metrics, longer repeated measurements,
  resource isolation or dynamic/stochastic training graphs. No hardware speedup
  follows from a tiny single-host comparison.

## ADR-0036 — Packed loss sums and explicit valid-token scheduling

- **Date/status:**2026-11-01; CPU actual-data acceptance verified:code557e4da,522-test/7-job CI evidence.
- **Choice:**reuse TrainingJob/model/AdamW/metrics with strict PretrainingJob. Require
  matching actual tokenizer/data vocabulary/context and explicit demo admission.
  An explicit TokenSchedule owns warmup/stable/linear decay at completed valid-token
  window end; embedded legacy step schedule fields must be0. Step horizon remains
  an independent stop cap. No hidden padding-capacity or microbatch-mean LR axis.
- **Accumulation:**sum loss gradients across the actual window, normalize once by
  global valid targets; compensate DDP world-average, no_sync intermediate batches.
  A shorter final window uses its own actual count. Validation is a separately
  seeded weighted heldout probe with isolated Torch RNG, never train-cursor state.
- **Sources:**[verified] installed PyTorch2.9.1 DDP.no_sync/reduction behavior,
  autograd/clip APIs; actual independent single-global-batch gradient comparisons
  with uneven masks and a two-rank Gloo oracle (including a0-target rank/microbatch).
  WSD context/paper evidence remains in the Phase1 survey; this exact piecewise
  token recipe is a project choice, not an asserted optimum or quality result.
- **Revisit:**measured optimization studies, a different token-stop policy,
  packed FSDP/AMP variants or exhaustive evaluation requirements. Current CUDA
  production performance and shared-store behavior remain unverified.

## ADR-0037 — Atomic application checkpoints and split-bound loader cursors

- **Date/status:**2026-11-01; original checkpoint engine extended and locally tested.
- **Choice:**actual data/tokenizer canonical manifests replace numeric descriptor
  lineage only for packed runs. Store each rank's committed consumer state, resolved
  recipe and actual token counters with model/Adam/RNG in one manifest-last commit.
  Application validators run on all ranks before model state is loaded; numeric
  trainer rejects application checkpoints. Normal metadata/hash/version/topology
  checks still apply. Coordinate rank-local validation/IO errors before collectives.
- **Identity:**include split in loader fingerprint (an omission found during this
  integration); bind schedule/accumulation/source/precision/attention/batch/world.
  Exclude only worker count,prefetch,data mount path after actual data hash checks;
  keep those resource observations in attempt/checkpoint recipes. Unsafe state
  rewriting/cross-topology migration is not claimed. Full source digests reject
  older executable versions rather than silently treating them as exact replay.
- **Sources:**[verified] existing ArtifactStore/manifest/RNG/safetensors engine,
  actual partial-window/worker-change/corruption tests and two-rank packed optimizer
  replay; PyTorch2.9.1 installed state-dict/collective interfaces. No new serializer
  or copied third-party trainer implementation. Backend durability remains the
  tested local POSIX filesystem contract, not arbitrary shared storage.
- **Revisit:**storage backend/hardware evidence, explicit migration, authoritative
  dataset revision changes or stochastic-worker transformations requiring more state.

FP32 factory refinement (2026-11-01):an executed ambient-float64 probe showed the
old layer constructors inherited float64 parameters despite the RFC FP32 seeded
initialization contract. Embedding/Linear/norm factories now explicitly request
FP32 (the installed PyTorch2.9.1 dtype keyword was inspected), preserving ordinary
CLI values and making identical seeded weights independent of ambient default
dtype. Explicit later .to(dtype=...) remains available. A regression compares both
ambient settings tensor-exactly; this corrects the initialization contract rather
than changing architecture or claiming new mixed-precision hardware evidence.

## ADR-0038 — Writable project-local development caches

- **Date/status:** 2026-11-04; implemented for the managed CPU development environment.
- **Context/options:** the inherited uv cache is under a read-only home directory,
  so ordinary `make setup` fails before dependency installation. Requiring a home
  permission change or repeating absolute machine-specific paths is unnecessary.
- **Choice/rationale:** Make exports overridable `UV_CACHE_DIR` and
  `PRE_COMMIT_HOME` defaults under the ignored repository `.cache/` directory.
  Keep the pinned lock, CPU/CUDA profiles and dependency versions unchanged.
  Document shell exports for direct uv/Git-hook invocations; Make exports affect
  its child commands, not the caller's shell. No proxy or TLS policy changes.
- **Sources:** [verified] actual uv 0.12.23 read-only-cache error, successful frozen
  installation, Make-exported cache paths, hook installation and execution in this
  checkout. Explicit network permission enables the inherited managed proxy and
  local distributed sockets; cache configuration does not grant network access.
- **Revisit:** cache storage limits, changes to the execution sandbox or tool
  versions. CPU readiness does not establish CUDA or final-scale readiness.

## ADR-0039 — Whole-dialog assistant supervision and weights-only SFT initialization

- **Date/status:**2026-11-04; implemented; bounded CPU acceptance verified in validation_sft.md.
- **Choice:** reuse the original packed training/checkpoint engine. Layout1 retains
  all-token pretraining; layout2 derives assistant-only target masks from explicit
  role IDs. Whole-dialog segments avoid needing a role-state sidecar for split
  documents. Prompt tokens remain valid attention context. Only assistant content
  and end-turn supply loss; outer EOS, headers and padding are excluded.
- **Truncation:**right-prefix retention reserves masked outer EOS, never synthesizes
  a learned end-turn, and rejects a cut without a nonempty final assistant turn.
  This loses long response tails but never pretends they were completed. Reports
  retain truncation counts. Complete-turn truncation and fragment-sidecar schemes
  were alternatives; they would drop useful response prefixes or add storage state.
- **Initialization:**pin the canonical QVELT checkpoint manifest, verify registered
  files, exact architecture and tokenizer, load finite safetensors weights only;
  create new Adam/schedule/RNG counters. Ordinary resume retains the entire SFT
  application state. Bind the objective and base digest, exclude relocatable base
  path from recipe identity. Distinct SFT application kind prevents objective drift.
- **Evidence/sources:**[verified] QVELT controls for tokenizer versions1/2, original
  loader/loss/checkpoint implementations; executed explicit shifted-token loss and
  logsumexp gradient oracles. Read the pinned Orchestra TRL SFT guide for chat/
  prompt-completion/packing context; its APIs, speed and hardware claims were not
  imported or treated as verified installed behavior. No TRL dependency was added.
- **Scope:**original Apache-2.0 exact-answer EN/RU supervision, reused base tokenizer,
  family splits and fixed train/heldout probes. Family isolation is within SFT;
  base demo task families overlap and no benchmark-contamination claim follows.
  Existing source-digest/software checks still reject historical exact resume when
  executable code changes; adding recipe defaults does not waive this contract.
- **Revisit:**tool/reasoning roles, production data rights/audits, demonstrated long
  conversation needs or measured packing waste. CUDA/distributed SFT needs separate
  runtime tests before being advertised as validated.

## ADR-0040 — Selective Phase4 closure on top of current SFT

- **Date/status:**2026-11-04; selected user patch integrated; corrected-source CPU matrix/8-job CI41909315717 verified.
- **Context:**the supplied83910ae patch targets260b431 and reuses ADR0038/R015,
  which now identify environment recovery. Applying its state files would discard
  current M6.1 work. Select source/tests/docs hunks and retain current memory.
- **Choice:**linear position interpolation via ModelConfig.rope_scale(default1,
  finite>=1), preserving baseline architecture and parameter shapes. At-least-FP32
  norm/RoPE/reference-softmax retains FP64 for numerical parameter-Jacobian tests.
  Loss intentionally stays FP32. Existing Decoder/cache/execution/factory remain.
  Keep trust-boundary input validation while permitting compile graph breaks.
- **Alternatives:**omit the supplied improvements; import all files/old receipts;
  replace the chosen RFC with optional QK-norm/MoE/MLA/sliding-window variants.
  The selected small changes close actual validation gaps without those changes.
- **Evidence/sources:**[verified] installed PyTorch2.9.1 promote_types,gradcheck fast
  mode,functional_call and compile AOT/Inductor behavior through executed tests.
  Default packed/padded logits,gradients and seeded weights match pre-import
  FP32/BF16 reference/SDPA values tensor-exactly on recorded seeded inputs.
  [unverified] Chen et al.2023,"Extending Context Window of Large Language Models
  via Positional Interpolation" was not reread; only the explicit linear rule and
  its identity oracle are claimed,not long-context quality. The project-local
  nanoGPT architecture guide informed review only; no vendor model code or speed
  claim was imported. Current primary component sources remain in RFC/survey.
- **Identity:**new ModelConfig field changes config hashes even at default1, and
  executable-source identity changes. Existing source/config resume rejection
  remains mandatory; historical runs are not silently migrated. Shapes/seeded
  default values remain compatible,which is distinct from exact run resume.
- **Revisit:**real long-context data/evaluation,CUDA dispatch/VRAM evidence,
  compiler/version upgrades or a requirement for fully captured host validation.

## ADR-0041 — Explicit developer dependency for CPU Inductor

- **Date/status:**2026-11-04; pinned,lock updated; CPU3.11/3.12/3.13 and all8 CI41909315717 jobs verified.
- **Context:**actual CI41908101017 at886ce61 passed7/8 jobs,including full CPU3.12/
  3.13 and all data/training/SFT jobs. CPU3.11 had560 passed/2 optional SDK skips/
  5 GPU deselected but its real Inductor test failed:torch.utils.cpp_extension
  imports setuptools,which the locked Torch dependency declares only on Python>=3.12.
  Optional SDKs had masked this missing developer dependency locally.
- **Choice:**declare setuptools==84.2.1 in the dev group,already present at exactly
  that version in uv.lock. Update only project dependency edges/resolution ordering;
  no package versions or distribution hashes are upgraded. Keep the compile test,
  do not skip it on Python3.11 or substitute eager for Inductor.
- **Sources:**[verified] actual CI stack and installed Torch2.9.1 import behavior;
  existing lock markers; installed setuptools84.2.1 metadata (MIT,Python>=3.10),
  frozen synchronization and local Inductor execution. This is compiler tooling,
  not a new model architecture/runtime numerical dependency.
- **Revisit:**PyTorch/compiler upgrades,developer packaging policy or removal of
  runtime kernel compilation. Compiler/hardware requirements remain separate.

## ADR-0042 — Expanded pretraining recovery and measured portable controls

- **Date/status:**2026-11-04; implemented;all9 current-source CI42069685890 jobs pass.
- **Context:**the supplied Phase5 WIP predates current SFT and omits the requested
  acceptance experiment. Select compatible code and preserve objective/base lineage.
- **Choice:**reuse token-sum accumulation and atomic checkpointing; add validated
  WSD/cosine schedules,fused AdamW opt-in,block compile,source probes,samples,z-loss,
  spike/layer metrics,auto-resume/quarantine and explicit retention. Version both
  pretraining/SFT application states. TP/PP>1 and compiled FSDP remain rejected.
  FSDP2 uses bottom-up wrapping,FP32 master/reduction,BF16 gathered compute,DCP and
  portable DTensor AdamW; actual GPU validation is still required.
- **Recovery:**SIGTERM records a flag and checkpoints at a common window boundary.
  Recoverable bad windows complete backward collectives before discarding gradients.
  Check weights/Adam for nonfinite state after every update. Rollback restores finite
  model/Adam/RNG but keeps the advanced data cursor; good updates since the checkpoint
  are also discarded. Log consumed targets,scheduled windows,actual optimizer updates,
  failure cursors and rollback separately. Save an initial finite step0 checkpoint;
  reset DCP's lazily initialized Adam counter to0,with first-update parity test.
  Corrupt bytes may be quarantined; semantic/source mismatch must still fail closed.
- **Schedule deviation:**retain ADR0036 end-window LR evaluation rather than the
  original RFC warmup/tail split. Mask the global rank-ordered final-window prefix to
  finish at exactly D valid targets; consume its unused remainder without supervision.
  Skipped exposure advances the schedule. Batch sweeps match nominal capacity only;
  actual targets/order need measurement. Mixture phases remain loader-microbatch based.
- **Alternatives:**blindly import old state/SFT files; silently emulate TP/PP; treat
  failed windows as exact uninterrupted training; claim GPU results from CPU DCP.
  These would violate compatibility,run identity or evidence requirements.
- **Sources/evidence:**[verified] installed Torch2.9.1 fully_shard/MixedPrecisionPolicy,
  DCP state-dict/Adam initialization behavior and Module.compile interfaces inspected;
  local project FSDP2 skill/reference guidance reviewed. Numerical CE/z-loss gradients,
  schedule references,finite-state recovery and actual full tiny200-step SIGKILL replay
  executed. DCP/DTensor CPU coverage does not verify CUDA/NCCL/FSDP runtime.
- **Revisit:**hardware/runtime validation,production source clearance,measured compile
  benefit,and a separately versioned token-boundary mixture/schedule implementation.

## ADR-0043 — Original sigmoid DPO with explicit frozen SFT lineage

- **Date/status:**2026-11-05;portable CPU accepted;all10 CI42073726045 jobs pass.
- **Context/choice:**M6.2 follows accepted SFT and Phase5. Choose sigmoid DPO over
  continuing positive-only SFT or adding reward-model/PPO rollout machinery. IPO/
  hinge variants appear in the local guide but their primary sources were not
  verified;no alternative objective or vendor runtime code is imported.
- **Math:**minimize softplus(-beta*((logp_chosen-logp_rejected)policy-
  (logp_chosen-logp_rejected)reference)),beta>0. Sequence logp SUMs cover response
  content/end-turn only,prompt/roles/PAD/outerEOS excluded. Preserve FP64 references,
  at-least-FP32 low-precision accumulation. Reject invalid/empty/nonfinite inputs.
  Length averaging would change the chosen objective and is not applied.
- **Reference/replay:**policy and frozen eval/no_grad reference start from the
  same byte-verified SFT weights. Check SFT application kind and objective;an ordinary
  pretraining checkpoint cannot masquerade as SFT. No logp cache;the reference must
  remain available on resume. Reuse AdamW,step schedules and atomic checkpoint/RNG
  engine;bind pair/reference hashes,beta,job,software/source and cyclic pair cursor.
  Exclude relocatable paths from recipe identity. CPU FP32/single/JSON only;unsupported
  modes fail explicitly. No zero-training or fake distributed preference path.
- **Data/scope:**register D-PREF-01 original Apache-2.0 verifier-correct versus
  exact gold+1 labels before generation;check both with the existing bounded verifier.
  Split32 pairs by whole family:12 train,16 validation,4 unused test. Strict complete
  single-turn frames;do not truncate away decisions or silently accept duplicates.
  Prepared data retains source/permission/tokenizer identities and is demo-only.
- **Sources/evidence:**[verified] Rafailov et al.,2023,Direct Preference Optimization,
  arXiv2305.18290v2 section4 Eq7/gradient and appendixB read at
  https://arxiv.org/html/2305.18290v2 . Executed SymPy derivative identity and independent
  NumPy value/analytic-gradient checks;masked sequence and unpadded model references.
  Local pinned Orchestra TRL/DPO guide informed review;TRL was not installed and
  guide performance/hardware recommendations are not verified QVELT results.
- **Limits/revisit:**reference memory/additional forward,length effects,simplistic
  negatives and cross-stage family overlap. Training fixture learning with worse
  heldout DPO loss does not establish reasoning improvement or justify promotion.
  Revisit with independent licensed data,hardware and fixed-budget generation tests;
  reference caching,multiturn/GPU/distributed modes need separate acceptance.

## ADR-0044 — Bounded original outcome-GRPO,not DPO model promotion

- **Date/status:**2026-11-05;CPU prototype accepted;all11 current-source CI42081751910 jobs successful.
- **Context:**R020's identical generation control yields base0/SFT3/DPO0 correct
  of16. Preserve the DPO negative finding and use a fresh SFT reference. Historical
  pre-RoPE SFT manifests fail the current config-integrity contract;do not bypass
  it. Rebuild the80-update original SFT fixture under current schema for RL.
- **Choice:**outcome-GRPO with original exact-fraction correctness;no critic,
  learned reward,external teacher,tool execution or forced gold sampling. PPO would
  add a value model;DPO is an offline pair objective. Restrict to registered demo
  generator,CPU FP32,<1M parameters,one update/fresh on-policy batch,temperature1.
  Record policy/reference identities,all sampled actions/logps,stops,masks and RNG.
- **Math/source:**[verified] Shao et al.2024,DeepSeekMath,arXiv2402.03300v3,
  section4.1.1 Eq3/4 and4.1.2 read at https://arxiv.org/html/2402.03300v3 . Use
  clipped likelihood-ratio advantages and direct sampled exp(delta)-delta-1 KL
  penalty,per-completion normalization. SymPy derivative/curvature/minimum checks
  and independent NumPy values/analytic gradients executed. The local TRL online-RL
  guide informed review;vendor reward/speed/hardware examples are not imported.
- **Explicit extensions:**population std,max(std,1e-6),zero advantage for constant
  groups;one-second post-call budget for a trusted130-character exact parser.
  No claim of a preemptible arbitrary-verifier sandbox. Nonfinite/overflow errors
  stop;zero gradients skip Adam. Reference-only KL updates and policy-signal
  updates are counted separately. No stale multi-epoch reuse or alternate sampler.
- **Lineage/recovery:**reuse shared SFT provenance/weights checks and atomic
  checkpoint engine. New GRPO job has its own optimizer settings and pins parent
  SFT recipe bytes/reference hashes. Restore model/Adam/RNG,prompt cursor,counters
  and rollout digest. Verify reference stays unchanged and policy does not mutate
  during sampling. Keep SFT test families unused. CLI/smoke reject unknown options.
- **Evidence/limits:**8 real nonzero policy updates,745 actions,11/32 constant
  groups;stop4/resume8 exact losses/rollouts/weights/Adam/RNG/cursor. Validation
  remains1/8 correct. Near-zero scalar objective at ratio1 is not zero gradient
  or evidence of quality. No public benchmark,improvement,production/GPU/distributed
  or full-scale promotion claim. Revisit with independent licensed data,heldout
  fixed-budget gains,resource evidence and isolated tool/proof verifiers.

## ADR-0045 — Prospective frozen project holdout and original evaluation protocols

- **Date/status:**2026-11-05;freeze88db69e;CPU harness accepted on83a1adb,all12 CI42092305795 jobs successful.
- **Context:**RFC002's planned200-validation/800-final project set did not exist.
  Historical checkpoint learning already occurred;do not claim retrospective
  preregistration or public benchmark status. Freeze prospective IDs and checksums now.
- **Choice:**five original exact-answer templates,seed20261105,new family namespace
  and small/large latent parameter bands. Two-equation latent rationals have distinct
  integer parts across bands,avoiding reduced-ratio cross-partition collisions.
  Verify stdlib/Fraction gold with SymPy;200/800 unique questions;explicit eval/
  decontamination-only permission. Store metadata/IDs in Git,raw corpus outside.
- **Harness contract:**own strict YAML/manifest protocols,whole/final-line bounded
  answer parsing,choice likelihood and token-weighted rolling LM controls. Failed,
  invalid and truncated attempts remain in generation denominators. Declare n/k,
  isolated seeded sampling,exact combinatorial pass@k and uncertainty assumptions.
  Audit declared sources without claiming complete semantic contamination clearance.
- **Alternatives:**rename earlier tiny validation examples as final benchmark;
  silently adopt vendor task prompts/scores or execute custom task Python. Existing
  QVELT models do not need a Transformers/lm-eval dependency for these contracts.
- **Sources:**[verified] RFC002 section6;project-local pinned Orchestra evaluation
  skill/custom-task reference read for task/version/logging patterns. Chen et al.2021,
  Evaluating Large Language Models Trained on Code,v2 section2.1/appendixA Eq1,
  https://arxiv.org/html/2107.03374 read for the combination estimator only;no dataset
  was ingested. NIST e-Handbook PRC7.2.4.1 Wilson interval formula read at
  https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm ;metric implementation
  and independent mathematical checks remain separate acceptance gates.
- **Limits/revisit:**templates are syntactically distinct,not proven semantically
  novel. Historical training history is not certified clean. Final set stays unscored
  in acceptance;public corpus rights,GPU/sharded evaluation and promotion need
  additional contracts. Future training must register this snapshot's exclusions.
- **Executed evidence:**exhaustive n1..7 pass@k subset references,high-precision
  Wilson score-equation roots and scalar Python nonuniform teacher-forcing checks
  pass. Focused42 tests;Ruff/format,mypy146 files and packaging pass. Repeated
  untrained controls/validation give identical predictions/metrics/audits.
  Historical SFT80 gets5/200 with one constant answer;all200 complete. No gain or
  promotion. Raw-source superset lexical alerts0 do not establish complete history.
  Context/generation CPU acceptance budgets differ explicitly from RFC002's512-token
  quality proposal. See validation_evaluation.md;full/CI receipts follow separately.
- **Final acceptance:**CPU3.11/3.12/3.13 each664 passed,2 optional SDK skips,7 GPU
  deselected;SDKs independently verified. Dedicated evaluation paired outputs/
  metrics/audits exact on digest0f742e0d9c94a3165f9fc2205d894a9a58a0e2c0a659a5d149e2fa2ac09209fd,
  weights unchanged,final800 unscored. Current-source200-step SIGKILL and SFT/
  DPO/GRPO replay jobs pass. Local665 passed/1 packed-DDP timeout remains reported;
  CI success does not diagnose that local issue. No production promotion.

## ADR-0046 — Matched-lineage CPU study and fail-closed promotion controls

- **Date/status:**2026-11-05;design89a0350 preregistered before learning;CPU tooling accepted on e2c628f,all13 CI jobs successful. Production acceptance remains open.
- **Context:**M7.1 exposes a constant-answer SFT policy;old DPO and GRPO use
  different SFT parents. Scores/training-loss reductions do not justify promotion.
- **Choice:**shared corpus/tokenizer,three distinct initialization seeds,base3/
  SFT80/DPO40/GRPO8 with matched parent for each seed. A paired DPO half-LR arm
  changes exactly one factor. Freeze common validation/robustness/control budgets,
  preserve final800 and record actual stage costs without claiming compute parity.
- **Interfaces:**strict YAML study,checksum-bound inputs,verified full run lineage,
  existing eval reports,per-case/family paired deltas and isolated CPU measurements.
  Necessary regression checks cover math,short general controls,whitespace,
  bounded verifier exploitation and inference RSS/time. No pickle analysis.
- **Alternatives:**compare historical mismatched parents,select thresholds after
  results,call optimizer-seed changes independent when deterministic updates do
  not change,or promote demo wins directly. These confound evidence or overclaim.
- **Sources:**[verified] project RFC002 evaluation/promotion section and local
  pinned Orchestra evaluation workflow read;use fixed prompts/budgets/sample logs.
  Thresholds in promotion_design.md are prospective engineering policy choices,
  not statistical significance or literature-derived quality guarantees.
- **Limits/revisit:**correlated templates/variants,short original LM/choice controls,
  tiny CPU models and validation reuse. Pass means eligible for further validation;
  production requires actual user-run model/hardware and licensed independent
  evidence. No automatic promotion or paid/GPU/full-scale run is authorized here.
- **Developmental checks:**focused35 pass;full local688 pass/1 old packed-DDP
  timeout/7 GPU deselected in311.24s. Ruff/format,mypy153 files and build pass.
  First one-seed control rejects DPO regression and unchanged half-LR/GRPO;
  no thresholds were selected from those results. See validation_promotion.md.
- **Additional architecture control:**3d379fc preregisters exploratory RoPE1/2
  before its learning,with exact equal initial weights and paired base/SFT data/
  ordering/budgets. Standalone script keeps the validated executable package
  unchanged and binds script SHA separately. Earlier validation is exposed,so
  this is not confirmatory. Runtime seeds also control sampler order across
  replications;within paired arms ordering agrees. Whole-process RSS has launch
  context effects and is not incremental model memory. No thresholds are changed.
- **Actual outcomes:**R024 three seeds give SFT5/1/0,DPO0/0/3,half-LR5/0/0,
  GRPO5/1/0 of200;all candidates reject. All15 repeated prediction/metric/audit
  outputs and comparison gates match exactly;timings are separate measurements.
  R025 RoPE2 gives1/0/0 versus5/1/0,also reject,with exact initial parameters
  and paired target budgets. No long-context/general improvement claim. Package
  digest9ca8a8fd993965196dbaaaca1f1d8b4f22a13339644b3c73f784fa7ad5601195 stays
  unchanged for architecture controls;script SHA is separately recorded. Source
  e2c628f GitHub42100151962 has all13 green CI jobs;CPU3.11/3.12/3.13 each689
  passed,2 optional SDK skips,7 GPU deselected,SDKs separately verified. Matched
  and RoPE two-seed studies and full200-step real-SIGKILL replay pass remotely.
  Local packed-DDP timeout remains unresolved;production acceptance remains open.

## ADR-0047 — Native bounded CPU generation and inference snapshot v1

- **Date/status:**2026-11-06;native CPU core accepted on699ff0b,all14 CI42239279899 jobs successful.
- **Context:**M7.2 rejects candidate promotions,but generation/export correctness
  is an independent dependency-ready milestone. Decoder cache has CPU reference
  coverage;no GPU or external-runtime compatibility is established.
- **Choice:**own greedy/seeded CPU engine with private per-request RNG/cache,
  genuine left-padded batched execution,explicit logical positions/validity,
  cooperative cancellation/time limits and immutable snapshot stream events.
  UTF-8 partial decodes may revise snapshots;do not promise append-only deltas.
  Reuse safetensors/strict manifests for self-contained native inference export.
- **Alternatives:**serial requests labeled batching,shared/global cache/RNG,
  copying training pickle into an inference runtime,or claiming untested HF/GGUF/
  vLLM compatibility. These do not satisfy correctness/provenance boundaries.
- **Sources:**[verified] local RFC001 cache contract,docs/model.md and pinned
  Orchestra vLLM skill read for batching/cache/resource patterns. Vendor speed,
  hardware and quantization examples are not QVELT measurements or installed APIs.
- **Limits/revisit:**CPU FP32,<20M parameters,8 batch/2048 context and bounded
  logical KV payload;copy-on-append/transient/RSS differs. Timeouts do not preempt
  kernels. Serving/continuous batching,GPU/quantization and model release rights
  need separate acceptance. See inference_design.md for exact interfaces/tests.
- **Executed references:**independent unpadded prefix logits/tokens and scalar
  categorical probabilities,actual KV tensor bytes and symbolic byte identity,
  close/weakref/mutation/Unicode/stopping/nonfinite/budget and snapshot corruption
  guards. Pre-final full local733 pass/7 GPU deselected;final guards focused53
  pass,Ruff/mypy163/build pass. Developmental native export has zero reload logit
  delta and identical CLI/API/stream/cache outputs with original sources hidden.
- **Final acceptance:**clean package
  2adb1ec750a1e03d4f859597edd7e4d344d7a8b948fdd648e76cdaa143b77bba;actual
  19.34M RFC tiny read-only export/reload delta0,cache/full max2.861e-6 and fresh
  CLI parity,source weights unchanged. Extracted-wheel generation matches under
  the same locked dependencies. CPU3.11/3.12/3.13 each735 passed,2 optional SDK
  skips,7 GPU deselected;SDKs verified separately. Native inference and200-step
  actual-SIGKILL/training/evaluation jobs pass. No quality/GPU/serving/model release
  claim;old Gloo instability is not diagnosed by a successful regression run.

## ADR-0048 — Expanded math post-training with bounded native CPU interfaces

- **Date/status:**2026-11-06;portable CPU extensions accepted on8e9b44c,
  all15 CI42253772192 jobs pass;model quality/promotion rejected.
- **Context:**the user's expanded Phase6 follows accepted M8.1 and adds requirements
  beyond original M6.1–M6.3. Reuse the existing packed SFT,DPO,GRPO and checkpoint
  engine rather than introduce TRL/Transformers/PEFT dependencies or fork loops.
- **Adapters/chat:**original LoRA on all seven attention/FFN block linears,
  frozen W/embedding/norm,train A/B,scale alpha/r. Gaussian A variance1/input_width,
  zero B;full SFT remains default. CPU FP32/single/no compile only;full base+adapter
  checkpoints preserve replay,not adapter-only release. Dense merge is explicit.
  No quantized/QLoRA substitute is faked. Chat schema2 separates optional assistant
  reasoning and final content,using explicit v2 control IDs8/9;legacy controls stay
  reserved. Shifted supervision includes reasoning/control/end-turn,not prompts,
  role IDs,outerEOS or padding. Prefix truncation never fabricates termination.
- **Verifiers/data:**exact Fraction and rational-tolerance comparisons;allowlisted
  AST rational-polynomial construction (x/y,total degree8,nodes48,chars128) before
  SymPy QQ identity. No eval/string sympify/parse_expr or arbitrary CAS expressions.
  Domain-changing cancellation through variable denominators is unsupported.
  Rejection uses actual pinned own checkpoints/native sampling with every candidate,
  fixed budgets/seeds,answer-only verification,difficulty pass@k and within-problem
  lexical dedup. Correct answers do not establish trace/process correctness.
- **Rights:**reviewed immutable original human-written GSM8K train subset<=64,
  pinned payload17f347dc.../MIT86bbb73e...,D-GSM8K-TRAIN-BOUNDED-01. No optimization
  on it or production clearance. Test/socratic/model-solution files remain outside
  training approval. External synthetic ingress is separately default-off and
  requires exact local source/teacher/revision/license/terms permission;no API calls
  or legal rights are inferred from a switch. Full notices/provenance are retained.
- **RL:**native Phase8 engine replaces repeated-prefix sampler,with KV caching
  and sampled-action logps recomputed by the frozen behavior policy. Private sampler
  seeds derive from checkpointed Torch RNG. One fresh-window update,temperature1;
  original clipped GRPO/KL math unchanged. Optional bounded format/length shaping,
  separate correctness/format/cost/KL logs and question/gold/text audits. Fourteen
  fixed verifier controls run outside optimization before training/checkpoints.
  Initial policy may be a checked DPO descendant;frozen reference remains SFT.
- **Offline:**checksum-bound bounded JSONL windows must match current model,
  reference,step,prompts,seeds,actions,stops,text,rewards and recalculated logps.
  Recompute canonical values so tolerated rounding cannot alter updates. Consume
  identical sampler-seed cursor. This is on-policy replay,not generic stale-data RL.
- **Evidence/sources:**[verified] Hu et al.(2021),*LoRA:Low-Rank Adaptation of Large
  Language Models*,arXiv2106.09685v2 section4.1,read2026-11-06 at
  https://arxiv.org/html/2106.09685v2 :frozen W0,BA,alpha/r,Gaussian A/zero B;
  variance1/d is our explicit choice,not a claimed paper constant. Independent
  dense values/gradients/frozen tensors/merge/replay are executed tests.
  [verified] installed SymPy1.14.0 Poly.from_dict/QQ and bounded exact reference
  identities;no user text is evaluated. [verified] immutable GSM8K README/LICENSE
  revision3101c7d... human train release/MIT text,actual bytes/checksums acquired.
  Prior ADR0043/0044 source equations and local pinned SFT/PEFT/SymPy guides inform
  review;vendor speed/hardware/quality claims are not adopted. Lean remains design
  only;toolchain/APIs/sandbox behavior unverified.
- **Outcome/limits:**first fixed127104-parameter arithmetic control improves sampled
  training reward .3125->.709375 (first/last10),validation15/64->26/64;DPO heldout
  loss .693147->.711595 worsens. Online/offline model/Adam/RNG/losses match exactly.
  Two-answer local fixture,known diagnostic outcomes before final clean rerun;
  no independent preregistration,general reasoning gain or model promotion.
  Final800 untouched. CPU GRPO remains below1M;larger/GPU/distributed/arbitrary
  problem/QLoRA/Lean tiers require new evidence and rights/resource gates.

ADR0048 review update: equivalent final-number spellings are not distinct traces.
Selection now compares only reasoning bodies and retains at most one answer-only
row/problem. Correct-candidate/pass@k counts are computed before dedup. This
closes an observed lexical-diversity loophole without changing training/rewards.

ADR0048 final acceptance:8e9b44c/package28addb6ad1641aebd047a16931b4177db276022609a39eae9e63ca5e273a3bc3,
all15 CI42253772192 jobs pass;each CPU798 passed/2 optional SDK skips/7 GPU
deselected,SDKs separately checked. Final focused88 pass,mypy179/build pass.
Clean control/LoRA20/source ingress/offline replay receipts committed. Quality
rejected:all59 parseable post-RL numbers0,answer1 stays0/32;26/64 validation is
below the32/64 constant-zero baseline. All12 selected rejection examples gold0.
Reward rise does not establish reasoning;coverage/model promotion remains blocked.
Local798 pass/2 historical Gloo failures and prior CI3.13 timeout are retained;
remote green does not diagnose their root cause. Final800 remain unscored.

## ADR0049 — Expanded offline evaluation, checkpoint regression and result receipts

- **Date/status:**2026-11-06;implemented CPU reference, acceptance in progress.
- **Context:**the expanded Phase7 request needs actual standard-harness integration,
  reproducible math/choice/code/Russian controls, actual-shard audits and an auditable
  ledger. Existing v1 reports/frozen final holdout must remain compatible.
- **Decision:**add explicit qvelt-eval-v2 using the native cached engine; preserve
  legacy v1 digest. Exact/numeric/GSM-strict answer extraction, Wilson intervals,
  unbiased pass@k, fixed-prefix plurality@k with first-seen tie diagnostics, explicit
  reasoning-token lengths and conservative failure witnesses. Incorrect free text
  without evidence stays unclassified. Code controls interpret a bounded pure
  integer-expression AST;never exec generated Python or label these HumanEval scores.
- **Harness:**pin optional lm-eval0.4.13 and implement its real LM API. Likelihood
  has numerical PyTorch references;rolling uses the actual upstream window helper;
  generation uses native inference. Actual ConfigurableTask/evaluator.evaluate runs
  on local original tasks. Reject unsupported options/retokenized continuation
  boundaries/context overflow,not silent changes to benchmark semantics.
- **Data/protocol sources [verified]:**installed lm-eval0.4.13 GSM8Kv3/MMLU/ARC/
  HellaSwagv1/HumanEvalv1 task definitions,filters,choice-normalization metrics and
  HellaSwag preprocessing source read. GSM8K publisher revision3101c7d5072418e28b9008a6636bde82a006892c
  README/full MIT and exact test bytes;MMLU publisher HFc30699e8356da336a370243923dbaf21066bb9fe
  card + original Hendrycks4450500f923c49f1fb1dd3d99108a0bd9717b660 README/LICENSE/
  evaluate.py read. Default5-shot protocols are explicitly different from our
  zero-shot first16 development controls. Full source pins/hashes in DATA_CARD and
  benchmark inventory;no official full benchmark score claim.
- **Blocked sources:**MATH loaders/equivalence source985bdc1696e88e8643f081a0ff4719da39f2ae2a
  read;its MIT code license does not clear MATH-500 data (reviewed card6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
  lacks explicit license). AMC/AIME reuse rights unresolved. HellaSwag publisher451
  respected without mirror bypass. AllenAI official ARC URL redirects to publisher
  HF210d026faf9955653af8916fad021475a3f00453 CC-BY-SA4 card;full legal-text403,
  terms review pending,no data. HumanEval b9453e8f1865df0a0057a1383767caa400acfbcf
  full MIT/README read,but arbitrary Python isolation unimplemented. These are
  explicit unavailable benchmark gates,not fake toy implementations.
- **Audit:**copy original usage policies into prepared artifacts;validate permissions,
  checksums,tokenizer and complete train-only frame coordinates before decoding.
  N-gram/exact/Jaccard/family findings never join documents. V2 full-scan bounds
  fail explicitly. Encoded preference frames have their own validated path.
  Unknown earlier model history and semantic leakage remain unverified.
- **Checkpoint regression:**optional pinned recipe through shared save/load hooks,
  evaluated every committed supported checkpoint with RNG restoration. Crash after
  checkpoint publication leaves weights safe;resume/repair completes reports and
  hash-bound PNG/JSON curves. CPU FP32/single<=20M only;unmerged LoRA/GPU/DDP reject.
- **Ledger:**verified receipts preserve exact metrics,audit,configs,source/inputs and
  reproduction argv without raw corpus/weights. Deterministic RESULTS.md and README
  numeric-result guard are checked in CI. Comparison key requires identical semantic
  evaluation/data/tokenizer/evaluator source;no paper scores copied. Release coverage
  requires every registered benchmark and actual training-shard audit,currently
  blocked;even passing artifact coverage is not quality promotion or history proof.
- **Dependencies:**Matplotlib3.11.2 supplies real standalone curves;optional lm-eval
  pins its actual dependencies via uv.lock. HuggingFace datasets constrains fsspec,
  so lock resolves2026.9.0->2026.7.0;other previously locked package versions retained.
  Default native evaluation needs no optional datasets/harness or external services.
- **Alternatives:**fake harness interfaces,unsafe Python execution,unreviewed mirror
  datasets and heuristic cognitive labels rejected. No remote service/GPU performance
  claim. Revisit full task coverage after rights,context,code sandbox and hardware
  validation;revisit audit bounds before large production corpora.

ADR0049 ARC review closure:full CC-BY-SA4.0 legal text read from official Creative
Commons legalcode4d2de083a6dcc5095ed1a30885bc48f868b9bb55 (20005 bytes,
a0bd498de6b7991f563789715e3cf9f9a77860b6bb5b2c2b072fb3bb36361008) after
web403. Publisher AllenAI HF card explicitly declares CC-BY-SA4.0,although its
body license subsection is unfinished. Register bounded first16 ARC-Challenge
only,evaluation/decontamination,no training. Preserve full license,AllenAI/source
attribution and conversion change indication;converted data remains CC-BY-SA.
Actual publisher payload203808 bytes/1172 rows,62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9,
read only after DATA_CARD approval. Native character-without-delimiter mode excludes
the stored leading target space from the denominator,matching upstream acc_norm;
raw choice likelihood remains unchanged. No raw corpus/predictions in public receipts.

ADR0049 CI correction:initial42302984907 evaluation-suite fails because fresh
checkout has no ignored runs directory. Create only requested parent directories,
retain exclusive output creation,and add actual nested-parent/reuse-rejection smoke.
Record actual relative output/source-cache argv,not a fixed example. Final focused
71 pass20.68s;new source-clean receipts supersede initial f741151 acceptance.

ADR0049 final portable acceptance:acced9071be0896b2fa6e915c7ed9d1b0f73e9d9,
source17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3,
all17 CI42303284852 jobs green. CPU3.11/12/13 each823 pass,6 optional skips,
7 GPU deselected;actual harness11 tests and optional SDKs separately pass.
Focused71 pass20.68s;preference-frame train-only audit separately covered in
8-test ledger/source set4.28s. Source-clean123904-param debug control yields
GSM8K0/16,MMLU4/16,ARC raw2/16 and normalized3/16,with context failures recorded.
No full benchmark/quality promotion. Nine report-bound receipts,hash-bound plot/
points and generated RESULTS published without raw corpus/predictions/weights.
Initial fresh-checkout failure and two local distributed timeouts retained.
Main-run hardware,blocked public source/full-code execution and release gates open.

## ADR0050 — Standard Llama export,safe HQQ artifacts and bounded OpenAI serving

- **Date/status:**2026-11-07;portable CPU reference accepted onb5deaf4,CI42356828424 all19 successful.
- **Context:**expanded Phase8 adds sampler controls,HF interoperability,weight-only
  quantization,HTTP batching,SSE,long-context evaluation and reproducible containers
  to the existing trusted CPU engine. Existing source/recipe/report contracts remain.
- **Choice:**preserve default sampling/cache behavior;add penalty->temperature->
  top-k->nucleus->min-p controls with independent scalar oracles. Engine protocol
  is shared by native HTTP and on-policy RL. Fixed padded batches remain the
  reference;no custom paging/kernel or unsupported throughput claim.
- **Export:**builtin Llama config/safetensors/fast tokenizer,not remote Python.
  Adjacent-pair native RoPE requires per-head Q/K row permutation to split halves.
  Tied head,norm eps,GQA,SwiGLU and linear interpolation map explicitly. Native
  config/lock/source manifest stay bound alongside HF files;do not weaken the
  shared artifact verifier to accept a foreign config under its native role.
- **Quantization:**HQQ0.2.8.post1 actual CPU pure-PyTorch path,group32,4/8-bit
  projections only;embedding/head/norm FP32. Quantized file contains only numeric
  packed weights/scales/zeros and ordinary gains/embedding. Its library's encoded
  dtype loader uses eval;never pass artifact dtype text to it. Regenerate trusted
  metadata from pinned config/library and validate complete tensor keys/shapes/
  dtypes/finiteness before restoring. Clone HQQ inference tensors to ordinary
  parameters for native version-counter/cache contracts. No custom quant kernels,
  GPU speed or HF/vLLM loading of the separate native HQQ artifact is claimed.
- **Evaluation reuse:**explicit EvalJob quantization_bits/context_extension fields;
  defaults omitted from digest to preserve historical v1/v2 receipts. Same native
  scorer/audit/receipt path,actual packed-payload SHA and library/config recorded.
  Ledger derives quantization deltas only for identical source checkpoint,prompts,
  tokenizer,budgets,scorer/source;no parallel invented metrics/receipt framework.
- **Serving:**one model worker,bounded queued microbatches,private cache/RNG,
  end-to-end deadlines/cooperative cancellation,strict OpenAI JSON/SSE,body bounds,
  constant-time bearer hook and loopback-only explicit auth opt-out. Hide unstable
  trailing UTF-8 replacement chars and stop prefixes before emitting append-only
  SSE;final concatenation must equal native text. No prompt/key/body logging.
- **Dependencies [verified]:**actual PyPI metadata/install of Transformers5.1.0,
  HQQ0.2.8.post1,FastAPI0.162.0/Uvicorn0.54.0;existing tokenizers0.22.2/Hub1.33
  remain compatible. New Starlette1.10 TestClient recommends httpx2;pin2.16.0 in
  development dependencies instead of suppressing its deprecation warning.
  Optional SDK imports are explicitly isolated for default CPU type checks.
- **Sources [verified]:**installed HF5.1 Llama constructor/linear-RoPE implementation,
  HQQ quantize/dequantize/state/dtype decoder APIs;actual offline AutoModel/
  AutoTokenizer full/cache/tokenizer parity and dense HQQ reference tests pass.
  vLLM0.33.0 commit4ba787ec9959e566d3b587a796dff07673324da5 registry/builtin
  Llama QKV/tied head/neox-style rotary/linear-scaling source inspected. This is
  structural compatibility evidence,not executed vLLM/GPU support. Local pinned
  Orchestra guides are guidance;their speed/quality numbers are not adopted.
- **Containers:**pin Python3.12.14-slim-bookworm and uv0.12.23 image digests;
  use signed Debian20261106 snapshot for training Git,uv.lock for Python,nonroot
  runtime,allowlisted source-only context and transient CA mount. First diagnostic
  build exhausted local storage;collapse large-venv layers/use hardlinks and clean
  only build caches. Actual container acceptance remains required.
- **Revisit:**GPU/sharded/large inference,paged allocation or faster quant backends
  after profiling;external engine runtime after hardware/compatibility tests;
  long-context quality after an appropriate trained checkpoint/independent data.

ADR0050 implementation follow-up:direct dense checkpoint export verifies all
registered bytes but never deserializes optimizer/RNG pickle;validates exact
FP32 dense shapes before materializing tensors;streams Q/K conversion into indexed
HF shards,target128MiB. This admits RFC tiers through large independently of the
native20M CPU cap. DDP replicated dense state is admissible;DCP shards/unmerged
adapters fail explicitly. Large execution and sharded consolidation remain unrun.
Inspected vLLM FlashAttention source accepts divisible-by8 head sizes through256,
FP16/BF16 and capability>=8.0;RFC head64 is structurally eligible,not runtime proof.
Installed metadata licenses:Transformers/HQQ Apache2,FastAPI MIT,Starlette/Uvicorn/
HTTPX/HTTPX2 BSD3. No upstream implementation code was copied.
Pinned Uvicorn re-raises SIGTERM after graceful shutdown;emit sanitized operational
counters in lifespan cleanup and require them before accepting Unix signal exit.

ADR0050 final acceptance:local all-extras864 passed/7 GPU deselected331.00s;
CI CPU3.11/12/13 each846 pass,18 optional skips,7 GPU deselected. Dedicated actual
HF/HQQ/API34 tests plus full smoke pass. All3 Docker targets build;nonroot actual
training/HTTP runtime passes in CI. Local vfs ENOSPC is retained as failed.
Seven clean source artifacts/six native reports,quantized payload identity match,
HF max error2.384e-7 and real HTTP workload are registered. PPL deltas are tiny
fixture measurements,all generation/retrieval accuracy0;no quality promotion.
Final test-only historical reasoning extension10 pass7.75s;runtime unchanged.
All15 native eval receipts regenerate RESULTS. Final800 remains unscored.

## ADR0051 — Phase9 bounded release readiness and security upgrade (2026-11-07)

- **Context:** the user requested documentation,hardening,clean reproduction and
  release plumbing. Framework engineering acceptance does not establish trained
  model quality,rights or safety.
- **Decision:** strict MkDocs1.6.1 plus canonical-document generation;original
  Mermaid sources and offline SVGs;real browser navigation. Use actual RFC tiny
  for the documented bounded base/resume/SFT/eval/HF/HTTP tutorial. Add seeded
  tokenizer/packing properties and bounded verifier fuzzing;measure critical
  branch coverage without exclusions. Original bilingual safety data is a design
  fixture with family splits and exact assistant masks,not safety acceptance.
- **Security:** normalize CPU/CUDA local version suffixes during advisory review.
  The audit required Torch2.13.0,Transformers5.10.4 and pytest9.0.3. Preserve
  unpatched optional sqlitedict CVE-2024-35515 as a failed all-extras gate. Never
  carry old security-vulnerable pins merely to preserve exact resume. Start a new
  lineage from verified safetensors when the runtime changes.
- **Loaders:** inspected installed Torch2.13 DCP reader/planner still use unrestricted
  metadata pickle and weights_only=False byte fields. Substitute a bounded static
  data-class unpickler,manifest-bound relative file extents and explicit weights-only
  byte loads;test real tensor/optimizer reload and rejected executable payloads.
  Trusted ranks/network remain required for object collectives. This is not a
  hostile-file resource sandbox. Numeric-only HQQ restore remains required.
- **Plumbing:** semantic framework0.1.0 candidate,archive-content/version gates,
  draft-only GitHub release workflow and sanitized issue/PR forms. No tag,public
  release,PyPI or model upload is created. Optional Hub automation is deferred
  until a model is eligible. Existing HF export is not a release permission.
- **Alternatives:** duplicating runbooks/results would drift;remote diagram/highlight
  CDNs would weaken offline reproducibility;ignoring CVEs or deleting test failures
  would create false evidence. MkDocs-gen-files0.5.0 avoids an unrelated fork.
- **Sources [verified,limited scope]:** installed Torch2.13.0 FileSystemReader,
  DefaultLoadPlanner and metadata APIs;MkDocs1.6.1 templates;pip-audit2.10.1 advisory
  artifacts and PyPI normalized Torch2.9.1/2.13.0 JSON;Transformers5.10.4 installed
  builtin Llama API exercised by tests. This verifies inspected interfaces and
  recorded CPU checks,not CUDA13.0 deployment or advisory completeness.
- **Revisit:** upstream DCP/API changes,advisory updates,a real training server or
  eligible independently evaluated model. Re-run profile-specific audits and
  recovery before adopting the new runtime on private hardware.

ADR0051 version follow-up:the original heldout manifests were produced by
0.1.0.dev0. SemVer0.1.0 changes only that envelope field and therefore its digest.
Do not rewrite the committed freeze/generator or ignore manifest identities.
The new freeze-identity verifier checks all actual files,requires the current
producer,reconstructs the complete original0.1.0.dev0 envelope hash and compares
all registry fields exactly. Payloads,rights,timestamps,case IDs,counts,generator
and license remain bound to the original preregistration. Final cases are not
model-scored. Source and metric protocols remain independent of framework version.

ADR0051 final acceptance:current runtime8c08214 has all22 CI42384342511 jobs
successful,including full three-version CPU,fresh tutorial,recovery,optional
interfaces,containers and hardening.114 focused tests and89.2562% critical
combined coverage,zero exclusions;16 native receipts regenerate RESULTS. Fresh
local7ef8bfd/new venv completed the full RFC tiny tutorial477.817426s;profiled
decode optimization preserves its entire two-source audit. Native+HF+serving
advisory audit exit0;all-extras sqlitedict remains failed. Current-tree offline
secret gate132 reviewed findings,zero unexpected. Installed-wheel native stream
parity passes. Local numeric-DDP timeout persists and is not declared fixed by
remote green. No GPU/model-quality/safety acceptance or actual release/upload.

## ADR-0052 — Frontier horizon and gated validation ladder

- **Date/status:** 2026-11-07; owner-approved design direction; documentation only.
- **Context/links:** [ADR-0008](#adr-0008-select-a-dense-tied-gqa-baseline-and-exact-size-family)
  selects the dense reference family; [ADR-0010](#adr-0010-use-adamwwsd-and-staged-budgets-as-tested-hypotheses)
  treats budgets as hypotheses; [ADR0051](#adr0051-phase9-bounded-release-readiness-and-security-upgrade-2026-11-07)
  separates portable framework acceptance from model release.
- **Decision:** the long-term research target is a frontier-class general model
  for reasoning and software engineering, with multi-trillion total parameters,
  likely sparse/MoE. The existing 19M/125M/493M/1.5B configurations are the
  validation ladder, not the destination. Preserve the dense CPU reference as
  the numerical oracle. Architecture selection, expert counts and active size
  remain unimplemented research decisions, not accepted performance results.
  Design accelerator interfaces for NVIDIA H100/H200/B200-class hardware while
  retaining backend abstraction; no such hardware is validated by this ADR.
- **Product budgets:** 1M means exactly 1,000,000 total tokens, with tokenized
  prompt plus all generated tokens <= 1,000,000 and output <= 128,000. System
  instructions, history, documents/files, tool results, internal framing and all
  reasoning/tool-call generation consume the applicable budgets.
  These are future targets, not current server limits or measured capabilities.
  **Owner amendment 2026-11-07, Q14:** 128,000 replaces the earlier 131,072;
  hidden reasoning does not provide an extra uncounted output allowance.
- **Gates:** first require CPU values/gradients, masks, initialization and replay
  tests for each new component; then actual few-GPU tests, multi-node composition
  and failure/recovery rehearsal on available hardware. Advance through measured
  architecture/optimizer/data/hyperparameter trials before a costly run. Report
  compute estimates with formula, active/total parameter accounting, assumptions
  and exclusions. No larger run or expense is authorized by this horizon.
- **Owner amendment 2026-11-07, Q1-Q19:** the strategic spec replaces EN/RU-only
  planning with [multilingual coverage and a measured mix](#adr-0058-multilingual-coverage-and-measured-mixture),
  and specifies [source-specific governance](#adr-0059-data-rights-consent-and-legal-review-gates).
  Per-run spending, approved corpus revisions and human reviewers remain undecided.
  Funding/access/team figures in STATE are owner statements, not resources that
  agents may spend or configure. Retain [ADR-0005](#adr-0005-use-explicit-provenance-and-approval-before-ingesting-data).
  Initial internal tool interfaces
  are read_file, search, apply_patch, exec for tests/lint/build, and git diff;
  execution requires a separate isolated runner, never unrestricted host access.
  Q11's broader file/archive/web/API/agent actions are later requirements with
  permissions for destructive actions and publication, not current capabilities.
  Image/audio/video and other modalities remain separate later phases/ADRs.
- **Product decisions (owner Q15-Q17, 2026-11-07):** use QVELT as the product/model/
  ecosystem name, retaining code/package paths. Web first; accounts, session/access
  control, history and consumer/enterprise separation from the product's first
  design. Desktop/local agent and broader product API come later. Payments are
  required before commercial launch, not the first closed version. Existing
  single-key framework serving is not an accounts or product implementation.
  Trademark/domain clearance remains a legal launch gate. Competitive superiority
  is an owner ambition, never a fact without independent measured evaluation.
- **Rationale/alternatives:** a gated ladder tests transfer and systems assumptions
  without treating parameter count as quality. Alternatives were retaining the
  laptop family as the destination or jumping directly to a cluster-scale run.
- **Evidence:** [verified] owner decisions in the Step A request and current
  [architecture RFC](docs/research/architecture_rfc.md), reviewed at main
  `901cb9b03632a5a3dc36beb3ef0e1fc29ce71b2e`. [unverified] frontier quality,
  large-run cost, accelerator support and 1M/128K operation. No claims about
  other labs' parameter counts or internal architectures follow.
- **Delivery gate/revisit:** Steps A through E are separate changes in that order;
  stop for owner review after each. This change is Step A only. Review this
  direction when hardware, budget, source permissions or measured transfer change.
  **Amendment acceptance:** only the documents change; Steps B-E remain unstarted.
  Read Q1-Q19/principles 1-5 in full as owner-supplied text on 2026-11-07. This
  verifies receipt, not funding, hardware access, staffing, deadlines or quality.

## ADR-0053 — Tokenizer v3 control block and explicit new lineage

- **Date/status:** 2026-11-07; accepted design; codec/artifacts not implemented.
- **Context/links:** [ADR-0009](#adr-0009-freeze-the-tokenizer-budget-not-untested-tokenizer-quality)
  remains the historical vocabulary baseline; [ADR-0029](#adr-0029-versioned-controls-and-exact-original-math-environments)
  preserves v1/v2 controls. This decision adds a future v3, without rewriting
  either existing version, its IDs, merges, artifacts or checkpoints.
- **Decision:** retain byte-BPE coverage and literal-spelling isolation. Only the
  validated structured codec inserts control IDs; ordinary text containing their
  spellings remains ordinary text. Allocate the following 128-entry v3 block:

  | IDs | Tokens, in ID order |
  | --- | --- |
  | 0-7 | `<\|pad\|>`, `<\|bos\|>`, `<\|eos\|>`, `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|end_turn\|>`, `<\|tool\|>` |
  | 8-12 | `<think>`, `</think>`, `<\|tool_call\|>`, `<\|end_tool_call\|>`, `<\|tool_result\|>` |
  | 13 | `<\|end_tool_result\|>` |
  | 14-17 | `<\|fim_prefix\|>`, `<\|fim_suffix\|>`, `<\|fim_middle\|>`, `<\|fim_end\|>` |
  | 18-21 | `<\|file_start\|>`, `<\|file_end\|>`, `<\|path\|>`, `<\|section\|>` |
  | 22-27 | `<\|effort_none\|>`, `<\|effort_low\|>`, `<\|effort_medium\|>`, `<\|effort_high\|>`, `<\|effort_extra_high\|>`, `<\|effort_maximum\|>` |
  | 28-127 | `<\|reserved_000\|>` through `<\|reserved_099\|>` |

- **Vocabulary:** V includes controls and byte symbols and must be configurable;
  no final V or production default is selected. The later measured decision
  considers 32768/65536/131072/262144 plus larger candidates such as 524288 and
  1048576 on an owner-approved corpus sample; these are candidates, not selected
  defaults or claims of current support. Record requested versus achieved V.
  Current TokenizerSettings caps V at262144; testing larger v3 candidates will
  require a separately reviewed v3-specific bound and resource validation while
  preserving v1/v2. Allocation of controls does not establish trained effort,
  FIM or tool capabilities. Positions and
  document IDs remain metadata, not a special token per context position. File
  markers require a declared corpus format; image tokens are outside this ADR.
- **Lineage/migration:** decide and freeze production v3 before real pretraining.
  Bind control map, byte/merge vocabulary, preprocessing, recipe, ordered corpus
  identity and implementation versions into tokenizer provenance. v3 starts a new
  lineage and requires retokenized data and matching model/export identities.
  Preserving IDs 0-12 does not make remaining IDs or embeddings compatible.
  Any old-weight transfer is a separate explicit initialization experiment;
  never relabel an old checkpoint or claim exact resume after retokenization.
- **Acceptance before use:** Step B must preserve v1/v2 loading and behaviour,
  test every control's literal-spelling isolation and deterministic artifact hash,
  and exercise a real v3 tokenizer with a CPU model <=20M parameters. Recompute
  model sizes for the selected test V; historical tier counts do not carry over.
  **Owner-spec amendment 2026-11-07:** reserve six effort IDs before Step B freezes
  its table, as recommended in [ADR-0057](#adr-0057-effort-conditioned-training-and-enforced-compute-budgets).
  This replaces the earlier unfrozen reservation of IDs22-127; block size remains
  128. Ordinary text cannot set effort. Require per-language fertility and the
  [language/data audit](#adr-0058-multilingual-coverage-and-measured-mixture) before
  production selection, plus [named human freeze review](#adr-0061-named-human-review-and-hardware-validation-gates).
- **Rationale/alternatives:** a versioned reserved block stabilizes framing while
  leaving compression a measured decision. Alternatives were keeping 32 controls,
  silently consuming old reserved IDs, or fixing a larger V without measurements.
- **Evidence:** [verified] owner-approved control-block draft and
  [current tokenizer](src/qvelt/tokenizer/core.py) at `901cb9b`; v3 runtime,
  compression and downstream effects are [unverified]. The three-digit reserve
  spellings make the new block explicit; they do not rename v1/v2 reserves.
- **Revisit:** measured tokenizer study, new protocol requirements or independently
  justified migration; never mutate an already frozen artifact in place.

## ADR-0054 — SFT schema 3 with typed tool blocks and shifted supervision

- **Date/status:** 2026-11-07; accepted design; schema/codec not implemented.
- **Context/links:** extend [ADR-0039](#adr-0039-whole-dialog-assistant-supervision-and-weights-only-sft-initialization)
  and [ADR-0048](#adr-0048-expanded-math-post-training-with-bounded-native-cpu-interfaces)
  while retaining their existing schema 1/2 behaviour; use
  [ADR-0053](#adr-0053-tokenizer-v3-control-block-and-explicit-new-lineage)
  for new framing. Keep DPO's distinct objective/reference contract in
  [ADR-0043](#adr-0043-original-sigmoid-dpo-with-explicit-frozen-sft-lineage).
- **Decision:** schema 3 has schema_version=3 and messages with a role and ordered
  typed blocks. Roles are system/user/assistant/tool. A text block contains text;
  an assistant tool_use block contains id, name and an arguments object; a tool
  tool_result block contains tool_use_id, is_error and text. Optional reasoning
  belongs only to an assistant reasoning block, separate from final text.
  Tool definitions are a separate validated input field and masked context.
- **Framing:** use role IDs and end-turn boundaries; tool_use uses tool_call /
  end_tool_call, tool_result uses tool_result / end_tool_result. Encode structured
  payloads with one canonical JSON/ID policy; literal content cannot inject
  controls. Keep inference action framing consistent with SFT, including assistant
  turns containing only a call. Tool results never become assistant targets.
  Initially permit one outstanding call: require unique IDs and the matching
  result before assistant continuation. A failed result with is_error=true is
  valid context. Reject orphan/duplicate results, role/block mismatches, unknown
  fields and malformed arguments. Parallel tool calls need a separate extension.
- **Shifted target table:** attention validity is independent of supervision;
  derive flags against next-token targets, never against unshifted input positions.

  | Target | Loss |
  | --- | --- |
  | System/user content, tool definitions, role headers | false |
  | Trusted effort control in the assistant prefix | false |
  | Assistant final text | true |
  | Assistant tool_use payload and its opening/closing delimiters | true |
  | Tool result payload, delimiters and tool end-turn | false |
  | Assistant reasoning and its delimiters | true only under explicit approved reasoning-supervision policy |
  | Completed assistant end-turn | true |
  | Outer BOS/EOS and PAD | false |

- **Truncation/errors:** never synthesize completed JSON, a call/result delimiter
  or end-turn after cutting a block. Reject truncated tool/reasoning structures;
  any retained plain assistant prefix must obey an explicit truncation rule with
  masked outer EOS. Reject examples without assistant targets. Preserve document
  isolation and report rejects/truncation. Step C must cover the full mask table,
  valid error results, orphan/duplicate/order cases and truncation boundaries with
  independent target/gradient oracles. Report observed chat schema version(s);
  fix the current hard-coded schema-1 report for schema-2 input with a regression.
- **Reasoning/data policy:** retained raw reasoning has a separate field/record
  and explicit access, retention and deletion policy; no duration is assumed.
  Clients receive a summary by default once that interface is implemented.
  Never present raw reasoning as a summary or infer trace correctness from a
  correct answer. Dataset permission and supervised-mask policy remain hash-bound.
  **Owner-spec amendment 2026-11-07, Q9/Q12/principle5:** effort conditioning
  follows [ADR-0057](#adr-0057-effort-conditioned-training-and-enforced-compute-budgets).
  Any schema3 effort metadata and its prefix placement must be explicit and
  validated; do not invent labels for historical schema1/2 data. Retention,
  consent/deletion and enterprise handling follow [ADR-0059](#adr-0059-data-rights-consent-and-legal-review-gates).
  Honesty/initiative/factuality requirements are [post-training/eval work](#adr-0060-measurable-honesty-initiative-and-factuality),
  not effects of this tokenizer/schema change. The schema2 reporting fix remains
  Step C, not this amendment.
- **Rationale/alternatives:** typed blocks make loss masks and call/result links
  testable. Alternatives were raw chat strings, supervising tool results, or
  silently reinterpreting schema 2. Tool execution remains outside SFT parsing.
- **Evidence:** [verified] owner decisions and [chat codec](src/qvelt/posttrain/chat.py),
  [SFT preparation](src/qvelt/posttrain/data.py) at `901cb9b`. Schema 3, execution
  isolation, summary generation and retention enforcement are [unverified].
- **Revisit:** parallel calls, structured tool-result payloads, measured truncation
  needs or an approved reasoning-retention policy; version protocol changes.

## ADR-0055 — Truthful deployment capabilities and shared token budgets

- **Date/status:** 2026-11-07; accepted design; endpoint extension not implemented.
- **Context/links:** extend [ADR0050](#adr0050-standard-llama-exportsafe-hqq-artifacts-and-bounded-openai-serving)
  using [ADR-0052](#adr-0052-frontier-horizon-and-gated-validation-ladder)
  targets without advertising them as current support. Token/control availability
  under [ADR-0053](#adr-0053-tokenizer-v3-control-block-and-explicit-new-lineage)
  or SFT schema 3 alone does not enable an HTTP feature.
- **Decision:** each /v1/models entry retains the existing model identity and adds
  capabilities_schema_version, context_window, max_input_tokens, max_tokens and
  a capabilities object. Derive values from the loaded model/tokenizer snapshot,
  backend and deployment policy, not a model-name lookup or aspirational config.
  Report additional body/message/cache limits where they constrain admission.
- **Limits:** context_window=C is the served total token ceiling. Prompt count P
  includes template/control tokens; generation budget R includes reasoning and
  tool-call tokens. Require P+R<=C, R>=1 and R<=max_tokens. max_input_tokens=C-1
  is the prompt upper bound with at least one output token reserved, not a promise
  that maximal prompt and maximal output fit together. Smaller requested outputs
  do not waive body/message/cache limits. Compute the effective output ceiling
  from model/backend/policy; never exceed the configured cap.
- **Current population:** native serving C<=min(snapshot.context,2048), output
  <=256, plus existing admission limits. A current tiny deployment with C=512
  could advertise max_input_tokens=511 and max_tokens=256, subject to P+R<=512.
  This is a derived contract example, not a running-server measurement. Future
  target budgets remain 1,000,000 total and output<=128,000 only after validation
  (owner Q14 amendment, 2026-11-07; replaces 131,072).
- **Capabilities:** use explicit support details for streaming, tools, reasoning,
  structured_outputs and image_input. Current tools/structured_outputs/image_input
  remain supported=false. Reasoning distinguishes historical reasoning-field
  input, separate output, summaries, adaptive operation and effort_levels;
  history support follows the tokenizer and codec, not just token spellings.
  The current separate-output/summary/adaptive features remain false and
  effort_levels=[]; reject effort requests instead of ignoring them. No supported
  flag implies trained quality. Step D must test snapshot/backend-derived limits,
  false unsupported capabilities, effort rejection and ordinary JSON/SSE parity.
- **Future HTTP contract:** typed tools/tool_choice/tool_calls and tool_call_id,
  structured outputs with a bounded declared JSON-schema subset, and distinct
  reasoning_content/reasoning_summary fields need their own implementation tests.
  Default clients receive summaries, never raw reasoning; any raw access and
  persistence require an explicit authorized retention/access policy. Until an
  actual summarizer is implemented, advertise no summary capability. Effort
  follows [ADR-0057](#adr-0057-effort-conditioned-training-and-enforced-compute-budgets),
  not a prompt-only label or synonym for raising max_tokens.
- **Owner-spec effort amendment (2026-11-07):** capabilities.reasoning publishes
  effort_levels, default_effort, budget_policy_revision and per-level enforced
  reasoning/output/step or time ceilings once implemented and measured. Initial
  aliases normalize none/instant, low, medium, high, extra_high, maximum; names
  may change before freeze. Until support is real, effort_levels=[],
  default_effort=null and budget_policy_revision=null, with unsupported flags
  false and all explicit effort requests rejected. Step D must not advertise
  these levels merely because Step B reserves tokens or Step C accepts metadata.
  Output includes hidden reasoning, final text, summary and generated tool-call
  framing/payload; tool results and later histories consume input context.
  Account all additional model/tool work separately as well; retries, summaries
  and subcalls cannot bypass an aggregate request budget. Native C<=2048 and
  output<=256 remain current bounds. Discovery is an interface contract, not a
  measured guarantee of reasoning quality or instant latency.
- **Rationale/alternatives:** products discover effective limits and unsupported
  features. Alternatives were hard-coded client limits, token-presence-based
  flags or silently accepting unsupported options.
- **Evidence:** [verified] owner request, [HTTP models/schema](src/qvelt/inference/api.py),
  [admission](src/qvelt/inference/generation.py) and
  [serving setup](src/qvelt/inference/cli.py) at `901cb9b`. Extended discovery,
  tools, effort, summaries and multimodal operation are [unverified].
- **Revisit:** new backend validation, policy/model limits or implemented protocol
  features. Keep discovery and actual request validation in agreement.

## ADR-0056 — Separate exact replay, topology migration and statistical reproducibility

- **Date/status:** 2026-11-07; accepted guarantee taxonomy; migration not implemented.
- **Context/links:** retain [ADR-0003](#adr-0003-define-reproducibility-within-explicit-execution-conditions),
  [ADR-0026](#adr-0026-checkpoint-completed-updates-with-explicit-replay-compatibility),
  [ADR-0037](#adr-0037-atomic-application-checkpoints-and-split-bound-loader-cursors)
  and [ADR-0042](#adr-0042-expanded-pretraining-recovery-and-measured-portable-controls).
  Current checkpoint and loader identities bind world size; a taxonomy does not
  relax their checks or implement elastic restart.
- **Decision:** name and record the guarantee for every recovery/study:

  | Class | Contract |
  | --- | --- |
  | Exact replay | Bitwise trajectory only within explicitly tested source/runtime/device/precision/topology/batch/kernel conditions; restore model, optimizer, scheduler/counters, RNG and committed data position. |
  | Topology migration | Explicit parent-linked execution lineage; restore logical tensors/optimizer state and data progress under a validated placement mapping; no bitwise continuation claim. |
  | Statistical reproducibility | Repeated runs with declared seeds/protocol and predeclared comparison metrics/tolerances; report distributions and uncertainty, not equality from one run. |

- **Exact boundary:** checkpoint only a common completed optimizer boundary with
  verified committed bytes. Keep source/runtime/config/tokenizer/data checks.
  Same topology alone does not establish determinism. Rollback-and-skip changes
  exposure/updates and must remain separately logged, not called uninterrupted
  exact replay. Hardware/version changes require explicit compatibility review.
- **Migration boundary:** preserve logical parameter/expert names, shapes, dtype
  meaning, optimizer moments/step, schedules, global data cursor and applicable
  router/precision/RNG state. Verify no lost/duplicated samples and a complete
  committed checkpoint. Declare RNG mapping and changed reduction order. A
  changed tokenizer/objective/architecture is not a placement-only migration.
  Reset optimizer state only as an explicitly different initialization workflow.
  Do not bypass the restricted DCP reader or claim cross-version format support.
- **Rationale/alternatives:** recoverable state and identical floating-point
  trajectories are different contracts. Alternatives were promising exactness
  across arbitrary world sizes, weakening hashes, or prohibiting all migration.
- **Acceptance/revisit:** future migration needs CPU tensor/layout/cursor oracles,
  GPU reshard tests, actual topology-change recovery and failure injection.
  Statistical claims need repetitions. None are established by this docs change;
  retain current strict resume behaviour until separately reviewed implementation.
  **Owner-spec amendment 2026-11-07:** [ADR-0061](#adr-0061-named-human-review-and-hardware-validation-gates)
  requires named ML review for precision/lineage/freeze decisions and first actual
  small-GPU validation before dependent GPU work. The owner-stated future cluster
  does not establish cross-topology, cross-precision or statistical acceptance.
- **Evidence:** [verified] owner-approved taxonomy, [checkpoint compatibility](src/qvelt/utils/checkpoints.py),
  [loader fingerprint](src/qvelt/data/loader.py) and
  [restricted DCP reader](src/qvelt/utils/safe_dcp.py) at `901cb9b`.
  Cross-topology recovery and cluster statistical reproducibility are [unverified].

## ADR-0057 — Effort-conditioned training and enforced compute budgets

- **Date/status:** 2026-11-07; proposed implementation design for owner review;
  no effort levels implemented or measured. Owner principle5 requires actual
  compute control; Q12 requires understandable explanations, not a raw thought dump.
- **Links:** amends [v3 controls](#adr-0053-tokenizer-v3-control-block-and-explicit-new-lineage),
  [SFT schema3](#adr-0054-sft-schema-3-with-typed-tool-blocks-and-shifted-supervision)
  and [capability discovery](#adr-0055-truthful-deployment-capabilities-and-shared-token-budgets).
  Historical v1/v2 and current HTTP behaviour remain unchanged.
- **Mechanisms considered:**

  | Mechanism | Role and limitation |
  | --- | --- |
  | Effort control tokens | Stable learned conditioning; reserve six IDs before Step B. Tokens alone do not enforce or prove different compute. |
  | System-level/API field | Authoritative request/policy input and normalization; plain prompt wording alone fails the requirement. |
  | Training at different reasoning budgets | Teach conditioned concise/extended responses using approved data; measure cost and task outcomes separately. |
  | Length-penalized or budget-aware RL | Later correctness-gated cost shaping; can reward premature/incomplete answers if designed poorly. Requires a verifier and ablations. |
  | Inference budget forcing | Enforce reasoning/output/step/time limits; truncation is not evidence of a correct answer or complete call. |
  | Routing to different compute paths | Could change model/expert/search work; defer until architecture, lineage, routing and cost are independently validated. |

- **Recommendation:** combine a validated system-level reasoning_effort field,
  one trusted effort token immediately after the assistant role in its generation
  prefix, budget-conditioned training, and actual backend-enforced ceilings.
  Reserve IDs22-27 for none/low/medium/high/extra_high/maximum; instant is an alias
  for none, not a seventh token. Untrusted text cannot set this field or insert a
  control. Input controls are masked SFT context. Missing historical labels are
  not guessed; any schema3 conditioning/label policy is explicit and hash-bound.
  The unfrozen draft table changes now; production freeze needs ADR-0061 review.
- **Budget meaning:** none suppresses an explicit generated reasoning segment,
  not ordinary neural computation; it does not promise instant latency. Other
  levels select measured finite budgets, not unlimited resources or guaranteed
  quality. Do not assign numeric per-level defaults from the owner's funding
  statements. Actual tokens, model forwards/rollouts, tool work and elapsed time
  must be logged to test that the modes change compute beyond their prompt.
  Higher effort is not assumed better on every task.
- **Accounting:** P includes system/history/files/tool results/controls; generated
  reasoning, final answer, summary and tool-call payloads share R. For the target
  main sequence P+R<=1,000,000 and R<=128,000. The reasoning allowance is a subset
  of R with an explicit remaining final-answer allowance, not an extra budget.
  Additional calls each obey their context ceiling and aggregate request quotas;
  account their prompts, outputs, compute and retention, including hidden work.
  On exhaustion report truncation/incompletion honestly; do not fabricate closing
  delimiters, execute partial tool JSON, or pass incomplete output as verified.
- **Acceptance:** CPU prefix/masking/accounting/limit tests and adversarial literal
  isolation precede a trained checkpoint test comparing actual compute across
  levels under identical tasks/seeds/budgets. Accelerator execution needs the
  first-small-GPU gate; deployment claims require measured backend enforcement.
  Until then publish effort_levels=[] and reject explicit effort requests.
- **Rationale/alternatives:** conditioning plus enforcement gives a testable
  contract. Prompt-only effort, tokens without training, inference caps alone
  and immediate multi-path routing do not establish the requested capability.
- **Evidence/revisit:** owner-supplied principle5/Q12 read in full on 2026-11-07;
  this verifies the requirement only. All proposed mechanisms, their benefits,
  budgets and effort quality are [unverified]. Revisit labels, numeric policies,
  RL/routing and retention after human review and measurements.

## ADR-0058 — Multilingual coverage and measured mixture

- **Date/status:** 2026-11-07; owner requirements recorded, experiments not run.
- **Links:** extend [ADR-0009](#adr-0009-freeze-the-tokenizer-budget-not-untested-tokenizer-quality),
  [ADR-0012](#adr-0012-treat-data-mixtures-as-conditional-proposals-not-approvals),
  [ADR-0052](#adr-0052-frontier-horizon-and-gated-validation-ladder) and
  [ADR-0053](#adr-0053-tokenizer-v3-control-block-and-explicit-new-lineage).
- **Owner Q5 requirement:** target 100+ languages, including low-resource languages;
  English, Ukrainian and Russian are mandatory priorities. This is a future
  coverage/quality requirement, not measured language support. The starter list
  also includes German, French, Spanish, Portuguese, Italian, Polish, Czech,
  Dutch, Swedish, Norwegian, Danish, Finnish, Greek, Romanian, Bulgarian, Serbian,
  Croatian, Slovak, Slovenian, Hungarian, Lithuanian, Latvian, Estonian, Albanian,
  Basque, Catalan, Icelandic; Chinese (simplified/traditional), Japanese, Korean,
  Hindi, Bengali, Urdu, Punjabi, Vietnamese, Thai, Indonesian, Malay, Filipino,
  Burmese, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam; Arabic, Persian,
  Hebrew, Turkish; Swahili, Amharic, Somali, Yoruba; Kazakh, Mongolian, Georgian
  and Armenian. This list is not the final 100+ registry; distinguish language,
  script and variety rather than counting script variants as extra languages.
- **Code languages (owner Q5):** Python, JavaScript, TypeScript, Java, C, C++, C#,
  Go, Rust, PHP, Swift, Kotlin, SQL and a reviewed extension list. Validate code,
  documentation, identifiers, JSON, LaTeX and whitespace/indentation separately.
- **Data-availability gate:** per language audit legally usable quantity, quality,
  deduplication, provenance, domain coverage, tokenizer fit and independent eval
  supply before making support promises. Adequate permissive data may not exist
  for some low-resource languages. Record missing/insufficient sources and ask
  for collection/licensing decisions; neither oversampling nor a larger V proves
  competence. Native-language review and independently held-out cases are needed.
- **Mixture hypothesis (owner Q6):** 40% general/knowledge, 30% code/engineering,
  20% math/science, 10% instructions/tools/agents is an initial research proposal,
  not a config/default or fixed schedule. Define non-overlapping source categories,
  stage/objective and denominator before trials; report post-filter optimization
  token shares. General/science overlaps need one declared assignment policy.
  Within general, weights are 0.40*alpha_language with sum(alpha_language)=1;
  alpha values are measured decisions informed by rights, data supply, fertility,
  domain evals and the EN/UK/RU priorities, not an assumed equal or fixed split.
- **Tokenizer study requirements:** include V=32768/65536/131072/262144 and larger
  candidates such as 524288/1048576. No V is selected. Report requested/achieved V,
  per-language bytes/token, fertility in tokens/Unicode character and tokens/word
  under a declared word-segmentation policy, sample sizes and tail distributions.
  For scripts without whitespace word boundaries use a reviewed segmenter and
  retain the character metric. Report script/code-switching coverage and code/
  math/JSON/LaTeX/whitespace metrics without hiding weak languages in an aggregate.
  Estimate downstream projection/attention/token-storage costs from measured
  lengths and explicit model shapes, formulas and assumptions; small fixed-budget
  model studies follow approval. Byte coverage alone is not language competence.
  V>262144 needs new v3 bounds/resource tests, not a claim about today's codec.
- **Evaluation:** per-language factuality, reasoning, instruction/tool following
  and safety/calibration sets with source/translation provenance and human review;
  code tests per named language in an isolated runner. Report failures/coverage,
  not an unsupported blanket 100+ language quality claim.
- **Rationale/alternatives:** measure allocation and trade-offs instead of freezing
  an EN/RU-only scope, equal language shares or a vocabulary cap from laptop needs.
- **Evidence/revisit:** owner Q5/Q6 and amendment request read on 2026-11-07;
  [verified] current [V bound](src/qvelt/tokenizer/core.py) and
  [token storage dtype selection](src/qvelt/data/shards.py) at `f15b827`.
  Language rights/data sufficiency, tokenizer results and per-language quality
  are [unverified]. Review the registry, corpus sample and criteria before study.

## ADR-0059 — Data rights consent and legal review gates

- **Date/status:** 2026-11-07; owner B/Q7-Q9 governance requirements; not legal clearance.
- **Links:** retain [ADR-0005](#adr-0005-use-explicit-provenance-and-approval-before-ingesting-data),
  [ADR-0012](#adr-0012-treat-data-mixtures-as-conditional-proposals-not-approvals)
  and [ADR-0048](#adr-0048-expanded-math-post-training-with-bounded-native-cpu-interfaces).
  Broad owner interest in web/books/papers/code/partners is not ingestion approval.
- **Registry and gate:** for each dataset and code repository/revision record
  publisher/provenance, payload identity, license and notices, terms/acquisition
  method, allowed uses, personal-data basis, transformations, exclusions, review
  date and named approving legal reviewer where required. A legal review gate
  precedes production ingestion/release, including jurisdiction and scraping terms.
  Public access, a repository license badge or purchased access alone is insufficient.
- **Default until named legal sign-off:** candidate sources are own data, open
  permissively licensed data and explicitly licensed data only, with verified
  rights scope and recorded source approval. This is an eligibility policy, not
  clearance for a named corpus; ambiguity blocks use. Prior explicitly approved
  authored fixtures remain bounded historical approvals. Do not broaden them.
  No new corpus is acquired or ingested by this docs change.
- **External model outputs:** default OFF. Require named legal review of the
  specific model/provider/revision/terms, permitted training/redistribution uses,
  acquisition provenance and contamination controls before enabling a particular
  source. The owner's permission to consider lawful outputs does not waive this
  gate. Own synthetic generation retains model/data lineage and verifier audits.
- **User data:** account creation, service use, reasoning retention and product
  history do not imply training consent. Separate voluntary opt-in is required;
  allow refusal/withdrawal/deletion and retain an auditable consent record. Before
  any training export, enforce permission, secret/password/key/PII filtering,
  minimization/anonymization and re-identification review. Define retention and
  deletion propagation through raw stores, derived shards, caches/backups and
  future training sets. Do not claim filtering is perfect or that deleting source
  records removes influence from already trained weights; model-level deletion
  needs its own verified policy and limitations disclosed before consent.
- **Enterprise:** default no training on enterprise data; require a separate
  authorized enterprise agreement and data-owner rights review. Separate tenants,
  storage/access/retention rules and audit logs; an individual consumer opt-in
  does not authorize an enterprise corpus. Named legal review must settle scope.
- **Reasoning:** raw reasoning, if retained, uses a separate controlled field with
  explicit retention/access/deletion policy; no duration is assumed. Summaries
  do not grant access to raw records or permission to train on either. Account
  and consent separation must be designed from the first product version.
- **Acceptance:** approved registry entries and notices, consent/withdrawal/
  deletion tests, secret/PII filter error audits and enterprise isolation tests
  precede any claim of enforcement. Legal decisions require named humans; an
  agent cannot certify legal compliance. All controls remain unimplemented here.
- **Rationale/alternatives:** source-specific rights and consent preserve existing
  fail-closed provenance. Alternatives were blanket owner approval, automatic
  opt-in, scraping anything public or enabling external teachers by default.
- **Evidence/revisit:** full owner section B and amendment instructions received
  2026-11-07; historical ADRs describe limited prior reviews, not new clearance.
  Contracts, legal sign-offs and privacy enforcement are [unverified]; reviewer
  names, jurisdictions, corpus permissions and retention details remain open.

## ADR-0060 — Measurable honesty initiative and factuality

- **Date/status:** 2026-11-07; owner principles1-4 translated into proposed
  post-training/evaluation requirements; none declared met.
- **Links:** extend [ADR-0048](#adr-0048-expanded-math-post-training-with-bounded-native-cpu-interfaces),
  [ADR0049](#adr0049-expanded-offline-evaluation-checkpoint-regression-and-result-receipts),
  [ADR0051](#adr0051-phase9-bounded-release-readiness-and-security-upgrade-2026-11-07)
  and [ADR-0058](#adr-0058-multilingual-coverage-and-measured-mixture).
  These behaviours are data/training/evaluation work, not tokenizer/architecture
  decisions or abilities conferred by special tokens.
- **Proposed measurable suites:**

  | Requirement | Probe and measurement |
  | --- | --- |
  | Honest, non-sycophantic assessment | Paired neutral/leading/flattering prompts with the same known facts/code defects; measure false agreement, correct corrections, false disagreement, fabricated praise and evidence-supported explanations. |
  | Initiative with stated disagreement | Seed flawed architecture/security plans; score defect detection, justified alternatives, explicit disagreement and adherence to authorization. More unsolicited actions is not a success metric. |
  | Factual accuracy without partisan agenda | Citation-grounded claims, matched identity/party swaps with unchanged evidence, currentness checks and human adjudication; distinguish facts, hypotheses, opinions, attributed claims and legal interpretation without false equivalence. |
  | Uncertainty and competence limits | Answerable/unanswerable/outdated-source cases; measure unsupported certainty, appropriate abstention, evidence retrieval and calibration when confidence is collected under a declared protocol. |
  | Refusal calibration | Benign, unsafe, privacy and tool-permission cases; measure over-refusal and appropriate refusal with human-reviewed labels. Political controversy alone is not a reason to refuse factual explanation. |
  | Verifiable work reports | Compare claimed commands, edits and tests with actual tool transcripts; measure fabricated verification and omitted material limitations. |

- **Factual-claims set:** each item requires claim, scope/date, language, cited
  source/passage, source revision or captured hash, evidence category, limitations,
  expected distinctions and named human adjudication. Contested political/legal
  items require appropriate domain review, not an agent-generated gold label.
  Preserve held-out cases, counter-framings and inter-reviewer disagreements;
  publish denominators/uncertainty and preregister acceptance thresholds.
- **Owner examples to source and review:** Ukraine probes must distinguish
  internationally recognized Ukrainian territory (including Crimea, Donetsk and
  Luhansk regions) from dated de facto control/occupation. Taiwan probes must
  distinguish de facto self-government/institutions, sovereignty claims and the
  recognition/relations policy of each named state. Do not collapse these into
  one diplomatic slogan or infer all legal status from UN representation alone.
  These record owner-required distinctions, not a completed adjudicated dataset.
- **Seed citations and access scope (2026-11-07):** proposed UN primary documents
  [A/RES/68/262](https://docs.un.org/en/A/RES/68/262) and
  [A/RES/2758(XXVI)](https://docs.un.org/en/A/RES/2758%28XXVI%29)
  returned403 and remain [unverified] in this pass. The
  [Taiwan MOFA diplomatic-relations page](https://en.mofa.gov.tw/AlliesIndex.aspx?n=1294&sms=1007)
  was read [verified, page content only]; it is that institution's account of
  relations, not independent adjudication of sovereignty. Collect dated governing
  documents and individual states' diplomatic/legal positions before human gold
  approval; a source citation alone is not adequate coverage or training rights.
- **Limits:** subjective opinions must be marked subjective; respectful honesty
  is not gratuitous insult. Photo-based owner examples remain for the later image
  phase. No competence, political-neutrality, safety or comparison claim is allowed
  before independent measured evaluation under an approved protocol.
- **Rationale/alternatives:** use evidence, paired probes and human review instead
  of hard-coded political answers, prompt-only virtue labels or self-scored success.
- **Evidence/revisit:** owner principles1-4 read fully on 2026-11-07; probe design
  is proposed and [unverified]. Named reviewers, source snapshots, legal approval,
  thresholds and actual evaluation remain prerequisites, not claimed results.

## ADR-0061 — Named human review and hardware validation gates

- **Date/status:** 2026-11-07; owner Q4/Q18/Q19 and amendment review requirements.
- **Links:** governs [ADR-0052](#adr-0052-frontier-horizon-and-gated-validation-ladder)
  through [ADR-0060](#adr-0060-measurable-honesty-initiative-and-factuality),
  retaining [ADR-0003](#adr-0003-define-reproducibility-within-explicit-execution-conditions)
  and [ADR0051](#adr0051-phase9-bounded-release-readiness-and-security-upgrade-2026-11-07).
- **Human accountability:** owner plans for an international team, chief technical
  lead and independent reviewers are owner statements, not evidence that these
  people are appointed. A named qualified human retains responsibility for critical
  architecture/security/training/release decisions. Agent agreement or passing
  tests does not replace human review. Reviewer identities are not yet supplied.

  | Irreversible or costly-to-reverse decision | Required review before freeze/use |
  | --- | --- |
  | Tokenizer-v3 control table and production V/merges | Named human with ML expertise; measured corpus/tokenization evidence and migration impact. |
  | Dense/MoE/attention architecture family | Named ML expert; CPU oracles, training/system evidence and active/total accounting. |
  | Data lineage, source/license and public release | Named ML expert plus named legal reviewer for rights; provenance, consent, exclusions and downstream deletion limits. |
  | Precision recipe/backend changes | Named ML expert; numerical/recovery checks on actual target hardware and replay-class declaration. |

- **Interim rule:** until the owner appoints the ML reviewer, every step stops
  for owner review. Reversible drafts/codec fixtures do not constitute a final
  production freeze or expert sign-off. High-cost production freezes/runs remain
  blocked without the required named human reviews and explicit owner budget.
  Preserve v1/v2 artifacts; do not treat approval of a roadmap as approval of
  later steps, expenses, dataset acquisition or destructive/public actions.
- **Hardware gate:** current environment has no GPU; future large NVIDIA clusters
  are owner-stated plans only. Before any GPU-dependent acceptance, require an
  owner-authorized small rented multi-GPU node with a stated spend ceiling and
  named ML review. First validate inventory/runtime, BF16 numerical parity,
  DDP/FSDP and checkpoint/recovery on that node. Cluster composition/performance
  and failure tests need a separate available cluster and budget. CPU tests are
  required references but not substitutes for those executions. Schedule no paid
  run or reservation in this documentation task.
- **Reports:** owner is not a programmer. Every step report must include a plain-
  language summary of what changed, why, what passed/failed, limitations and the
  next decision, alongside commit, files, exact command output and risks. State
  missing evidence simply; do not hide it behind technical terminology.
- **Rationale/alternatives:** explicit responsibility and hardware gates prevent
  irreversible decisions from being inferred from agent autonomy or unverified
  funding. Alternatives were agents approving themselves or treating CPU results
  as cluster validation.
- **Evidence/revisit:** owner specification and amendment instructions read on
  2026-11-07. Local Torch2.13.0+cpu reports CUDA available=False/GPU count=0.
  Reviewer appointments, legal approval, rented nodes, future clusters and scale
  guarantees remain [unverified]. Review gates when named humans and approved
  resources actually become available, not when an aspiration is restated.

## ADR-0062 — Evidence-gated improvement after each real training run

- **Date/status:** 2026-11-07; owner-requested standing workflow; awaiting real-run inputs.
- **Context/links:** [ADR-0010](#adr-0010-use-adamwwsd-and-staged-budgets-as-tested-hypotheses),
  [ADR-0046](#adr-0046-matched-lineage-cpu-study-and-fail-closed-promotion-controls)
  and [ADR-0061](#adr-0061-named-human-review-and-hardware-validation-gates)
  require measured hypotheses, baseline comparisons and human/resource gates.
- **Decision:** use the [post-run loop](runbooks/10_continuous_improvement.md) after
  each owner-identified real run: diagnose, compare competing causes, rank cheap
  discriminating experiments, implement only required changes with tests, append
  evidence/decisions and report in Russian. Analyze loss/validation, throughput/
  MFU and comparable eval/scaling evidence without converting hypotheses to facts.
  Preserve baselines and history; one major factor per experiment. Prefer checking
  evaluation/data quality over architecture novelty at small scale. Improvement
  requires held-out decontaminated evidence and repeated seeds where noise matters.
- **Boundary:** missing artifacts block run-specific diagnosis/ranking and training
  commands. Existing synthetic/CPU controls cannot stand in for a new owner run;
  R018's short fixture fit is not a general scaling law. Append intake status
  without inventing an ML experiment/result. Read-only analysis need not await
  model release, but implementation/production/data/legal/budget gates remain.
  This workflow does not approve the pending tokenizer Step B or paid execution.
- **Rationale/alternatives:** evidence-first attribution minimizes confounding and
  wasted compute. Alternatives were automatic architecture changes, assuming a
  cause from a symptom, or extrapolating fixture losses to a production corpus.
- **Evidence/revisit:** [verified] owner Phase10 instruction and current
  [collector](scripts/summarize_run.py)/[archive allowlist](src/qvelt/train/report.py)
  read at `fe5ee52`. No new real-run logs/config/eval were supplied in this intake;
  run-specific causes, gains, budgets and scaling predictions are [unverified].
  Revisit the queue only after identifying and validating the actual run inputs.

## ADR-0063 — Explicit v3 codec selection without changing legacy identities

- **Date/status:** 2026-11-08; Step B implemented for CPU engineering validation;
  production freeze and Step C remain separate owner/human review gates.
- **Context/links:** owner continuation authorizes the next reversible step of
  [ADR-0053](#adr-0053-tokenizer-v3-control-block-and-explicit-new-lineage),
  with [ADR-0057](#adr-0057-effort-conditioned-training-and-enforced-compute-budgets)
  markers and [ADR-0061](#adr-0061-named-human-review-and-hardware-validation-gates).
- **Choice:** add TokenizerV3Settings with mandatory requested vocab_size and a
  bounded384..1048576 ceiling, preserving the128-control map. Accept explicit
  recipe schema2 for v3; schema1/default TokenizerSettings remains the old v2
  training contract and v1 remains readable. Existing recipe bytes/digest stay
  unchanged; no default version field is inserted into old serialized settings.
  Bundles use their existing generic reserved_tokens field and validate actual
  controls against recipe type. No checkpoint or lockfile format change.
- **Trust/numerics:** retain byte-BPE/digit/normalization policy and literal text
  isolation; all v3 ordinary IDs>=128. Strict loading checks the complete table,
  flags, policy, contiguous vocabulary and byte coverage. Bound serialized v3
  payloads before writing to the existing64MiB reader limit. The requested ceiling
  does not prove a million-entry artifact fits or a corpus supplies enough merges.
- **Consumers:** native in-memory plain-v1 Engine works with a matching small
  decoder. Explicitly reject v3 at existing SFT/chat/eval-chat/HF-chat and native
  snapshot-export boundaries; do not infer tool/effort support from token IDs.
  Existing packed-data version guards remain. Implement those protocols only in
  separately reviewed work; the Step C reporting mismatch is not fixed here.
- **Reproducibility:** require equal tokenizer payload hashes in three processes
  with different hash seeds/thread counts; preserve recorded pre-change minimal
  v1/v2 payload hashes and legacy recipe identity. Creation-time/provenance-bearing
  manifest hashes are not expected to match across fresh bundle creations. Source
  changes still invalidate historical exact resume under the existing contract.
- **Rationale/alternatives:** separate settings plus recipe version avoid changing
  old asdict/digest identities or silently selecting a production V. Alternatives
  were adding a serialized default version to old recipes, choosing a final V,
  or letting v3 enter unsupported chat/export paths. None are needed for Step B.
- **Evidence:** [verified] installed tokenizers0.22.2 APIs and legacy hash captures
  on09446b4, approved local fixture bundle/CLI tests, literal/control corruption
  tests, independent-process hashes and real CPU decoder/Engine checks. The local
  [tokenizers skill](.agents/skills/orchestra/02-tokenization/huggingface-tokenizers/SKILL.md)
  informed API review; its speed/quality claims are not QVELT measurements.
  See [Step B evidence](EXPERIMENTS.md#e-step-b-tokenizer-v3-cpu-codec-integration).
- **Limits/revisit:** no final V, full million-entry BPE, multilingual quality,
  GPU execution, production corpus or model release was validated. Revisit under
  Step E measured corpus criteria and named human freeze review; stop before C.
- **Maintenance evidence, 2026-11-08:** [verified by CPU execution] the critical
  [CI coverage selection](.github/workflows/ci.yml) omitted the existing
  [v3 behavior tests](tests/unit/test_tokenizer_v3.py). Include that suite while
  keeping source scope, threshold89 and exclusions0 unchanged:114 tests/88.082437%
  (report exit2) ->277 tests/89.695341% (exit0). No test definitions, runtime,
  legacy identities or artifacts change. No dedicated Makefile critical target
  exists; full CPU discovery already includes v3. This is gate maintenance,
  not new architecture, tokenizer training, quality or GPU acceptance.

## ADR-0064 — Canonical token accounting and the token-count API

- **Date/status:** 2026-11-08; owner-approved direction, implementation pending;
  this entry specifies a contract, not an available endpoint.
- **Context/links:** extend [ADR-0055](#adr-0055-truthful-deployment-capabilities-and-shared-token-budgets)
  using [schema3](#adr-0054-sft-schema-3-with-typed-tool-blocks-and-shifted-supervision)
  and [effort accounting](#adr-0057-effort-conditioned-training-and-enforced-compute-budgets).
  Token accounting is the first context-efficiency implementation workstream;
  preserve the separately reviewed C -> D -> E sequence in [PLAN](PLAN.md).
- **Decision:** count and generation must use the same canonical validated
  renderer and token IDs. Count all system/history/content, tool definitions and
  results, retained reasoning, task metadata rendered into the prompt, and chat
  framing. No silent normalization, hidden truncation or client-side estimates
  substituted for actual encoding. Bind counts to model/checkpoint revision,
  tokenizer payload/provenance hash, protocol/template and budget-policy revision.
  Cache reuse never reduces logical context occupancy.
- **Proposed endpoint:** POST /v1/token-count accepts model/messages and only
  options actually supported by the loaded codec/backend. Future tools/task/effort
  fields require their own support; do not accept them merely for counting.
  Return accounting_schema_version, identity fields, prompt_tokens, a disjoint
  prompt_tokens_breakdown, context_window, max_input_tokens, max_tokens,
  max_completion_allowed, fits and constraint violations. Content, historical
  reasoning, tools and framing categories sum to prompt_tokens; document category
  ownership rather than counting the same tokens twice. No generation, cache
  warming, cache-presence probing or history mutation occurs. Authenticate and
  bound body/tokenization resources before processing; preserve strict errors.
- **Admission meaning:** with actual rendered prompt P, served context C and
  output cap O, max_completion_allowed=max(0,min(O,C-P)). fits requires at least
  one output token, all static request limits, and any explicit requested budget
  R to satisfy 1<=R<=O and P+R<=C. Distinguish context/body/message validity from
  transient queue/batch/cache admission: a count response reserves no capacity
  and is not a promise that a later request will be admitted. Count endpoints
  must not require a client to guess or insert special-token spellings.
- **Usage contract:** retain prompt_tokens, completion_tokens and total_tokens;
  total=P+completion. prompt_tokens_details.cached_tokens is a subset of P,
  zero only when reuse is known absent; unsupported/unknown measurements are
  explicit, never fabricated. completion_tokens includes all sampled actions,
  including terminal controls and output suppressed by stop rendering. Add
  completion_tokens_details.reasoning_tokens only with a validated semantic
  classifier; otherwise null with its capability false. Define disjoint final,
  reasoning, tool-call, summary and control/other output categories. Cached reads,
  cache writes and physically recomputed input are separate observations, not
  interchangeable measures of FLOPs or context. Auxiliary compaction, retries,
  model subcalls and tool work need a separate aggregate request ledger; never
  hide them in a shorter visible answer or rewrite earlier usage retrospectively.
- **Discovery:** extend ADR-0055 with versioned token_count, prefix_cache and
  compact support objects. Publish tokenizer/template identity and actual limits;
  compact includes supported and versions. Until implemented, all three remain
  unsupported; effort_levels=[], default_effort=null, and unsupported effort,
  tools, structured outputs and image input are rejected. Existing historical
  reasoning input is distinct from separate reasoning output or summaries.
  Current native C<=min(snapshot.context,2048), O<=256 are code-inspected bounds,
  not long-context measurements; the owner target remains total<=1,000,000 and
  output<=128,000 only after validation. Output reasoning shares R, not an extra
  allowance. Numeric production budgets are not selected here.
- **Acceptance/hardware:** CPU-only contract fixtures must compare count with
  actual admitted prompt IDs, including Unicode, framing, supported reasoning,
  limits, unknown fields, literal controls and JSON/SSE totals. Test unsupported
  v3 chat until C is implemented, cancellation, stops and auxiliary accounting;
  property tests use independent enumerated token/category references. Native
  tests do not prove a later GPU adapter reports the same contract correctly.
- **Rationale/alternatives:** one authoritative renderer prevents app/server
  budget disagreement. Alternatives were text-length heuristics, a separate
  counting template, advertising future limits, or discounting cached tokens.
- **Evidence/revisit:** [verified by reading] owner decisions on 2026-11-08 and
  [chat framing](src/qvelt/inference/chat.py), [admission](src/qvelt/inference/generation.py),
  [HTTP schema/usage/discovery](src/qvelt/inference/api.py) at
  `cfa0aa55f39aef6a79d0496e6e32e04ba88f3861`. The endpoint, extended usage and
  capabilities are [unverified/unimplemented]. Revisit after each renderer,
  backend or policy revision; do not weaken existing resume/lineage checks.

## ADR-0065 — Tenant-isolated prefix-cache identity and engine acceptance

- **Date/status:** 2026-11-08; owner-approved requirements for future engine
  selection; no prefix store or deployment is implemented by this decision.
- **Context/links:** extend [ADR-0056](#adr-0056-separate-exact-replay-topology-migration-and-statistical-reproducibility),
  [accounting](#adr-0064-canonical-token-accounting-and-the-token-count-api),
  [privacy](#adr-0059-data-rights-consent-and-legal-review-gates) and
  [hardware gates](#adr-0061-named-human-review-and-hardware-validation-gates).
- **Adoption boundary:** do NOT build a custom block-store prefix cache now.
  Later compare pinned vLLM/SGLang revisions and adopt an engine only after
  source/license/dependency/security review and execution against these
  requirements. Engine choice, block size, layout and performance are undecided;
  a vendor guide or compatible export alone does not establish acceptance.
- **Private identity:** cache hits require exact compatible token prefixes and
  an authenticated server-derived tenant namespace plus chat scope/deletion epoch
  and a server-controlled salt. A chat-specific namespace is the first-version
  default requirement so deletion has an unambiguous scope; cross-chat sharing
  would need separate retention/deletion review. Include verified checkpoint/
  weights/config and adapters, tokenizer payload/provenance, protocol/template/
  task version, RoPE/positions/document masks, dtype/KV quantization/layout and
  compatible execution/kernel policy. Client-supplied tenant/salt fields cannot
  grant access to another scope. The present single bearer key is not proof of
  tenant identity; a trusted identity adapter is a prerequisite.
- **Hash requirements:** use a collision-resistant, deterministic cryptographic
  prefix chain, such as SHA-256 over a domain-separated canonical identity and
  then previous digest, block index/length, fixed-width token IDs and relevant
  position/mask metadata. Never Python's built-in hash. Review the selected
  engine's actual algorithm/serialization and collision handling; compare stored
  tokens/identity when accepting a candidate or demonstrate an equivalent safe
  validation. Do not accept arbitrary client KV tensors. Salt/namespace material
  must not leak in logs. Sampling seed/temperature need not enter deterministic
  prefill identity; RNG and repetition history remain per request. Disable reuse
  for unvalidated stochastic/batch-dependent prefill paths.
- **Lifetime:** first version is fully private, memory-only KV, with a short,
  finite TTL and deletion with the chat. No KV disk spill, checkpoint, backup or
  durable cache logs. Actual duration and idle/absolute expiry policy need owner
  and security/legal review; this ADR selects no seconds/default. Expiry,
  deletion, access revocation and model revision changes invalidate access and
  release retained blocks; active work must cancel or stop retaining deleted
  state under an explicit policy. Do not claim forensic memory erasure. Raw
  originals and summaries follow user retention/deletion; enterprise retention
  is configurable and legal review remains pending. Retention never implies
  permission to train. Shared public prefixes are deferred; any later exception
  must be an explicit immutable server allowlist, never inferred from user text.
- **Resource/security contract:** record physical shared/private/temporary memory,
  allocator overhead and logical request budgets; bound tenant quotas and cache
  admission, eviction and cancellation. Test cross-tenant/chat access, salt and
  stale-state mismatch, poisoning, forced hash collision, malformed prefixes,
  deletion/expiry and no prompt/hash logging. Partitioning hits does not prove
  absence of timing leakage via shared hardware, load or eviction; review those
  channels and use stronger isolation where required. Avoid public cache probes.
- **Replay boundary:** CPU cached/uncached equivalence uses identical canonical
  execution and prefill chunk schedules and requires bitwise equality there.
  Different batching/kernel/device/precision schedules require declared measured
  tolerances, not an exactness claim. Preserve model ownership/stale-weight
  validation and next-token boundary computation; KV alone is not next logits.
  Engine caches are disposable serving state, not training checkpoint contents.
  Changes to RL rollout execution/log-probabilities require separate lineage and
  replay review; ADR-0056 does not authorize bypassing compatibility guards.
- **Invalidation/accounting:** changing old content or clearing a tool result
  invalidates the affected block and subsequent causal states. Do not splice old
  suffix KV after an edit/summary. Compare compaction/refetch/rebuild cost against
  the warm-cache baseline over expected later turns; no universal token-saving
  threshold is established. Report cached tokens under ADR-0064 without reducing
  logical context or implying measured acceleration.
- **Acceptance/hardware:** CPU-only fixtures can specify identity, isolation and
  deterministic numerical oracles without building a store. Actual engine
  acceptance needs a small rented GPU node after G1, explicit owner budget and
  pinned model/engine/runtime/workload. Priority is repeated long prefixes, then
  multi-user load, then single-chat latency. Measure cold/warm parity, cached
  input counts, memory and latency distributions. Cluster-length acceptance is
  separate; no paid run is scheduled.
- **Rationale/alternatives:** adopt tested storage/scheduling primitives while
  keeping QVELT privacy and replay requirements authoritative. Alternatives were
  a custom store now, unsalted cross-tenant reuse, persistent private KV or
  assuming an engine's default prefix cache satisfies these requirements.
- **Evidence/revisit:** [verified by reading] owner decisions on 2026-11-08 and
  [request-local cache](src/qvelt/inference/generation.py),
  [cache validation](src/qvelt/model/cache.py), [HTTP auth](src/qvelt/inference/api.py)
  at `cfa0aa55f39aef6a79d0496e6e32e04ba88f3861`. No selected GPU engine, TTL,
  privacy enforcement, performance or cluster claim is verified. Revisit upon
  engine proposal, identity integration, retention review and actual hardware.

## ADR-0066 — Versioned compaction with source-linked memory and drift evaluation

- **Date/status:** 2026-11-08; owner-approved format/evaluation direction;
  implementation limited to future task/schema and CPU fixtures. Do NOT train a
  compaction model now or claim the current model can summarize faithfully.
- **Context/links:** extend [ADR-0054](#adr-0054-sft-schema-3-with-typed-tool-blocks-and-shifted-supervision),
  [accounting/capabilities](#adr-0064-canonical-token-accounting-and-the-token-count-api),
  [cache invalidation](#adr-0065-tenant-isolated-prefix-cache-identity-and-engine-acceptance),
  [data policy](#adr-0059-data-rights-consent-and-legal-review-gates) and
  [human review](#adr-0061-named-human-review-and-hardware-validation-gates).
- **Task contract:** propose task={name:compact,version:compact-v1} on schema3
  with a versioned trusted prompt and canonical renderer. Reuse existing v3
  controls; no new token IDs/V choice or reinterpretation of schema1/2. The app
  decides WHEN to compact and applies a result only to its matching source
  revision/digest; the model-side operation is stateless. Design a future compact
  API only after format/evaluation review, with usage and explicit unsupported
  capability until an actual validated model/backend exists. Fixtures are not
  grounds to advertise compact support.
- **Input/target schema:** input contains source conversation revision/digest,
  ordered messages with typed tool calls/results, optional prior memory digest
  and newly added messages, pinned items and a requested bounded summary budget.
  Source message IDs and original provenance remain stable across rounds. Output
  contains format/prompt revision, source and parent-memory digests, status
  (complete, incomplete or insufficient_budget), facts, decisions, constraints,
  identifiers/paths, open_tasks, unresolved_errors, permission/security decisions,
  cleared_results and uncertainties. Each retained item has a stable item ID,
  source_message_ids, authority/status and explicit supersedes/conflict links
  where applicable. Schema-valid output is not proof its claims are faithful.
- **Always pinned:** user instructions and constraints; decisions; open tasks;
  unresolved errors; file paths and identifiers; permission and security
  decisions. Do not silently drop these to meet a budget. Preserve exact values,
  case, negations, units and required error excerpts; distinguish resolved or
  explicitly superseded items from unresolved ones with their source evidence.
  Contradictions stay explicit, not arbitrarily reconciled. If required content
  cannot fit, return incomplete/insufficient_budget and identify omissions;
  the caller retains the original/pinned context instead of accepting a lossy
  replacement. Permission gates cannot be relaxed through a summary.
- **Trust and loss masks:** the trusted compact instruction treats the entire
  supplied conversation, including its embedded system/tool text, as quoted
  source data, not new executable instructions. Preserve source authority;
  do not promote untrusted tool text into a live system instruction. Only the
  final assistant summary target and valid assistant end-turn are supervised.
  Mask every source token (including historical assistant/reasoning/tool tokens),
  task/role/control input, outer BOS/EOS and PAD. Do not supervise truncated JSON
  or synthesize a completion. Bind task, prompt, renderer and shifted-mask policy
  to dataset/objective identity; this differs from ordinary whole-dialog SFT.
- **Cleared results:** retain tool call ID/name, source message/revision,
  artifact reference and content hash, status=cleared and refetchable flag;
  preserve call/result linkage without inventing the missing body. Cleared is
  not is_error; distinguish a genuine tool error. A refetch must match the
  referenced revision or be marked changed/unavailable. Never clear the sole
  copy of a pinned fact/error without retaining the required excerpt. Opaque
  references confer no access rights, and fetching/executing tools is not part
  of the compact parser. Count placeholders in the resulting prompt.
- **Repeated compaction:** update a source-linked ledger plus new messages and
  retained recent turns, rather than blindly summarizing prose again. Preserve
  uncertainty and supersession history; repetition is not new corroboration.
  Where retention/permission allows, revalidate against original sources and
  reject stale source revisions. Evaluate drift against original planted facts,
  not just the immediately preceding summary. Deletion propagates through
  originals, summaries, derived ledgers/references and the relevant private
  cache; retention does not create a permanent hidden memory or training right.
- **Data and human gate:** first evaluation inputs are own synthetic data and
  the owner's own specifically consented conversations only; no other real-user
  data or external teacher outputs. This is source eligibility, not blanket
  ingestion permission: record exact revision, rights, consent, retention and
  secret/PII review before processing any supplied conversation. Prefer safe
  synthetic fixtures for committed tests; do not commit private transcripts.
  Originals/summaries follow user deletion/retention; enterprise retention is
  configurable, separately reviewed, and not an approved corpus. Legal review
  remains pending. Until a named ML reviewer exists, the owner reviews samples
  using the plain-language rubric below; that is not production expert sign-off.

  | Owner sample-review question | Record |
  | --- | --- |
  | Are all required instructions, decisions, tasks, errors, paths/IDs and permission/security limits preserved? | Missing/changed item and its source; critical omission blocks replacement. |
  | Does each claimed fact match its source, including numbers and negations? | Unsupported claim, contradiction or exact-value error. |
  | Are unknowns, conflicts and cleared tool results clearly identified? | False certainty, invented content or broken result link. |
  | Can work continue correctly, and does the next compaction preserve the same essentials? | Follow-up task result and differences from the original, not only the last summary. |
  | Is the summary useful within its budget, or does it honestly report that it cannot fit? | Token counts, readability and incomplete status; brevity alone does not pass. |

- **CPU acceptance fixtures:** deterministic source/gold-ledger generators with
  held-out templates/families, language and fact-position slices; immutable seeds,
  corpus/prompt hashes and matched token budgets. Measure critical and overall
  planted-fact retention, unsupported claims/contradictions, exact identifiers,
  source links, follow-up QA correctness, refetch status, source/summary token
  ratio and complete compaction cost. Compare repeated-round decay against the
  original facts, and truncation/extractive baselines at equal retained budgets.
  Include prompt injection, supersession, conflicting facts, cleared vs error,
  orphan tool links, insufficient budget and stale/deleted sources. Evaluator
  fixtures test known answers without training; scripted oracle summaries are
  labelled as fixtures, never model-generated evidence. Human entailment review
  remains necessary outside deterministic planted-fact checks. Numerical fixture
  sizes and quality acceptance thresholds need a reviewed experiment plan.
- **Hardware/lineage:** format, masks and fixture evaluators are CPU-only. They
  change no current artifact or training objective in this docs task. Future
  compact training is deferred and needs separate data/budget/ML review and a
  new objective lineage; GPU claims need G1 and a small rented node, target
  long-context quality needs a cluster. None is scheduled or validated here.
- **Rationale/alternatives:** source-linked typed memory makes omissions and
  drift inspectable. Alternatives were untracked prose summary-of-summary,
  automatic server history rewriting, truncation alone or training immediately.
- **Evidence/revisit:** [verified by reading] owner answers on 2026-11-08 and
  [existing schema/masks](src/qvelt/posttrain/chat.py) at
  `cfa0aa55f39aef6a79d0496e6e32e04ba88f3861`; source/license policy is ADR-0059.
  No compact dataset, evaluator implementation, trained capability or quality
  gain is verified. Revisit after format/fixture owner review and named legal/ML
  approval; legal clearance and measured quality cannot follow from this ADR.

## Maintenance note — Current summary reconciliation (2026-11-08)

[Verified by reading] source inspected at
`334b4c65847d68e2fcf814c650c16f0f03b92e65`: scoped benchmark/train ingress,
portable SFT/DPO/GRPO/evaluation and CPU HTTP/HF/HQQ paths already exist.
The [current STATE summary](STATE.md#current-implementation-summary) and subsystem
READMEs distinguish them from unverified deployment/quality; old STATE and
DATA_CARD prose is explicitly archived with dated pointers, not removed.
No historical ADR is edited by this note. Correct only the ServingConfig error
text to "Invalid post-validation queue/generation timeout"; body processing
precedes [deadline creation](src/qvelt/inference/batching.py). Bounds/timing and
all artifact formats/identities remain unchanged. Source bytes change, so strict
source-bound exact-resume checks remain authoritative. This maintenance grants
no data approval, feature implementation, paid run or quality acceptance.
