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
PyTorch is not installed in this instance. M2.1 must select and verify a supported
Python/PyTorch/toolchain combination and recheck relevant APIs.

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
