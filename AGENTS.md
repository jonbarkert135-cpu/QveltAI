# Qvelt contributor and agent instructions

## Start every session

1. Read this file, `STATE.md`, `PLAN.md`, and relevant entries in `DECISIONS.md`.
2. Inspect Git status and preserve existing user work. Use the existing checkout;
   cloud tasks are already isolated. Do not create a Git worktree unless the user
   explicitly requests one.
3. Restate the task and acceptance criteria in Russian. Write a short design
   describing behavior, rationale, and interfaces before implementing code.
4. Work on the next dependency-ready milestone. The initial session is explicitly
   documentation-only; further implementation awaits the user's next prompt.

## Scope and evidence

- Communicate with the user in Russian. Write code, comments, and repository
  documentation in English.
- Implement core model and training code originally. Reference public designs
  and attribute dependencies; do not copy substantial repository code.
- Record every non-trivial design decision in `DECISIONS.md`, with context,
  alternatives, rationale, date, sources, and revisit conditions. Cite paper
  titles and years or official library documentation.
- Mark each source claim `[verified]` only after reading the relevant source;
  record its version or revision and verification scope. Use `[unverified]` for
  unread or unavailable evidence. Verification of a document is not verification
  of an implementation, a benchmark, or support for the installed version.
- Do not invent APIs, papers, metrics, results, performance, or hardware support.
  Confirm the installed APIs and tool versions before using them.
- An ambitious engineering goal does not establish parity with any commercial
  lab. Describe implemented capabilities and measured evidence separately.

## Implementation standards

- Use Python 3.11+, PyTorch 2.x, type annotations, math-aware docstrings, small
  composable modules, and YAML configuration validated through dataclasses.
  Select and pin actual versions in M2.1, after compatibility checks.
- Every implementation module requires unit tests. Numerical modules additionally
  require independent reference comparisons, including gradients where relevant.
  Avoid circular tests that merely repeat the implementation.
- Keep CPU/tiny correctness paths usable without CUDA. Add accelerator-specific
  optimizations behind capability checks with tested portable alternatives.
- Seed runs, define deterministic data ordering, and make checkpoints resumable.
  State reproducibility limits by version, device, precision, and world size;
  do not promise bitwise identity across hardware or PyTorch releases.
- No placeholder functions, fake results, or `TODO: implement` code. Record future
  work with dependencies and acceptance criteria in `PLAN.md` instead.
- Pin dependencies and use lockfile-preserving installation. Keep secrets, raw
  corpora, credentials, large checkpoints, and machine-specific paths out of Git.
- Do not choose a dataset by popularity alone. Record its exact revision,
  license, permitted use, and provenance in `DATA_CARD.md` before ingestion.
  Unclear licenses block use; do not scrape against site terms.

## Validation and delivery

1. Make small, reviewable commits with clear messages when a coherent change is
   verified. Do not reset, overwrite, or publish the user's changes.
2. Run relevant unit/reference tests, Ruff, and mypy once M2.1 supplies these
   tools. Fix failures; report actual counts and distinguish skips and unrun tests.
   Documentation-only changes require document/link/consistency checks; do not
   describe those checks as ML or runtime validation.
3. Execute tiny CPU or available single-GPU smoke tests for implemented training
   workflows. Capture configuration, revision, seeds, command, logs, and results
   in `EXPERIMENTS.md`. Do not launch final-scale training yourself.
4. Require explicit user authorization for expensive compute, paid services,
   irreversible decisions, or choices requiring private hardware/budget details.
   Continue reversible independent work when possible. Never request secret
   values in chat.
5. At the end of every work session update `PLAN.md`, `STATE.md`,
   `DECISIONS.md`, `EXPERIMENTS.md` when experiments ran, and `RISKS.md` as needed.
6. Report what changed, what actually passed or failed, what remains uncertain,
   and the next recommended action in Russian. Follow explicit stop boundaries.
