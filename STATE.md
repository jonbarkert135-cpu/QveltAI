# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase

Documentation bootstrap M0 is complete. Phase 1 (research and architecture
decisions) is next and has not started. The explicit first-session boundary is
documentation only; wait for the user's next prompt before implementation.

## Last completed task

Inspected the empty repository and established project memory: `PLAN.md`,
`DECISIONS.md`, `EXPERIMENTS.md`, `RISKS.md`, `DATA_CARD.md`, `AGENTS.md`, and
`README.md`, with this state file. The roadmap covers all ten requested phases
and records owners, statuses, dependencies, tasks, and acceptance criteria.

## Observed repository and environment

- Checkout: `/workspace/newai_cddevel`.
- Origin identity: `github.com/jonbarkert135-cpu/newai_cddevel`.
- Working branch: `work`; the initial inspection found an unborn HEAD, no
  tracked/project files, no pre-existing user changes, and no `AGENTS.md`.
  No branch reset, force operation, or push was performed.
- Baseline tools observed: Python 3.12.14, Git 2.52.0, Node.js 24.19.0,
  npm 11.9.0. Node is not a selected Qvelt dependency.
- Python package metadata: PyYAML 6.0.3 is present; torch, pytest, Ruff, and mypy
  are not installed. No dependency versions have been selected or installed
  for Qvelt. Hardware capabilities have not been profiled.
- Managed instance reports running/connected with enforced network policy.
  Official PyTorch v2.6.0 documentation/source was read through the platform
  proxy; see `DECISIONS.md` for the narrow verified claims.

## Verification of this delivery

- A standard-library documentation checker passed: eight expected Markdown files,
  seven relative file links, ten ordered phases, 23 milestones with required
  metadata, four completed bootstrap tasks, six decision records, and 16 risks.
  It also checked task status consistency, newlines, whitespace, and code fences.
- No Python/model code, configs, tests, data, tokenizer artifacts, weights,
  dependency lockfiles, or CI were added. No ML training or benchmark ran.
- Unit tests, numerical tests, Ruff, and mypy are **not run / not applicable to
  this documentation-only delivery**. Their tools and project configuration are
  scheduled in M2.1; these checks are not reported as passing.
- This delivery is a local documentation commit. Nothing has been pushed,
  published, or saved as an environment installation/startup draft.

## Next three tasks

1. **M1.1:** read primary architecture sources and log verified findings; compare
   a dense causal-decoder baseline and alternatives without implementing them.
2. **M1.1:** write the chosen architecture/interface specification, including
   equations, tensor shapes, masks, initialization, parameter counts, and reference
   tests for tiny, approximately 100M, and planned 500M–1.5B tiers.
3. **M1.2:** define resource/replay contracts and a held-out mathematical evaluation
   protocol, then prepare the evidence needed for M2.1 dependency/scaffold work.

## Open blockers and uncertainty

- No blocker prevents the next source-review and design work.
- No training servers, GPU inventory, memory/interconnect measurements, or budget
  are supplied. Real distributed/performance validation and final training require
  those prerequisites. Do not infer that full-size training fits on weak devices.
- Architecture, tokenizer, optimizer, RL objective, dependency pins, datasets,
  project license, and export/runtime formats remain undecided. No dataset is
  approved. Qvelt brand/trademark/domain availability has not been checked.
- `[unverified]` reading-queue entries are research candidates, not accepted
  designs. No benchmark or parity claim with other labs is supported.

## Reproduce the last verified state

From the checkout:

```bash
cd /workspace/newai_cddevel
git status --short --branch
git log -1 --oneline
git ls-files
git diff --check
```

Expected: a clean `work` branch with a documentation commit and eight root
Markdown files; no executable Qvelt package. Inspect the roadmap headings with:

```bash
rg '^## Phase [0-9]+|^### M[0-9]+|^Owner:|^Acceptance:' PLAN.md
rg '\[verified\]|\[unverified\]' DECISIONS.md
```

There are intentionally no train/install/test commands yet. Add real commands
when the associated implementations exist and have actually been exercised.
