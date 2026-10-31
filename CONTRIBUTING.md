# Contributing to Qvelt

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md and the relevant research RFCs.
AI agents must apply the project-local ponytail skill and read relevant skill
instructions; upstream examples must be checked against installed versions.

Use Linux/Python 3.11–3.13 with uv 0.12.23. The project's default interpreter is
3.12.14. From a checkout with uv and make available:

```bash
make setup
make lint
make typecheck
make test
make smoke
make hooks
```

Use `make format` to format first-party code. `make hooks` installs local hooks;
setup does not overwrite existing developer hooks automatically. `make test`
selects non-GPU cases, including actual two-process CPU integration. Optional
TensorBoard/W&B tests skip when extras are absent; validate both adapters with:

```bash
make EXTRA_ARGS="--extra tensorboard --extra wandb" setup
uv run --frozen --extra cpu --extra tensorboard --extra wandb pytest tests/unit/test_metrics.py -q
```

For CUDA use `PROFILE=cuda` instead of cpu and follow runbooks/00_server_setup.md.
GPU cases are explicitly marked and must run on real devices; mocks/CPU DCP
success do not validate NCCL or FSDP2. Do not launch paid/final-scale runs without
the user's authorization.

Every new implementation module requires meaningful tests; numerical code needs
independent value/gradient references. Preserve validation and failure handling.
Update the roadmap/state, decisions with verified source/version citations,
experiments with actual results, and risks when behavior changes. Keep generated
runs/data/credentials out of Git; use fresh output roots. Never weaken tests to
pass, silently overwrite a checkpoint, infer dataset rights from a badge, copy
external model implementations or invent benchmark/training results.

Preserve public imports when moving modules; the primary layout is documented
in docs/development/foundation_design.md. Resolved dataclasses are the config
truth. CLI leaf overrides are validated before hashes are computed. A schema,
source/library or topology change is not an exact-resume-compatible edit.

The project owner currently requests verified development directly on main;
follow the latest AGENTS/user publication policy, preserve others' work, and
never force-push. A PR is also acceptable for external contributions. Original
code/docs use Apache-2.0; retain separate third-party and fixture permissions.
