# Managed CPU development environment recovery

Date: 2026-11-04. Starting source: `260b43184e776a00919f87313482b39e076a90ce`.
Validation ran with the cache/README changes in the working tree; model/trainer
source and the dependency lock are unchanged. This is environment validation,
not a new model milestone or GPU/production-data acceptance.

## Diagnosis and changes

The default command sandbox could not reach the managed HTTP proxy. Explicit
network permission restored authenticated GitHub fetch and push dry-run, without
changing proxy addresses, credentials or TLS checks. The local `main` now tracks
`origin/main`, which matched the starting revision when fetched.

`make setup` initially failed because uv's inherited home cache was read-only.
Make now exports overridable repository-local uv/pre-commit caches under the
ignored `.cache/` directory. Direct uv commands and Git commits using the installed
hook need the shell exports documented in README; Make exports apply to children.
The environment uses Python 3.12.14, uv 0.12.23 and PyTorch 2.9.1+cpu. Frozen
installation includes TensorBoard 2.19.0 and W&B 0.19.11 for offline SDK tests.

## Executed checks

- Frozen CPU/optional-SDK setup: 72 packages installed/checked; lock unchanged.
- Ruff lint and format: pass, 97 files formatted.
- Strict mypy: pass, 97 source files.
- `uv lock --check --offline` and `uv pip check`: pass.
- CPU doctor: pass, checksum 8.0; CUDA unavailable.
- `make hooks` and `pre-commit run --all-files`: pass (lint and strict types).
- Wheel and sdist build without build isolation: pass.
- Cache ignore checks and `git diff --check`: pass.
- Full CPU/optional-SDK tests: 524 passed, 5 GPU deselected, no skips, in 94.75s.

The first default-sandbox CPU suite was interrupted at 16 passed/5 GPU deselected
in 93.22s while local distributed communication waited. It is not a full-suite
pass. The replacement full suite uses explicit network permission for local
Gloo/DDP rendezvous. No sandbox or network-policy bypass is used.

## Bounded training/replay checks

Commands (run with the documented cache exports and optional SDK extras):

```bash
uv run --frozen --extra cpu --extra tensorboard --extra wandb python scripts/smoke.py --directory "$QVELT_CHECK_ROOT/numeric"
uv run --frozen --extra cpu --extra tensorboard --extra wandb python scripts/pretraining_smoke.py --directory "$QVELT_CHECK_ROOT/packed" --repository . --steps 20
```

`QVELT_CHECK_ROOT` must identify a writable parent and the child paths must not
exist. The commands use original offline demo data and the default JSON training
sink; installing SDKs does not claim full SDK stochastic-state replay.

- Numeric: 117056 parameters, 100 steps, 6400 targets; loss
  5.6883039474487305 -> 5.023181438446045. Stop37/resume100 weights and Adam state
  are bit-exact against uninterrupted training.
- Packed BPE: 127104 parameters, 20 steps, 40 microbatches, 4959 valid targets;
  256-target heldout LM probe 6.126703500747681 -> 5.756516695022583.
  Stop6/resume20 weights, Adam and consumer state match exactly.
- Machine-readable smoke receipts are retained in
  `docs/development/evidence/environment-2026-11-04.json`.

## Limits

No NVIDIA GPU is present; five GPU tests remain unrun. This does not establish
CUDA/NCCL/FSDP, multi-node/shared-store or large-model resource readiness. No paid
compute, final-scale training, external corpus or online metrics upload was used.
Networked commands still need the executor's network permission in future turns.
