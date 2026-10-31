# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current phase and last completed task

Phase 2: M2.1 is complete; M2.2 artifact/provenance work is next. The typed
package, strict YAML config, seeding/NumPy streams, JSON logging, diagnostic CLI,
locked CPU/CUDA profiles and CPU CI definition are implemented. No tokenizer,
model, trainer, serving, or checkpoint implementation exists.

Checkout: `/workspace/newai_cddevel`, branch `main`; canonical repository
`jonbarkert135-cpu/QveltAI`. Implementation commit: `9348442` plus the following
documentation commit. The user's publication pause remains active: no push,
cloud draft edit, old setup-script replay or republication occurred.

## Verified state

- `uv.lock` pins all dependencies/artifacts and selects mutually exclusive CPU
  or CUDA profiles. Python support >=3.11,<3.14; default pin 3.12.14.
- Actual Linux CPU suite: **121 passed** on each of Python 3.11.14, 3.12.14,
  3.13.5; strict mypy passed on 12 files for all three. Ruff/format green.
- CPU PyTorch 2.9.1+cpu, NumPy 2.2.6, PyYAML 6.0.3 operational. Seed tests
  compare real RNG draws with library references; doctor runs an FP32 matmul.
- Fresh frozen installs, repeated lock-preserving sync, built wheel/source
  archive, typed marker, installed-wheel CLI and base-only/no-torch behavior
  passed. CUDA dependency graph resolves; no CUDA execution occurred.
- Source/commands/outcomes/hashes: `docs/development/validation_m2_1.md`;
  engineering experiment R002 in `EXPERIMENTS.md`. No ML training/benchmark ran.
- GitHub CPU CI is defined with pinned action commits; remote CI was not run.

## Selected research and hardware constraints

Dense pre-RMSNorm/RoPE/SwiGLU/GQA, tied embeddings, bias-free layers; V=32768.
Five validated configs include exact tiny/small/base/large research dimensions
and short-context laptop-debug. Counts remain analytical, not instantiated models.
The user has RTX 3060 Laptop/Core i5/16 GB RAM; exact VRAM/driver/OS and server
budget are unknown. CPU tests do not establish laptop/CUDA/Windows support.
No datasets are approved; no source corpus has been downloaded. Project license
and Russian-data priority remain undecided and do not block artifact scaffolding.

## Next three tasks

1. M2.2: design and implement versioned run/data/tokenizer/checkpoint manifests,
   code/config/environment/artifact identities, validation and corruption tests.
2. M2.2: atomic artifact writes/checksums and interruption/recovery tests; keep
   large outputs outside Git. Config digests and seeds alone are not replay.
3. M3.1 preparation: resolve source rights and build a licensed bounded-memory
   tokenizer workflow, then encode/test math/LaTeX/Unicode/Cyrillic fixtures.

## Reproduce the last verified state

```bash
cd /workspace/newai_cddevel
uv sync --frozen --extra cpu
uv run --frozen --extra cpu qvelt config validate configs/tiny.yaml
uv run --frozen --extra cpu qvelt doctor --device cpu
uv run --frozen --extra cpu pytest -q
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy
git diff --check
git status --short --branch
```

Expected: clean main with local commits; valid tiny config, checksum 8.0,
121 passing tests, lint/format/types green. See `docs/development/setup.md`
for base/CUDA profiles and explicit limits. No train command exists yet.
