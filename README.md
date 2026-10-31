# Qvelt

An original language-model framework under development, focused on reproducible
training and mathematical reasoning. Original code/docs: [Apache-2.0](LICENSE).
Third-party skills/assets and test data retain their separate notices.

The engineering foundation now includes an RFC decoder and **real synthetic
training**, not a general-purpose pretrained model. The CI smoke uses an explicit
117,056-parameter dimension override, 100 updates and 6,400 next-token targets.
The 19.34M/125.85M/493.55M/1.510B family parameter counts are checked on actual
meta-device models; only the tiny reference has a CPU forward/backward check.
No reasoning benchmark, real-corpus pretraining or GPU performance is claimed.

## Quickstart

Prerequisites: Linux, Git, make and uv **0.12.23**; Python **3.11–3.13**.
`.python-version` selects3.12.14; use UV_PYTHON to choose another tested interpreter.
From a clean checkout:

```bash
make setup
make lint
make typecheck
make test
make smoke
```

`make test` includes numerical references, actual tiny-model forward/backward,
checkpoint/RNG replay and two CPU ranks. GPU tests are deselected; optional
TensorBoard/W&B tests skip if their extras are absent. `make smoke` executes
100 real AdamW updates, then stop 37/resume 100, comparing weights and Adam states.
Outputs use a new temporary directory, printed with its actual metrics.

For CUDA setup use `make PROFILE=cuda setup` and the
[server runbook](runbooks/00_server_setup.md). CPU and CUDA extras are exclusive;
the pinned GPU build is PyTorch 2.9.1/CUDA 12.8. GPU/NCCL/FSDP2 hardware validation
is pending. `make hooks` installs the local pre-commit hooks; `make format` fixes
first-party formatting. Optional offline backend checks:

```bash
make EXTRA_ARGS="--extra tensorboard --extra wandb" setup
uv run --frozen --extra cpu --extra tensorboard --extra wandb pytest tests/unit/test_metrics.py -q
```

## Honest component status

| Component | Implemented | Actual verification / remaining gate |
| --- | --- | --- |
| Strict YAML/dataclasses/CLI overrides | Yes | CPU tests; resolved config/hash saved |
| Layout, lock, Make, pre-commit, lint/types/CPU CI | Yes | Local checks; current CI evidence in validation report |
| Byte-BPE tokenizer | Yes | Train/save/load/manifest/evaluation; Python 3.11–3.13 fixture replay |
| Cleaning/dedup/decontamination/shards/packing/mixing | Yes | Offline EN/RU end-to-end tests; public-benchmark production gate remains blocked |
| Multi-worker data resume | Yes | Consumer-committed cursor, 0/2-worker crash replay and uint16/uint32 pair references |
| Math fixtures and bounded exact rewards | Yes | Original bilingual templates, SymPy/stdlib cross-checks; no reasoning quality result |
| RMSNorm/RoPE/GQA/SwiGLU tied decoder | Yes | Independent value/gradient/causality/document-mask oracles; all four meta counts |
| Synthetic training/AdamW/token LR | Yes | 100 CPU updates; source/config/software/topology-compatible replay |
| DDP | Yes | Two CPU ranks, uneven-mask global-gradient oracle and runner replay |
| Atomic checkpoints/safetensors export | Yes | Corruption/incomplete outputs/RNG/Adam/weight replay on Linux local storage |
| DCP/DTensor shards | Yes | Two actual CPU ranks and DTensor shard round trip |
| GPU FSDP2/NCCL | API path and tests | Hardware tests provided; not run here |
| JSON/TensorBoard/W&B metrics | Yes | JSON/event files and actual W&B offline run; no online upload |
| SFT/preferences/reasoning RL | No | Phase6; directory documents scope only |
| Reasoning evaluation/serving/export formats beyond safetensors | No | Phases7–8; no fake commands |

See [current state](STATE.md), [roadmap](PLAN.md), [engineering design](docs/development/foundation_design.md),
[validation](docs/development/validation_foundation.md), [decisions](DECISIONS.md),
[experiment records](EXPERIMENTS.md) and [data rights](DATA_CARD.md).

## Commands and artifacts

```bash
uv run --frozen --extra cpu qvelt config resolve configs/model/tiny.yaml --set model.context=32
uv run --frozen --extra cpu qvelt doctor --device cpu
uv run --frozen --extra cpu qvelt train configs/train/smoke.yaml --directory /tmp/qvelt-run-01 --stop-after 37
uv run --frozen --extra cpu qvelt train configs/train/smoke.yaml --directory /tmp/qvelt-run-01 --resume latest
```

Choose a new run path; existing output is never silently overwritten. A training
job embeds the shared experiment schema, and `--set path=value` overrides only
known scalar leaves before full validation/hashing. Changing the configured
horizon is not exact resume. Checkpoints keep fully resolved experiment/job YAML,
run/config/data-descriptor identities, model/Adam/schedule/sampler/RNG, code and
package observations. `pip-freeze.txt` is a normalized installed-version snapshot
without editable paths/direct URLs; uv.lock is the installation authority.

The synthetic data/tokenizer identity slots explicitly hash numeric-fixture
**descriptors**, not the trained byte-BPE/data manifests from M3.1. The next real-data
trainer must consume the actual Phase3 artifacts; no dataset approval is implied.
Optimizer/DCP state is trusted project state, not an arbitrary uploaded checkpoint.
Generic `manifest verify` checks bytes; semantic training/tokenizer loaders add
compatibility checks. Metrics include valid targets/s, unclipped gradient norm,
clipping and LR. Estimated MFU stays null without a user-supplied peak FLOP rate;
it is not a measured hardware-utilization percentage.

Offline tokenizer/data preparation (no external download):

```bash
uv run --frozen --extra cpu python scripts/data_smoke.py --directory /tmp/qvelt-data-smoke-01
uv run --frozen --extra cpu qvelt data benchmark /tmp/qvelt-data-smoke-01/prepared/manifest.json --workers 0 2
```

The [data runbook](runbooks/01_data_preparation.md) documents source/license gates,
cleaning limits, exact/near dedup, complete planned-benchmark exclusion, binary
layout, deterministic mixing/cursor replay and laptop resource assumptions.
GSM8K test has a working evaluation-only downloader; MATH/AIME/MMLU rights/snapshots
remain blocked. The offline demo is explicitly non-production. The existing
training command uses numeric fixtures until the real-data trainer milestone.

Tokenizer workflow: [runbook](docs/development/tokenizer_workflow.md),
[earlier validation](docs/development/validation_m3_1.md),
[data validation and measured receipts](docs/development/validation_data.md).
Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

## Instructions for every AI agent

Read [AGENTS.md](AGENTS.md), STATE/PLAN and relevant RFC/decision entries before
work. Apply the user-supplied [ponytail](.agents/skills/ponytail/SKILL.md) on every
coding task (`full` default), plus relevant [installed skills](docs/development/skills.md).
409 entries are pinned by commit/file hash: 408 vendor entries plus ponytail.
Six vendor entries with restrictive or unspecified licenses are excluded.
Skills are guidance, not verified APIs, authorization for paid jobs or a replacement
for tests. Verify mathematical claims with executable exact arithmetic or independent
NumPy/PyTorch references. All project agents share this rule through AGENTS and
Claude/Gemini/Cursor entry files. Verified work/reports are published to main under
the user's current authorization; never force-push or expose secrets/raw corpora.
