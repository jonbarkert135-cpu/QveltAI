# Qvelt

An original language-model framework focused on reproducible training and
mathematical reasoning. Original code/docs: [Apache-2.0](LICENSE). Third-party
skills, assets and datasets retain their own notices.

Implemented: byte-BPE and packed data preparation, a decoder-only transformer,
pretraining with atomic resume, assistant-masked SFT and LoRA, DPO, bounded
verifiable-reward GRPO, native cached inference,versioned evaluation,builtin HF
export,HQQ weight-only quantization and a bounded OpenAI-compatible chat server.
The checked CPU workflows establish engineering behavior. They do not establish
general reasoning quality or production readiness.

Measured evaluation results live exclusively in the generated [RESULTS.md](RESULTS.md),
with checkpoint/config/source hashes, commands and verified receipts. Comparisons
require identical evaluation settings. Original controls and limited public
subsets are labeled separately from official full benchmark results.

## Quickstart

Prerequisites: Linux, Git, make, uv **0.12.23**, Python **3.11–3.13**.
The pinned interpreter is selected by `.python-version`; `UV_PYTHON` overrides it.

```bash
make setup
make lint typecheck test
make smoke
```

Managed environments with a read-only home use writable local caches:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
```

Network access and local distributed sockets require the environment's network
permission. CPU tests include actual decoder/reference comparisons, training,
checkpoint/RNG replay and distributed ranks. GPU tests are separately marked;
optional SDK tests require their extras.

```bash
make EXTRA_ARGS="--extra tensorboard --extra wandb --extra eval-harness" setup
uv run --frozen --extra cpu --extra eval-harness pytest tests/integration/test_eval_phase7_interfaces.py -q
uv run --frozen --extra cpu python scripts/evaluation_suite_smoke.py --directory runs/eval-example --repository .
```

Use `make PROFILE=cuda setup` for the separately pinned CUDA build. Actual GPU,
NCCL/FSDP,large-model quantization and high-throughput GPU serving validation remain open. CPU and CUDA extras
are exclusive. No paid compute or final-scale training is launched automatically.

## Workflows

| Workflow | Guide | Scope |
| --- | --- | --- |
| Data/tokenizer | [Data preparation](runbooks/01_data_preparation.md) | Source permissions, cleaning, packing, exclusions and committed loader position |
| Model | [Block math](docs/model.md) | RMSNorm, RoPE, GQA, SwiGLU, cache and initialization references |
| Pretraining | [Server commands](runbooks/02_pretraining.md) | AdamW, schedules, rollback, signal checkpoints and resume |
| Post-training | [Math workflow](runbooks/03_posttraining.md) | Full/LoRA SFT, bounded verifiers, rejection, DPO and native GRPO |
| Evaluation | [Suite and ledger](runbooks/04_evaluation.md) | Native evaluators, actual optional standard harness, shard audit and checkpoint curves |
| Inference | [Native export](runbooks/09_inference_export.md) | Portable data-only snapshots, private KV/RNG, batching and streaming |
| Serving | [HTTP,exports and containers](runbooks/05_serving.md) | HF interoperability,HQQ,OpenAI JSON/SSE,auth,microbatching and context diagnostics |
| Comparison | [Promotion controls](runbooks/08_model_promotion.md) | Matched lineage/settings, negative findings and necessary quality gates |

Evaluation-only benchmark approval does not grant training permission. Reviewed
source pins and blocked rights/execution gates are in [DATA_CARD.md](DATA_CARD.md)
and [the benchmark registry](configs/eval/benchmarks.yaml). The prospective final
holdout remains unscored. A rise in arithmetic training reward previously
coincided with answer collapse; no model promotion is claimed.

New output directories are never silently overwritten. Checkpoints bind model,
Adam, schedule, RNG and consumer cursor to config/data/tokenizer/source identities.
Optimizer pickle is trusted project state; portable inference exports use data-only
weights. Estimated MFU needs an explicit hardware peak; it is not measured utilization.

## Project memory

Read [AGENTS.md](AGENTS.md), [STATE.md](STATE.md), [PLAN.md](PLAN.md),
[DECISIONS.md](DECISIONS.md), [EXPERIMENTS.md](EXPERIMENTS.md) and
[RISKS.md](RISKS.md) before making changes. Project-local skills are indexed in
[the usage policy](docs/development/skills.md); agents apply
[ponytail](.agents/skills/ponytail/SKILL.md) and relevant guides. Guides do not
replace installed-API verification, licensing review or tests.

The user authorizes verified work and reports on GitHub main. Never force-push,
expose secrets/raw corpora, or claim unsupported hardware or model quality.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.
