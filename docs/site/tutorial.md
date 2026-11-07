# Train your first tiny model in one hour

This is a bounded engineering tutorial on the **actual RFC tiny configuration**:
eight layers,hidden320,GQA5/1,FFN896,vocabulary32768,context512. It performs real
optimization,restart,SFT,evaluation,export and HTTP serving. It does not train a
useful reasoning assistant or establish safe refusal behavior.

## Prerequisites and budget

Use Linux x86-64,Git,make,Python3.11–3.13 with venv support,outbound package access,
and a private local working directory. Allow at least8GiB RAM and8GiB free disk
for the environment,data,checkpoints and exports. These are conservative operating
budgets,not measured hardware minima. A CPU run is sufficient;no GPU or API key is
needed. Cold downloads or a constrained machine can exceed the one-hour target.

The tutorial keeps the full tiny dimensions. Original generated nonce/math fixtures
provide enough train-only diversity to fit the real vocabulary. They are not a
recommended production corpus. No benchmark/final-holdout text is trained on.

## Install from a fresh checkout

```bash
git clone https://github.com/jonbarkert135-cpu/QveltAI.git
cd QveltAI
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv==0.12.23
export PATH="$PWD/.bootstrap/bin:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
uv sync --frozen --extra cpu --extra hf --extra serving
uv run --frozen --extra cpu qvelt doctor --device cpu
```

The repository pins Python3.12.14;uv obtains it if necessary. Record the checkout
commit and keep it unchanged during optimization/resume. A dependency or source
edit is not an exact-resume-compatible operation. Use a filesystem with adequate
space;do not put this run on a nearly full container root.

## Execute the complete tiny pipeline

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
  uv run --frozen --extra cpu --extra hf --extra serving python scripts/first_model.py \
  --directory runs/first-model --repository . --steps 6
```

Choose a new directory. Do not silently overwrite an existing model. The script:

1. Creates licensed original domain fixtures and a train-only byte-BPE tokenizer.
2. Resolves `configs/model/tiny.yaml` exactly and prepares document-isolated shards.
3. Runs six real pretraining updates,stopping after three and resuming the checkpoint.
4. Prepares the original bilingual safety/helpfulness design fixture with family
   splits and assistant-only masks;performs three full SFT updates from base weights.
5. Runs fixed original math/choice/LM/code controls and audits both actual training
   sources. The resulting numbers are diagnostic,not a safety or reasoning score.
6. Writes verified native/HF exports and checks native versus real HF logits.
7. Starts a real authenticated loopback server,checks JSON/SSE,concurrent batching,
   disconnect and graceful shutdown. Its ephemeral auth key is never printed.

The small update count keeps the exercise bounded. Meaningful training needs a
separately approved data mix and budget. Extending this fixture run is not the
recommended way to obtain a capable model.

## Inspect and verify artifacts

```bash
uv run --frozen --extra cpu qvelt manifest verify runs/first-model/snapshot/manifest.json
uv run --frozen --extra cpu qvelt results runs/first-model/receipts/tutorial.json \
  --output runs/first-model/RESULTS.md --check
uv run --frozen --extra cpu qvelt generate runs/first-model/generate.yaml --stream
```

`result.json` records elapsed time,environment,source/config/checkpoint hashes,
actual resume point,HF error and HTTP workload. `pretraining.yaml`,`sft.yaml` and
`evaluation.yaml` contain exact resolved settings. Each training directory has
its metrics and committed checkpoints. The local generated RESULTS points to its
native receipt;contamination details are in `evaluation/audit.json`.

Healthy engineering outcomes:all assertions pass,base resume starts at the saved
step,evaluation leaves weights unchanged,HF logits agree within the recorded
tolerance,SSE matches JSON and shutdown pending count is zero. Nonsensical text,
failed answers and no useful refusal behavior are expected at this budget.
Never interpret a lower training loss alone as generalization or safety.

## Continue with real training

Follow [the ordered readiness checklist](responsible-release.md) before the
[small server pilot](reference/runbooks/02_pretraining.md). First establish
rights/data quality,complete required exclusions,hardware correctness and recovery,
then run a short pilot with independent evaluation. Do not use prospective final
holdout data for choosing data,learning rates or stopping steps.

For failures,use [Troubleshooting](troubleshooting.md) and share sanitized logs,
manifest hashes and the exact command. The [status page](status.md) states remaining
hardware,license,security and model-quality gates.
