# Reproducible evaluation and release evidence

## Supported scope

The native evaluator runs offline on CPU/FP32 checkpoints up to20M parameters.
The actual optional `lm-eval==0.4.13` adapter supports likelihood, rolling
likelihood and generation through the same native model/engine. It rejects
unsupported settings and context overflow rather than truncating silently.
This is an engineering reference, not GPU/distributed benchmark throughput.

Read `DATA_CARD.md` D-EVAL-PHASE7-01 and `configs/eval/benchmarks.yaml` before
acquiring data. Evaluation permission does not authorize training. The frozen
prospective final800 are not used by this workflow. Do not enable `allow_final`
to select hyperparameters or recover from poor development scores.

## Install and run

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
make setup
uv run --frozen --extra cpu python scripts/evaluation_suite_smoke.py --directory runs/eval-original --repository .
```

This creates a real original BPE fixture, trains three short updates, evaluates
every checkpoint, plots a curve, stops a second run after step2 and resumes to3.
Weights, Adam and RNG must match; sampled predictions/metrics/audits are repeated
exactly and evaluation must leave model weights unchanged. Its original EN/RU
math, choice, LM and code controls are not public benchmark scores.

For the real standard-harness bridge, install and retain its extra:

```bash
make EXTRA_ARGS="--extra eval-harness" setup
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --frozen --extra cpu --extra eval-harness python scripts/evaluation_suite_smoke.py --directory runs/eval-harness --repository . --with-harness
uv run --frozen --extra cpu --extra eval-harness qvelt eval-harness runs/eval-harness/final.yaml
```

The bridge uses actual `lm_eval.api.model.LM` and local ConfigurableTask datasets;
`evaluator.evaluate` computes metrics. No hosted inference or remote datasets are
needed for parity. Arbitrary upstream tasks still need reviewed data, compatible
prompts/budgets and explicit local task registration; the adapter does not grant
rights or pretend every task fits this small context.

## Approved external development subsets

Acquire ONLY the immutable source/license URLs and byte hashes in the registry.
Stage locally as `runs/eval-source-cache/gsm8k-test.jsonl`, `gsm8k-LICENSE`,
`mmlu-abstract-algebra-test.parquet`, `mmlu-LICENSE`,
`arc-challenge-test.parquet`, `arc-LICENSE`. These are ignored, private
inputs; do not commit them. MMLU Parquet conversion needs the optional harness
extra; converted native evaluation itself needs no harness.

```bash
uv run --frozen --extra cpu --extra eval-harness python scripts/evaluation_suite_smoke.py --directory runs/phase7-reproduce --repository . --with-harness --external-sources runs/eval-source-cache
```

This imports the first16 test rows of each reviewed source with full copied license
and payload pins. No optimization/tokenizer fitting uses these rows. To import a
separate bounded subset, use `qvelt eval-import BENCHMARK PAYLOAD --sha256 HASH
--license-file LICENSE --license-sha256 LICENSE_HASH --count 16 --directory NEW`.
ARC conversion retains CC-BY-SA4.0,AllenAI attribution,source link and change
indication;it is not relicensed Apache. The publisher card body has an unfinished
license field,but its explicit CC-BY-SA frontmatter is the recorded determination.
Full legal text was read from the official Creative Commons GitHub repository.
Both hashes must equal the reviewed registry, not a caller-selected replacement.

Verified protocol definitions and explicit deviations:

| Task | Source protocol | Implemented development protocol |
| --- | --- | --- |
| GSM8K | lm-eval0.4.13 taskv3, Question/Answer prefix, default5-shot, strict first `####` numeric capture and normalized literal exact match | Same strict extraction, zero-shot/first16, short declared generation limit; token-limit completions may score if strict answer exists, as upstream |
| MMLU abstract algebra | Original evaluate.py5-shot, subject description, A-D choices, argmax answer-letter likelihood, accuracy | Same raw likelihood and subject/question/options/Answer shape, leading-space letters, zero-shot/first16; only one subject, not full MMLU |
| ARC-Challenge | Installed harness taskv1 Question/Answer prefix + choice text likelihood, acc and character-normalized acc_norm | First16 zero-shot;stored delimiter space excluded from normalization denominator,matching raw choice character length |
| HellaSwag | Installed harness taskv1 contextual completion, acc/character-normalized acc_norm | No corpus/evaluation; publisher HTTP451 respected, no mirror bypass |
| MATH/MATH-500 | Original MATH LaTeX/string normalization differs from numeric equivalence | No corpus/evaluation; data rights and general LaTeX metric unresolved |
| AMC/AIME | Contest-specific exact final answers | No corpus/evaluation; reuse rights/exact snapshot unresolved |
| HumanEval | Original raw Python prefix, executable tests, unbiased pass@k | No full HumanEval execution; arbitrary-code sandbox not implemented |
| Original code controls | Authored integer-function prompts and exact tests | AST-interpreted bounded expression subset; rejects calls/imports/loops/attributes, never executes Python |
| Original Russian | Authored EN/RU arithmetic and language/choice controls | Explicit language labels, per-language accuracy; not a standardized Russian benchmark |

Upstream package/source revisions and full license checks are recorded in the
registry, DATA_CARD and ADR0049. Official metric review is not dataset approval.
Code DSL rejection counts as failure; it is not claimed equivalent to HumanEval.

## Resolved jobs and metrics

`suite_job` resolves `configs/eval/quick.yaml` into strict `EvalJob` YAML containing
manifest paths/hashes, protocol, prompts, extraction, sampling, seeds, budgets,
tolerances and audit settings. Evaluate an existing approved recipe:

```bash
uv run --frozen --extra cpu qvelt evaluate runs/phase7-reproduce/final.yaml --directory runs/new-report --repository .
uv run --frozen --extra cpu qvelt eval-audit runs/phase7-reproduce/final.yaml --output runs/new-audit.json
```

Use a fresh output directory. A report commits metrics, predictions, audit, exact
model/evaluation configs, checkpoint/tokenizer/dataset snapshots, command and
Git/source/environment provenance. Incomplete directories are not valid reports.

- Accuracy: mean correct problems, Wilson95% interval. Nonrandom prefix subsets
  and correlated templates limit population interpretation; intervals are descriptive.
- pass@k: `1 - C(n-c,k)/C(n,k)` per problem, then mean. `c` is the number of correct
  attempts among the fixed `n`; no filtering of failed samples. Exhaustive small
  combination tests compare against brute-force sampling without replacement.
- Majority-vote@k: firstk attempts, canonical answer plurality; invalid attempts
  remain in the budget. Ties select the first encountered answer and are counted.
  This is a documented plurality convention, not a claim of strict >half support.
- Reasoning length: exact generated token span between a unique valid pair of
  reasoning control IDs, plus decoded character length. Completion length is
  separate. No reasoning delimiter means zero measured reasoning span.
- Failures: truncation, format error, code failure, witnessed arithmetic slip,
  explicitly declared wrong method, or incorrect_unclassified. A wrong final
  number alone cannot establish a cognitive cause. Arithmetic witness mismatches
  are logged even when the final answer happens to be correct.
- Choice: raw continuation likelihood and explicitly selected token/character
  normalization,with an explicit delimiter-excluded character mode for ARC; official ARC/HellaSwag normalization is character length.
- LM: token-weighted held-out NLL/perplexity, tokenizer/window identity required.

Never compare rows with different dataset, prompt, context/generation budget,
sampling, extraction, tokenizer-dependent LM definition or evaluator source.
The ledger's comparison key checks common settings; it is not a statistical
significance or identical-training-history certificate. No paper scores are copied.

## Training regression and recovery

Add both fields to a supported training job:

```yaml
training:
  regression_recipe: runs/phase7-reproduce/regression.yaml
  regression_sha256: EXACT_FILE_SHA256
```

The recipe must be approved v2, deterministic single-sample, at most32 cases and
32 generated tokens, CPU/FP32/single rank, at most20M parameters, no final cases.
The shared checkpoint saver evaluates every committed checkpoint, pins the actual
training source from checkpoint lineage, restores Python/NumPy/Torch RNG, and
writes hash-bound `regression/curve-step-*.json` and PNG. Unmerged LoRA regression
is rejected. DDP/GPU/large asynchronous checkpoint evaluation is not implemented.

A failed evaluation leaves the valid training checkpoint intact and fails loudly.
Resume repairs the report before continuing optimization; explicit repair:

```bash
uv run --frozen --extra cpu qvelt eval-repair runs/phase7-reproduce/training --repository .
```

Plots show fixed-control accuracy against optimizer steps. A flat or zero curve
is a valid negative measurement; CI checks integrity/replay, not forced improvement.

## Contamination and release ledger

The audit validates checksums, copied training/decontamination permissions and the
exact tokenizer, then decodes train-only frames from actual packed token shards.
Coordinates must cover every stored training token exactly; documents are not
joined across boundaries. Encoded preference chosen/rejected frames have a separate
validated path. V2 never silently crops a long fragment: bounds fail explicitly.
N-gram, exact normalized, Jaccard and template-family flags are separate.

Only declared sources are covered. Unknown earlier checkpoint history, semantic
paraphrases and latent memorization remain unverified. Presence of an audit is
not a cleanliness certificate. Old prepared bundles without copied usage policy
must be regenerated; permission is never inferred from a manifest name.

```bash
uv run --frozen --extra cpu qvelt eval-receipt runs/new-report/manifest.json --sha256 REPORT_SHA --benchmark original-en-ru-code --output runs/receipt.json
uv run --frozen --extra cpu qvelt results runs/receipt.json --output runs/RESULTS.md
uv run --frozen --extra cpu qvelt results docs/development/evidence/evaluation-phase7-*.json --output RESULTS.md --readme README.md --check
```

Receipts retain exact report-bound metrics/audit/config/provenance and input
manifests, excluding predictions/raw public text/model weights. `--check` rejects
stale ledgers and numeric README result prose. Generate root RESULTS only from the
registered receipt list; do not hand-edit numbers. Each entry records exact model
payload/checkpoint/tokenizer/dataset/source hashes and reproduction argv.

`qvelt results RECEIPTS... --release` additionally requires every registered
benchmark plus an untruncated actual-training-shard audit for each. This release
gate currently fails because several benchmarks are blocked. Even a future artifact
coverage pass is not automatic quality promotion or proof of complete history.

To send evidence for analysis, supply the report manifest/hash, resolved recipes,
metrics/audit/provenance, regression curve JSON/PNG, checkpoint lineage and ledger
receipts. Keep training data, credentials and raw licensed benchmark text private.
