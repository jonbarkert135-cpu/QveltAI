# Phase6: math post-training on the supported CPU path

This is the consolidated entry point for chat/SFT, reasoning-data construction,
DPO and outcome-GRPO. It extends [SFT](04_supervised_finetuning.md),
[preferences](05_preferences.md) and [RL](06_reasoning_rl.md). The original
decoder/train/checkpoint implementation is reused; TRL/PEFT/HF are not installed.
No paid job or final-scale training is part of these commands.

## Execute the complete arithmetic control

From a checkout with the locked environment and a new output directory:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo
mkdir -p runs
make setup lint typecheck
uv run --frozen --extra cpu python scripts/posttraining_smoke.py --directory runs/phase6 --repository .
```

This trains a real reduced127104-parameter decoder/V445/T128: base3, SFT20,
DPO4, then GRPO60. Base/SFT/DPO seed20261031,RL seed20261106; greedy/stochastic
evaluation uses explicit budgets and stochastic validation seed20261107. Each RL
window has4 prompts x8 samples,temperature1,max4 sampled actions,LR0.001,KL0.01,
clip0.2,zero weight decay. The DPO policy descends from the same SFT used as the
frozen RL reference; reference and initial policy are different concepts.

The original arithmetic is a-a+b,b in {0,1},with32 EN/RU problems and eight
operand strata. This deliberately easy fixture tests reward learning. Neither a
larger reward nor longer output establishes general mathematical reasoning.
SFT uses whole-stratum splits; DPO consumes only SFT-training strata, with its
own pair-train/validation split. RL's fixed64-sample evaluation uses SFT-validation
strata unused by all optimizers. The project final800 and GSM8K test are untouched.
The local specification was written before diagnostics; it was not an independent
timestamped preregistration, and the task was chosen to be easy.

Artifacts include:

- `result.json`: actual stage results, every RL-window reward, fixed-seed
  validation before/after, rejection report and online/offline equality.
- `sft/sft.yaml`, `dpo.yaml`, `grpo.yaml`, `offline.yaml`, `reasoning.yaml`:
  resolved recipes with input hashes and real paths.
- `rl/metrics-*.jsonl`, `rl/rollouts-*.jsonl`: each window's loss, reward,
  correctness, formatting, length cost, truncation, gradients and all samples.
  Each rollout contains readable question/gold/text plus actions/logps/seeds.
- `rl/verifier-audit.json`:14 public fixed adversarial/numeric/symbolic controls
  excluded from optimization. Checks run before training and at checkpoint times.
- `reasoning/{manifest.json,report.json,audit.jsonl}`: actual own-checkpoint
  rejection sampling on training prompts only, including every rejected candidate.

The script independently trains the offline replay and compares all losses,
model,Adam and RNG exactly on the same locked CPU setup. Run source/hardware/
precision/world-size identity remains mandatory for exact resume; initialization
from checked weights is distinct from restoring a training run.

## Full SFT and LoRA

Full fine-tuning is the default/main method. It reuses packed document attention,
assistant masks, gradient accumulation, clipping, schedules, Adam and atomic
checkpoint recovery from pretraining. To repeat the generated full-SFT job:

```bash
uv run --frozen --extra cpu qvelt sft runs/phase6/sft/sft.yaml --directory runs/full-sft --repository .
```

For an actual adapter experiment using the same source and base checkpoint:

```bash
uv run --frozen --extra cpu qvelt sft runs/phase6/sft/sft.yaml --directory runs/lora-sft --repository . --set lora_rank=4 --set lora_alpha=8.0 --stop-after 6
uv run --frozen --extra cpu qvelt sft runs/phase6/sft/sft.yaml --directory runs/lora-sft --repository . --set lora_rank=4 --set lora_alpha=8.0 --resume latest
```

`lora_rank: 0` selects full training. Positive rank1..64 adapts all block
attention/FFN linears with W_eff=W+(alpha/r)BA; embeddings/norms/base W stay
frozen. A uses Gaussian variance1/input_width; B starts at zero. Rank must fit
every target matrix. The shared optimizer excludes frozen parameters. Checkpoints
retain base tensors plus adapters for exact standalone replay; they are **not**
small adapter-only redistribution packages. `merge_lora(model)` provides tested
in-memory dense merging; discard existing KV caches after merging. Unmerged
adapter checkpoints are rejected by the standard native export/DPO/GRPO loaders.
No adapter-export compatibility or quantization result is implied.

LoRA acceptance is single-process CPU FP32,compile=false. QLoRA is unimplemented:
we have no tested quantized decoder/bitsandbytes path, compatible manifest or
GPU evidence. An unknown QLoRA configuration fails instead of silently falling
back to full training. GPU/distributed SFT and production small runs need their
own hardware/source acceptance; these commands exercise the reduced control.

## Exact chat masks and reasoning fields

Schema1 conversations retain their original behavior. Schema2 supports an
optional nonempty `reasoning` field on assistant messages only:

```json
{"schema_version":2,"messages":[{"role":"user","content":"2+3?"},{"role":"assistant","content":"Final answer: 5","reasoning":"Add two and three."},{"role":"user","content":"And 5+1?"},{"role":"assistant","content":"Final answer: 6","reasoning":"Increase five by one."}]}
```

Tokenizer-v2 explicit controls8/9 encode think boundaries. Text inside a user
message that spells `<think>` or a role is ordinary byte-BPE text. Loss includes
assistant reasoning, delimiters, final content and end-turn; prompts, role IDs,
system/user end-turns, outerEOS and padding contribute zero. Whole conversations
pack together with document isolation. Right-prefix truncation reserves masked
EOS and never invents a completed assistant response. A partial reasoning prefix
can supply SFT targets; it never earns a complete-answer RL reward. Legacy
tokenizer-v1 reserved IDs are never reinterpreted as reasoning controls.

## Own-checkpoint rejection sampling

The smoke produces a real resolved recipe and source. Re-run it into a new path:

```bash
uv run --frozen --extra cpu qvelt reasoning-data runs/phase6/reasoning.yaml --directory runs/rejection-repeat
```

`ReasoningJob` pins checkpoint/tokenizer and optional local problem manifest.
Without a problem source it uses the registered original exact generator.
Budgets:1..64 problems,1..8 independent samples/problem,1..128 actions/sample;
CPU engine context/cache/time limits also apply. Private seeds are recorded.
`framed: true` requests optional `<think>...</think>` followed by exactly one
`Final answer: ANSWER`; selected SFT records preserve both explicit reasoning and
the final-answer format. Bare mode accepts a whole numeric/polynomial answer.

Verifiers:

| Mode | Scope | Limits |
| --- | --- | --- |
| exact | rational equality, including finite decimal spellings | ASCII,bounded digits,no expression evaluation |
| numeric | abs(error)<=atol+rtol*abs(gold) using exact rational arithmetic | explicit tolerances0..0.1,no NaN/inf/exponents |
| polynomial | rational-coefficient identity over x,y,verified with SymPy QQ | <=128 chars,48 AST nodes,total degree<=8;division by constant only |

No generated string reaches `eval`, `sympify`, `parse_expr` or tool execution.
These checks do not prove free-text reasoning, arbitrary algebra, variable-domain
equivalence or excluded denominator points. Bad grammar, repeated final answers,
incomplete turns and wrong results are rejected. Every candidate is retained in
the audit. Deduplication is lexical case/whitespace/token-set Jaccard on reasoning
bodies within each problem, not semantic diversity. Different equivalent final
number spellings do not create different traces;bare answers yield at most one
selected row/problem. Correct counts for pass@k include duplicates
before selection; per-difficulty estimates use1-C(n-c,k)/C(n,k),with counts and
sample budgets. Imported unrated problems stay unrated.

A selected training-only bundle is not a complete validation dataset. Combine
it with separately frozen validation families, retaining source licenses and
split identity, before preparing a new SFT dataset. Never sample validation/final
gold into rejection-training data. Empty selection is a valid negative result;
do not fabricate traces or automatically increase budgets until something passes.

## Licensed original GSM8K train ingress

Read D-GSM8K-TRAIN-BOUNDED-01 in DATA_CARD and the pinned publisher LICENSE/README
first. Only the exact approved original train bytes/license are accepted; the
adapter retains16 questions/numeric gold labels by default,maximum64. It does
not verify the publisher's worked solutions or infer difficulty labels.

```bash
mkdir -p runs/gsm8k-source
curl --fail --location https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl --output runs/gsm8k-source/train.jsonl
curl --fail --location https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE --output runs/gsm8k-source/LICENSE
uv run --frozen --extra cpu qvelt import-gsm8k runs/gsm8k-source/train.jsonl --sha256 17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465 --license-file runs/gsm8k-source/LICENSE --license-sha256 86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b --directory runs/gsm8k-problems --count 16
```

Use the returned manifest hash in `ReasoningJob.problems` with
`approved_for_training: true`. Many prompts exceed the reduced control's128-token
context; budget rejection is intentional. This approval does not waive the
production benchmark registry/overlap audit or permit GSM8K test for training.
The full source payload/license checksums and imported count appear in
`import.json`; raw corpus and model weights stay outside Git.

Other sources require their own DATA_CARD revision/license/usage approval and
a normal bounded raw bundle with question text,ID,family,difficulty,gold.
The permissive-license allowlist and `approved_for_training` flag enforce declared
policy, not independent legal verification. A repo license is not automatically
the license of every mirrored dataset or model output.

## Optional external synthetic traces

Off by default. No external teacher/API has been invoked. To import already
reviewed local teacher traces, register their exact manifest and teacher/revision,
license, terms URL and training permission first. Raw records must contain strict
single-turn chat JSON plus a gold numeric answer in document metadata.
Provide an <=8KiB YAML terms acknowledgement with exactly:

```yaml
teacher: exact-reviewed-teacher-name
revision: exact-reviewed-revision
license_id: Apache-2.0
terms_url: https://publisher.example/immutable-terms
training_permitted: true
reviewed: true
```

Then use `qvelt ingest-traces --help`: both `--external-synthetic` and
`--approved-for-training` plus exact source/terms SHA256 are required. It checks
final answers, copies source license/terms evidence and logs rejects. It grants
no rights by itself. Review output-use, distillation/training, redistribution,
attribution, provider terms and upstream data restrictions; unclear permission
blocks use. The code switch does not authorize a paid API or teacher request.

## DPO, GRPO and stability interpretation

DPO is the original sigmoid objective with **summed** assistant response logps
and a frozen verified SFT reference; no alternate objective is silently selected.
Read runbook05 for exact masks and replay commands. It stores a second FP32
model; policy+gradient+Adam+reference require roughly20 bytes/parameter before
activations/logits/temp buffers (no claim of measured peak RAM/VRAM). For19.34M
this is386.8MB decimal; for125.85M small about2.52GB,with potentially much larger
training peaks. There is no logp-cache/offloading/quantized-reference workaround.

GRPO uses fresh temperature1 native-engine rollouts,group-relative population-std
advantages,zero advantages for constant groups,clipped ratios and sampled
exp(delta)-delta-1 KL. Each batch receives one update; no value model. KL0..1,
format_weight0..0.2,length_penalty0..0.2 are explicit YAML controls. Shaped reward
is max(0,(1-format_weight)*correctness+format_weight*format-length_penalty*n/maxn).
Malformed/incomplete responses get no correctness/format credit. Wrong but valid
answers may earn only the declared format component; log it separately. Longer
responses cannot win merely by length. Truncated sampled actions remain in the
policy/KL loss with zero outcome credit; no terminal token is synthesized.

Current GRPO is single CPU FP32,original environments,<1M parameters,groups2..8,
prompts1..8,max64 actions and at most100 windows. Native KV caching is reused,
but collection is sequential by completion with teacher-forced logp recomputation;
this is a correctness path, not a high-throughput rollout server. No HF fallback
or vLLM acceleration is claimed. GPU/distributed and arbitrary external-problem
RL environments remain unverified/unimplemented in this prototype.

Online mode writes readable JSONL windows. Offline mode requires `mode: offline`,
`rollout_file` and its SHA256 (at most8MiB/100 windows). It revalidates current policy/reference/prompt IDs,
questions,actions,seeds,stops,decoded text,rewards and behavior logps. It consumes
the same sampler seed cursor and recomputes canonical logps before each update.
It is replay of matched on-policy windows, not stale arbitrary off-policy RL.
Wrong-policy/tampered data aborts. On resuming offline mode,keep the exact file,
recipe,software and topology; checkpoint state restores the window cursor/RNG.

Watch correctness and completion separately from shaping. Inspect question/gold/
text,answer frequencies,constant-output collapse,zero-variance groups,truncation,
KL/gradients and per-window reward. A scalar centered GRPO loss near zero can
have nonzero gradient. Rising training reward with flat independent accuracy,
longer traces,format-only reward or verifier-control failures is a stop signal.
Do not promote a checkpoint without the existing fixed-budget evaluation gates.

Send `result.json`,resolved YAML,manifests,metrics JSONL,verifier audit and a small
reviewed sample audit,plus GPU/CPU/software details. Retain checkpoints locally;
do not send credentials,private corpora or unreviewed personal text.

## Expected quality and optional formal verification

The19.34M tiny and125.85M small architectures have no demonstrated general math
gain here. Prior broader200-case studies rejected every candidate; DPO heldout
loss can worsen even when its training objective improves. SFT can teach format
and short familiar arithmetic; LoRA saves optimizer state,not missing pretrained
knowledge. GRPO redistributes probability among solutions the policy can sample;
all-wrong groups supply no correctness preference. Reliable multi-step competition
math gains are not justified by a two-answer fixture or a few thousand tokens.
No numerical forecast is defensible without actual pretraining quality,licensed
data and independent evaluations across seeds at the intended model size.

Lean is [design only](../docs/research/lean_verifier_design.md): pinned toolchain,
isolated compiler,statement/axiom auditing,licenses and realistic scale limits.
No installed Lean checker or theorem-proving score is claimed.
