# Bounded original outcome-GRPO prototype

Scope:original exact-answer fixtures,one CPU process,FP32,decoder below1M parameters.
This is a correctness/recovery prototype,not a server/production reasoning-RL recipe.
No teacher,reward model,model-text execution,tools,paid compute or GPU is required.
DPO failed its generation control;the prototype starts from fresh SFT instead.

## Reproduce

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
make setup
mkdir -p runs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/grpo_smoke.py --directory runs/grpo-demo --repository . --steps 8
```

Use a new output directory. The script first builds the original base/SFT fixture
and80-update SFT reference,then executes8 GRPO windows and stop4/resume8. It checks
real nonzero policy-gradient updates and changed weights,plus exact losses,
rollouts,model,Adam,RNG and application cursor. Reduced127104-parameter V445/T128
model,seed20261105,4 prompts/window,4 candidates/prompt,max16 generated tokens,
LR1e-6,clip0.2,KL0.01,gradient clip1,weight decay0. No server feasibility is implied.

A separate branch from the generated frozen reference:

```bash
uv run --frozen --extra cpu qvelt grpo runs/grpo-demo/grpo.yaml \
  --directory runs/grpo-branch --repository . --stop-after 4
uv run --frozen --extra cpu qvelt grpo runs/grpo-demo/grpo.yaml \
  --directory runs/grpo-branch --repository . --resume latest
```

Both the SFT recipe bytes and SFT checkpoint are pinned. The source SFT recipe
supplies model/tokenizer/data lineage;its optimizer,LR,schedule,batch,seed and step
settings do not configure the new RL optimizer. GRPO's resolved job explicitly uses
one CPU thread,constant configured LR,no decay/step warmup,clip1 and rollout batch
prompts*groups. Parent execution backend/recomputation are retained;compile is off.
The supported environment is the fixed registered seed20261104 EN generator,
filtered through the original SFT training families. SFT validation families only
are monitored;its test families are not scored or used for optimization.
This is not an adapter for arbitrary external SFT corpora or environments.

## What is optimized

Sample from the current policy at temperature1;do not force gold,filter unsuccessful
candidates or reuse stale multi-epoch batches. Other temperatures are unsupported
because behavior/policy logp must represent the same distribution. All rollouts
finish before updating weights;the recorded policy byte digest is checked unchanged.
The reference stays frozen. Every sampled action,including end-turn/outerEOS,is
included in RL loss;prompt and padding are masked. This differs intentionally from
SFT/DPO outerEOS supervision masking.

Group rewards are mean-centered/population-std normalized,with a1e-6 floor and
exact zero advantage for constant groups. The clipped surrogate and sampled KL
penalty are averaged per completion,then per group/prompt. Old/reference logps and
advantages are detached. Numeric overflow/nonfinite state fails;it is not clamped
into an apparently successful run. There is one optimizer update per fresh batch.

At ratio1,group advantages average to0,so the initial scalar surrogate is near0
while its policy gradient can be nonzero. Do not expect a CE-style falling RL loss.
Check gradient norms,changed weights,actual policy_updates,reward distribution,
zero-variance groups,format/completeness,truncation and fixed validation controls.
All-zero gradients skip Adam entirely;KL-only updates are counted separately from
windows with a policy-reward signal. No fake optimizer step is used to warm state.

## Verifier and recovery limits

Only a complete end-turn6 response can score1. Premature outerEOS2,truncation,
empty text,reasoning prose,decimal/NaN/inf/expressions/code and malformed answers
score0. Parse bounded ASCII integer/p/q exactly with Fraction,zero tolerance;
at most130 characters and60 digits per integer. No model text reaches eval,
sympify,an interpreter or shell. The trusted bounded parser has a1s post-call wall
budget;this is not preemptible sandboxing for an arbitrary replacement verifier.
A future tool/proof verifier needs a separately reviewed isolated execution contract.

Checkpoint only after a complete rollout/update window. Resume binds recipe,
parent/reference/data/tokenizer/source/software and restores model,Adam,RNG,prompt
cursor,actual updates/skips and last rollout digest. Preserve parent/reference
artifacts. Strict latest resume fails on incompatible/corrupt state. This prototype
does not add pretraining's SIGTERM handler,automatic rollback or distributed support;
a hard interruption replays from a valid existing boundary. Retain raw rollouts and
metrics for diagnosis;they contain original questions/answers and checkpoint hashes.

Actual8-window local check:8 policy updates,745 generated action targets,11/32
zero-variance groups;all loss/rollout/state replay exact. Fixed8-problem validation
stays1 correct before/after. This is not a reasoning improvement,public benchmark
score or justification for full-scale model promotion. See validation_grpo.md.
