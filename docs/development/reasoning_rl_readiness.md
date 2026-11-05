# M6.3 model-promotion readiness control

Date:2026-11-05. Predeclared bounded engineering control following M6.2. The
training RFC requires a useful SFT checkpoint and heldout verifier checks before
a reasoning-RL pilot. A low fixture training DPO loss alone does not satisfy this.
This control is independent of future objective/rollout software implementation.

Compare the actual engineering base,30-update SFT and40-update DPO checkpoints
from R019 on exactly the same16 original DPO-validation problems. No final-test
families or public benchmarks are used. The base/SFT can have seen the problem
families earlier;there is no project-final contamination clearance or new benchmark.
Pin all checkpoint/weights/tokenizer/model/source identities in the receipt.

Greedy B1 CPU FP32 completion uses the exact single-turn user/assistant prefix,
16 new tokens,context128,raw full-prefix decoder evaluation,no sampling/tools or
prompt changes per model. End-turn6 is the required complete-response marker;
premature outerEOS2,empty/invalid/non-integer-or-fraction/truncated answers fail.
The existing bounded exact-fraction verifier scores the generated answer,never
executes model text,and has independent unit tests and exact gold checks.

Necessary (not sufficient) readiness:at least one complete correct answer at this
fixed budget and no harness/scorer error. Record every invalid/truncated result and
retain negative evidence. Passing this tiny necessary control would still not
establish useful general reasoning or authorize expensive training. A failure
blocks model promotion to a reasoning-RL pilot;portable algorithm/reference work
can proceed separately without a quality claim.

## Executed control

Use the source-bound R019 run and a new receipt:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/preference_generation_control.py --run runs/preferences-demo \
  --output runs/preferences-demo/generation-control.json
```

Actual reduced fixture result:base0/16 correct,0 complete;SFT3/16 correct,
8 complete;DPO0/16 correct,0 complete. Base/DPO each exhaust16-token budgets on
all16 problems;SFT exhausts8. All completions and hashes are retained in
[evidence/reasoning-readiness.json](evidence/reasoning-readiness.json).
The necessary tiny control passes for SFT only. Keep SFT as the candidate for a
bounded engineering RL check;do not promote DPO or claim useful general reasoning.
Reference/tokenizer/model identities are verified and receipts are exclusive.
