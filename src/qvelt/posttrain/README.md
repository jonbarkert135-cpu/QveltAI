# posttrain

Expanded Phase6 adds chat-v2 explicit reasoning,unquantized CPU LoRA in the packed
SFT loop,bounded numeric/polynomial verifiers,own-checkpoint rejection selection,
immutable original GSM8K train ingress and default-off external-trace terms gates.
GRPO now uses Phase8 GenerationEngine with KV caching,format/length components,
human-readable audits and checksum-bound current-policy offline replay. See
runbooks/03_posttraining.md. QLoRA/Lean/GPU/arbitrary-problem RL are not implemented
or validated by these extensions. The two-answer mechanism control is not a
general math benchmark or production promotion.

M6.1 implements strict version1 chat formatting,family-isolated licensed demo
preparation,assistant-only masks and whole-dialog packing. The packed trainer
supports pinned weights-only SFT initialization and atomic resume.

M6.2 adds single-turn licensed demo preferences,original sigmoid DPO,response SUM
log probabilities,a frozen SFT reference and strict atomic model/Adam/RNG/cursor
replay on single-process CPU FP32. Use qvelt dpo with actual generated YAML or
scripts/preference_smoke.py. No reference cache or GPU/distributed DPO is claimed.

See runbooks/04_supervised_finetuning.md and runbooks/05_preferences.md. Fixture
loss/implicit pair scores do not establish public instruction/math quality;the
DPO heldout negative finding does not authorize model promotion or reasoning RL.

M6.3 adds the bounded original exact-answer CPU outcome-GRPO prototype:temperature1
rollouts,group-normalized/clipped objective,sampled KL,frozen SFT lineage and exact
rollout/model/Adam/RNG/cursor replay. See runbooks/06_reasoning_rl.md. The verifier
never executes model text;no tool/proof/GPU/distributed or reasoning gain is claimed.
