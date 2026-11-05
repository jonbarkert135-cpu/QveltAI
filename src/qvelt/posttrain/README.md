# posttrain

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
