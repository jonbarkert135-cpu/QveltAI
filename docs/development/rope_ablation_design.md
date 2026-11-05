# Exploratory single-factor RoPE ablation

Date:2026-11-05. Source6b0d4cc,package9ca8a8fd993965196dbaaaca1f1d8b4f22a13339644b3c73f784fa7ad5601195.
Preregister this additional architecture comparison before its learning. Earlier
validation scores are already exposed;this is explicitly exploratory,not an
independent confirmatory test or a threshold chosen from RoPE results.

Compare the registered base3->SFT80 chain with an otherwise identical chain
using model.rope_scale2 instead of1. Use existing shared data/tokenizer and the
same three runtime seeds20261105/6/7. Runtime seeds affect initialization and
sampler ordering;within each paired arm both are identical. Assert initial model
weights are bit-exact before learning,and source/model/data/optimizer/schedule/
seed settings match after removing only rope_scale and derived SFT parent identity.
Verify equal processed targets at both stage boundaries. No DPO/GRPO change.

Reuse all247 cases,greedy16/context128/LM64,existing adversarial/paired/gate
helpers and isolated CPU resource protocol. Reuse thresholds from89a0350;
record per-seed/family deltas,failures and unchanged/negative findings. All800
final cases remain unscored;no public corpus/new dependency or GPU run.

Implement orchestration as a versioned standalone script so the already frozen
model/trainer/evaluator package remains unchanged. Bind script SHA and Git code
identity separately from executable-package SHA. Commit script before final
learning. Preserve full training/evaluation artifacts,write the final manifest
last and publish only summary receipts. This short-context interpolation ablation
cannot validate long-context extension or justify production promotion.
