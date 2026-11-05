# M7.2 controlled comparison and promotion design

Date:2026-11-05. Base605ddee. Reuse the original CPU qvelt-eval-v1 harness and
existing training engines. Ponytail/full and pinned Orchestra evaluation guide
inform review;no vendor benchmark runtime,new dependency,teacher or paid/GPU job.

## Study contract

Preregister before learning:three model-initialization seeds20261105/6/7,one
shared licensed older original corpus/tokenizer,SFT supervision and preference
pairs. Per seed:base3 updates,SFT80,DPO40 beta0.1/LR0.001,GRPO8(LR1e-6,T1,P4/G4,
max16),plus a single-factor DPO LR0.0005 arm. All post-training arms use the same
SFT checkpoint for that seed. The two DPO arms differ only in LR;models/data/order/
update count/batch/seed/reference/beta agree. Stage compute differs by design;
report actual target/action counts and optimizer windows,not compute parity.

Freeze200 validation cases from88db69e,40 whitespace robustness variants(two
variants of four cases per each of five families) and7 existing controls. Keep
all800 final cases unscored. Variants retain parent gold/family and are correlated
with parents;do not inflate an independent sample count. Register original source
and eval-only permission in DATA_CARD before generating. Standard inference is
greedy chat-v1/whole-v1,max16,context128,LM window64,B1 CPU FP32,same evaluation
seed. Exactly the same dataset/tokenizer/model shape/budget applies to every arm.

A strict YAML study binds the evaluation recipe bytes and named seed/checkpoint
manifests. Require base/SFT/DPO/GRPO for each seed;optional DPO half-LR arm.
Preflight full run/checkpoint lineage,matching seed/tokenizer/config,data identities
and actual parent references. Do not silently compare historical DPO/SFT30 to
GRPO/SFT80. Fail before any report is committed on malformed/mismatched evidence.
Reports retain evaluation manifests,training budget ledgers,paired case/family
deltas and resource measurements. Never deserialize checkpoint pickle for analysis.

## Necessary gates,not automatic promotion

Compare DPO/GRPO/half-LR to SFT on matched seeds. Preregister necessary checks:
macro-family exact accuracy improves at least0.01 on every initialization seed,
no primary family declines,no increase in invalid/truncated/failed generations,
robust accuracy/invariance do not regress,web/code choice sum/token-mean counts do
not decline,and each web/code/math LM PPL is finite and <=1.05 times SFT.
Execute bounded verifier controls including malformed/multiple/unit/NaN/hostile
text and partial answers. Those are parser controls,not arbitrary code isolation.

Measure fixed CPU inference in a fresh process per checkpoint:prefix8,16 forced
greedy forward steps,no early stop,2 warmups/5 timed repetitions,eager full-prefix,
one thread. Record all timings/median,tokens/s,weights bytes and Linux process
peak RSS(import/load/model included,not isolated tensor allocation). Necessary
budget:peak RSS<=2GiB,median<=4x paired SFT;these are smoke limits,not a server SLA.
Do not compare timing across machines;no wall-clock reproducibility promise.

At least two distinct initialized base models,200 primary cases/five families,
40 robustness controls,three LM domains and web/code choices are needed even for
an `eligible_for_extended_validation` result. Missing coverage gives inconclusive;
measured regressions give reject. All passing necessary checks only permit more
validation,never production promotion. Public/source-clean final evidence,actual
user-run GPU/small checkpoints and resource acceptance remain open requirements.
Report per-seed/family deltas and range,not an iid significance claim over
templated/paired cases. Keep negative and identical findings. Selection using this
validation set cannot turn it into an untouched final test.

Acceptance:independent count/delta/gate references and adversarial config/lineage
tests;actual shared-lineage multi-seed CPU study,one-factor LR ablation,immutable
artifact checks and honest regressions;frozen dependency/lint/types/build and clean
current-source CI. Publish only summarized receipts,not protected prompts/answers.
