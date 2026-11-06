# Qvelt current state

Updated:2026-11-06. Owner:Lead agent.

## Environment readiness (2026-11-04)

Managed CPU development is restored: Python3.12.14/uv0.12.23/Torch2.9.1+cpu,
frozen optional metrics SDKs, writable Make caches and installed Git hooks.
Authenticated GitHub fetch/push dry-run works with explicit network permission;
local main tracks origin/main. CPU/optional-SDK suite:524 passed,5 GPU deselected,
no skips in94.75s. Lint/format,mypy97 files,lock/dependency checks,hook checks and
wheel/sdist build pass. Numeric100-step and packed BPE20-step exact replay smokes
pass. See docs/development/validation_environment.md and its receipts. Direct uv/
Git-hook invocations need the README cache exports in read-only-home environments;
network operations/local distributed sockets need execution network permission.
M6.1 CPU acceptance is recorded below; no GPU or paid/final-scale run was performed.

## Current task

Expanded Phase7 portable evaluation accepted onacced9071be0896b2fa6e915c7ed9d1b0f73e9d9;
all17 CI42303284852 jobs successful. CPU3.11/3.12/3.13 each823 passed,6 optional
skips,7 GPU deselected;actual optional harness11 tests and metrics SDKs separately
verified. Final focused71 pass20.68s;additional preference-shard audit covered by
8-test ledger/source set4.28s. Lint/format,mypy195,lock/dependencies and build pass.
Native v2 metrics/majority/failure witnesses,actual lm-eval0.4.13 bridge,original
EN/RU/code controls,pinned bounded GSM8K/MMLU/ARC imports,train-only actual-shard
audits,per-checkpoint regression/repair/plots and verified generated RESULTS.
Clean source17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3:
123904-parameter debug model,three training updates,stop2/resume3 model/Adam/RNG
and curve values exact;sampled predictions/metrics/audits byte-equal,weights unchanged.
All three input runs and nine native report receipts dirty=false. Negative bounded
zero-shot prefixes:GSM8K0/16 (4 context overflows),MMLU algebra4/16 (5 failed
contexts),ARC raw2/16,normalized3/16 (5 failed contexts). Original generated math/
code all fail;regression0/0/0. No official full benchmark score or model promotion.
Nine receipts,acceptance and curve JSON/PNG committed;root RESULTS generated/checks.
MATH-500/AMC-AIME/HellaSwag/full HumanEval block release artifact coverage;rights,
access,LaTeX metric/full Python sandbox remain explicit gates. Final800 unscored.
Pre-final-ARC local full822 passed/2 two-rank120s timeouts/7 deselected505.11s;
no suppression. Initial f741151 CI caught missing runs/ parent;corrected with nested
parent/exclusive-output test. Remote green does not diagnose historical Gloo faults.
See runbooks/04_evaluation.md,validation_evaluation_phase7.md and source CI receipt.
Next dependency-ready implementation:M8.2 measured serving;large/GPU/DDP evaluation
and production quality/release gates remain separate.

Previous expanded Phase6 acceptance follows.



Expanded Phase6 portable extensions accepted on8e9b44c;all15 CI42253772192 jobs
successful. CPU3.11/3.12/3.13 each798 passed,2 optional SDK skips,7 GPU deselected;
SDKs verified separately. Final focused88 passed in47.29s;lint,mypy179/build pass.
Chat-v2 reasoning masks,shared full/LoRA SFT,bounded exact/numeric/SymPy polynomial
verifiers,own-model rejection/difficulty pass@k,default-off teacher/terms ingress,
pinned original GSM8K train adapter,native-engine GRPO/offline replay and Lean design.
Clean source28addb6ad1641aebd047a16931b4177db276022609a39eae9e63ca5e273a3bc3:
SFT20/DPO4/RL60 on127104params;training reward .3125->.709375,validation15/64->26/64.
Quality rejects:all59 parseable post-RL numeric outputs0,answer1 remains0/32;
26/64 is below32/64 always-zero baseline. DPO heldout loss worsens. Rejection12
of64 candidates (46 correct before dedup),all selected gold0;not reasoning traces
or a promotion. LoRA20 rank4/alpha8 stop6/resume20 exact weights/Adam/RNG/cursor,
9728 trainable parameters,base127104 unchanged. Actual16-row GSM8K train import,
no optimization on it. Native offline all60 losses/model/Adam/RNG exact. All five
control stage manifests clean on8e9b44c. Final800 unscored. See runbook03,
validation_posttraining_phase6.md and eight posttraining evidence receipts.
Pre-selector local full798 passed/2 historical distributed failures/7 deselected;
prior CI3.13 also had a numeric-DDP timeout. Remote success does not diagnose/fix
Gloo. No tests suppressed. GRPO stays original-environment CPU<1M;QLoRA/GPU/
distributed/arbitrary-problem RL,Lean execution and production source/model gates
remain open. Next dependency-ready implementation:M8.2 measured serving.

Historical M8.1 acceptance follows.

M8.1 native CPU generation/export is accepted on699ff0b;GitHub42239279899 has
all14 jobs successful. CPU3.11/3.12/3.13 each735 passed,2 optional SDK skips,
7 GPU deselected;SDKs separately verified. True left-padded batches,private KV/RNG,greedy/seeded sampling,stream
snapshots,stop/context/output/cache limits and cooperative cancellation/timeouts.
Native safetensors/tokenizer/config/source snapshots reload without original
directories;no optimizer/RNG pickle. Developmental123904-parameter V395 export
has logit delta0 and matching cached/uncached/CLI/stream outputs. Pre-final full
local733 passed,no skips,7 GPU deselected in236.44s;final guards focused53 pass
in19.80s,Ruff/format,mypy163 and build pass. Existing Gloo issue is not declared
fixed by this one green local run. Clean source
2adb1ec750a1e03d4f859597edd7e4d344d7a8b948fdd648e76cdaa143b77bba passes the full
standalone workflow. Historical19338560-parameter RFC tiny export/reload delta0,
cache/full max delta2.861e-6,unchanged weights and matching fresh CLI generation.
Extracted-wheel import/generation matches on the same locked dependencies.
Current-source200-step actual-SIGKILL and all training/evaluation replays pass CI.
See validation_inference.md/runbook09. No model promotion/GPU/HTTP/quantization.
Next:M8.2 measured serving interface;quantization/external runtime gates stay open.

Historical M7.2 acceptance follows.

M7.2 matched-lineage CPU study and necessary promotion controls are implemented.
Preregistered89a0350:shared training/tokenizer/SFT/pairs,initialization seeds
20261105/6/7,base3/SFT80/DPO40/GRPO8 and a DPO half-LR arm. New strict compare
CLI verifies actual parents,seeds,data/config/source and one-factor ablation;
paired family/robustness/general/verifier/resource gates never auto-promote.
Focused35 checks pass including actual one-seed training/analysis and fresh-process
CPU measurements. First one-seed diagnostic rejects all candidates;DPO regresses
5/200 to0/200. Full local688 passed,1 existing packed-DDP timeout,7 GPU deselected
in311.24s;Ruff/format,mypy155 files and existing build pass. Three-seed actual
results:base0/0/0,SFT5/1/0,DPO0/0/3,half-LR5/0/0,GRPO5/1/0 of200 each;all
candidates reject. Repeated15-model analysis gives exact predictions/metrics/
audits/decisions;inputs unchanged. Additional controlled RoPE1/2 ablation gives
SFT5/1/0 versus1/0/0,also reject. Initial weights/data order/target budgets match
within paired arms. Runtime seed couples initialization and sampler across runs.
Package9ca8a8fd993965196dbaaaca1f1d8b4f22a13339644b3c73f784fa7ad5601195;
sourcee2c628f GitHub42100151962 has all13 CI jobs successful. CPU3.11/3.12/3.13
each689 passed,2 optional SDK skips,7 GPU deselected;SDKs independently verified.
Matched2-seed/RoPE2-seed studies and200-step actual-SIGKILL/SFT/DPO/GRPO replays
all pass. Local packed-DDP timeout is not fixed by remote success. Production
acceptance still needs actual user-run small/GPU and independent licensed evidence.
Next independent implementation:M8.1 inference/export. See validation_promotion.md
and runbook08 for the necessary gates and observed seed/RSS limits.

Historical M7.1 acceptance follows.

M7.1 original CPU evaluation is accepted on source83a1adb;GitHub42092305795 has
all12 jobs successful. CPU3.11/3.12/3.13 each664 passed,2 optional SDK skips,
7 GPU deselected;SDKs separately verified. Versioned exact/numeric/choice/LM protocols,seeded sampling/pass@k,
manifest-bound reports and declared-source lexical audits. Prospective1000-case
freeze88db69e:200 validation/800 unscored final,seed20261105,SymPy gold verification.
Focused42 tests pass;Ruff/format,mypy146 files and build pass. Full local665 passed,
1 existing packed-DDP resume timeout,7 GPU deselected in281.85s. Historical SFT80
scores5/200 with one constant answer,not a reasoning gain;repeat outputs/metrics/
audits exact,weights unchanged. See validation_evaluation.md,runbook07 and ADR0045.
Paired evaluation outputs/metrics/audits exact on both local and CI hosts;weights
unchanged,0 final cases scored. Dedicated200-step actual-SIGKILL and SFT/DPO/GRPO
replays also pass. Source0f742e0d9c94a3165f9fc2205d894a9a58a0e2c0a659a5d149e2fa2ac09209fd.
Local Gloo timeout remains unresolved;remote success is not a local fix.
Next:M7.2 controlled comparisons/promotion and ablation gates;demo checkpoints
cannot justify production promotion. Public rights/GPU/user-server gates remain open.

Historical M6.3 acceptance follows.


M6.3 bounded CPU outcome-GRPO is implemented from the SFT branch. Actual8 real
policy-gradient/Adam updates,745 generated actions,11/32 constant groups;stop4/
resume8 all losses,rollouts,weights,Adam,RNG and prompt cursor exact. Reduced
127104-parameter model,V445/T128,P4/G4,max16 actions,temperature1,LR1e-6,KL0.01.
Validation1/8 before/after:engineering acceptance,not reasoning improvement.
Final-source combined DPO/GRPO33 tests pass in22.23s;Ruff/format,mypy135 files pass.
Pre-final full622 passed,1 numeric-DDP timeout,7 GPU deselected in258.63s;root cause
of the intermittent local Gloo problem remains unresolved. Current-source38a74bf GitHub42081751910:all11 jobs successful. CPU3.11/3.12/
3.13 each622 passed,2 optional SDK skips,7 GPU deselected;SDKs verified separately.
Dedicated GRPO,DPO,SFT and200-step SIGKILL replays pass. Source4775afbb264baa8c842cf40b67e0372162ee4e690f1c3738e227db42e7136b1b.
See validation_grpo.md and runbooks/06_reasoning_rl.md. Next:M7.1 evaluation harness,
with original fixture scope first;production/public benchmark rights still blocked.

Historical M6.2 acceptance follows.


M6.2 original CPU DPO is implemented after accepted Phase5. Full CPU/optional-SDK
614 passed,no skips,7 GPU deselected in158.37s;focused23 passed;Ruff/format,mypy128
files and build pass. Actual40 updates/80 pairs/1582 targets,127104 parameters,
V445/T128,beta0.1,LR0.001;stop13/resume40 all losses/weights/Adam/RNG/cursor exact.
Train DPO0.693147->0.352214,validation0.693147->0.843191:negative heldout result,
not a quality promotion. Frozen reference is byte-verified SFT and checked unchanged.
Source c3c18aa16e68110c5cb7e4b9750773a942d822984d537307e87f9b5f666dda08;
Current-source3b3ef4a GitHub42073726045:all10 jobs successful. CPU3.11/3.12/3.13
each612 passed,2 optional SDK skips,7 GPU deselections;SDKs separately green.
Both40-step DPO and200-step SIGKILL replays are exact independently on CI.
See validation_preferences.md and runbooks/05_preferences.md.
M6.3 readiness control is executed:base0/16,SFT3/16,DPO0/16 complete correct
answers at identical16-token greedy budget on original validation-only problems.
Use SFT for a future bounded engineering RL check;DPO is rejected for promotion.
The control is a necessary tiny check,not proof of general reasoning.
Reproduction:reasoning_rl_readiness.md and scripts/preference_generation_control.py.
Hardware/source gates remain separate;the subsequent GRPO prototype is recorded above.

Historical Phase5 acceptance follows.


Expanded Phase5 is implemented selectively on top of SFT. Actual full RFC tiny
19,338,560 parameters,V32768,T16/B2:200 optimizer updates/6386 valid targets.
SIGKILL89,restore75;all200 losses differ by0.0;weights/Adam/RNG/application match
bit-exactly. Final source f14b266,digest
ba3bb6d89d5aa74dbea7a80d3a1f43e1fdfab309ac8a0f04d0c635859de45e50.
Local full run586 passed,3 distributed-process failures/timeouts,7 GPU deselected
in365.00s;do not call it green. Final200-step repeat,compile and four-size measurements pass. Ruff/format,
mypy122 files,frozen lock/dependencies and build pass. GitHub42069685890 atf14b266:all9 jobs successful,CPU3.11/3.12/3.13 each
589 passed,2 optional SDK skips,7 GPU deselected;SDKs separately green. Dedicated
200-update hard-kill replay independently bit-exact. Final local restart run590
passed,1 packed-DDP timeout,7 GPU deselected in318.63s;Gloo root cause still open.
Portable Phase5 closed;M6.2 was the next milestone and is now accepted above.
See validation_phase5.md and runbooks/02_pretraining.md. No GPU/full small/paid run.

Historical Phase4 acceptance follows.


Selective Phase4 closure is implemented on top of M6.1:RoPE scale(default1),FP64
reference numerics,whole-model gradcheck,random-length scaled cache,compile/depth/
overfit/RFC-count tests and operator profiling.22 focused tests passed in63.36s;
final profiler subset3 passed in3.36s;Ruff/format,mypy105 files and build pass.
Default FP32/BF16 logits/gradients/weights match pre-import seeded values exactly.
Final local regression:562 passed,1 existing numeric-DDP exit SIGABRT,5 GPU
deselected in104.85s. Packed replay/model/SFT tests pass;root cause of the local
DDP process-exit fault remains unresolved,also observed before this import.
Corrected-source GitHub41909315717 atfee1e3d has all8 jobs successful:
CPU3.11/3.12/3.13 each561 passed,2 optional SDK skips,5 GPU deselections;optional
metrics separately3 passed. Numeric/data/packed/SFT smokes are green. CPU3.11's
initial missing setuptools was fixed by pinning already-locked84.2.1 in dev;bare
local3.11.17 Inductor without SDKs passes in44.92s. No dependency versions/hashes
upgraded. M4.3 closes for verified CPU scope;GPU remains unverified.
Do not call the local full suite green. See validation_phase4_closure.md,phase4_import_review.md and
new docs/model.md. Supplied old state/profile claims were not imported.


M6.1 at0abfcd3 is complete for original demo data/single-process CPU: strict chat formatting,
assistant-only masks,whole-dialog packing/truncation,licensed family-heldout data,
pinned base weights and atomic SFT replay. Full CPU/optional-SDK suite:543 passed,
5 GPU deselected,no skips in97.32s;Ruff/format,mypy103 files,build/lock/dependency
and hook checks pass. Fixed69-target train probe5.89794546->0.14195556;
16-target heldout probe5.81094980->4.76135039;80 updates/2520 assistant targets.
Stop26/resume80 weights/Adam/RNG/consumer match exactly. Final test execution uses
GLOO_SOCKET_IFNAME=lo,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 after earlier distributed
process-exit stalls;root cause of those stalls is not established. See R016,
docs/development/validation_sft.md and runbooks/04_supervised_finetuning.md.
M6.1 remote GitHub41905361820 at0abfcd3 has all8 jobs successful.
M6.2 portable implementation is recorded above. No reasoning-accuracy claim,
production corpus,external teacher,paid run or GPU/distributed-SFT validation.


M4.1/available-CPU M4.2 and M5.1 are complete. Implementation557e4da; actual
GitHub run41290979913 has all7 jobs successful, including CPU3.11/3.12/3.13 and
real packed BPE optimization/replay. M5.2 CPU/DDP portion is verified; CUDA/FSDP/
shared-store/multi-node tiers remain blocked by hardware validation. Production
sources/benchmarks remain blocked in M3.3. Publication to main is active. No final
model/paid training or old cloud-environment configuration change was performed.

## Last completed work and evidence

- Portable RFC decoder:all four actual meta counts,tiny CPU backward,independent
  value/gradient/causality/mask oracles,seeded residual initialization and combined
  tied-embedding gradients. Constructors explicitly initialize FP32 parameters
  even under ambient float64. Non-finite losses stop before optimizer/checkpoint.
- Reference/SDPA runtime and non-reentrant block recomputation; compact explicit
  request-local KV-cache with packed/multi-token/full parity,stale-state checks,
  weak ownership and CPU BF16 coverage. CPU graph/storage profile recorded.
- Actual packed-data trainer with token-sum accumulation/partial windows,valid-token
  LR schedule,heldout probe,clipping/throughput and actual data/BPE canonical IDs.
  Atomic checkpoint stores model,Adam,RNG and each rank's committed loader state.
  Split-bound fingerprints and worker/prefetch/mount-safe replay are tested.
- Actual two-rank Gloo uneven-count gradient/global-batch reference,coordinated
  rank-local failure and SDPA+checkpoint optimizer/cursor/RNG replay. Numeric
  engineering trainer and CPU DCP/DTensor checks remain intact.
- Fresh clone:522 tests passed,2 optional SDK skips,5 GPU deselections in122.79s;
  Ruff/format and mypy97 files pass;frozen dependency check and wheel/sdist build
  succeed. Remote3.11/3.12/3.13 each522 passed;optional SDK job independently green.
- Clean actual-BPE smoke:127104 parameters,20 updates,40 microbatches,4959 valid
  targets,heldout256-target LM probe6.1267035->5.7565165. Stop6/resume20 has exact
  weights/Adam/consumer/RNG state. This is original demo language modeling,not a
  public benchmark or mathematical-answer-quality result.
- Existing tokenizer/curation/shards/weighted loading/exact math generators remain
  verified. No external training corpus/teacher;GSM8K is exclusion/evaluation only.

Evidence:docs/development/validation_pretraining.md and its committed receipts;
EXPERIMENTS R012–R014,runbooks/01_data_preparation.md,02_model_execution.md and
03_packed_pretraining.md. Earlier foundation/data evidence remains separate.

## Open limits

Five supplied GPU tests did not run. No CUDA/NCCL,packed FSDP,real laptop/shared-
store/multi-node validation or final-scale memory/throughput measurement. The
functional cache copies appended state;graph-storage bytes are not allocator peak.
Exact replay is limited to the documented source/software/topology/hardware scope;
unsafe .data/cache mutation and arbitrary uploaded optimizer files are unsupported.

Full checksum preflight requires input I/O budget. HF BPE pair statistics can grow
with corpus diversity. LID/PII/toxicity/LSH/lexical exclusion need real-source audits;
no trained quality classifier. MATH/AIME/MMLU and frozen project-final exclusions
are incomplete,so production preparation remains blocked. The heldout LM probe is
weighted/limited and may reuse source blocks. Reasoning RL,public evaluation and
serving/export beyond safetensors are still future milestones.

## Next three tasks

1. M7.1:build the versioned evaluation harness on original fixtures first;preserve
   fixed budgets,invalid/truncated counts and benchmark/source clearance gates.
2. M5.2–M5.3:actual CUDA/DDP/FSDP/shared-store/multi-node and resource validation
   when hardware is supplied;no costly/final-scale run without authorization.
3. M3.3:clear exact training/benchmark rights,freeze the distinct project final
   heldout and calibrate corpus-scale quality/dedup/PII/semantic-leakage audits.

## Reproduce

```bash
make setup
make lint
make typecheck
make test
mkdir -p runs
uv run --frozen --extra cpu python scripts/pretraining_smoke.py --directory runs/bpe-smoke-new --repository . --steps 20
```

Use a new output path,uv0.12.23,Linux/Python3.11–3.13. CPU is default. Operator
commands/actual-data identities/schedule/replay limitations are in runbook03.
