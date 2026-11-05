# M6.2 original preference optimization acceptance

Date:2026-11-05. Baseae3dee7 plus original preference source;
source digestc3c18aa16e68110c5cb7e4b9750773a942d822984d537307e87f9b5f666dda08.
[Design](preferences_design.md),ADR0043 and[operator runbook](../../runbooks/05_preferences.md)
define the supported single-process CPU FP32/demo scope.

## Implemented and verified

Strict typed YAML jobs bind pair/tokenizer/SFT-reference identities,beta,job and
source/software;relocatable paths are excluded from recipe identity. Raw preparation
requires explicit approval,training policy and source license;prepared loads recheck
permission and artifact bytes. Duplicate IDs/prompts,identical candidates,mismatched
prefixes,malformed frames,partial end-turns and context overflow fail. Whole families
split before optimization. Payloads are bounded;ordinary pretraining checkpoints
cannot be mislabeled as SFT references. No reference cache,production data,GPU,
distributed or compiled DPO is silently emulated.

Response log probabilities SUM content/end-turn,masking user prefix,role markers,
PAD and outer EOS. Independent NumPy probability/analytic-gradient tests cover masks,
sequence-length effects and policy gradients;frozen reference receives no gradient.
Naive unpadded model scoring agrees with padded batching. SymPy confirms the scalar
loss derivative under finite real margins and positive beta. The exact same pinned
SFT weights initialize policy and frozen eval reference;separate tests verify reference
weights remain unchanged throughout optimization. Reused AdamW grouping,step schedule
and atomic checkpoint engine preserve model,Adam,RNG,pair cursor and target accounting.

## Actual40-update experiment

Original D-PREF-01 approved before ingestion:32 EN generated exact-answer problems,
gold preferred versus exact gold+1 rejected,both checked with the bounded verifier.
DPO families:12 train/16 validation/4 unused test pairs. SFT/base family exposure is
not excluded across stages;these are not decontaminated public benchmarks.

Python3.12.14,Torch2.9.1+cpu,FP32,seed20261031,one thread. Reduced127104-parameter
V445/T128 decoder,B2 pairs/update,beta0.1,LR0.001,weight decay0. Fresh3-update
engineering base then30-update SFT initialize the reference. DPO consumes80 pairs/
1582 response targets. Stop13/resume40 matches all40 losses and final weights,
Adam,RNG and consumer/application state exactly. Final-source receipt is
[evidence/preferences-40.json](evidence/preferences-40.json).

| Fixed split metric | Initial | Final |
| --- | ---: | ---: |
| Train mean DPO loss | 0.69314718 | 0.35221379 |
| Validation mean DPO loss | 0.69314718 | 0.84319099 |
| Train reference-relative positive-margin fraction | 0(all ties) | 0.916667 |
| Validation reference-relative positive-margin fraction | 0(all ties) | 0.375 |

This is fixture learning/replay,not improved generation or mathematical reasoning.
Heldout preference loss worsens;retain the negative finding and do not promote
this checkpoint. Pair positive-margin fractions are not sampled win rates or
measured KL. Tiny overfit LR differs deliberately from the RFC5e-6 small-model
hypothesis and does not tune that server recipe.

## Checks and reproduction

Focused23 tests passed in9.11s. Full CPU+optional-SDK regression614 passed,no skips,
7 GPU deselected in158.37s;no tests disabled or weakened. Ruff/format,strict mypy128
files and wheel/sdist build pass. The earlier local distributed faults recorded for
Phase5 remain historical observations;this green run does not prove their root
cause is fixed. Current-source3b3ef4a GitHub42073726045 has all10 jobs successful.
CPU3.11/3.12/3.13 each612 passed,2 optional SDK skips,7 GPU deselections;
SDKs are independently verified. Dedicated40-update DPO and200-update hard-kill
replay jobs both pass with exact within-host state/loss comparisons. CI DPO train
0.35152350/heldout0.84594192 differ slightly from local0.35221379/0.84319099;
no cross-host bitwise identity is promised.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/preference_smoke.py --directory runs/preferences-demo --repository . --steps 40
```

First preparation failed on a missing required data artifact role and was fixed.
Three adversarial helper failures were exclusive commit-marker replacement in
copied fixtures;helpers unlink only their copied marker before republishing a
correctly checksummed hostile bundle. Runtime validation was retained. Final suite
checks strict permissions,replay/source/reference changes,CLI updates,malformed
pairs,sequence boundaries,reference freezing and independent math references.

## Remaining scope

Two resident models and extra reference forward passes cost memory/compute;no
laptop/server feasibility result is implied. Reference logp caching,multiturn/tools,
GPU/distributed preference training and production/human preference data require
separate acceptance. Length sensitivity and simplistic verifier negatives remain.
Reasoning-RL promotion needs generation/verifier validation,not just a lower train
DPO loss. Paid/final-scale runs remain outside this engineering check.


The subsequent source-unchanged generation control is reproducible through
scripts/preference_generation_control.py;16 validation-only problems,B1,greedy,
16 new tokens,complete end-turn required. Base0/SFT3/DPO0 correct. No DPO promotion;
see reasoning_rl_readiness.md for all completions and input/weight identities.
The diagnostic script has verifier/context/identity self-checks and strict output
exclusivity;Ruff/format and mypy129 files pass after adding it.
