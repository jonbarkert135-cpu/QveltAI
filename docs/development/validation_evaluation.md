# M7.1 original evaluation acceptance

Date:2026-11-05. Prospective freeze88db69e follows base6bcefbb;original CPU
reference harness,not a public benchmark integration. See
[design](evaluation_design.md),ADR0045 and[runbook07](../../runbooks/07_evaluation.md).

## Corpus and reference contracts

D-EVAL-01 was registered before generation;1000 unique original questions/IDs,
200 validation and800 final,seed20261105,five independent stdlib/SymPy gold
checks. Validation/final template-band family labels are disjoint. Frozen registry
includes generator/license/partition payload+manifest hashes and all IDs,not raw
prompts/answers. Freeze happened after historical models learned;prospective
means future training only. New templates are not proof of semantic novelty.
Final800 are generated/exported for identity checks but never scored in acceptance.

Strict versioned jobs bind actual checkpoint/tokenizer/evaluation/audit hashes,
prompt/extraction/sampler protocols,seeds,temperature,budgets and tolerances. Own
CPU full-prefix generation,choice likelihood and rolling token-weighted LM PPL
reuse existing verified Qvelt helpers. No extra dependency,teacher,judge,generated
code/CAS execution or checkpoint Adam/pickle deserialization. Unsupported modes
fail;reports publish their manifest only after successful evaluation/provenance.

Independent references execute exhaustive k-subset enumeration for n1..7,
high-precision SymPy roots of the Wilson score equation,exact Fraction/Decimal
edge cases,uniform log(vocabulary) losses and scalar Python teacher-forcing
log-sum-exp on nonuniform real decoder outputs. Tests cover stop/truncation/
nonfinite/context failures,unchanged input hashes,global RNG/thread restoration,
per-case seed independence,denied rights,final guard,corruption,tokenizer mismatch,
literal control safety and deliberate exact/near lexical overlap. No test drops
failed cases from its primary denominator. Wilson bounds are descriptive under
iid problems;family clustering prevents population confidence claims.

## Actual engineering and historical diagnostic runs

Self-contained script creates a real untrained BPE decoder checkpoint,using only
older approved original training text for tokenizer fitting. It evaluates200
validation cases and7 controls(web/code choices,math/numeric,three LM domains).
Greedy and categorical n3/k1,2/T1 controls plus greedy validation repeat twice.
Predictions,metrics and audit must be byte-exact;input weights stay unchanged.
Artificial overlap calibration is not a source actually trained by the model.
Untrained scores,short-control PPL and choice counts are engineering evidence only.

Read-only historical80-update SFT diagnostic:127104 parameters,V445/T128,
CPU FP32,B1,greedy chat-v1,whole-v1,max16 actions,all200 validation problems.
5/200 correct,all200 complete,0 invalid/truncated/context failures,400 generated
actions. **One distinct answer across all200 problems**;all5 correct are remainder
cases. This exposes a degenerate constant-answer policy,not reasoning improvement.
Two evaluations give identical predictions/metrics/audits;input weights unchanged.
Checkpoint abce0cee3c274003912aae25d641ff32e866661ab7cc87c9576199524d7b31d2,
tokenizer47d18d7614373b9d5a49c91b7f9cb690eb0e1adbf9fb2735095dcdbbc0b46ece.

Audit declares three registered raw-source supersets:42 EN,42 RU and64 SFT
documents. Exact/13-gram/Jaccard>=0.85/family alerts are0,no prefix truncation.
Unknown history and semantic leakage remain unverified;zero alerts are not a
cleanliness certificate. No case is filtered using those results. No final800,
public dataset,GPU,distributed evaluator,production score or promotion is claimed.

## Validation status

Focused evaluation42 tests pass in4.51s before final packaging checks. Ruff/format,
strict mypy146 files and wheel/sdist build pass. Full local CPU/optional-SDK
regression:665 passed,1 packed-training DDP resume timeout,7 GPU deselected in
281.85s. No skips and no green local-full claim. The timed-out torchrun worker
was cleaned up after its parent failed. Lock93 resolved/pip72 compatible.
Current-source acceptance receipts and clean GitHub matrix follow below.
The previous intermittent managed Gloo issue remains an environment risk until
separately diagnosed;do not skip/weaken distributed tests to hide it.

Final executable source83a1adb,package digest
0f742e0d9c94a3165f9fc2205d894a9a58a0e2c0a659a5d149e2fa2ac09209fd.
Clean-checkout self-contained acceptance produces identical predictions/metrics/
audits for each paired run,and untouched weights;see
[local receipt](evidence/evaluation-smoke.json). Untrained math0/200 is expected
and not hidden. The3 LM controls have89 actual targets,total NLL531.28185654,
PPL391.29394781;this is tokenizer/model/short-corpus-specific,not general quality.
Sampled controls have6 attempts,24 actions,5 token-limit stops/1 premature control;
all6 invalid remain in pass@1/pass@2 denominators. Greedy choice sum score2/2,
token-normalized1/2 illustrates length effects on these two tiny controls only.

Historical SFT diagnostic was repeated on that clean source and matches its
earlier results exactly;see[diagnostic receipt](evidence/evaluation-sft-diagnostic.json).
GitHub42092305795's dedicated evaluation job also succeeds on this digest with
all paired comparisons exact,weights unchanged and0 final cases scored;see
[remote receipt](evidence/evaluation-ci-smoke.json). This is separate per-host
repeatability,not a cross-hardware promise. Complete matrix acceptance follows.

GitHub[42092305795](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42092305795)
has **all12 jobs successful** on83a1adb. CPU3.11/3.12/3.13 each664 passed,
2 optional SDK skips,7 GPU deselected in216.17s/240.53s/167.01s. SDK event/offline
tests pass in their separate job. Full RFC tiny200-step real-SIGKILL recovery
matches the complete loss curve(max delta0),weights,Adam,RNG and application
state after kill89/restore75. SFT,DPO and GRPO replay jobs also succeed. See
[matrix/recovery receipt](evidence/evaluation-ci.json). No test is skipped or
weakened to conceal the local packed-DDP timeout;that runtime issue remains open.
M7.1 closes for its original CPU reference scope. Next:M7.2 controlled comparisons
and promotion gates,retaining negative findings and requiring actual quality data.
