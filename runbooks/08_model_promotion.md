# Matched model comparisons and necessary promotion gates

This is a bounded original CPU study,not a production release command. Design
and thresholds were preregistered in89a0350 before M7.2 learning;see
[design](../docs/development/promotion_design.md) and ADR0046. Use the pinned CPU
environment;no new dependency,external corpus,judge or model-generated code runs.

## Reproduce the three-initialization study

```bash
mkdir -p runs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/promotion_study.py --directory runs/promotion-01 --repository .
uv run --frozen --extra cpu qvelt manifest verify runs/promotion-01/analysis/manifest.json
```

Choose a new directory. Default initialization seeds20261105/6/7 share exactly one
registered older original corpus,byte-BPE tokenizer,SFT dataset and preference
pairs. For each seed,the script trains base3 updates then SFT80;DPO40,GRPO8 and
DPO40 at half LR branch from that exact SFT checkpoint. DPO beta0.1,LR0.001 or
0.0005,B2,WD0. GRPO LR1e-6,KL0.01,clip0.2,P4/G4,T1,max16. SFT LR0.003. Model
shape is the existing reduced127104-parameter,V445/context128 engineering fixture.
This is not the RFC19.34M model or a full small-model run.

`--seeds 20261105 20261106` runs the explicit two-seed CI subset. At most three
distinct integer seeds are accepted. An optimizer RNG seed on a fixed deterministic
model is not counted as independent initialization:actual base weights must differ.
`prospective-plan.json` is written before tokenizer fitting/model updates,with
suite identity,seeds and stage budgets. The committed design fixes thresholds;
do not adjust them after seeing this validation result and claim preregistration.

The two DPO arms differ only in LR,including identical seed,parent,pairs,beta,
update count and processed target count. Across stages,data/objectives and compute
budgets intentionally differ;the ledger reports this rather than implying equal
FLOPs. Processed targets/actions omit reference/attention/backward cost;they are
not measured training FLOPs. GRPO optimizer boundaries can include skipped
windows;actual counters are retained. No exact training replay is newly claimed
by this study;existing trainer replay acceptance remains a separate gate.

## Repeat analysis without training

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu qvelt compare \
  runs/promotion-01/training/study.yaml --directory runs/promotion-analysis-02 --repository .
uv run --frozen --extra cpu qvelt manifest verify runs/promotion-analysis-02/manifest.json
```

Strict study YAML binds evaluation recipe **file bytes** and named checkpoint
canonical manifests by SHA256. Each entry has role,initialization seed and
`checkpoint:{path,sha256}`. Roles base/sft/dpo/grpo are required for every seed;
dpo-half-lr is optional but must be present for all selected seeds. Preserve the
full run/checkpoint layout:the comparator verifies the parent run manifest and
application recipe,not only model weights. A moved standalone checkpoint lacks
that study lineage. Updating paths requires regenerating recipe byte hashes.
Do not edit/overwrite existing manifests or bypass integrity checks.

The evaluation recipe is a protocol template;its checkpoint is replaced by each
explicit study entry. Tokenizer,case order/IDs/golds,prompts,inference seed,budgets,
scorers,audit sources and model shape stay fixed. Unsupported sampled/final/capped
protocols,wrong seed/parent/kind,changed pairs/tokenizer/data or additional LR-arm
differences fail preflight. All learning remains CPU FP32 with original demo data.
Unexpected errors abort;only the top-level manifest commits a completed analysis.
Partial directories/checkpoints are preserved after failures;rerun analysis to a
new directory rather than silently deleting/reusing outputs.

## Read results

The suite has200 frozen validation questions,five40-case families;40 whitespace
controls(two variants of four questions per family) and7 earlier controls. The
variants preserve parent gold/family and use leading spaces or a trailing newline.
They are40 transformations of20 existing questions,not40 independent problems.
Final800 are never scored or selected for this study. No public benchmark name
is attached to these examples. Raw prompts/answers stay outside Git.

`result.json` includes:

- Per-role/seed evaluation manifest and metrics,actual training recipe settings,
  checkpoint/parent/data/tokenizer/source identities,optimizer boundary,processed
  targets/actions and available skip/update counters.
- SFT versus base,each candidate versus SFT,and half-LR versus ordinary DPO;
  matched problem wins/losses,per-family counts and macro-family accuracy delta.
  Macro delta averages family accuracy differences equally;it is not a weighted
  micro average when family counts differ. Initialization-seed min/max are
  descriptive;no significance/power or population confidence claim is made.
- Robust correctness and answer invariance to the parent. Invariance requires
  complete valid numeric outputs and Fraction-equal answers;stable wrong answers
  can be invariant,so invariance alone never demonstrates quality.
- Short web/code choice scores(sum and token-mean),three domain LM PPL probes,
  bounded positive/adversarial verifier controls and actual inference resources.

Necessary checks are frozen in qvelt-promotion-v1:macro-family exact accuracy
gain>=0.01 on **every** initialization seed,no primary family decline,no increased
invalid/truncated/failed count,no robust correctness/invariance decline,no web/code
choice count decline,and finite per-domain LM PPL<=1.05x SFT. Resource limits are
CPU whole-process peak RSS<=2GiB and median fixed-workload duration<=4x SFT.
These are necessary engineering controls,not statistically established release
margins. Any measured failure yields reject;missing coverage yields inconclusive
unless an observed regression already warrants rejection. At least two distinct
base initializations,200 primary/five families,40 robustness cases,three LM domains
and web/code choices are required for eligible_for_extended_validation. Missing
gate evidence is an error,never a pass. Lower training loss cannot override them.

Passing all checks permits further validation only. `automatic_promotion:false`
is deliberate:actual user-operated small/GPU checkpoints,licensed independent
quality/contamination evidence and an untouched final protocol remain required.
Do not deploy/promote a model because this original-fixture report exists.

## Resource and artifact interpretation

Each model is measured in a fresh Linux process,CPU FP32/one thread,eager
full-prefix. Workload:fixed8-token prefix,16 greedy forward steps ignoring stop
controls,2 warmups and5 timed repetitions. Record every duration,median and16/
median tokens/sec. This fixed-action workload controls output-length differences;
it is not end-to-end generation latency,KV-cache performance or a serving SLA.
Linux ru_maxrss includes imports,checkpoint loading and model;it is not incremental
tensor memory. Child startup/load are excluded from the timed inner loop but
included in whole-process RSS. Timing changes with load/hardware;do not expect
bitwise timings or compare thresholds across machines. No CUDA/MFU/VRAM is measured.

The top manifest commits child evaluation reports,resource JSON,recipe/lock/config
and summarized results. Verification hashes all registered files;optimizer/RNG
pickle is never deserialized by analysis. Protected predictions remain under each
child report. Share summarized result,manifest and relevant recipes/provenance
after reviewing paths;keep raw final cases,predictions,weights and credentials
private. Negative/unchanged findings must remain in EXPERIMENTS. General controls
are too small for real language-quality assurance;zero lexical audit hits do not
certify complete training-history or semantic cleanliness.
