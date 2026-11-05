# Original versioned evaluation

CPU FP32 read-only decoder evaluation:qvelt-eval-v1 configuration,licensed raw
task records,exact/numeric answer scoring,choice likelihood,rolling LM perplexity,
seeded sampled pass@k and declared-source lexical overlap audits. The prospective
1000-case snapshot has200 validation/800 unscored final cases. No public benchmark,
GPU/sharded evaluator,arbitrary-code verifier or model promotion claim.

Matched-seed studies add strict parent/data/config checks,paired family deltas,
whitespace/general/verifier regression controls,one-factor DPO LR ablation and
fresh-process CPU resource measurements. See
[promotion runbook](../../../runbooks/08_model_promotion.md). Necessary gates
reject regressions or permit further validation;no automatic production release.

See [runbook07](../../../runbooks/07_evaluation.md),
[design](../../../docs/development/evaluation_design.md) and
[acceptance](../../../docs/development/validation_evaluation.md).
