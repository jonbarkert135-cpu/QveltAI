# Original versioned evaluation

## Current scope

Updated 2026-11-08 against code at334b4c6. The CPU reference supports explicit
qvelt-eval-v1/v2 protocols under schema1, native exact/numeric scoring, choice
likelihood, rolling LM perplexity, pass@k and declared-source overlap audits.
The optional [lm-eval adapter](harness.py) implements actual local-task parity;
the [benchmark registry](../../../configs/eval/benchmarks.yaml) records bounded
GSM8K, MMLU abstract-algebra and ARC evaluation imports and blocked sources.
These subsets are not official full benchmark scores; approvals remain scoped
in [DATA_CARD](../../../DATA_CARD.md#current-inventory-summary).
The suite also provides diagnostics, checkpoint regression and provenance
receipts; v2 explicitly permits HQQ/RoPE variants. See
[configuration](config.py), [runner](runner.py) and
[the evaluation runbook](../../../runbooks/04_evaluation.md).
GPU/sharded evaluation, full arbitrary-Python execution and independently
measured model-quality acceptance remain unverified. The prospective final
holdout remains unscored; this documentation task runs no new benchmark.

Matched-seed studies add strict parent/data/config checks,paired family deltas,
whitespace/general/verifier regression controls,one-factor DPO LR ablation and
fresh-process CPU resource measurements. See
[promotion runbook](../../../runbooks/08_model_promotion.md). Necessary gates
reject regressions or permit further validation;no automatic production release.

See [runbook07](../../../runbooks/07_evaluation.md),
[design](../../../docs/development/evaluation_design.md) and
[acceptance](../../../docs/development/validation_evaluation.md).

## Archived initial scope

Superseded on 2026-11-08 by [Current scope](#current-scope); retained verbatim:

> CPU FP32 read-only decoder evaluation:qvelt-eval-v1 configuration,licensed raw
> task records,exact/numeric answer scoring,choice likelihood,rolling LM perplexity,
> seeded sampled pass@k and declared-source lexical overlap audits. The prospective
> 1000-case snapshot has200 validation/800 unscored final cases. No public benchmark,
> GPU/sharded evaluator,arbitrary-code verifier or model promotion claim.
