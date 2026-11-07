# Evaluation and provenance

Native evaluators score bounded math,numeric,choice,LM and integer-function code
controls without external services. The optional standard harness adapter is thin;
its persistent SQLite cache is outside the supported safe workflow.

Metrics include failure counts,descriptive confidence intervals,unbiased pass@k,
majority vote and reasoning/completion lengths. Compare only matched prompts,
shots,budgets,scorers and settings. Original controls and limited public prefixes
are not official benchmark scores. Standard benchmark rights and metric differences
are recorded explicitly;blocked data is not silently replaced by a mirror.

Every published number belongs to a verified native receipt and generated
[RESULTS](../reference/RESULTS.md),with config/checkpoint/tokenizer/source identities
and the reproduction command. Per-checkpoint regression is fixed and plotted.
Release contamination audits cover declared actual train shards,not unknown history.

Read the [evaluation runbook](../reference/runbooks/04_evaluation.md). The prospective
final holdout remains unscored;do not use it for tutorial tuning or safety-data design.
