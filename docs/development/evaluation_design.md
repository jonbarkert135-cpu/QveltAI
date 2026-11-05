# M7.1 versioned evaluation design

Date:2026-11-05. Base6bcefbb. Read-only original Qvelt CPU FP32 harness;reuse
manifest/checksum/tokenizer/decoder/chat/response-logp helpers. No lm-eval,judge API,
external corpus,model-generated code execution or paid/GPU run. Ponytail and pinned
Orchestra evaluation/custom-task guidance inform review,not vendor benchmark scores.

## Contracts and acceptance

Freeze the RFC's1000 project-authored exact-answer cases now:200 validation and800
final,new prompt-template families with disjoint validation/final parameter bands.
Record seed,source/version,license,IDs,raw/manifest checksums before any new learning.
This happens after historical M1–M6 checkpoints;no retroactive pretraining-heldout
claim. Family labels distinguish templates/parameter bands,not semantic novelty.
Only validation controls are evaluated in acceptance;final access requires an
explicit future protocol. Freeze metadata/IDs in Git,raw questions/answers outside.

A strict YAML job binds one checkpoint,tokenizer,evaluation dataset and optional
raw training-audit manifests by actual SHA256. Version prompt/answer/metric/LM
window protocols,context/token/sample limits,sampler temperature,seeds,tolerances
and subset size. Unsupported GPU/sharded models/public benchmark adapters fail;
local registered examples never become official benchmark results by naming them.

Dataset types:exact integer/rational math,bounded decimal numeric tolerance,
choice-continuation likelihood controls,and plain-document LM loss/perplexity.
Strict whole-answer or single final-answer-line extraction;no CAS/eval of output.
Context overflow,nonfinite logits,invalid answers,truncation and premature stops
are explicit failures,never removed from generation accuracy denominators. Unknown
runtime/artifact errors abort and leave no committed successful report.

Greedy pass@1 uses one attempt. Categorical samples use isolated per-case/attempt
CPU generators,temperature-bound distribution and explicit n/k;no greedy duplicates
or plug-in biased pass@k. Report all attempts,first-attempt pass@1,success counts,
exact combination estimator1-C(n-c,k)/C(n,k) and uncertainty with declared assumptions.
Wilson95% bounds are descriptive conditional iid problem intervals;family clustering
invalidates population-level interpretation. Perplexity is exp(total NLL/targets),
with disjoint rolling target windows,one previous token at resets,BOS first/EOS last;
no corpus wrapping,repetition or per-document unweighted averaging.

Audit normalized exact containment,contiguous math-aware n-gram overlap and lexical
shingle Jaccard against explicitly declared training source bundles. Record source
IDs,family overlaps,counts and bounded coverage;unknown training history/semantic
leakage remains unknown. Never label a run clean solely because declared lexical
checks return0. Do not filter the primary evaluation set using audit hits.

Publish write-once predictions/metrics/resolved recipe/checkpoint+tokenizer+dataset
identities/software/source provenance under a committed report manifest. Preserve
input checkpoints;never load their Adam/pickle payloads. Test metrics against exact
subset enumeration/high-precision Wilson references,numeric edge/adversarial cases,
naive LM/choice references,corruption/license/final guards and actual repeatable tiny
CPU generation with matching predictions and metrics. Observe clean CI and update
state/decisions/experiments/runbook. Public/final model promotion remains separate.
