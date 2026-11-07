# Responsible release and first real run

A framework candidate and a trained-model release are separate decisions. A
framework package provides code and bounded engineering evidence. It does not
clear data,weights,safety or a public model claim. Current debug models are not
eligible for quality/safety promotion.

## Artifact-specific checklist

| Gate | Required evidence | Current disposition |
| --- | --- | --- |
| Framework code/license | Apache LICENSE/NOTICE,dependency/vendor notices,clean source | Reviewable;candidate build and CI evidence required |
| Data rights | Exact sources/revisions/licenses/usage policies and redistribution rights | Original fixtures approved;production mix remains separate |
| Benchmark coverage | Every required licensed snapshot and official scorer/prompt | MATH-500,AMC/AIME,HellaSwag and full HumanEval remain blocked |
| Contamination | Actual train shards for every benchmark and parent lineage | Native reports cover declared sources;unknown history/semantic overlap unverified |
| Quality | Independent matched-setting evaluation and readable failure analysis | Current toy reasoning/retrieval is inadequate |
| Safety | Bilingual risk/benign contrast evaluation and independent human review | Design fixture only;trained refusal behavior unverified |
| Runtime security | Current advisories,loader/secrets checks and deployment boundaries | Native profile checked;optional harness SQLite risk retained |
| Hardware | Actual intended driver/precision/DDP/FSDP/kernel/resource tests | CPU evidence;new CUDA13.0 profile not accepted on a server |
| Publication | Filled cards,provenance,checksums,notes and operator action | No tag,model upload or public release performed |

Read the [data card](reference/DATA_CARD.md),[model card template](reference/MODEL_CARD.md),
[results ledger](reference/RESULTS.md) and [security review](security.md).
Evaluation-only approval grants no training permission. The ARC development
source's CC-BY-SA terms require artifact-specific attribution/change/share-alike
review;this is not permission to redistribute every derived model. Blocked or
ambiguous rights remain blocking rather than inferred from repository popularity.

## Contamination evidence

Use the native evaluator's required audit artifact,including actual pretraining
and SFT sources. Multiple audit-source manifests are order/hash-bound to the eval
policy. Retain counts,coverage,source identities and truncation diagnostics for
each benchmark. A complete lexical report cannot certify semantic cleanliness
or unknown parent checkpoints. Keep the prospective final holdout unscored until
the declared final protocol;never use it for selecting hyperparameters.

## Small safety SFT implementation

`qvelt.posttrain.safety` supplies24 original EN/RU examples across12 families,
balanced refusal/helpfulness before family splitting. Topics include unauthorized
access,secret extraction,harassment,self-harm support,weapon/fraud requests,and
benign account recovery,password protection,phishing education,privacy redaction,
math and basic chemistry. Refusal responses contain no operational harmful details.

The builder records rights/design data and uses the existing real SFT preparation
pipeline. Translations share a family and split;whole chats preserve boundaries.
Tests verify that supervised labels are exactly the assistant response plus end-
turn,that families are disjoint and that no example was truncated. The tiny
tutorial briefly optimizes this fixture as a plumbing test. This is not a claim
of trained safety,coverage of a threat model or independent human review.

## Content-safety evaluation plan

Before a real release,freeze an independent bilingual set with safe/unsafe pairs,
paraphrases and multi-turn contexts. Keep all safety-eval families out of training.
Review direct harmful compliance,privacy/credential leakage,targeted harassment,
self-harm support,unsafe code/action suggestions and over-refusal on benign math,
security education and user-owned tasks. Include prompt injection/control tokens,
truncation and long-context variants within a documented risk budget.

Human reviewers should judge concrete harmful assistance and useful alternatives,
not just refusal keywords. Use at least two independent reviewers for material
categories,resolve disagreements and record per-language/category denominators,
uncertainty and examples. Do not request graphic or operationally harmful details
just to make an automated score larger. No such human evaluation has been executed
here;no pass threshold is invented for the current fixture model.

Stop promotion on answer collapse,unverifiable traces,unsafe compliance,unexplained
over-refusal,leakage,missing audit coverage or invalid contexts. Separate sampled
training reward from correctness,format,length and readable audits.

## Recommended order for the first real training run

1. Reproduce the [tiny tutorial](tutorial.md),including restart and exported serving,
   from the exact candidate commit/lock on your machine.
2. Inventory RAM/SSD/GPU/driver;run actual device numerics,distributed recovery and
   representative-context memory profiles for the selected security-updated runtime.
3. Approve exact production data rights and benchmark exclusions;curate/deduplicate,
   inspect samples and train the tokenizer only on training data. Keep full lineage.
4. Verify vocab/context/masks/loader replay and run fixed held-out domain probes.
5. Execute the [small200-step server pilot](reference/runbooks/02_pretraining.md).
   Inspect loss,gradient norms,NaN/spike/cursor recovery,throughput,memory and samples.
6. Freeze the accepted recipe and continue its full horizon. Do not repair a failed
   pilot by hiding tests,using eval text as training or silently changing resume inputs.
7. Evaluate the pretrained checkpoint independently before SFT/preferences/RL.
   Re-run data/safety/quality gates for every promoted lineage,then choose a release.
