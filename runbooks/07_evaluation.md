# Versioned original evaluation

Scope:qvelt-eval-v1 is a single-process CPU FP32 reference harness for Qvelt
unsharded FP32 checkpoints,at most20M parameters. It evaluates original licensed
fixtures. CUDA,sharded/distributed evaluation,public benchmark adapters,judges,
proof/code execution and production model promotion are outside this acceptance.

## Reproduce the engineering acceptance

From the checkout with its frozen CPU environment:

```bash
mkdir -p runs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/evaluation_smoke.py --directory runs/evaluation-01 --repository .
uv run --frozen --extra cpu qvelt manifest verify runs/evaluation-01/validation-a/manifest.json
```

Use a new directory. The script regenerates and verifies the committed freeze,
fits a byte-BPE tokenizer only on the older approved original training fixture,
saves a seeded **untrained** real decoder checkpoint,and evaluates the200
validation cases plus7 independent controls. Greedy and categorical(n3,k1/2,T1)
controls and greedy validation each run twice. Assertions compare predictions,
metrics and audits byte for byte and verify untouched input weights. Audit
calibration deliberately includes a known collision in an artificial source which
was never trained. This proves the detector sees that collision,not that the model
memorized it. No optimizer update and no final800-case scoring occur.

`result.json` binds report manifest identities. Each report commits
`evaluation.yaml`,resolved model/runtime config,lock,predictions,metrics,audit,
checkpoint/tokenizer/dataset/audit-source manifests and actual software/source
provenance. Timestamps/run IDs differ across reports;predictions/metrics/audits
must match on the same installed source/device/software. No cross-host or
cross-Torch bitwise guarantee is made. Checkpoint optimizer/RNG files are hashed
for integrity but never deserialized by evaluation.

## Evaluate an existing local checkpoint

Register source revision,license and permitted evaluation use in DATA_CARD before
ingestion. Required data contract:raw Qvelt bundle with license plus
`usage.json` allowing evaluation,aligned identifier/family metadata,and one JSON
object per document with exactly these keys:

```json
{"schema_version":1,"identifier":"example-01","kind":"math","domain":"math","family":"original-family","partition":"validation","prompt":"Return one half as a fraction.","target":"1/2","choices":[]}
```

Kinds:math exact integer/rational,numeric bounded decimal/rational,choice exact
target option with2..8 distinct choices,lm plain prompt text with empty target and
choices. Partitions:validation,control,final. Final is rejected unless a future
recorded evaluation protocol explicitly sets allow_final. Current acceptance
leaves all800 final cases unscored. Questions/answers and raw corpora stay outside
Git;the prospective1000-case metadata registry is
`docs/development/evidence/evaluation-frozen-v1.json`,frozen in88db69e. New original
families/bands are not proof of semantic independence from historical training.
Raw bundles retain the deterministic fixture export epoch in created_at;the
registry's freeze_date2026-11-05 and freeze commit establish the actual freeze,
not a backdated preregistration claim.

Set actual local paths,then derive the canonical manifest identities rather than
copying raw-file hashes into the recipe:

```bash
export QVELT_EVAL_CP=/absolute/run/checkpoints/step-00000080/manifest.json
export QVELT_EVAL_TOKENIZER=/absolute/data/tokenizer/manifest.json
export QVELT_EVAL_DATA=/absolute/evaluation-01/heldout/validation/manifest.json
uv run --frozen --extra cpu python - <<'PY'
import os
from pathlib import Path
from qvelt.eval.config import ArtifactRef, EvalJob, dump_eval
from qvelt.utils.artifacts import ArtifactStore
from qvelt.utils.manifests import manifest_digest

def ref(variable):
    path = Path(os.environ[variable]).resolve(strict=True)
    manifest = ArtifactStore(path.parent).read_manifest(path.name)
    return ArtifactRef(str(path), manifest_digest(manifest))

job = EvalJob(ref('QVELT_EVAL_CP'), ref('QVELT_EVAL_TOKENIZER'),
              ref('QVELT_EVAL_DATA'), approved_for_evaluation=True,
              context_limit=128, lm_window=64, max_new_tokens=16, max_cases=200)
with Path('evaluation.yaml').open('x') as stream:
    stream.write(dump_eval(job))
PY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu qvelt evaluate \
  evaluation.yaml --directory runs/evaluation-existing-01 --repository .
uv run --frozen --extra cpu qvelt manifest verify runs/evaluation-existing-01/manifest.json
```

Review the resolved YAML before running. Pin optional training_sources to actual
registered raw manifests allowing both training and decontamination;the job
defaults to an empty declared audit set and explicitly unknown history. To audit
SFT/preference raw sources,the harness extracts message contents/prompt fields;
other records are scanned as text. Auditing a raw-source superset is conservative
lexical coverage,not evidence every document trained. Path relocation preserves
semantic recipe identity;changing hashes,budgets or sampler changes it.

## Protocol and metric definitions

- chat-v1 prompt: `[BOS=1,USER=4,encode(prompt),END_TURN=6,ASSISTANT=5]`.
  plain-v1 prompt:`[BOS=1,encode(prompt)]`. No added prose,few-shot examples or
  silent truncation. Literal control spellings encode as ordinary text;unexpected
  control IDs from ordinary encoding are rejected. Choice options are encoded
  separately as explicit continuations;leading spaces are meaningful.
- Complete generation requires END_TURN6(chat) or EOS2(plain). Other controls,
  nonfinite logits,context overflow,token/8192-character limits are logged.
  Invalid/failed/truncated outputs remain in the primary denominator. Counts
  overlap:one attempt can be both invalid and truncated. Full generated IDs,
  stop reason,seed,text,extracted answer and correctness are retained.
- whole-v1 requires one complete integer/rational(math),or bounded decimal/
  exponent(numeric). final-line-v1 requires exactly one `Final answer:` marker
  on the last nonempty line. No generated expression runs in Python or SymPy.
  Exact math uses Fraction equality;numeric uses
  `abs(pred-gold)<=atol+rtol*abs(gold)`,default1e-8/1e-6. Parser bounds130
  characters,60 mantissa digits,decimal scale magnitude100.
- Greedy:n1,T0,k1. Categorical:n<=32,0<T<=2,independent per-case/attempt
  CPU generators derived from version,seed,case ID and attempt. Sampled pass@k
  is the per-problem mean of `1-C(n-c,k)/C(n,k)`,n>=k,with all failures in n;
  first-attempt pass@1 is separately named. It is not `1-(1-c/n)^k`.
- Wilson95 intervals describe first-attempt/choice counts under an iid-problem
  assumption. These templated problems are clustered;the bounds do not establish
  population confidence or commercial/public benchmark parity. No unsupported
  uncertainty interval for sampled pass@k is invented. Reports list family counts.
- Choice scores sum conditional token log-probabilities and also divide by option
  token length. No end-turn is scored. Exact ties choose the first option.
  Different length normalization can change rankings;report both counts.
- LM uses plain documents regardless of chat format. Targets are document tokens
  followed by EOS2,each counted once. Disjoint windows of min(lm_window,effective
  context) use BOS1 first and the previous target token on later resets. No corpus
  wrap or overlapping target duplication. This is a specified bounded-context
  PPL,not full-history PPL. `PPL=exp(sum(NLL)/sum(targets))`;FP32 token log-softmax/
  window sums and Python-float window accumulation. A failed LM document makes
  aggregate PPL null. Overall and web/code/math per-domain counts/losses are logged.

The default engineering budget16 is deliberately below RFC002's proposed512-token
quality protocol;the CPU prototype caps generation128/context2048 and records the
effective model context. Changing these requires a new recipe/comparison,not
relabeling a score. Choice controls and three short LM documents are sanity
fixtures,not MMLU or representative production domains.

## Overlap interpretation and logs

NFC/case-folded math-aware tokens preserve digits and operators. Audit flags
exact normalized containment,shared contiguous n-grams(default13),5-token
shingle Jaccard(default0.85) and declared family matches. Per-source checksums,
document counts and prefix coverage are logged. Documents above8192 normalized
tokens are prefix-scanned and counted as truncated. Sources are bounded64MiB/
10000 documents,max16 per recipe. These are lexical alerts,not semantic proof;
unknown lineage remains unknown even with zero hits. Audit never removes cases
from the main score. Duplicate or malformed evaluation IDs,denied rights,bad
hashes,wrong tokenizer/config and unexpected runtime errors abort publication.

For analysis,attach the report's metrics,audit,provenance,resolved recipe and
manifest. Predictions include protected prompts/answers;share them only when
intended and keep final holdouts private. Do not attach optimizer/checkpoint
weights,credentials or unrelated raw data. Future production training must
register all relevant benchmark exclusions before learning;this runtime audit
alone does not enforce that upstream gate.

References:[verified] Chen et al.2021,arXiv2107.03374v2 Eq1/appendixA
(https://arxiv.org/html/2107.03374),and NIST e-Handbook PRC7.2.4.1 Wilson score
interval(https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
These sources justify formulas,not Qvelt model scores. Independent executable
subset enumeration,high-precision score-equation roots and scalar teacher-forcing
tests verify this implementation;see tests/unit/test_evaluation* and integration.
