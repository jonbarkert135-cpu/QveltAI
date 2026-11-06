# Expanded Phase7 evaluation acceptance

Date:2026-11-06. Scope:offline CPU correctness,standard harness interoperability,
source rights,report integrity and checkpoint regression. No final-scale model,
quality promotion,GPU performance or official full public benchmark score.

## Implementation and verified sources

See ADR0049,DATA_CARD D-EVAL-PHASE7-01 and runbooks/04_evaluation.md.
The optional lm-eval0.4.13 bridge executes the actual LM API/ConfigurableTask
path;it is not an API-shaped mock. Native math/choice/LM/code scoring remains
usable without that dependency. Original EN/RU controls are authored;the code
track interprets a bounded integer-expression subset without executing Python.
Benchmark/source imports verify publisher payload/license pins and preserve
full notices. MMLU/GSM8K imports are limited first16 zero-shot development rows;
MATH/competition/HellaSwag/full HumanEval gates remain explicitly blocked.

Native v2 preserves v1 digest/frozen holdout behavior. Metrics include exact and
numeric accuracy,Wilson95 intervals,unbiased pass@k,first-prefix plurality@k with
tie diagnostics,exact explicit-reasoning token spans and conservative failure
witnesses. Actual training-shard frames are decoded train-only after checksums,
usage policies,tokenizer and full boundary/coverage checks. Audits make no claim
about unknown prior history or semantic cleanliness.

Every supported training checkpoint can run a pinned fixed regression recipe.
The shared save hook publishes the checkpoint before evaluation;missing reports
are repaired at resume or with eval-repair. The dedicated interruption test fails
after step2 checkpoint publication,then proves repaired reports and exact final
weights,Adam and RNG versus both uninterrupted and evaluation-disabled runs.
Plots bind each point to its checkpoint/report hashes.

Verified portable receipts exclude predictions/raw external text/model weights.
RESULTS.md is deterministically generated from them with configs,hashes and argv;
tampered metrics,source,inputs,commands and settings fail verification. Release
artifact coverage fails when any registered benchmark is missing. The README
contains no numeric measured-result prose;CI checks both inventory and ledger.

## Executed diagnostics before source publication

The mixed suite ran with actual original BPE data,three optimizer updates and
step2 stop/resume. Model/Adam/RNG and curve values match exactly. Repeated seeded
sampled predictions,metrics and audits match byte-for-byte;weights remain unchanged.
Actual harness raw/character-normalized original choice metrics match native.

The diagnostic result is negative:all original generated math/code answers fail,
GSM8K0/16 (four context overflows),MMLU abstract algebra4/16 (five failed contexts).
Small context256 and completion8,limitedzero-shot prefixes and essentially untrained
weights preclude a useful quality interpretation. These numbers are not paper-
comparable scores. No tuning/selection on them;final800 untouched.

Initial focused evaluator/reference/regression tests:66 passed in22.20s.
Added CLI/source-boundary coverage:8 passed in4.08s. Ruff/format pass;mypy and
full CPU acceptance are finalized below. Wheel/sdist build and frozen lock pass;
installed dependency consistency passes using writable UV_CACHE_DIR. An initial
unscoped uv pip check hit the read-only home cache;rerunning with the documented
cache environment passed without package changes.

Source-clean evidence and remote CI closure are recorded after the implementation
commit;diagnostic runs have dirty=true and are not represented as clean acceptance.

Final local focused set after ARC integration:70 passed in14.08s,including
hand-built MMLU/ARC conversions,delimiter-excluded normalization and actual harness
parity. The ARC diagnostic yields raw2/16,normalized3/16 with5 failed contexts;
this is another limited negative development control,not a full official score.
Full local CPU run before the final ARC-only extension:822 passed,2 failures,
7 GPU deselected in505.11s. Both failures are unmodified two-rank integration
tests:NaN recovery and packed pretraining replay,timeouts120s. No tests suppressed.
After results were recorded,two confirmed orphan workers were sent SIGTERM;this
cleanup did not alter outcomes or artifacts. Remote all-source CI remains required;
remote success cannot itself diagnose the historical intermittent Gloo issue.
Ruff/format,mypy195 files,lock consistency,installed dependency check and build pass.
The harmless read-only-home fontconfig warning appears during actual successful
PNG generation;plot metadata and image hashes are verified.

Initial source f741151 passed a clean local standalone repeat,all input run/report
provenance dirty=false. First CI42302984907 exposed a fresh-checkout assumption:
`runs/` did not exist and exclusive bundle creation requires an existing parent.
The suite now creates parent directories while still refusing an existing output;
actual nested-parent end-to-end test added. Reproduction argv also records the
actual supplied relative directory/source-cache path instead of a fixed example.
Final focused71 pass20.68s,mypy195/format/lint pass after this correction.
The first CI failure is retained;it was not a metrics/scoring or distributed failure.

## Final clean source acceptance

Implementation correction:acced9071be0896b2fa6e915c7ed9d1b0f73e9d9.
Runtime source SHA256:17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3.
Exact command,after staging the reviewed cache:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

All three input run manifests and all nine report receipts record dirty=false and
the exact correction commit. The repeated sample reports match byte-for-byte;
weights/Adam/RNG and three curve values match stop2/resume3. Evaluation leaves
weights unchanged;prospective final scoring count remains0.

Public artifacts:

- [Acceptance receipt](evidence/evaluation-suite-acceptance-v2.json)
- [Regression points and report/checkpoint hashes](evidence/evaluation-suite-curve-v2.json)
- [Standalone regression plot](evidence/evaluation-suite-curve-v2.png)
- [Generated results ledger](../../RESULTS.md),backed by nine evaluation-phase7
  receipts in evidence;no public raw questions,predictions or weights.

The final measured results reproduce the diagnostics:original generated math/code
0 correct,GSM8K0/16 (4 context overflows),MMLU4/16 (5 failed contexts),ARC raw2/16
and normalized3/16 (5 failed contexts). The fixed regression curve is0 at all three
steps. These are diagnostic results on a debug model with context256/completion8,
not an accuracy improvement or full public benchmark score. No quality promotion.

The ledger regenerates and checks exactly. Default release coverage deliberately
fails with missing amc-aime,hellaswag,humaneval,math-500. Original and three approved
public development sets all have actual train-shard audits. Those audits certify
neither unknown prior history nor semantic cleanliness.

Final-source CI42303284852 has passed evaluation-suite and eval-harness jobs;
remaining source acceptance is recorded once all jobs complete.

## Source CI closure

[CI42303284852](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/42303284852)
completed successfully:all17 jobs. [Verified job/count receipt](evidence/evaluation-suite-ci-v2.json).
CPU3.11/3.12/3.13 each823 passed,6 optional skips,7 GPU deselected in273.77s/
241.92s/256.25s. The skips are optional SDK/harness/Parquet tests,not hidden core
failures;actual harness11 tests pass13.46s in its dedicated offline job,SDKs in
another. Both original evaluation and expanded standalone suite jobs pass.
Current-source RFC tiny200-step actual SIGKILL89/resume75 still has all losses
exact (max difference0),model/Adam/RNG/application exact. All old SFT/DPO/GRPO/
inference/promotion workflows remain green. The local Gloo timeouts are not
claimed fixed by this remote run.

A final test-only extension exercises the encoded-preference audit path using
actual approved pair preparation:only train chosen/rejected frames are decoded,
counts/token coverage and exact text match an independent bundle decode. The full
ledger/source module passes8 tests4.28s;runtime source remains identical to acced90.
Final Ruff/format,mypy195,build,ledger/inventory and changed guide-link checks pass.
