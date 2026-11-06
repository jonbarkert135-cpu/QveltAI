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
