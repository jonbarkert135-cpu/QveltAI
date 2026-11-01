# Packed-data pretraining and atomic application progress

Date:2026-11-01. Owner:Lead agent. Status:design before implementation.

Reuse TrainingJob/model/AdamW/metrics/atomic checkpoint machinery, adding a strict
PretrainingJob:experiment+training, pinned packed data path/canonical identity,
explicit allow_demo, accumulation_steps, optional finite total_batches and an
explicit TokenSchedule in valid-target units. Legacy warmup_steps/decay_steps must
be0 for this job, preventing silently competing schedules. If total_batches is set,
ceil(total_batches/accumulation_steps) must equal the configured optimizer steps;
the final window may be partial. Demo training requires explicit permission and
is labeled; missing production-source/benchmark gates cannot be bypassed.

Verify tokenizer bundle linkage and exact model vocabulary/context versus data;
store actual data/tokenizer manifests, resolved application recipe, and per-rank
consumer loader state with model/optimizer/RNG in one manifest-last checkpoint.
Extend the existing engine with optional application state/counts and validators;
legacy numeric fixtures retain their explicit descriptor identities/default path.
Restore model and every cursor/counter only after hash/schema/recipe/source/env/
topology checks. Reject a packed checkpoint opened by the numeric-fixture trainer.
Saving a loader cursor separately is not training resume. Bind cursor identity to
split as well as artifact:the existing loader identity omitted split, which this
integration must correct and test before using validation.

Accumulate loss SUM gradients, then divide by the total valid targets in the whole
window, including a short final window and uneven masks. Under DDP no_sync skips
intermediate synchronization, then compensate its world-average by world/count.
Finite-loss failures must be communicated before backward; clip only after global
normalization. Explicit schedule uses actual cumulative valid targets, not padding
capacity or microbatch means. CPU/1GPU path first; distributed extensions need
independent uneven-count/replay references before any support claim.

Validation uses the heldout data split and summed loss/count without touching the
training consumer cursor or model RNG. Save validation metrics with step provenance.
A tiny original BPE corpus learning/replay smoke is permitted, not final training
or a public benchmark. No paid/GPU/final-model run is authorized by this design.

## Verified implementation

Code557e4da passes actual BPE optimization/replay,partial-window/split/worker
identity tests and two-rank SDPA+checkpoint Gloo references. The shared engine
keeps legacy numeric descriptors separate from actual data/tokenizer canonical
identities and validates application state before loading model/optimizer. Explicit
FP32 factories fix the observed ambient-float64 parameter initialization issue.
Final acceptance/receipts are in validation_pretraining.md (522 CPU tests,7 green
CI jobs,20 actual-BPE updates). GPU and production-source gates remain unvalidated.
