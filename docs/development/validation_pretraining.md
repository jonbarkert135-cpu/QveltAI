# Phase4/5 packed-data training acceptance

Date:2026-11-01. Owner:Lead agent. Scope:original CPU debugging data and actual
same-topology optimizer/data replay, not a final model or public reasoning score.
Code557e4da5a4a9ba3d800afc374af1ac5be44ce471; measured code tree clean.

## What works

- Phase4:model component/gradient oracles, seeded FP32 initialization even under an
  ambient float64 default, tied-embedding combined gradients and finite-loss guards.
  Reference/native SDPA, non-reentrant block recomputation and bounded request-local
  compact KV-cache pass CPU FP32/BF16 comparisons. Cache rejects stale/cross-model/
  malformed state. Source/CPU profile remains [phase4_design.md](phase4_design.md).
- Phase5:actual byte-BPE/sharded records, explicit valid-token schedule, sum/count
  accumulation including shorter windows and unequal masks, fixed heldout LM probe,
  clipping/throughput and actual artifact identities. Model/Adam/RNG/per-rank
  consumer state share one atomic manifest-last checkpoint. Worker-count/mount
  changes preserve semantic replay; source/software/schedule/topology changes fail.
- Actual two-process Gloo reference includes0-target local microbatches and verifies
  gradients against one global masked batch. A rank-local failure makes both ranks
  stop before subsequent work. Packed SDPA+checkpoint full/resumed optimization
  matches weights/Adam and per-rank cursor state. RNG equality is also checked.

## Commands and observed results

```bash
make setup
make lint
make typecheck
make test
uv lock --check
uv pip check
make build
uv run --frozen --extra cpu python scripts/pretraining_smoke.py --directory /tmp/qvelt-pretraining-clean-557e4da --repository . --steps 20
```

A separate clean clone/frozen environment passed lint/format,mypy97 files,dependency
checks,wheel/sdist build and **522 passed,2 optional SDK skips,5 GPU deselections
in122.79s**. No tracked changes resulted from validation. Earlier full521 tests
preceded the final FP32-factory regression/provided GPU case; the final counts
above include them. Existing CPU tiny/count/DDP/DCP/data tests remain green.

The clean actual-BPE smoke instantiated **127104 parameters**, verified against
actual saved tensor enumeration,20 optimizer updates,40 microbatches,4959 valid
targets (the nominal5120 includes padding). Full versus stop6/resume20 had
bit-exact weights/Adam/consumer state; direct checkpoint comparison also confirms
Python/NumPy/Torch RNG equality. Fixed256-target heldout probe loss decreased
6.126703500747681 ->5.756516456604004. This is tiny original-data language modeling,
not correct-answer accuracy or evidence for a strong mathematical model.

Machine-readable evidence:

- [Smoke result](evidence/pretraining-smoke-557e4da.json).
- [Code/environment/checkpoint/cursor and RNG proof](evidence/pretraining-provenance-557e4da.json).
- [Direct GitHub job evidence](evidence/pretraining-ci-557e4da.json).

Actual data canonical hash:
`47c240756969f8601a83e356efaa45c208a2f1857ef8e9a6e2079a1541e3e0fe`;
tokenizer:`1df21693cd5216cb93c9ad5a910897672312a74fd75052332193890bbf454041`.
They are real manifests, unlike the separate numeric fixture's descriptor slots.
Run UUID/time/provenance can change artifact identities across rebuilding; only
specified payloads/updates are compared, not arbitrary manifest bytes across hosts.

## Actual remote CI

[GitHub run41290979913](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/41290979913)
completed with all7 jobs successful:CPU3.11/3.12/3.13,numeric100-step smoke,offline
data smoke,packed20-step pretraining and optional offline metrics. Each CPU job
reports522 passed/2 optional skips/5 GPU deselections (85.51/96.48/103.55s).
The direct packed-job log confirms the same127104 parameters/4959 targets and
exact replay; its heldout probe6.12670373916626 ->5.756516695022583 differs slightly
from the local host, consistent with no cross-hardware bitwise guarantee. Actual
within-run replay remains exact. The separate optional job tests installed SDKs;
skips in the lean matrix are not treated as passed SDK tests.

## Limits and next gates

No actual CUDA/NCCL,packed FSDP,shared-store,multi-node or laptop test ran. Five GPU
cases are provided but deselected here. Full dataset checksum preflight and cache
concatenation/storage costs require representative hardware/I/O profiling. Graph
saved-storage metrics are not allocator peak memory. No approved external training
corpus; MATH/AIME/MMLU and frozen project-final exclusions remain incomplete, so
production preparation stays blocked. SFT/preferences/RL/evaluation/serving retain
their roadmap gates; this delivery does not mark them implemented.

Run instructions:[runbook03](../../runbooks/03_packed_pretraining.md).

Additional explicit RNG regressions compare all saved Python/NumPy/Torch state
fields for partial windows and both CPU-DDP ranks:3 targeted cases passed in39.27s.
These assertions supplement the existing optimizer/cursor proofs without changing
training code. Exact replay demonstrations use the default JSON metric backend;
the optional SDK job verifies offline output files, not stochastic SDK/trainer
restart parity. CUDA numerical replay requires its separate actual-hardware gate.
