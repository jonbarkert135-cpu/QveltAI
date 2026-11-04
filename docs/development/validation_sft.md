# M6.1: original conversational SFT acceptance

Date:2026-11-04. Base revision7eed79c with the implemented SFT changes in the
working tree. Exact executable source digest, recipe, data/tokenizer/base identities,
seed, counts and results are in evidence/sft-smoke-2026-11-04.json. Receipt paths
are normalized relative to the run root; originals are retained in local run
artifacts. No weights/raw corpora/optimizer files or machine paths are committed.

## Implemented behavior

Version1 chat formatting uses shared Qvelt role controls with explicit message
validation and plain-text control isolation. Layout2 masks all prompt/role/system/
padding/outer-EOS targets while retaining full valid attention context. Whole-dialog
segments prevent role state and causal context being lost across packed blocks.
Right truncation never fabricates an end-turn and rejects a prompt-only prefix.
Layout1 pretraining behavior remains unchanged.

Preparation requires license records, explicit training permission and operator
approval recorded in DATA_CARD; writes source/tokenizer/license lineage and family
splits, target/truncation counts and honest demo-only scope. An empty registered
demo benchmark set explicitly supplies no production contamination clearance.
Existing strict YAML/dataclasses support SFT plus pinned initial checkpoint fields.
`qvelt sft` uses verified matching base weights with a new optimizer/schedule, then
uses existing atomic application checkpoints for ordinary exact SFT resume.

## Local checks

- Focused new chat/SFT tests:19 passed in9.78s before the final explicit-approval
  assertion was added; full acceptance includes that assertion.
- Ruff lint/format and strict mypy:pass,103 first-party files.
- Full CPU/optional TensorBoard/W&B suite:543 passed,5 GPU deselected,no skips in97.32s.
- Wheel/sdist build:pass; all three new posttrain modules present in the wheel.
- Frozen lock/dependency consistency:pass; uv.lock unchanged,72 packages compatible.
- CI YAML parsed and a dedicated80-step SFT smoke job added, preserving pinned
  existing actions and CPU3.11/3.12/3.13 coverage. Remote run:GitHub41905361820 at0abfcd3,all8 jobs successful (including the
  three CPU Python versions and the dedicated80-step SFT smoke).

The first full diagnostic run had a numeric-DDP process-exit timeout and two new
packing-test oracle mistakes (shifted length and corrupting the wrong split's
shard); those test expectations were corrected. Numeric-DDP standalone rerun
passed in26.47s. A subsequent full run was interrupted when a packed-DDP subprocess
remained after checkpoint creation. The final suite uses explicit network permission,
`GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. This binds local Gloo
traffic to loopback without changing system proxy/routing/TLS policy. Root cause of
the earlier process-exit stalls is not established; no general network fix is claimed.

New references independently enumerate multi-turn shifted mask offsets and compare
loss/gradients to explicit per-target logsumexp, including exact zero masked-logit
gradients. Storage tests exercise multiple dialogs, truncated responses, label
boundaries, padding and actual checksummed token corruption. Further tests cover
invalid schemas/roles/policies/approval, family isolation, model/tokenizer/objective
mismatch, base corruption, CLI/resume, RNG/Adam/consumer replay and worker restart.

## Actual tiny optimization and replay

Command with README cache exports,Python3.12.14/uv0.12.23/Torch2.9.1+cpu:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu --extra tensorboard --extra wandb python scripts/sft_smoke.py --directory "$QVELT_SFT_OUTPUT" --repository . --steps 80
```

Use a new output path under an existing writable parent. The script trains a
three-step original engineering base with context128 and then initializes SFT
from its pinned checkpoint, using the same445-token BPE vocabulary. Supervision:
64 EN/RU original exact-answer dialogs,seed20261104,24 train/16 validation/24 test,
with disjoint SFT problem-family lists across languages. No truncation in this
acceptance dataset. Data generation independently verifies exact gold values with
Fraction/math and SymPy; no external teacher or model API is called.

Actual SFT:80 optimizer steps,80 microbatches,2520 assistant targets,LR0.003,
B2/T128,FP32,seed20261031,one CPU thread. Fixed69-target train probe decreases
5.897945459338202 ->0.14195556225983993. Distinct16-target family-heldout probe
5.810949802398682 ->4.761350393295288. First/last optimizer-window losses are
5.889200846354167/0.1138413429260254. Stop26/resume80 versus uninterrupted run:
weights,Adam,RNG and committed application/consumer state all bit-exact.

## Limits and next step

This demonstrates original-fixture SFT overfit and restart, not answer accuracy,
proof correctness or general instruction following. The validation probe is small
and may reuse blocks; base demo task families overlap the later SFT heldout task
types. Production corpus/license/decontamination gates remain open. Layout2 does
not support tool/reasoning controls or splitting long dialogs. Arbitrary imported
solutions receive structural validation, not automatic proof verification.

Single-process CPU SFT is validated; distributed SFT,CUDA/NCCL/FSDP,multi-node,
shared storage,SDK stochastic restart and final-scale resources are unverified.
Old-source checkpoints remain subject to source/software exact-resume rejection.
Next dependency-ready portable milestone:M6.2, verified preference objective and
licensed original preference fixtures; public model-quality evaluation is later.
