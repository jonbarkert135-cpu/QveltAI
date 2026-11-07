# Distributed pretraining: first small server run

Scope:Linux,Python3.11–3.13,uv0.12.23,PyTorch2.13.0;original Qvelt decoder and
manifest-pinned data. CPU correctness and restart have executable tests. Real
CUDA/NCCL/FSDP2,shared-store and multi-node validation require your actual server;
this document is an operator recipe,not a claim that those hardware runs occurred.
No paid/full-scale training is launched by the development agent.

## 1. Freeze inputs and inspect the server

Before a production run,register exact source revisions/licenses in DATA_CARD and
prepare purpose=production data with all required licensed exclusion snapshots.
MATH/AIME/MMLU and project-final clearance remain blocked in this repository.
`allow_demo=true` never upgrades a demo into production. Do not start a first full
run using placeholder hashes or the nonce/math test fixture.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
make PROFILE=cuda setup
uv lock --check
uv pip check
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
uv run --frozen --extra cuda qvelt doctor --device cuda
uv run --frozen --extra cuda python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported(including_emulation=False)
assert torch.distributed.is_nccl_available()
print(torch.__version__,torch.version.cuda,torch.cuda.device_count())
PY
```

Use driver/setup instructions in00_server_setup.md;do not install drivers inside a
managed container. BF16 is required by this preset;unsupported hardware must use
an explicitly validated different precision recipe,not silent emulation. Inventory
RAM/SSD and retention capacity as well as VRAM. A full small Adam state is roughly
16P bytes plus activations/workspace;actual peak memory must be measured.

## 2. Create a real recipe and run a short pilot

Run from the checkout root. Set QVELT_DATA to an actual approved prepared manifest,
with vocabulary32768 and packing context2048. Set N to the actual selected GPU
count;global sequences64 must divide N*per-GPU-batch. Example below deliberately
starts B1 with block recomputation. The factory derives accumulation after overrides.

```bash
export QVELT_DATA=/absolute/path/to/approved/prepared/manifest.json
export N=1
mkdir -p runs/small
uv run --frozen --extra cuda python scripts/make_pretrain_recipe.py \
  --preset configs/pretrain/small.yaml --data "$QVELT_DATA" --world-size "$N" \
  --output runs/small/pretraining.yaml \
  --set job.training.batch_size=1 \
  --set job.training.execution.activation_checkpointing=true
```

The output must be new;existing resolved recipes are not overwritten. With N1,
strategy is single and accumulation64;N8 gives DDP and accumulation8. Model is
125854464 parameters,L16,d768,GQA12/4,SwiGLU2048,tied V32768,T2048. Default peak
LR6e-4,beta0.9/0.95,eps1e-8,clip1,projection decay0.1,norm/embedding decay0.
Fused AdamW is explicit in this server preset;portable AdamW remains available.
All of these are starting hypotheses,not demonstrated optimal server settings.

```bash
uv run --frozen --extra cuda torchrun --standalone --nproc-per-node="$N" \
  -m qvelt pretrain runs/small/pretraining.yaml --directory runs/small/run \
  --repository . --resume auto --stop-after 200
uv run --frozen --extra cuda python scripts/summarize_run.py runs/small/run
```

Inspect hardware utilization/peak allocator memory separately with the operator
profiler and GPU tools;CPU graph-storage bytes are not a VRAM estimate. Run actual
GPU numerical/replay tests before promoting this pilot. For two selected GPUs:

```bash
uv run --frozen --extra cuda pytest -m gpu -q
uv run --frozen --extra cuda python scripts/profile_model.py \
  --model configs/model/small.yaml --device cuda --bf16 --recompute \
  --batch 1 --length 512 --steps 5
```

The profiling length512 is a distinct resource diagnostic,not evidence that full
T2048 fits. Supply `--peak-tflops` only with an identified device/precision peak;
otherwise MFU stays null. Do not invent a GPU peak from CPU throughput.

## 3. Continue the first full horizon

Only after rights/resource/correctness/pilot acceptance,continue the same run:

```bash
uv run --frozen --extra cuda torchrun --standalone --nproc-per-node="$N" \
  -m qvelt pretrain runs/small/pretraining.yaml --directory runs/small/run \
  --repository . --resume auto
```

The preset has an82000-window safety cap and a10B supervised-target horizon:
100M warmup,stable through9B,1B WSD decay to0. Counts exclude PAD/BOS and include
repeats. The last window's global rank-ordered target mask is truncated to end at
exactly D;the unused final remainder is consumed but never supervised. Warmup/
tail LR is evaluated at completed-window end(ADR0036),rather than the original
RFC proposal of splitting every boundary. This deterministic convention is part
of run identity. Steps count scheduled windows;`optimizer_updates` counts actual
applied updates after skips/rollback. Skipped target exposure advances the schedule.

Cosine control is `schedule.shape=cosine,min_lr_ratio=0.1`,with decay from warmup
end to the same horizon;0.1 and zero endpoints are different experiments. Generate
a new resolved recipe/run for any schedule/model/batch/topology change. Data mixture
stages remain committed-loader-microbatch based;token-phase mixture changes require
explicit coordinated preparation,not an inferred conversion from the LR schedule.

No absolute healthy-loss targets are known for your corpus. Uniform-vocabulary CE
is ln(32768)=10.3972 nats as a rough reference,not a predicted starting value.
Healthy warmup has finite loss/gradients and a stable LR ramp;the stable phase
should improve fixed heldout controls over meaningful token intervals;decay should
remain finite without all domains worsening. Compare to the pilot and matched
seeds,not a promised loss of2 or3. PPL is exp(mean token CE),not math answer accuracy.

## 4. Stability and recovery playbook

Metrics include CE,z-loss separately(if enabled),LR,preclip global norm,clipping,
actual target counts,tokens/s/device,nominal MFU,nonfinite/rollback counters and
spike z-score. Optional layer RMS/abs-max describes the last microbatch;rank-mean
RMS is not an exact global-token-weighted statistic. Compile mode logs layer
parameter gradients without activation hooks. Sample outputs are bounded greedy
uncached continuations and can contain ordinary special tokens;inspect them rather
than calling them a quality score.

Source-specific heldout CE/PPL uses actual validation source names. Web/code/math
coverage exists only if those approved sources have heldout splits. The authored
three-domain fixture is a harness test,not coverage of real web/code/math corpora.
Fixed probes can wrap/reuse tiny source blocks;the public final evaluation remains
separate. Greedy samples reserve bounded response room and label prompt truncation.

Red flags:nonfinite windows or checkpoint quarantine;frequent spikes;clipping on
more than half of100 updates;gradient norms>10x their run median;persistently
worse heldout domains;throughput collapsing below70% of its earlier median. These
are documented heuristics,not validated universal thresholds. Inspect source/sample
IDs,normalization,masks,data/precision changes and layer statistics before changing
LR. Change one variable and branch a new lineage. z-loss(default0) is an optional
ablation;do not simultaneously change QK normalization or invent that support.

The preset selects rollback. An initial finite step0 snapshot permits first-window
recovery. Nonfinite loss/gradient windows complete synchronization hooks,discard
grads,skip the optimizer and advance the committed data cursor. Nonfinite persistent
model/Adam state is detected after the step and rolled back. Restore finite weights,
Adam and RNG from the last checkpoint while keeping the forward cursor;good updates
since that checkpoint are also discarded. This is a documented data-skip policy,
not exact uninterrupted learning. Counters and failure cursor receipts stay loud.
The failure budget stops repeated bad windows. Validation failures and arbitrary
runtime/communication errors fail coherently where possible;they are not silently
converted to skipped training examples.

`--resume latest` is strict:corrupt latest is an error. `--resume auto` verifies
registered bytes,newest first,quarantines incomplete/corrupt checkpoints with logs
and selects the newest valid one. Recipe/source/software/tokenizer/topology semantic
mismatch still fails;it is not bypassed by choosing an older semantic identity.
Quarantine is retained. `keep_checkpoints=0` keeps all;explicit positive retention
removes older committed checkpoints,not quarantine. Allocate disk accordingly.

SIGTERM only records a stop request;all ranks save at a common completed window,
then the CLI exits3. Allow time for the longest window/collective/checkpoint in the
scheduler grace period. SIGKILL cannot save;auto-resume replays from the last
manifest-committed checkpoint,including an interrupted partial write. Keep the same
software,precision,seed,world size and batch/accumulation;cross-topology/version
migration and arbitrary optimizer uploads are not supported.

## 5. FSDP2 and unsupported combinations

FSDP2 is an implemented CUDA-only API path with bottom-up sharding,FP32 master
weights/reductions,BF16 gathered compute and DCP sharded state. It has not run on
this CPU machine. Create a separate validated recipe with:

```bash
# Add these to make_pretrain_recipe.py BEFORE creating a new FSDP run:
# --set job.training.strategy=fsdp --set job.training.fused_optimizer=false
```

Use at least two actual GPUs,run FSDP replay/clipping/memory checks on your shared
storage,and inspect tied embeddings and per-rank cursors. Portable per-parameter
AdamW is used for DTensors;fused DTensor kernels are unverified. Compiled FSDP2 is
explicitly rejected pending hardware validation. DDP/single compiled blocks retain
parameter names and have CPU correctness controls;exact resume still binds compile
settings. TP/PP>1 are rejected as unimplemented,no silent emulation or fake launcher.
No multi-node performance/shared-store durability claim follows from CPU DCP.

## 6. Sweeps,compile and mini scaling controls

Validate and inspect the generated small recipe and GPU count before launching:

```bash
uv run --frozen --extra cuda python scripts/sweep.py configs/sweeps/small-lr-batch.yaml
# The shipped launcher assumes8 GPUs;edit its argv to your actual N first.
# This next command starts12 short server jobs sequentially;inspect budget first.
uv run --frozen --extra cuda python scripts/sweep.py configs/sweeps/small-lr-batch.yaml --execute
uv run --frozen --extra cuda python scripts/sweep.py configs/sweeps/small-lr-batch.yaml --summarize
```

The grid spans0.5x/1x/2x LR,two accumulation sizes and two seeds. `preserve_tokens`
is the historical key for matched nominal sequence capacity:actual valid targets,
mixture and ordering can differ with batch shape. Compare logged counts and token
horizons. Reruns use auto-resume;no failed child job is a successful result. These
small/server sweep jobs were not launched by the development agent.

For bounded CPU controls,use an existing approved demo dataset:

```bash
# Keep CPU diagnostics separate from an active CUDA run environment.
export UV_PROJECT_ENVIRONMENT="$PWD/.cache/cpu-diagnostics"
uv run --frozen --extra cpu python scripts/data_smoke.py --directory runs/diagnostic-data
uv run --frozen --extra cpu python scripts/benchmark_training.py \
  --data runs/diagnostic-data/prepared/manifest.json --directory runs/compile-control
uv run --frozen --extra cpu python scripts/scaling_sanity.py \
  --data runs/diagnostic-data/prepared/manifest.json --directory runs/scaling-control --steps 20
```

Alternatively use
`runs/bpe-smoke-01/data/prepared/manifest.json` from03_packed_pretraining.md.
The compile benchmark reports total wall time including initial compilation and
steady update throughput separately,checks final-weight tolerance and leaves CPU
MFU null. The scaling script runs four explicitly reduced tiny models with the
same original data/token exposure,then fits log(loss)=a+b*log(P). Four short runs
are a descriptive sanity check,not a general scaling law or proof of transfer to
small. Positive slope/poor residual fit is a red flag,not a reason to fabricate a
monotone curve. Record losses,finite state and clipping before choosing server LR.

## 7. Reproduce the200-step kill experiment and send logs

```bash
mkdir -p runs
export UV_PROJECT_ENVIRONMENT="$PWD/.cache/cpu-diagnostics"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/pretraining_acceptance.py --directory runs/tiny-kill-200 --repository .
```

This uses the actual19,338,560-parameter RFC tiny with train-only BPE32768,T16/B2,
200 real optimizer updates and three authored diagnostic domains. It compares a
continuous run with a separate subprocess killed by SIGKILL after step89,restarted
from its last committed checkpoint. It asserts the entire200-point loss curve and
weights/Adam/RNG/application state,not just a final scalar. Source fixtures and
permissions are in DATA_CARD. This is engineering recovery,not mathematical quality.

```bash
unset UV_PROJECT_ENVIRONMENT  # Return to the original CUDA environment.
uv run --frozen --extra cuda python scripts/summarize_run.py runs/small/run \
  --archive runs/small/diagnostic-logs-01.tar.gz
```

Use a new archive filename. The allowlist includes manifests,resolved recipes,
metrics,samples,failure cursors and checkpoint metadata;it excludes tensor/optimizer
payloads and unrelated text files. Files over64MiB are omitted. Review the archive:
prompts,generated samples and paths can be private. Remove/redact private text from
a separate diagnostic copy,preserving originals for replay. Attach the reviewed
archive in this chat with the GPU/driver inventory and what happened. No upload,
email or external message is sent automatically. I can inspect loss/LR/norms,
clipping,domain PPL,samples,throughput and restart boundaries from that evidence.
