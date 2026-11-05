# Tiny packed BPE pretraining and exact optimizer/data resume

Scope:original approved math/code EN/RU fixture on the CPU first. This is an
engineering smoke, not final-scale training or mathematical reasoning evaluation.
Runtime execution/cache/recomputation is described in02_model_execution.md;
expanded recovery/server controls are in02_pretraining.md; data rights and
production benchmark gates remain in runbook01/DATA_CARD.

## 1. End-to-end offline smoke

After frozen CPU setup, from the repository root, choose a new directory:

```bash
make setup
mkdir -p runs
uv run --frozen --extra cpu python scripts/pretraining_smoke.py --directory runs/bpe-smoke-01 --repository . --steps 20
```

The script generates the approved original dataset, fits byte-BPE on train only,
curates/partitions/packs it, writes a fully resolved pretraining.yaml with actual
manifest hash/vocabulary/context, then performs20 optimizer updates. It compares a
continuous run with stop6/resume20. Each update accumulates two microbatches, batch2,
context64; actual targets exclude PAD/BOS and include EOS. The final JSON must say
weights_bit_exact=true, optimizer_bit_exact=true, consumer_state_exact=true.
Source scope remains demo. No external corpus/model, remote service or GPU is used.

The generated data and both runs are retained under the selected directory.
Check the printed continuous.initial_validation/final_validation for the fixed
heldout language-model probe, not answer accuracy. On resume, initial_validation
means the checkpoint model at the start of that attempt, not the untrained baseline.
Actual seed/code/environment, data/tokenizer identities and exact valid-token
counts accompany checkpoints. Runtime/disk depend on corpus/model/configuration;
this small fixture is suitable for CPU debugging, not a capacity test for large tiers.

## 2. Use the generated recipe directly

```bash
uv run --frozen --extra cpu qvelt pretrain runs/bpe-smoke-01/pretraining.yaml --directory runs/bpe-run-02 --repository . --stop-after 6
uv run --frozen --extra cpu qvelt pretrain runs/bpe-smoke-01/pretraining.yaml --directory runs/bpe-run-02 --repository . --resume latest
```

PretrainingJob embeds the existing TrainingJob under job, adds pinned data_manifest/
data_sha256, TokenSchedule, accumulation_steps, optional total_batches, worker/
prefetch/boundary controls and a heldout validation budget. Defaults/overrides are
fully validated. The generator fills actual values; no fake checksum template
must be edited into a runnable config. For another approved prepared corpus,
replace the input path/hash, exact vocabulary/context and scope after verification.

Packed jobs require the embedded legacy warmup_steps=decay_steps=0; their explicit
TokenSchedule owns valid-token warmup/decay. The old `qvelt train` numerical fixture
continues to use its original schedule and descriptor identities. It cannot load
a packed-data checkpoint and mistake it for a numeric stream.

`job.training.steps` counts scheduled optimizer windows (`optimizer_updates`
separately counts applied steps after skips/rollback); stage mixing start_step counts
committed loader microbatches. By default batches=steps*accumulation_steps. An
explicit total_batches must satisfy ceil(total_batches/accumulation_steps)=steps;
this permits a shorter last window. Never normalize by a mean of microbatch means.
For example a debug recipe may use3 updates, accumulation4, total_batches9.

## 3. Loss, schedule and validation semantics

Each microbatch contributes a loss sum. Gradients are accumulated then divided by
all valid targets in the entire window; DDP's world averaging is compensated by
world_size/global_targets. Intermediate DDP microbatches use no_sync; the last
synchronizes. Local target counts may differ or be0; a wholly empty window is
invalid. The default abort policy stops on nonfinite loss/gradients. Optional skip/rollback
completes synchronization hooks before discarding the window;see02_pretraining.md. Partial/uncommitted windows never produce checkpoints.

TokenSchedule explicitly specifies warmup_tokens, decay_start_tokens, decay_tokens.
LR is evaluated at the end of the new global valid-token window:linear warmup,
constant peak, then linear decay clipped at0. It uses actual targets rather than
padded capacity. The optimizer-step horizon is an independent stop cap:changing
steps does not silently stretch this token schedule or promise LR0 on the last
step. Set all boundaries deliberately for an actual training recipe.

Validation is a fixed seeded weighted heldout probe with its own loader/state and
RNG isolation, not an exhaustive public benchmark. It may intentionally reuse source
blocks after exhaustion. It never advances the training cursor and restores model
mode. Reports label validation_probe_loss/targets and record the step. The original
project final-evaluation suite remains separate/unprepared.

Metrics record summed-token loss, LR, preclip norm, clipping, cumulative actual
targets, committed microbatches and global throughput. MFU is a nominal formula
only if an explicit peak TFLOP budget is supplied; it excludes recomputation/
padding/IO and is not measured utilization. Otherwise MFU is null.

## 4. Atomic resume and identities

A checkpoint contains model.safetensors, Adam state, rank RNG, schedule/counters,
base config/job, run manifest and application-rank-N.json for every rank. Application
state includes the resolved packed recipe, semantic digest, committed loader cursor,
actual data/tokenizer canonical hashes and actual processed tokens. The top manifest
is published last. With strict resume,incomplete directories are ignored and corrupt latest is an
error. Auto-resume explicitly quarantines invalid writes and selects verified state.

Run metadata copies data/tokenizer manifests and the application recipe; corpus
payloads remain in the external immutable prepared directory. Keep that directory
available, or relocate it with identical content/hashes. Full input checksums are
verified at startup:large corpora require an I/O preflight budget. These copies
are provenance metadata, not a duplicate of the entire corpus inside each checkpoint.
Optimizer/RNG files are trusted Qvelt state, not arbitrary uploaded pickle inputs.

Restore checks hashes, semantic recipe, source/package versions, topology,
precision, cursor split/seed/batch/mixing/attention and token counters before
loading model/optimizer. Loader identity now binds train/validation/test as well
as artifact hash. Older incompatible cursor fingerprints are rejected.

Worker count/prefetch and data mount path may change after actual artifact checks;
these resource selectors are excluded from semantic replay identity and preserved
in per-attempt/checkpoint recipe observations. Optimizer/model/schedule/accumulation/
world size changes reject exact resume. No cross-version/topology migration is
advertised. rank-local read/transfer/checkpoint errors are coordinated before the
next collective; hard process/node failures still need relaunch from a checkpoint.

## 5. Distributed and accelerator limits

Actual two-rank Gloo tests compare uneven-count accumulated gradients against one
global reference batch, verify coordinated rank-local failure and compare packed
continuous/resumed weights/Adam/per-rank cursors. Production CUDA/NCCL/shared-store/
multi-node behavior is still unvalidated. Packed FSDP2 has a CUDA-only implementation with mixed precision and DCP;
its hardware correctness remains unverified. Compiled FSDP2 is explicitly rejected.
The earlier CPU numeric DCP checks remain separate. Do not equate CPU success with GPU throughput/durability.

An actual distributed launch uses the same validated recipe on every rank, sets
job.training.strategy=ddp and an appropriate global valid-token schedule:

```bash
uv run --frozen --extra cpu torchrun --standalone --nproc-per-node=2 -m qvelt pretrain runs/bpe-smoke-01/pretraining.yaml --directory runs/bpe-ddp-01 --repository . --set job.training.strategy=ddp
```

This is a CPU debug example. World-size changes are new runs, not exact resume.
For CUDA follow runbook00 and capability diagnostics first; no server/paid run is
started here. A production dataset must already have purpose=production with every
planned approved benchmark snapshot; setting allow_demo=true never upgrades demo
scope to production approval. No external training corpus is currently approved.

A supplied single-GPU packed replay test can be run explicitly with the workspace
setting established before Python starts:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --frozen --extra cuda pytest tests/integration/test_pretraining.py -m gpu -q
```

It uses small FP32 shapes and explicit numerical tolerances. This test has not run
on this CPU host; skipping for missing GPU/workspace setup is not GPU validation.

Relative data_manifest paths resolve against the recipe file's directory. Resolved
absolute mount paths are preserved in attempt/checkpoint recipes, while semantic
replay uses the verified canonical artifact hash. Exact restart demonstrations use
the default JSON logger; optional SDK offline-output tests do not establish SDK/
stochastic-training restart parity.
