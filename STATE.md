# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current task

Expanded Phase 2 foundation M2.3 is complete: clean-clone Make/hooks/build/smoke
and actual GitHub run 41242952575 passed for code 9ee9443. The user now requests
expanded Phase 3 tokenizer/data pipeline. M3.2 is in progress; implementation design
and license/benchmark gates are next. M3.1 baseline remains complete. Ponytail
full is mandatory; original code/docs use Apache-2.0 by delegated user choice.

## Implemented and observed

- Primary config/tokenizer/data/model/train/utils packages; compatible old imports
  and model-preset paths; future posttrain/eval/inference scope documents only.
- Make/hooks/markers/strict YAML/CLI scalar overrides, resolved job/config and
  code/environment/normalized pip-compatible package snapshot next to checkpoints.
- Actual pre-RMSNorm/RoPE/GQA/SwiGLU tied decoder; independent value/gradient/mask
  references, all four meta parameter counts, real tiny CPU forward/backward.
- Real 117056-parameter synthetic trainer,100 updates,6400 targets; early smoke
  loss 5.6883 to 5.0232, exact stop 37/resume 100 weights and Adam states.
- 356 tests passed with optional backends ; 2 GPU cases deselected. Two actual CPU
  ranks verify DDP/global-token gradients/replay and DCP/DTensor shards. Actual
  TensorBoard event and W&B offline run checked. No online SDK upload.
- Local validation details: docs/development/validation_foundation.md. Checkpoints
  validate config/source/software/topology, completed steps, scheduler/sample
  cursor, RNG and integrity; corrupt latest fails, partial directories ignored.

## Limits

Synthetic descriptor hashes are not actual BPE/data manifests; real-corpus
pretraining, accumulation/packing, SFT/RL, reasoning evaluation, cache/serving and
non-safetensors export remain later work. GPU/NCCL/FSDP2/Slurm/shared-store and
multi-node training durability are unvalidated. Native Windows symlink layout is
unvalidated. User laptop VRAM/OS/driver/server resources remain unknown. No paid
or final-scale run and no environment draft change. No external corpus approved.
MFU is an estimate only with supplied peak FLOPs; null otherwise.

## Next three actions

1. Phase 3 design: versioned reasoning/tool tokenizer controls, verifiable authored
   math sources and explicit production benchmark/source approval registry.
2. Implement/test cleaning, exact/near dedup, n-gram decontamination and compact
   binary shards/document packing with deterministic mixing/loader cursor replay.
3. Offline end-to-end demo, worker throughput/fertility/digit experiments, data
   preparation runbook and current CI; keep production gates fail-closed.

## Reproduce

```bash
make setup
make lint
make typecheck
make test
make smoke
```

Use uv 0.12.23 and Linux/Python 3.11–3.13. CPU is default; GPU setup instructions
and exact all-reduce launcher are in runbooks/00_server_setup.md. Publication to
main is authorized/active. Do not reapply old cloud-environment configuration.

Final foundation evidence:354 passed/2 optional skips/2 GPU deselections in the
clean CPU clone;356 passed with local extras; actual remote three-Python CPU
jobs plus smoke and optional SDK job all successful. See validation_foundation.md.

Phase3 implementation checkpoint: strict recipe/raw metadata/usage policy,
seeded LID/PII/heuristics, SQLite exact+math-aware MinHash-LSH, complete benchmark
gate and n-gram removal, original exact EN/RU math, compact shards, mixed-source
0/2-worker consumer cursor replay and data CLI/smoke are implemented locally.
67 focused data tests passed; CPU Make setup/lint/types/test passed with421
passed,2 optional SDK skips and2 GPU deselections. Experiment/remote CI acceptance
is still in progress. One explicitly approved
GSM8K evaluation snapshot acquired/pinned; no external training corpus approved.
