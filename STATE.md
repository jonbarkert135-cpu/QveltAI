# Qvelt current state

Updated: 2026-10-31. Owner: Lead agent.

## Current task

User requested the expanded Phase 2 engineering foundation after M3.1. M2.3 is
in progress: implementation/local checks complete, clean-clone/remote current-CI
acceptance pending. M3.1 remains complete. Real-data M3.2 and full M4/M5 gates are
not bypassed by the synthetic path. Ponytail full and relevant installed skills
remain mandatory; original code/docs are now Apache-2.0 by delegated user choice.

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

1. Clean clone: make setup && make lint && make typecheck && make test; real
   make smoke, build, pre-commit checks and package validation.
2. Publish to main and observe current-revision GitHub Actions. Update M2.3 status
   with actual CI/clone evidence; preserve logs/negative diagnoses in experiments.
3. Resume M3.2 deterministic real-data preparation, dedup/splits and token shards
   on approved local fixtures before any external corpus or production training.

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
