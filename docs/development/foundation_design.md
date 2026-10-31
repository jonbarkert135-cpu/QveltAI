# Engineering foundation completion design

Date: 2026-10-31. Owner: Lead agent. Status: CPU foundation acceptance complete; actual CI/clone evidence in validation_foundation.md.

The user explicitly requests the expanded Phase 2 deliverables after M3.1.
Implement these without replaying completed tokenizer/research work. This adds
an RFC decoder and synthetic training earlier than the original phase order to
make the required 100-step CI smoke a real training run. Real-data preparation,
post-training, reasoning evaluation and serving remain later work.

Preserve old import/config paths through small compatibility exports and local
configuration symlinks. Put primary code in config/, tokenizer/, data/, model/,
train/, utils/. posttrain/, eval/, inference/ contain scope documents, not fake
functions. New tests live in unit/ and integration/; existing fixture IDs stay
unchanged. All data/model/runtime choices remain dataclasses and strict YAML.
A training-job dataclass embeds the existing experiment dataclass to preserve
schema-1 config/manifest compatibility; CLI overrides resolve before hashing.

Implement the RFC's tied, bias-free pre-RMSNorm/RoPE/GQA/SwiGLU decoder and sum/
count masked causal loss. Independent math/gradient/causality/packing tests precede
100 synthetic CPU updates. The smoke uses a documented small dimension override,
not a claim of full-scale pretraining or a useful language model.

Structured local metrics are always written, with optional TensorBoard/W&B
adapters (W&B offline). Record valid targets/s, estimated model FLOPs utilization
only with user-supplied peak FLOPs, unclipped gradient norm, loss and LR. Do not
invent utilization or compare it to measured hardware efficiency.

Single/DDP checkpoints save safetensors weights, optimizer, schedule/token/sample
positions, rank RNG, resolved configs, code/environment and a safe pip-compatible
installed-version snapshot. FSDP2 uses DCP model/optimizer shards. Manifest-last
publication and checksums gate resume-from-latest; corruption and compatibility
mismatch fail closed. Resume is same code/software/device/precision/topology only.
FSDP/NCCL require hardware tests; a CPU DCP test is not GPU FSDP validation.

Make and pre-commit reuse the pinned uv tools; CI runs lint/types/CPU tests and
an actual 100-step synthetic smoke with checkpoint replay. The server runbook
covers Linux/NVIDIA prerequisite checks, CPU/GPU all-reduce, tmux/Slurm and storage;
server-specific commands/requirements are stated as assumptions and not executed
on the user's laptop. Apache-2.0 is the user-specified default for original code;
retain separate vendor/fixture licenses and provenance.

Acceptance: requested Make chain from an isolated clean checkout, numerical and
resume reference tests, two CPU ranks, actual optional offline backend checks,
100-step logs/checkpoints and a verified build. Observe remote CI if credentials
allow; never equate local success or a workflow file with green remote jobs.
