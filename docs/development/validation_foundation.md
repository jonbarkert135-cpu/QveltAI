# Expanded engineering foundation validation

Date: 2026-10-31. Owner: Lead agent. Status: local implementation checks recorded;
clean-clone and current-head GitHub CI gates are in progress (M2.3 in PLAN.md).

Actual local evidence so far: 356 non-GPU tests passed with TensorBoard/W&B extras,
2 GPU cases deselected,45.25s on Linux/Python 3.12.14. This includes all 315 prior
cases plus new numerical/config/data/training/distributed/backend tests. Actual
100-update smoke: 117056 parameters, 6400 targets, fixture CE5.6883039474487305 to
5.023181438446045, exact stop 37/resume 100 weights and Adam states. Initial logs
are /tmp/qvelt-foundation-smoke-first; final clean-commit evidence will supersede
its dirty-source provenance. No general-purpose model or benchmark score.

Two CPU ranks passed integer all-reduce, uneven-loss-mask global-gradient oracle,
DDP runner exact restart and real CPU DTensor/DCP shard round trips. Optional
metrics tests write/read actual TensorBoard events and actual W&B offline output.
GPU/NCCL/FSDP2 cases exist but have not executed. No Slurm job or server driver
change was made. Driver/runbook source evidence is in ADR-0028.

Diagnoses retained: stale gradients on an emptied optimizer caused PyTorch DCP's
initializer to leave its restore template empty; clearing grads before load fixed
it, verified against saved Adam tensors. Concurrent per-rank stdout JSON records
could interleave; CLI now prints rank 0 summaries while all ranks validate. Moving
modules required mock targets to follow primary modules; malformed test typing
was corrected without relaxing strict mypy or numerical assertions.

The prior published commit b214fc2 had successful CI run 41236858997, actually
observed through GitHub API. That does not validate this new code. Final current-
revision CI results and clean-checkout Make commands will be recorded here after
those gates pass.
