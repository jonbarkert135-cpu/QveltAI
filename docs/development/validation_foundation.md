# Expanded engineering foundation validation

Date: 2026-10-31. Owner: Lead agent. Status: foundation acceptance passed for code commit
9ee94434d4ebc50361850689e5041b603cdc30b0. GPU/cluster limits remain explicit.

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

## Final clean-checkout acceptance

An isolated ordinary Git clone at /tmp/qvelt-foundation-clean-3rbeIn ran exactly
make setup, make lint, make typecheck, make test, then make smoke, make build,
pre-commit run --all-files, uv lock --check and uv pip check. All exited 0;
Git remained clean. Python 3.12.14:354 passed,2 optional SDK skips,2 GPU cases
deselected,55.04s. Ruff/format and strict mypy pass on 60 files; both hooks pass.
The base profile omits optional SDKs deliberately; the all-extras local run passed
356 cases and the dedicated remote SDK job executed all 3 backend tests.

The clean-clone smoke repeated the recorded loss/counts and exact weights/Adam
replay with actual clean Git provenance. Run directory: /tmp/qvelt-smoke-3_8kwcqd.
Normalized package snapshot SHA256:
20f947f688a0075bced2386566f464b044fadffae7199f29d394f44ef89edb4f.
Run manifest raw SHA256:675f7066dc1ecd9b79105185ef6016aeb43a9de579a08734cffe03ad860bc9c8.
Lock SHA256:770ad3e399d2190d3b05c8ce4d420cc3a53b5691f5548ddea7a45fb4e12f330d.
Tested clean-build wheel SHA256:
357029f8521ec599235c708b2809079cd1de7c44532d5c7c41cb161ecdd4928c.
All wheel Python module bytes match the checkout; py.typed and LICENSE are present.
A later README-only build may change wheel metadata, not the tested source.
Temporary receipt/log: /tmp/qvelt-foundation-clean-receipt.json and
/tmp/qvelt-foundation-clean-make.log. Checked-in source/tests and Make commands are
the durable reproduction path; temporary instance files are not required.

## Actual GitHub Actions result

Observed through GitHub API and downloaded logs:
[run 41242952575](https://github.com/jonbarkert135-cpu/QveltAI/actions/runs/41242952575),
head 9ee94434d4ebc50361850689e5041b603cdc30b0, all five jobs completed/success.

| Job | Observed result |
| --- | --- |
| cpu3.11 |354 passed,2 skipped,2 deselected in 37.42s |
| cpu3.12 |354 passed,2 skipped,2 deselected in 32.09s |
| cpu3.13 |354 passed,2 skipped,2 deselected in 41.58s |
| smoke |100 real updates,6400 targets, exact weights/Adam replay |
| optional-metrics |3 passed in 6.27s, actual JSON/TensorBoard/W&B offline outputs |

The remote smoke reported the same first/last loss as the local smoke. The uploaded
synthetic-smoke-logs-and-checkpoints artifact exists, ID 13058222452,6736814 bytes,
not expired at inspection. GPU cases did not run. These are actual job/log results,
not an inference from the workflow definition. Further source changes need new CI.
