# Troubleshooting

| Symptom | Likely boundary | Action |
| --- | --- | --- |
| Read-only home cache error | uv/audit/tool cache | Set UV_CACHE_DIR and PRE_COMMIT_HOME under the checkout;pass pip-audit --cache-dir explicitly |
| Permission/network error or async hang | Managed loopback/proxy policy | Enable execution network permission;keep inherited proxy and CA trust |
| Artifact directory already exists | Exclusive write protection | Choose a new output directory;do not remove an unknown run |
| Vocab/sequence mismatch | Actual tokenizer/data differ from config | Inspect resolved recipe and packed manifest;never pad by relabeling metadata |
| Resume compatibility failure | Source/lock/config/world size changed | Use original commit/environment or start an explicit new lineage |
| Corrupt newest checkpoint | Incomplete or modified bytes | Preserve quarantine and logs;auto-resume a valid committed predecessor |
| DCP metadata/global rejected | Unsupported or executable pickle | Do not disable restrictions;inspect provenance and exact Torch version |
| API401 | Missing or wrong bearer binding | Set the named secret privately;never send keys in a bug report |
| API429/504 | Queue/deadline exhausted | Reduce concurrency/budgets;inspect sanitized counters;avoid hiding failures |
| Docker ENOSPC with vfs | Image root copies exhaust disk | Use adequate build/runtime disk or the verified hosted runner;do not prune user data |
| Tiny generated answers nonsense | Very short original-fixture training | Expected engineering limit;prepare real licensed data and independent evaluation |
| All needle results zero | No useful retrieval baseline | Do not claim long-context preservation;evaluate a suitable trained checkpoint |

## Reporting a failure

Include the commit,lock hash,Python/Torch/device/world size,exact sanitized command,
manifest hashes and the relevant log. Keep raw private data,environment dumps,
weights,authorization headers and credentials private. Preserve the original
failure;successful retries do not prove its root cause is fixed.
