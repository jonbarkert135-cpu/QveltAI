# Training and recovery

`qvelt.train.pretraining` owns the packed-data loop,accumulation,precision,AdamW,
token-count schedules,gradient clipping,validation,stability diagnostics and
checkpoint consumer state. Full SFT reuses it with assistant-only target masks.

The checkpoint manifest is a commit marker. Incomplete writes are quarantined;
auto-resume chooses the latest valid committed checkpoint. Resume restores model,
optimizer,schedule,RNG and data cursor,with strict config/source/library/topology
checks. SIGTERM saves cooperatively;SIGKILL recovery uses an earlier checkpoint.
Do not edit runtime Python while a resumable experiment is running.

Single-process CPU and actual two-rank CPU correctness have executable evidence.
GPU/NCCL/FSDP2 and real-server memory/throughput require hardware acceptance.
The CUDA profile's new security-updated runtime must be checked on the intended
driver before a pilot;old CUDA success is not inferred.

Read the [first small-run commands](../reference/runbooks/02_pretraining.md),
[training RFC](../reference/docs/research/training_rfc.md) and
[version compatibility policy](../reference/docs/development/release_policy.md).
MFU requires an identified hardware peak and is an estimate,not measured utilization.
