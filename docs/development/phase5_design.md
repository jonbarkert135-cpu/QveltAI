# Expanded Phase5 integration design

Date:2026-11-04. Start:b82131c. Review supplied Phase5 WIP,retain current SFT and
original decoder; never replace state files or accept unexecuted hardware claims.
Use existing token-normalized loop and atomic checkpoint engine. Select compatible
WIP modules and three-way merge conflicts against260b431,then test the final code.

## Behavior and interfaces

Strict recipes add cosine/WSD floor,fused AdamW,block compile,spike/z-loss/layer
metrics,source-heldout perplexity,greedy sample logging,nonfinite policies,retention
and auto resume. Current SFT objective,base-weight lineage and train probes remain.
Default fail-stop remains available; the server preset explicitly selects rollback.
Reserve TP/PP only as rejected unsupported settings; FSDP2 requires actual CUDA,
bottom-up wrapping,FP32 master/reduction and DCP. Do not claim GPU acceptance.

Signal handlers only record a flag; all ranks stop and commit at a common completed
window. Hard kill resumes a verified committed state and replaces incomplete writes
through quarantine. Corrupt bytes may fall back with loud logging; semantic/source/
recipe mismatch still fails closed. Validate every checkpoint/quarantine path against
symlinks. Retention is explicit/default off and never deletes quarantine evidence.

Complete backward synchronization even in recoverable nonfinite windows so DDP/
FSDP cannot retain unfinished reducer state. Discard gradients and optimizer updates
on bad windows. Rollback restores finite model/Adam/RNG from the last checkpoint,
retains the forward data cursor and counts skipped targets. Save an initial step0
checkpoint for rollback before the first update. Log scheduled windows separately
from actual optimizer updates; fatal errors and unsupported hardware are not hidden.
Check persistent state before committing it. Version the expanded application state.

Keep the established ADR0036 LR-at-window-end convention rather than silently
changing historical update math. Document its deviation from RFC boundary splitting:
warmup/tail boundaries remain window-based. At the final horizon,mask a globally
rank-ordered prefix of the last window to end at exactly D supervised targets,
consuming but not supervising its remainder;no future update uses that remainder. Sweeps match
nominal capacity,not an invented claim of identical valid targets/order across batch
sizes. Production data rights/exclusions and real resource validation remain gates.

## Required acceptance

Independent schedule/z-loss/gradient references; recovery/rollback/finite-state,
signal,checkpoint corruption and multi-rank tests; all existing SFT/model tests;
actual200-update full RFC tiny(19,338,560 parameters) CPU process SIGKILL/resume
versus continuous loss curve and model/Adam/RNG/cursor comparison. Use only original
registered source fixtures and a real32768-vocabulary tokenizer,short sequence
length for bounded CPU work; never relabel a reduced model as RFC tiny.

Add a bounded compile benchmark and four-size original-data scaling diagnostic
with an explicitly descriptive log-log fit. Generate and validate LR/batch sweep
commands; do not run small/full server training. Supply runbooks/02_pretraining.md
with exact conditional commands,healthy-trend diagnostics and a reviewed logs-only
archive workflow. Publish verified work/evidence to main. GPU/NCCL/FSDP/multi-node
validation remains unverified without hardware; no paid/final-scale run.
