# Expanded Phase5 acceptance

Date:2026-11-04. Start:b82131c. Selectively integrated supplied WIP against260b431;
current chat/SFT,base-weight lineage and source/software/data trust checks retained.
No GPU,paid compute,external teacher or full small-model run was performed.

## Implemented behavior

Token-normalized gradient accumulation,FP32 loss/norm numerics,BF16 autocast,
clipping,no-decay norms/bias/embeddings,AdamW fused opt-in,WSD/cosine floors and
strict unsupported TP/PP guards. Single/DDP compile operates per decoder block;
FSDP2 is a bottom-up CUDA-only API path with FP32 master/reduction,BF16 gathered
compute and sharded DCP. Compiled FSDP is rejected. Two actual-GPU FP32/BF16
packed replay tests exist but cannot run on this CPU host.

Auto-resume verifies committed artifact bytes and semantic compatibility,quarantines
incomplete/corrupt snapshots and retains explicit checkpoint counts. SIGTERM records
a flag;all ranks checkpoint a completed window. NaN/Inf loss/gradient windows finish
backward synchronization before discarding gradients. Model/Adam state is also checked
after updates. Rollback restores finite model/Adam/RNG but retains the advanced cursor;
good updates since that snapshot can be discarded. Failure receipts,counters and
actual optimizer_updates make this data-skip policy explicit. Initial step0 rollback
DCP initialization preserves Adam time0,with independent first-update parity test.

Optional vocabulary z-loss,spike detector,layer statistics,source-heldout CE/PPL,
greedy samples,tokens/s/device and conditional analytical MFU are integrated.
Logs-only archives use an explicit allowlist;bounded sweeps use argv without a shell.
The real-data factory fills actual checksums,checks V/T and derives accumulation
after batch overrides. The small recipe has125,854,464 actual parameters;the12-point
LR/batch/two-seed sweep was expanded against real32768/T2048 authored demo shards
without launching any optimizer/server job. Demo inputs remain blocked for production.

## Actual hard-kill experiment

Python3.12.14,Torch2.9.1+cpu,FP32,seed20261031,one thread. Original registered
D-TRAIN-200 data:96 own web-style/code/math documents plus seeded nonce payloads,
train-only BPE32768,T16,B2,full RFC tiny19,338,560 parameters,L8/d320.
The nonce is a tokenizer-cardinality fixture,not a representative training corpus.
Two independent process histories each perform200 real optimizer updates and6386
valid targets. The interrupted subprocess receives actual SIGKILL after observing
step89(returncode-9),resumes from the last committed step75 and reaches200.
All200 losses match with maximum absolute difference0.0;final weights,Adam,RNG and
application/loader state are bit-exact. Fixed probe CE10.47552204->9.00020885;
final code/math/web PPL3163.41/2402.49/2653.38 are fixture harness outputs,not public
benchmarks or mathematical-answer accuracy. See
[evidence](evidence/phase5-kill-resume-200.json).

After compiler/provenance correction,the identical protocol is rerunning against
source4202244ee1e28ac839d934600bbb70b7c5c2f595c0202ebc86664777b2e2572a;
final-source receipt will replace the earlier receipt before acceptance closes.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/pretraining_acceptance.py --directory runs/tiny-kill-200 --repository .
```

## Measured compile and four-size controls

Eight-step actual CPU optimizer benchmark,V445/T64,962 valid targets each:
eager median steady11,589.89 versus compiled13,509.28 targets/s/device;
wall0.811s versus6.386s including initialization/checkpoints/compilation. Existing
Inductor cache was retained from a preceding failed experiment;these are cached
startup measurements,not cold compilation cost. Maximum final weight difference
7.445e-6,within declared rtol1e-3/atol2e-5;fixed heldout loss5.874644711 for both.
Shared-host,one seed,eight steps:do not infer a robust speed ranking or GPU gain.
MFU is null without a named accelerator peak. The first attempt caught a real
checkpoint provenance bug:cpp_extension imports setuptools,which appends bundled
platformdirs4.4.0 metadata alongside installed4.14.2. Capture excludes only that
exact setuptools bundled path;real installed-version conflicts still fail. A fresh
subprocess regression verifies inventory equality before/after compiler import.

Four20-update models,identical2296-target exposure and LR0.001:13,344/38,976/
76,896/127,104 parameters;heldout losses5.991178/5.803633/5.660614/5.567677.
Descriptive fit log(loss)=2.10195744-0.03270178*log(P),R²0.99782621.
No positive-slope flag in this fixture. Four short points do not establish a
universal scaling law or hyperparameter transfer to the125.85M small model.

## Tests and remaining gates

Local pre-final regression586 passed,3 distributed-process failures/timeouts,
7 GPU deselected in365.00s. This is not an all-green local suite. Such Gloo exits/
stalls were also observed before this import;root cause remains unresolved.
No test is disabled,xfail'ed or replaced by a forced process exit. Focused final
recovery/reference/provenance tests and current-source GitHub CPU3.11/3.12/3.13,
optional SDK and dedicated200-update recovery jobs are pending observation.
Ruff/mypy/build results will be recorded with the final source revision.

RFC deviations are explicit in ADR0036/0042:LR is evaluated at completed-window
end rather than splitting warmup/tail boundaries;final target horizon is exact;
mixture phase remains loader-microbatch based;skips advance exposure;batch sweeps
match nominal capacity only. Hardware-resource/production-source acceptance remains
blocked. Exact commands,healthy trend interpretation,instability playbook and manual
reviewed-log attachment are in[runbook02](../../runbooks/02_pretraining.md).
