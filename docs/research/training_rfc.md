# RFC-002: Qvelt training and validation recipe

Date: 2026-10-31. Status: proposed numerical starting points; implementation and
empirical tuning pending. Owner: Lead agent. No training run is authorized or
claimed by this RFC. Sources are verified in `survey.md`; derived numbers are
reproducible with `estimates.py` and `estimates.json`.

## Hardware policy

Current user hardware: ASUS TUF Gaming F15, RTX 3060 Laptop, Core i5, 16 GB RAM.
Exact VRAM/power limit/OS are not supplied. Assume 6 GiB only for planning; check
`nvidia-smi --query-gpu=name,memory.total --format=csv` locally. Check selected
PyTorch/CUDA versions, `torch.cuda.is_bf16_supported()`, actual kernel support,
and allocator peak in a future dry run. No PyTorch API has been executed here.
CPU FP32 fixtures remain mandatory; unsupported BF16 must not be emulated and
presented as a GPU performance path. A separately validated FP16+scaler mode
can follow, with scaler state checkpointed; it is not the default fallback.

Notebook workflow: tiny at T=32–128 for debugging, then 512 if it fits; small
at T=256–512/B=1 only after profiling. Keep tokenizer/shard processing bounded,
0 workers initially, CPU thread counts measured against the actual i5, and no
simultaneous full-corpus preprocessing. 16 GB RAM excludes careless dataset
materialization and is not a strategy for offloading a large optimizer.

## 1. Token budgets and optimizer settings

Budgets count **processed valid next-token targets**, including repeats and
the annealing tail, not raw bytes or unique source tokens. Padding/ignored
targets do not advance LR. Tiny is a debug recipe, not compute-optimal pretraining.

| Size | Initial compute-reference gate ~20P tokens | Extended target D | Main / anneal targets | Global valid tokens/update | Peak LR | Warmup targets | Approx. updates to D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 0.387B (reference only) | 0.010B | 9M / 1M | 8192 | 1e-3 | 81,920 | 1221 |
| small | 2.517B | 10B | 9B / 1B | 131,072 | 6e-4 | 100M | 76,294 |
| base | 9.871B | 50B | 45B / 5B | 524,288 | 3e-4 | 500M | 95,368 |
| large | 30.202B | 150B | 135B / 15B | 1,048,576 | 2e-4 | 1.5B | 143,052 |

The ~20P column is a Chinchilla-inspired **decision gate**, not an instruction
to stop then resume a fully decayed schedule indiscriminately. Preserve stable
checkpoints and branch decay experiments. The extended sizes are ~79/101/99
tokens per parameter, inspired by inference-friendly overtraining [S13,S30,S35],
not a scaling-law prediction of benchmark quality. Do not attempt these final
budgets on the laptop as the first experiment. Even small's full budget could
take many weeks/months under the illustrative assumptions below.

AdamW baseline [S18,S13]: beta1=0.9, beta2=0.95, eps=1e-8; decoupled weight
decay 0.1 on block projection matrices; no decay on norms or tied embedding.
The tied embedding must appear exactly once in optimizer groups. Excluding it
from decay is a project choice also motivated by stability practice in OLMo 2
[S42], not a theorem. Global gradient clipping at L2 norm 1.0 **after** correct
global-token normalization/accumulation and before optimizer step. Log unclipped
norm and clipping rate. Clipping must count each sharded parameter exactly once.

Peak LRs/batches are starting hypotheses; use fixed-data, fixed-token sweeps
(0.5x,1x,2x LR) and at least two seeds before promoting a size. Never transfer
an LR blindly from tiny when depth, tokenizer, data quality, or batch changes.
No muP scaling is claimed [S17]. Batch ablations should measure validation loss,
gradient noise proxies, throughput, and time-to-target, not just utilization [S38].

## 2. Schedule and mixture transitions

Default: WSD-inspired [S16,S30] with explicit horizon D, warmup w, tail a=0.1D,
and u=valid targets already consumed before the update:

    lr(u) = peak * min(1, (u + current_update_targets)/w)    during warmup
            peak                                         w <= u < D-a
            peak * max(0, (D-u)/a)                       in tail

Use token-boundary-aware integration/splitting at phase boundaries: either split
the update at a boundary or record a deterministic rule evaluating LR at its
start. Chosen baseline: split updates at w, D-a, D; apply the resulting smaller
valid-token count without reweighting it to a full nominal batch. Do not mix
rank-local boundary decisions. The next update after u=D is forbidden.
For warmup, evaluate at its end as written, then use start-of-update LR elsewhere.
Record this convention; small rounding differences must not silently change replay.

The annealing mixture in `data_mix.md` starts at D-a; checkpoint the phase and
each source cursor. Extend only from stable checkpoints, not by relabeling a
decayed optimizer. A budget change creates a new resolved config/run lineage.

Cosine control [S13], same D/w, after warmup:

    lr(u) = peak * [0.1 + 0.45*(1 + cos(pi*(u-w)/(D-w)))]

This ends at 0.1*peak, whereas the default WSD tail ends at zero. State the
endpoint difference in results; add matched-zero-endpoint cosine when isolating
shape alone. For schedule-only tests hold the **mixture transition** identical;
for mixture-only tests hold the schedule fixed. Optional adaptive schedules are
out of scope until a fixed deterministic recipe is correct.

## 3. Loss, precision, data order, and replay

Use the architecture RFC's sum/count masked causal loss, no label smoothing.
Optional vocabulary z-loss=lambda*mean_valid(logsumexp(logits)^2), lambda=0
by default; 1e-5 is a stability ablation [S42]. Keep CE and z-loss separately
logged. QK-norm is another independent ablation. Do not change both and claim
one caused a gain. Preserve non-finite batches/artifact identity for diagnosis;
abort coherently across ranks and resume a known-good checkpoint when required.

Store weights/gradients/Adam moments FP32; autocast compatible GPU matmuls to
BF16; FP32 norm and loss/softmax accumulation. Implement optimizer updates in
FP32, without an unnecessary duplicate master model. Validate precision paths
against FP32 references. No FP8 baseline or unconditional fused-kernel requirement
[S29,S36,S46]. Application pins are selected in M2.1, not by a paper's version.

Deterministic sampler contract: seed=20261031, immutable manifest, source order,
shard permutation, document cursor, epoch/repetition count, mixture phase and
packing policy. Derive per-worker/rank RNG streams from explicit indices;
serialize every relevant Python/NumPy/PyTorch CPU/device RNG when used. Do not
rely on arrival order from asynchronous workers. Hold-out assignment precedes
sampling and duplicate removal across train/eval sources. Split chunks from the
same document/problem family cannot cross train/validation/test.

Checkpoint only at completed optimizer-update boundaries for version 1. Save
weights, optimizer, LR/token counters, resolved config/hash, software/environment,
tokenizer/data hashes, sampler/cursors, RNG states, world-size/rank identities,
and scaler state if a future FP16 mode uses one. No partially accumulated
gradients are supported; interrupted partial updates replay from the previous
boundary. Atomic manifest commit + artifact checksums precede marking a checkpoint
valid. Optimizer checkpoints are trusted artifacts, not arbitrary pickle uploads.

Initial exact replay support is same software/device/precision/world size and
sampler topology. Resuming across a different topology is an explicit migration
that must define new rank allocation and revalidate equivalence; no bitwise
cross-device guarantee (root ADR-0003). Use fixed-window failure/restart tests
and compare next sample IDs, LR, weights and optimizer states, with justified
tolerances for non-bitwise numerical paths.

## 4. Parallelism strategy

| Hardware / size | Default | Reason and required validation |
| --- | --- | --- |
| Laptop CPU/tiny | Single process, small T/B, FP32 | Correctness reference and short learning/replay fixture; no final-model training |
| Laptop RTX 3060/tiny | Single GPU, B=1 then measured increase, accumulation | Portable first GPU path; exact VRAM/precision support must be measured |
| Laptop/small | B=1,T=256–512 pilot, checkpoint blocks | Possible planning fit, not guaranteed; lower context is a different training config |
| Laptop/base or large | No full-state GPU pretraining baseline | State/activation requirements exceed assumed VRAM; do not promise practical RAM offload |
| One hypothetical 24 GiB GPU | small/base, checkpointing, accumulation | Profile before full horizon; large's baseline estimate is ~28.6 GiB |
| One 4–8 GPU node, sufficient memory | DDP at first, fixed effective token batch | Simple reproducible baseline; two CPU ranks then real CUDA tests |
| Memory-constrained multi-GPU / future bigger sizes | FSDP full parameter/gradient/optimizer sharding | Save/load, global clipping, tied embedding, prefetch/gather dtypes and rank weighting need actual tests |
| Multi-node | DDP or FSDP depending on memory/interconnect | Rendezvous, collectives, IO, timeouts, restart and networking measured before any scale claim |

Tensor/pipeline parallelism adds partitioning/communication/pipeline bubbles;
not justified as a v1 requirement for <=1.5B on 80 GiB devices. Preserve clean
module boundaries for later work, not untested configuration flags. CPU optimizer
offload is not a default laptop solution. FSDP is a memory tool, not a guaranteed
throughput win. Recheck API behavior for the selected PyTorch release; this RFC
does not select an unverified FSDP2 or third-party launcher API.

For DDP, if each rank sums token losses S_r over accumulation and DDP averages
gradients across n ranks, multiply accumulated gradients by n/sum_r(valid_r).
Perform one optimizer update per synchronized window. For a sharded backend,
verify its reduction convention rather than assuming the same factor. All ranks
must agree on window boundaries, skips, phase transitions, and final partial
batches. Compare uneven masks and accumulation with a one-process global batch.

## 5. Time and cost estimates — assumptions, not measurements

No training hardware benchmark has run. Let F be `estimates.json` FLOPs/token,
D the token horizon and R the **assumed sustained model-operation rate**:

    ideal_hours = F*D / (R_TFLOPs * 1e12 * 3600)
    planning_hours = ideal_hours * factor, factor in [1.5,3]
    cost = planning_hours * GPU_count * price_per_GPU_hour

F excludes recomputation/optimizer/IO; the factor provisionally covers these
and launch/communication effects, not a verified bound. Rates are scenario
inputs, **not specifications or measured GPU performance**. They may be
unattainable for small shapes, a thermally limited laptop, or poor interconnect.

| Size / full target | Ideal hours, assumed laptop 2–8 TF/s | Ideal hours, 24 GiB scenario 30 TF/s | Ideal hours, 8x80 GiB scenario total 800 TF/s | Server planning hours (1.5–3x) | Hypothetical server cost at $4/GPU-hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny, 10M | 0.046–0.183 | 0.012 | 0.00046 | Not useful: launch dominates, use 1 device | Not meaningful |
| small, 10B | 367–1468 | 97.867 | 3.670 | 5.5–11.0 | $176–352 |
| base, 50B | Not feasible under assumed laptop GPU memory | 2128.100 | 79.804 | 119.7–239.4 | $3,830–7,661 |
| large, 150B | Not feasible under assumed laptop GPU memory | Not baseline-fit on 24 GiB | 639.631 | 959.4–1918.9 | $30,702–61,404 |

The price is an illustrative variable, **not a current vendor quote**. The small
model may scale particularly poorly to eight GPUs, making the server estimate
optimistic. At 2–8 TF/s plus 1.5–3x factor, small's full default horizon spans
roughly 23–184 continuous days on a laptop if it fits; this is why the laptop
plan uses short pilots, not a promise of a competitive 125M model in hours.
Laptop electricity can be estimated as average wall-power kW * hours * local
tariff; all three are currently unmeasured. Include storage, preprocessing,
rejected/ablation runs, checkpoints and teacher/rollout costs separately. No
paid infrastructure has been acquired. Calibrate actual tokens/s and peak memory
with a 100–1000-update pilot before committing any full budget.

## 6. Post-training starting recipe and math evaluation contract

Only begin after a useful pretrained checkpoint and approved sources exist.

- **SFT:** one deterministic epoch initially, cap at 100M assistant target tokens
  for a pilot, max T=2048 (laptop shortened), effective 32 examples/update with
  token-normalized assistant loss; start LR=2e-5 (small/base), 1e-5 (large),
  3% warmup, cosine to 0.1x, weight decay=0, clip=1. These are hypotheses, not
  copied optimal settings. Compare concise worked solutions and answer-only data.
- **Rejection sampling:** generate at most four bounded traces per training
  problem initially; retain answer-verified, deduplicated traces with provenance.
  Never filter using held-out answers. Track acceptance and every sampling token.
- **DPO:** only for defensible preference pairs [S40]; start beta=0.1,
  LR=5e-6, 32 pairs/update, up to 1000 pilot updates; freeze/reference-hash the
  SFT model. Precompute reference log-probabilities only when pair/tokenizer/
  mask identity is immutable. This can save a second resident reference model
  on the laptop; it does not eliminate policy activation memory.
- **Reasoning RL:** defer until SFT passes verifier-based held-out checks. A
  GRPO-style [S26] pilot uses G=4 responses/prompt, max 512 generated tokens
  initially (increase only after capacity/correctness tests), four prompts per
  update through microbatches, LR=1e-6, ratio clip=0.2, KL coefficient=0.01,
  <=100 pilot updates. Freeze rollout policy per update and log old token
  probabilities; reference policy is fixed to the SFT checkpoint.

For the RL pilot use reward=1 only for a correctly parsed, verifier-correct
terminal answer; malformed/truncated responses get zero, no style/length bonus.
Population group std with epsilon=1e-6 defines A_i=(r_i-mean(r))/(std(r)+epsilon).
Drop zero-variance groups from the pilot update; if none remain, do not advance
optimizer/schedule, but count generated tokens and log the no-signal event.
Do not hide this by repeatedly regenerating until success. Review curriculum
instead if the model cannot obtain positive rewards.

Per-response, per-token clipped objective averages min(ratio*A,clip(ratio)*A)
over valid response tokens, then equally over retained responses, subtracting
beta_KL*k. With delta=log(pi_ref)-log(pi_theta), k=exp(delta)-delta-1 is the
sampled KL estimator used in [S26]; with old-policy sampling its finite-sample
value is not exact full-distribution KL. Compute safely in FP32, monitor
non-finite ratios/KL and fail rather than silently clip overflow. Length
normalization and group filtering are explicit project choices with bias risks,
not claims that GRPO universally improves reasoning. All objective/gradient
reference tests and frozen-policy identity checks precede actual rollout training.
RL memory/time is **not included** in pretraining tables; include generation,
reference scoring, repeated response passes and KV/optimizer memory separately.

Evaluation contract before any promotion:

1. Reserve a project-authored fixed set of 1000 arithmetic/rational/algebra
   problems: 200 validation/800 final test, generator seed/family IDs committed
   before learning. Train-generator templates/parameter families are disjoint;
   additionally evaluate out-of-family cases. These problems do not yet exist.
2. Frozen zero-shot prompt and answer syntax, greedy pass@1, maximum 512 new
   tokens initially. Report exact rational/numeric scoring (relative 1e-6 plus
   absolute 1e-8 only when floats are necessary), parse failures as incorrect,
   per-domain counts and binomial intervals. Verifier tests cover adversarial
   units, multiple answers, invalid fractions, NaN/inf and resource limits.
3. Do not treat symbolic equality over unrestricted user expressions as safe.
   Use a bounded grammar; isolation/resource review before adding a CAS or tools.
4. Compare base/SFT/DPO/RL with identical prompts/inference budgets. Report
   sampled pass@k only with sample count, seeds, estimator, and total tokens.
   Tool-assisted and Lean evaluations remain separately labeled optional tracks.
5. Public GSM8K/MATH-style benchmarks require exact revision/license approval,
   contamination audit, frozen scorer/prompts and general-language controls;
   they are not approved datasets by being named here. Keep final tests untouched
   by curriculum/schedule selection. No benchmark score is predicted by this RFC.

## Run manifest minimum and promotion gates

Manifest schema must include code revision/dirty state, resolved config/hash,
Python/PyTorch/CUDA/kernel versions, CPU/GPU/memory/world-size inventory, seeds,
tokenizer and data manifests/hashes/licenses, unique/seen/loss-token counts,
precision/backend/determinism settings, actual LR/batch/window semantics,
checkpoints, validation prompts/scorers and artifact locations. Fail closed on
incompatible tokenizer/config/data identities. M2.2 implements versioned schemas.

First prove 2–10 CPU fixture updates and exact-order replay; then a measured
laptop GPU pilot; then a fixed-data small-model learning study; then the ~20P
decision gate; only then consider extended and larger user-operated runs. Real
GPU/distributed smoke tests remain unrun; Phase 1 contains analytical checks only.
