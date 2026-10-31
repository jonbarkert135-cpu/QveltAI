# Phase 1 risks and excluded work

Date: 2026-10-31. Owner: Lead agent. Supplements the root `RISKS.md`.
Likelihood/impact assessments are engineering judgments, not measurements.

| Risk | Exposure | Mitigation / acceptance evidence |
| --- | --- | --- |
| Laptop VRAM smaller or less usable than assumed | High impact; no local inventory | Verify actual memory/precision support, use tiny and short contexts, measure peaks and free margin; do not sell 1.5B pretraining as a 3060 workflow |
| 16 GB host RAM, CPU/IO and laptop thermal/power limits | Likely throughput constraint | Stream/map shards, 0–2 workers initially, bounded caches, measure sustained tokens/s and wall power; no TB corpora on laptop |
| Parameter math correct but implementation differs | Open until M4.1 | Unique-parameter storage count on actual model; match formulas and tied optimizer groups, including final norm |
| GFLOP/time/memory estimates mistaken for results | High impact | Named throughput/price assumptions, excluded overheads, pilot calibration; never promise fit or vendor pricing from formulas |
| GQA loses quality at small scale | Unknown | MHA and grouping-ratio ablations with matching budgets and KV/memory measurements |
| Long-context, masks, packing or initialization fail | Open | Independent forward/gradient tests, perturb-future tests, packed-boundary and cache parity tests; no untested extension |
| Attention/logit growth, loss spikes, non-finite moments | Open | Log diagnostics, fixed replay, clip after global normalization; isolate QK-norm and z-loss ablations before adoption |
| Reported dataset license fails to cover underlying material | High impact; unresolved candidates | Record exact license texts, revisions and upstream obligations; no actual ingestion approval from wrapper metadata |
| Code-data slot has no approved corpus | Current recipe dependency | Resolve per-file rights or run explicitly renamed no-code ablation; do not silently import gated code |
| Math oversampling and synthetic errors | Likely at proposed large horizon | Source-capacity/repetition logs, exact-answer checks, concise-trace controls, teacher and source provenance, diversity/family caps |
| Benchmarks leak through web, translations or derivatives | High impact | Freeze IDs/families first, cross-source approximate checks and residual uncertainty; self-authored out-of-family tests |
| RL zero-signal groups, length bias or verifier exploitation | High impact | Logged zero-variance handling, response caps, adversarial verifier tests, no style bonus, fixed-budget held-out comparisons |
| Tool/CAS execution or autoformalization trust errors | High impact | No in-process generated execution; isolated bounded tools and pinned Lean track only after review; proof checks do not verify NL translation |
| Distributed normalization, shard replay or partial-update restart is wrong | Open | Single-process oracle and two-rank CPU tests, then actual GPU/multi-node tests; topology migration explicitly separate |
| Current sources do not cover every 2026 development | Known scope limit | Publish inspected-source dates/versions; update survey before expensive decisions, not an unsupported SOTA claim |
| Project/data release license, branding and paid compute not decided | External decision | Keep work local under current publication pause; no paid runs or final training; ask only when required by a concrete release/run |

## What we will NOT do in the baseline, and why

1. **No MoE/MLA baseline:** uncertain small-scale payoff and extra routing/cache/
   optimizer complexity without hardware evidence [S10,S14,S29].
2. **No FP8, custom CUDA kernels or unconditional compile dependency:** first
   establish FP32/BF16 correctness and laptop-compatible paths [S29,S36,S46].
3. **No advertised long-context extrapolation or sliding-window substitution:**
   train/evaluate the chosen context first; distant math evidence can be lost
   and RoPE changes require dedicated validation [S03–S05,S11,S33].
4. **No automatic muP/optimizer superiority claim:** AdamW reference first;
   Muon/Lion and transfer have real evidence but require matched tests and
   version-compatible state/sharding support [S17–S20].
5. **No tensor/pipeline/expert parallelism requirement at <=1.5B:** DDP/FSDP
   address our first memory/scaling problems with fewer moving parts. Revisit
   against actual server/interconnect and larger model goals.
6. **No copying pretrained architecture/trainer code or importing weights:**
   use source-attributed ideas and original tested implementation.
7. **No corpus approval from a README license badge alone, no gated-data bypass,
   no unauthorized scraping:** rights, provenance and actual access are gates.
8. **No expensive teacher APIs, final-scale pretraining, or paid infrastructure:**
   current work is design; laptop pilots precede explicit hardware/budget decisions.
9. **No process-reward model, autonomous tool execution or Lean baseline:**
   these are separate optional tracks with annotation/safety/toolchain costs.
10. **No claim that correct final answers prove valid reasoning or that more
    reasoning tokens mean better reasoning:** compare fixed-budget, held-out
    outcomes and audit traces [S23–S28].
11. **No default CPU offload as a promise of large training on 16 GB RAM:**
    memory/IO costs remain real; provide separate inference experiments later.
12. **No GitHub or environment publication in this session:** the user's latest
    publication instruction pauses it. Local commits preserve reviewable work.

Revisit exclusions through an ADR plus evidence, not hidden configuration flags.
None prevents building interfaces that can later support a larger server.
