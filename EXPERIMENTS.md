# Qvelt experiment register

Last updated: 2026-10-31. Owner: Lead agent.

**Completed ML experiments: 0. Training runs: 0. Benchmark results: 0.**
The initial session creates documentation only. Repository inspection and
document checks are engineering verification, not ML experiments; their evidence
belongs in `STATE.md`.

## Required record for an actual experiment

Create an entry when a command is actually run. Before launch, specify its
hypothesis, control, measurement, and affordable acceptance gate. Record:

- ID, date, owner, status (`planned`, `running`, `completed`, `failed`, or
  `inconclusive`), and linked milestone/decision.
- Hypothesis and what outcome would refute it; baseline/control where relevant.
- Exact code revision plus dirty-tree information, resolved config and hash,
  dependency versions, seed(s), device/precision/world size, and data/tokenizer
  revisions, licenses, and hashes.
- Working directory, complete command, expected steps/tokens/sample counts,
  resource limits, and log/artifact paths with checksums. Keep secrets out.
- Actual exit code, steps/tokens completed, test counts, failures/skips,
  measured metrics, uncertainty, and resource use when measured.
- Conclusion, limitations, reproducibility command, and next action. Retain
  negative results and interruptions; an interrupted run is not a completed run.

Do not use illustrative numbers as results or describe a runbook as an executed
experiment. Logs stored outside Git must have a persistent, documented location
before they are cited as release evidence.

## Queued experiments — not executed

| Queue ID | Hypothesis / purpose | Configuration needed | Acceptance | Dependency | Status |
| --- | --- | --- | --- | --- | --- |
| Q01 | A tiny portable decoder matches an independent numerical reference | M1.1 design, tiny FP32 CPU tensors, fixed seed, no external data | Forward/gradient errors within justified tolerances and causal tests pass | M4.1 | planned; no executable config yet |
| Q02 | The tiny training loop learns its deterministic fixture and resumes correctly | Tiny CPU training config, authored fixture, complete checkpoint state | Real updates, predeclared loss reduction criterion, matching next samples and supported update replay | M5.1 | planned; no executable config yet |
| Q03 | Two CPU ranks produce the reference global-token-weighted update | Fixed effective batch, unequal valid-token masks, Gloo, two processes | Distributed and single-process reference updates agree within justified tolerances | M5.2 | planned; no executable config yet |
| Q04 | The SFT loop learns only the intended assistant target tokens | Tiny CPU SFT config, authored chat/math fixture | Loss-mask oracle and real tiny update/resume checks pass | M6.1 | planned; no executable config yet |
| Q05 | The reasoning-RL loop can perform a correct bounded-reward update | Reviewed objective, tiny rollout config, authored math fixture, bounded verifier | Objective/gradient tests and actual rollout/update pass, including degenerate rewards | M6.3 | planned; no executable config yet |

Numerical thresholds and actual step counts must be chosen from the implemented
reference contract before these experiments run. None of these checks, even if
successful, would establish general mathematical reasoning quality.
