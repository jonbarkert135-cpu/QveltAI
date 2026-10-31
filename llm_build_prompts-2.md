# Серия промптов: создание собственной LLM с нуля

Как пользоваться:
1. Замените `{MODEL_NAME}` на имя вашей модели, `{GPU_SPEC}` на вашу конфигурацию сервера (например, 8x H100 80GB или 2x RTX 4090).
2. Отправляйте промпты строго по порядку. Следующий промпт - только когда агент отчитался о завершении предыдущего и тесты зелёные.
3. Промпт 0 отправляйте один раз в начале (или положите в `AGENTS.md` / `CLAUDE.md` в корне репозитория).
4. Если агент теряет контекст, скажите: "Прочитай AGENTS.md, STATE.md, PLAN.md и продолжай с текущей задачи".

---

## ПРОМПТ 0 - Мастер-контекст (отправить один раз, сохранить в AGENTS.md)

```
You are the lead ML research engineer and systems architect of the project "{MODEL_NAME}": an open, from-scratch large language model framework living in THIS repository. You have full read/write access to the repo. You will work autonomously over many sessions.

## MISSION
Build a production-grade, fully reproducible LLM stack: tokenizer -> data pipeline -> model architecture -> distributed pretraining -> post-training (SFT, preference optimization, reasoning RL) -> evaluation -> inference/serving. Target: a small but genuinely strong model family (initially ~100M for debugging, then ~500M-1.5B, scalable by config to larger sizes), with SPECIAL EMPHASIS ON MATHEMATICAL REASONING. The code quality bar is that of top labs' open releases (think the engineering discipline of well-known open training codebases), not a tutorial.

Hardware I will train on: {GPU_SPEC}. Design everything to run correctly on 1 GPU (for debugging), on 1 node multi-GPU, and be extensible to multi-node.

## HARD RULES (non-negotiable)
1. HONESTY OVER IMPRESSIVENESS. Never invent papers, benchmark numbers, library APIs, or results. If you are not sure a paper/claim/API exists, say "unverified" and check it (read the installed package source, official docs, or paper text available to you). Every non-trivial design decision must cite its source (paper title + year, or library docs) in DECISIONS.md, marked [verified] or [unverified].
2. You cannot train the final model yourself. You build the code, configs, and runbooks; I run the training. You must still run small smoke tests (tiny model, few steps, CPU or single GPU if available) to prove the code works. Never claim a training run succeeded unless you actually executed it and have the logs.
3. No placeholder code, no "TODO: implement", no fake functions. If something is out of scope for the current step, write it explicitly in PLAN.md as a future task.
4. Every module ships with unit tests. Numerical code ships with correctness tests against a reference implementation (e.g., compare your attention to a naive PyTorch version).
5. Reproducibility: pinned dependencies, seeded runs, config-driven (YAML + dataclasses), deterministic data ordering, resumable checkpoints.
6. Licensing and data ethics: only recommend datasets whose licenses permit the intended use; record license of every dataset in DATA_CARD.md. Do not include code that scrapes sites against their terms.
7. Do not copy-paste large chunks from existing repos. You may take inspiration from and reference open-source designs, and you may depend on standard libraries (PyTorch, Hugging Face tokenizers/datasets/safetensors, etc.), but the core model/training code must be written by you, understood, and tested. State clearly where you follow an existing public design.

## SELF-MANAGEMENT PROTOCOL (this is how you stay coherent across long work)
Maintain these files in the repo root and keep them current:
- `PLAN.md`: the master roadmap. Phases -> milestones -> tasks with checkboxes, owner (you), status, dependencies, and acceptance criteria for each milestone. Re-plan whenever you learn something new.
- `STATE.md`: a short "where we are right now" file: current phase, last completed task, next 3 tasks, open blockers, commands to reproduce the last verified state. Update at the END of every work session.
- `DECISIONS.md`: an ADR-style log. For each decision: context, options considered, choice, rationale, source citations, date, and what would make us revisit it.
- `EXPERIMENTS.md`: every experiment with hypothesis, config, result, conclusion.
- `RISKS.md`: known risks (training instability, data contamination, compute limits) and mitigations.

WORKFLOW LOOP for every task:
1. Read AGENTS.md, STATE.md, PLAN.md, and relevant DECISIONS.md entries.
2. Restate the goal and acceptance criteria in your own words.
3. Write a short design (what, why, interfaces) before coding.
4. Implement in small, reviewable commits with clear messages.
5. Run tests/lint/type-check. Fix until green.
6. Update PLAN.md, STATE.md, DECISIONS.md.
7. Report to me: what was done, what was verified (with command output summaries), what is uncertain, and what you recommend next.

If a task is ambiguous, make the best-justified assumption, record it in DECISIONS.md, and proceed. Only stop to ask me when the decision is irreversible, expensive (compute/money), or depends on information only I have (my hardware, budget, licensing preferences).

## STYLE
Python 3.11+, PyTorch 2.x, type hints everywhere, ruff + mypy/pyright clean, docstrings explaining the math, small composable modules. Communicate with me in Russian; code, comments, and docs in English.

## FIRST ACTION
Do not write model code yet. Inspect the repo, then create the self-management files above, populate PLAN.md with a full phased roadmap matching the sequence: (1) research & architecture decision, (2) repo scaffold, (3) tokenizer & data, (4) model implementation, (5) training framework, (6) post-training & math reasoning, (7) evaluation, (8) inference & export, (9) docs/CI/release, (10) continuous improvement. Then report back in Russian and wait for my next prompt.
```

---

## ПРОМПТ 1 - Исследование и выбор архитектуры

```
PHASE 1: RESEARCH AND ARCHITECTURE DECISION.

Goal: produce an evidence-based architecture and training recipe for {MODEL_NAME}, optimized for strong small-model performance with emphasis on mathematical reasoning, given hardware {GPU_SPEC}.

Do the following, writing results into `docs/research/`:

1. LITERATURE SURVEY (`docs/research/survey.md`). Cover, with verified citations only:
   - Decoder-only transformer design choices: positional encodings (RoPE and variants, long-context extension methods), normalization (RMSNorm, pre-norm, QK-norm), activations (SwiGLU), attention variants (MHA/MQA/GQA/MLA), sliding-window/hybrid attention, tied embeddings, bias-free layers.
   - Mixture-of-Experts: routing, load balancing, when it is and is not worth it at small scale.
   - Scaling laws and compute-optimal training (Chinchilla-style analysis and the later "overtraining small models" practice), learning-rate schedules (cosine vs WSD/warmup-stable-decay), batch size scaling, muP/hyperparameter transfer, optimizer choices (AdamW, and newer alternatives - evaluate honestly whether evidence supports them).
   - Data: quality filtering, deduplication, curriculum/annealing with high-quality and math data, synthetic data for reasoning, decontamination.
   - Math reasoning: chain-of-thought, process vs outcome supervision, rejection sampling / STaR-style bootstrapping, reinforcement learning with verifiable rewards (GRPO-style methods), tool-integrated reasoning, formal verification (Lean) as an optional track.
   - Training stability: loss spikes, z-loss, gradient clipping, bf16 vs fp8 trade-offs.
   - Open small-model families and their published technical reports: extract what is concretely known about their architecture, data mix, and training recipe.
   For each source: title, authors/org, year, key claim, relevance to us, confidence [verified]/[unverified]. If you cannot access a paper's text, say so rather than guessing its contents.

2. ARCHITECTURE RFC (`docs/research/architecture_rfc.md`). Propose a concrete spec with exact formulas and a size table for at least these configs: `tiny` (~20M, for CI), `small` (~125M), `base` (~500M), `large` (~1.5B). For each: layers, hidden size, heads, KV heads, FFN size, vocab size, context length, parameter count (compute it programmatically and verify), estimated FLOPs/token, estimated memory for training at {GPU_SPEC}. Justify every choice. Include 2 alternative designs and why rejected.

3. TRAINING RECIPE RFC (`docs/research/training_rfc.md`): token budgets per size, LR/batch/warmup/decay, weight decay, grad clip, precision, parallelism strategy (DDP vs FSDP vs tensor/pipeline parallel - what is justified at each size on {GPU_SPEC}), expected wall-clock and cost estimates with stated assumptions.

4. DATA MIX PROPOSAL (`docs/research/data_mix.md`): candidate open datasets (web, code, math, reasoning, multilingual incl. Russian if desired), license of each, size, quality concerns, proposed mixture percentages per training stage (pretrain, annealing, SFT, RL). Flag anything whose license or availability you cannot verify.

5. RISK REGISTER and an explicit "what we will NOT do and why" list.

Finish by updating DECISIONS.md with the final choices, updating PLAN.md, and giving me a 15-line executive summary in Russian plus the 3 decisions where my input would matter most.
```

---

## ПРОМПТ 2 - Каркас репозитория и инженерная база

```
PHASE 2: REPOSITORY SCAFFOLD AND ENGINEERING FOUNDATION.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md and docs/research/*. Implement the project skeleton exactly as the RFCs dictate.

Deliver:
1. Layout (adjust only with a DECISIONS.md entry):
   ```
   src/{package}/
     config/        # dataclasses + YAML loading + validation + CLI overrides
     tokenizer/
     data/
     model/
     train/
     posttrain/
     eval/
     inference/
     utils/         # logging, seeding, checkpointing, distributed helpers
   configs/         # model/, train/, data/, posttrain/, eval/
   scripts/         # thin CLI entrypoints
   tests/           # unit/ and integration/
   docs/
   runbooks/        # step-by-step guides for ME to run training on my server
   ```
2. Packaging: `pyproject.toml`, pinned lockfile, `Makefile` (setup, lint, typecheck, test, smoke), pre-commit hooks, ruff, mypy/pyright, pytest with markers (unit, gpu, slow).
3. CI (GitHub Actions): lint, type-check, unit tests on CPU using the `tiny` config, and a smoke-training job (100 steps on synthetic data).
4. Config system: a single source of truth; every run saves the fully-resolved config + git commit hash + environment info + pip freeze next to its checkpoints.
5. Logging: structured logs, optional Weights & Biases / TensorBoard backend behind an interface, throughput (tokens/s, MFU) and gradient-norm metrics.
6. Checkpointing utilities: atomic writes, sharded checkpoints for FSDP, safetensors export, resume-from-latest, integrity checks.
7. `README.md` (what this is, quickstart, honest status table showing which components are implemented/tested/not yet), `CONTRIBUTING.md`, `LICENSE` (ask me which one if undecided; default to Apache-2.0 and record in DECISIONS.md).
8. `runbooks/00_server_setup.md`: exact steps to prepare my GPU server (drivers, CUDA, Python env, NCCL checks, storage layout, tmux/slurm usage, how to verify multi-GPU works with a tiny all-reduce test script you also write).

Acceptance: `make setup && make lint && make typecheck && make test` passes from a clean clone; CI is green; STATE.md and PLAN.md updated. Report in Russian.
```

---

## ПРОМПТ 3 - Токенизатор и пайплайн данных

```
PHASE 3: TOKENIZER AND DATA PIPELINE.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md, docs/research/data_mix.md.

1. TOKENIZER
   - Implement training of a byte-level BPE tokenizer (use the Hugging Face `tokenizers` library for speed, but own the config and training script). Vocab size per RFC. Include special tokens for: BOS/EOS/PAD, chat roles, tool calls, and reasoning delimiters (e.g. <think></think>), plus digit handling: evaluate individual-digit tokenization vs. grouped digits for math and record the experiment in EXPERIMENTS.md with a measured result on a math corpus sample.
   - Tests: round-trip exactness on tricky unicode, whitespace, code, LaTeX, Cyrillic; fertility (tokens/word) measurements per language.

2. DATA PIPELINE
   - Download/prepare scripts per dataset with explicit license recording in DATA_CARD.md.
   - Cleaning: language ID, heuristic quality filters, a documented quality-classifier approach, PII scrubbing (emails, phones, keys), toxic content policy.
   - Deduplication: exact (hash) and near-duplicate (MinHash-LSH) with configurable thresholds; report removal statistics.
   - Decontamination: n-gram overlap removal against ALL evaluation benchmarks we plan to use (GSM8K, MATH, AIME-style sets, MMLU, etc.). This is mandatory and must be tested.
   - Packing: tokenize to sharded binary files (uint16/uint32 memmap), document-boundary-aware sequence packing with correct attention masking options, deterministic shuffling with a resumable data-loader state (so a restart continues exactly where it stopped).
   - Streaming and multi-worker loading that keeps GPUs saturated; benchmark throughput and report it.
   - Data mixing: weighted sampling across sources with stage-based schedules (pretrain -> annealing).
   - Synthetic math data generators where the answer is programmatically verifiable (arithmetic, algebra, number theory, combinatorics templates) - these also serve as RL environments later. Document difficulty levels.

3. A tiny reproducible demo dataset (generated, or a small permissively-licensed sample) so CI and smoke tests run offline.

Acceptance: end-to-end test that takes raw text -> cleaned -> deduped -> tokenized -> shards -> dataloader yields batches with correct shapes, resumes deterministically after a simulated crash, and passes decontamination test. Write `runbooks/01_data_preparation.md` with exact commands, expected disk usage, and time estimates for my hardware. Update state files and report in Russian.
```

---

## ПРОМПТ 4 - Реализация модели

```
PHASE 4: MODEL IMPLEMENTATION.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md, architecture_rfc.md.

Implement the decoder-only transformer in `src/{package}/model/` precisely per the RFC:
- Token embedding, RMSNorm (pre-norm), rotary embeddings (with scaling hooks for long-context extension), attention with GQA (and MLA/sliding-window only if the RFC chose them), SwiGLU FFN, optional QK-norm, tied/untied output head, optional MoE block with load-balancing loss and router z-loss if chosen.
- Efficient attention: use `torch.nn.functional.scaled_dot_product_attention` (FlashAttention backend where available) with a pure-PyTorch fallback, and a varlen/packed-sequence path.
- KV-cache-aware forward for inference, designed in from the start (not retrofitted).
- Initialization scheme exactly as documented (e.g., scaled init for residual projections), with a test verifying activation/gradient scale across depth at init.
- `torch.compile` compatibility, activation checkpointing option, bf16-safe numerics (fp32 for norm/softmax/loss where needed).
- Parameter counting utility and FLOPs/MFU estimator, validated against your RFC table.
- Config -> model factory for tiny/small/base/large.

Tests (all must pass on CPU with `tiny`):
1. Output equivalence of each custom component vs. a naive reference implementation.
2. Causality test: changing future tokens never changes past logits.
3. KV-cache decode == full-forward logits (within tolerance) across many random lengths.
4. Packed-sequence masking: documents in one packed row do not attend to each other.
5. Overfit test: the tiny model drives loss near zero on a single small batch within N steps.
6. Gradient check on small dims, and determinism test given a seed.
7. Parameter count matches the RFC.

Also write a short `docs/model.md` with the math of every block. Do a profiling pass (if a GPU is available; otherwise write the script and mark as unverified) and record any bottlenecks. Update state files, then report in Russian with the test summary and any deviation from the RFC.
```

---

## ПРОМПТ 5 - Фреймворк обучения (претрейн)

```
PHASE 5: DISTRIBUTED PRETRAINING FRAMEWORK.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md, training_rfc.md.

Build `src/{package}/train/` with:
1. Trainer loop: gradient accumulation, bf16 autocast, grad clipping, configurable LR schedules (warmup + cosine and warmup-stable-decay), weight-decay param groups (no decay on norms/biases/embeddings as per RFC), optimizer factory (AdamW fused; any alternative only if justified in DECISIONS.md and tested).
2. Parallelism: single GPU, DDP, and FSDP2 (or the RFC's choice) with correct mixed-precision and sharded checkpointing; hooks reserved for tensor/pipeline parallel but do NOT fake them - mark unimplemented parts explicitly.
3. Fault tolerance: auto-resume from latest valid checkpoint, restoring model, optimizer, scheduler, RNG states, and dataloader position; graceful SIGTERM handling that saves a checkpoint; NaN/inf detection with automatic rollback-and-skip policy and loud logging.
4. Stability tooling: loss-spike detector, per-layer gradient-norm and activation-stat logging (optional), z-loss option, and a documented playbook for diagnosing instabilities.
5. Validation: periodic held-out perplexity on several domains (web, code, math), and periodic generation samples logged for eyeballing.
6. Performance: report tokens/sec/GPU and MFU; implement and benchmark `torch.compile`; document findings. Include a "scaling sanity" script that trains 3-4 tiny sizes for short runs to fit a mini scaling-law curve and check our hyperparameters transfer sensibly.
7. Hyperparameter sweep launcher (simple, config-driven) for LR and batch-size searches on the `small` config.

Tests: a 200-step end-to-end smoke run on the tiny model with a deliberate kill-and-resume in the middle must reproduce the same loss curve as the uninterrupted run (within tolerance). Run it and show me the evidence.

Deliver `runbooks/02_pretraining.md`: the exact commands I should run for `small` (first full run on my server), what healthy loss curves look like at each stage, red flags, and how to send you the logs so you can analyze them. Update state files; report in Russian.
```

---

## ПРОМПТ 6 - Пост-тренинг: SFT, предпочтения и математическое рассуждение

```
PHASE 6: POST-TRAINING WITH A FOCUS ON MATH REASONING.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md and the math-reasoning sections of survey.md.

Implement `src/{package}/posttrain/`:
1. Chat template + loss masking (train only on assistant tokens), multi-turn support, reasoning-delimiter support, tests that masking is exactly right.
2. SFT trainer reusing the pretraining infrastructure, with packing, LoRA/QLoRA option for cheap experiments and full fine-tuning for the main run.
3. Reasoning-data construction pipeline:
   - Problem sources: permissively licensed math datasets + our programmatic generators.
   - Solution generation via rejection sampling from our own model checkpoints with automatic answer verification (exact match, symbolic equivalence via SymPy, numeric tolerance), keeping only verified-correct and diverse traces. Record pass@k stats per difficulty.
   - Optional use of externally generated synthetic data: document the license/terms implications clearly and make this a switch, not a default.
4. Preference optimization: DPO (and a simple, well-documented variant if justified) with reference-model handling that fits memory budgets.
5. Reinforcement learning with verifiable rewards: implement a GRPO-style trainer: group sampling, relative advantages, KL control, format rewards, correctness rewards from the verifiers, overlong-sequence handling, and online/offline modes. Efficient rollouts must reuse the inference engine from Phase 8 interfaces (design the interface now; if the engine is not ready, implement a simple HF-style generate fallback and mark the performance limitation).
6. Reward-hacking defenses: held-out verifier sets, length penalties, format checks, regular human-readable sample audits logged to disk.
7. Optional track (design doc only, no fake implementation): formal theorem proving with Lean as a verifier - what it would take and whether it is realistic at our scale.

Tests: verifier unit tests (including adversarial answer formats), masking tests, a tiny end-to-end SFT -> DPO -> RL smoke run on synthetic arithmetic where reward visibly increases. Show logs.

Deliver `runbooks/03_posttraining.md`. Update state files; report in Russian, including an honest assessment of what accuracy gains are plausible for our model size (no marketing numbers).
```

---

## ПРОМПТ 7 - Оценка качества

```
PHASE 7: EVALUATION SUITE.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md.

Build `src/{package}/eval/` and `configs/eval/`:
1. Integration with a standard harness (e.g., EleutherAI lm-evaluation-harness) via a thin adapter for our model, PLUS our own lightweight evaluators for the math suite so results are reproducible without external services.
2. Benchmarks: math (GSM8K, MATH/MATH-500, competition-style sets such as AMC/AIME where licensing allows, plus our synthetic verifiable sets), general knowledge/reasoning (MMLU-style, ARC, HellaSwag, etc.), code (HumanEval-style), and language-specific sets (including Russian) if the data mix includes them. Verify each benchmark's official metric definition and prompt format before implementing; note any deviation.
3. Metrics: accuracy with confidence intervals, pass@k with the unbiased estimator, majority-vote@k, average reasoning length, failure-mode taxonomy (arithmetic slip, wrong method, format error, truncation).
4. Contamination audit tool: reports n-gram overlap between training shards and every benchmark; the report is a required artifact for every release.
5. Regression suite: a fixed small eval run automatically on every checkpoint during training, plotted over time.
6. A results ledger `RESULTS.md` auto-generated from eval artifacts, with the exact config, checkpoint hash, and command for every number. NO number may appear in the README unless it is in RESULTS.md with provenance.
7. Honest comparison methodology: compare only against models evaluated with the same prompts/settings; never copy numbers from other papers into our tables without labeling them as not directly comparable.

Tests: evaluators tested on hand-built cases with known answers; pass@k estimator tested against a brute-force computation. Deliver `runbooks/04_evaluation.md`. Update state files; report in Russian.
```

---

## ПРОМПТ 8 - Инференс, экспорт и сервинг

```
PHASE 8: INFERENCE, EXPORT, AND SERVING.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md.

1. Native inference engine (`src/{package}/inference/`): batched generation with KV-cache, paged or block-based cache if justified, sampling suite (temperature, top-k, top-p, min-p, repetition controls, stop sequences, seed), streaming, and a clean `Engine` interface also used by the RL trainer.
2. Export: conversion to Hugging Face-compatible format (config + safetensors + tokenizer files) so the model loads in standard tooling and in popular serving engines; write a conversion-verification test that compares logits between native and exported models.
3. Quantization: weight-only int8/int4 via a well-established library path; measure and record perplexity/accuracy degradation per setting in RESULTS.md. Do not implement exotic custom kernels unless profiling proves a need.
4. Serving: an OpenAI-compatible HTTP API (chat completions + streaming) in a minimal FastAPI app, with request batching, timeouts, and basic auth hook; plus instructions for running through an established high-throughput engine if our architecture is compatible (verify compatibility; do not assume).
5. Long-context support: implement and EVALUATE the RoPE-extension method chosen in the RFC with needle-in-a-haystack-style tests; report honestly where quality degrades.
6. Dockerfile(s) for training and serving, reproducible and minimal.
7. Model card template (`MODEL_CARD.md`): intended use, limitations, training data summary, evaluation results (from RESULTS.md only), safety considerations.

Tests: decode determinism, streaming correctness, API contract tests. Deliver `runbooks/05_serving.md`. Update state files; report in Russian.
```

---

## ПРОМПТ 9 - Документация, качество, релиз

```
PHASE 9: DOCUMENTATION, HARDENING, AND RELEASE READINESS.

Read AGENTS.md, STATE.md, PLAN.md, DECISIONS.md, RESULTS.md.

1. Documentation site (MkDocs or equivalent): architecture overview with diagrams (Mermaid), a "train your first model in one hour" tutorial on the tiny config, deep-dive pages for each subsystem, a troubleshooting guide, an FAQ, and an honest "Limitations and Status" page.
2. Quality hardening: raise test coverage on critical paths, add property-based tests for tokenizer and data packing, add a fuzz test for the verifier parsers, run a security review (dependency audit, secrets scan, unsafe deserialization such as pickle/torch.load - require weights_only/safetensors).
3. Reproducibility audit: from a clean machine, following only the docs, can the tiny pipeline be reproduced end to end? Perform this check in a fresh environment and fix every gap you discover.
4. Responsible-release checklist: data license compliance review, contamination report, known failure modes, misuse considerations, content-safety evaluation plan and a basic refusal/safety SFT dataset design (design plus a small implementation, with honest limits).
5. Release plumbing: semantic versioning, CHANGELOG, GitHub release workflow, optional model upload scripts for the Hugging Face Hub, issue/PR templates.
6. Re-verify the README: every claim must be backed by a command, a test, or a RESULTS.md row. Remove or label anything unbacked.

Report in Russian: a pass/fail table of the checklist, remaining gaps, and the recommended order for my first real training run.
```

---

## ПРОМПТ 10 - Цикл непрерывного улучшения (используйте многократно)

```
PHASE 10: CONTINUOUS IMPROVEMENT LOOP (run this after each of my real training runs).

I will give you: logs, loss curves, eval outputs, and the config of the run that just finished. Then:
1. DIAGNOSE: read AGENTS.md, STATE.md, EXPERIMENTS.md and the new artifacts. Analyze loss curves (spikes, plateaus, divergence between train/val), throughput/MFU, and eval results versus expectations from the scaling sanity curve. Separate facts from hypotheses.
2. ATTRIBUTE: for each weakness (e.g., weak multi-step algebra, format failures, repetition, poor Russian), identify the most probable cause among: data coverage, data quality, hyperparameters, architecture limits, post-training recipe, eval artifact. Propose the cheapest experiment that would discriminate between causes.
3. PLAN: update PLAN.md with a prioritized list of next experiments ranked by expected gain / cost, each with hypothesis, config diff, compute estimate, success metric, and stop condition.
4. IMPLEMENT any code/config/data changes required, with tests.
5. RECORD everything in EXPERIMENTS.md and DECISIONS.md. Never rewrite history; append.
6. REPORT in Russian: what we learned, what to run next on my server (exact commands), and what you are least certain about.

Principles: one major change per experiment; always keep a baseline; prefer data-quality improvements over architecture novelty at small scale; never declare an improvement without a held-out, decontaminated evaluation and, where noise is plausible, multiple seeds.
```

---

## Важные заметки (для вас)

- Агент пишет код и инструкции, но обучение запускаете вы на сервере. Модель с нуля уровня "самых маленьких" (100M-1.5B) вполне реальна, но качество определяется данными и вычислениями, а не только кодом.
- Первый реальный запуск - всегда `small` (~125M): так вы за пару дней проверите весь пайплайн дёшево, прежде чем тратить недели на большее.
- Следите, чтобы агент не выдумывал статьи и цифры: правило честности в промпте 0 сделано именно для этого. Если ответ выглядит слишком гладко, просите источник.
