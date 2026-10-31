# Qvelt

Qvelt is a planned open, from-scratch language model framework with an emphasis
on mathematical reasoning. Its intended scope spans tokenization, data,
pretraining, post-training, evaluation, and inference.

**Current status: Phase 1 research/design complete; model implementation pending.**
This repository contains project governance, a development roadmap and
[source-audited research RFCs](docs/research/README.md) with tested analytical
size/resource calculations. It does not yet contain a tokenizer, model, training
loop, pretrained weights, or benchmark results. No training has run.

The initial model-size targets are approximately 100M parameters for development,
then 500M–1.5B parameters, with configuration-based scaling as a research goal.
These are targets, not implemented configurations or quality claims. No servers
are currently available. Tiny CPU configurations will establish correctness;
larger training runs will require separately assessed hardware and budgets.

## Project documents

| Document | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Working protocol for future sessions |
| [PLAN.md](PLAN.md) | Phases, tasks, dependencies, and acceptance gates |
| [STATE.md](STATE.md) | Current state, blockers, next tasks, and verification commands |
| [DECISIONS.md](DECISIONS.md) | Decisions, evidence, and unresolved research candidates |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Experiment register; no completed ML experiments yet |
| [RISKS.md](RISKS.md) | Technical, resource, licensing, and release risks |
| [DATA_CARD.md](DATA_CARD.md) | Dataset approval policy and provenance register |

Read `STATE.md` and `PLAN.md` before starting work. The next development session
starts with the package/config scaffold following the research RFCs.

There are no installation or application-start commands yet. Dependency pins,
an executable package, and CI belong to later milestones. A project license has
not yet been selected; public visibility alone does not grant reuse rights.
