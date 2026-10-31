# Qvelt literature survey

Date: 2026-10-31. Scope: the primary texts actually inspected, primarily
2015–2025; this is not an exhaustive survey of everything available in 2026.
Hardware is unspecified. No paper's performance is a Qvelt result.

## Evidence standard

All scientific sources cited below are **[verified]**: their PDF text was
retrieved and the abstract and relevant method/report passages were inspected.
This verifies the stated attribution, not independent replication or universal
applicability. Use the exact arXiv **v1** links, not silently changing latest
versions. `sources.json` records SHA-256 digests, access outcomes, and scope.
Some two-column PDF extractions interleave text; inspect the PDF for equations
before implementation. No third-party implementation was copied.

A lookup originally intended for another math paper returned an unrelated
structural-health-monitoring paper (`2407.14342v1`). It is excluded and recorded
as a title mismatch, not treated as evidence. Inaccessible license texts are
listed as unavailable in `data_mix.md`, not cited as verified legal evidence.

## 1. Decoder architecture

- **Position:** RoPE [S03] rotates paired Q/K coordinates so dot products encode
  relative offsets. It does not guarantee extrapolation beyond training length.
  PI [S05] rescales position indices and fine-tunes; YaRN [S04] changes frequency
  interpolation and attention scaling. Both require evaluation of retained short
  context and target long-context behavior. Start at 512/2048/4096 tokens with
  unscaled RoPE; do not advertise 32K/128K based on a theta change alone.
- **Normalization:** RMSNorm [S02] removes mean subtraction; pre-norm [S37]
  studies gradient behavior, but its experiments do not prove every modern LLM
  is stable without warmup. Use pre-RMSNorm plus warmup. Original QKNorm [S06]
  uses L2-normalized Q/K with a learned scale in translation; OLMo 2 [S42]
  uses RMSNorm on Q/K. These are distinct methods, not interchangeable names.
  Keep QK normalization off for the simplest baseline and test it if logged
  attention-logit growth or instability warrants the extra change.
- **MLP:** SwiGLU [S07] uses three projections and a gated SiLU product. The
  approximately 8d/3 FFN width matches a conventional 4d two-matrix FFN's budget;
  aligned widths introduce small differences. Gains on T5 are evidence for a
  candidate, not proof of superior Qvelt mathematical reasoning.
- **Attention:** MHA gives each query head separate KV. MQA [S08] shares one KV
  head; GQA [S09] interpolates. The GQA paper includes uptraining experiments;
  it is not a direct small-from-scratch guarantee. Choose GQA for KV footprint,
  then test an equal-compute MHA control. Tiny's single KV head is MQA, the
  boundary case of GQA. MLA [S10] compresses KV through a latent bottleneck and
  decouples rotary position information; attractive at scale but more complex
  to implement/cache correctly. Its large-MoE evidence does not settle our scale.
- **Local/hybrid attention:** Mistral [S11] combines GQA and sliding windows;
  layers expand the receptive field but are not equivalent to global attention
  for distant problem evidence. Phi-3-small [S33] reports alternating dense and
  block-sparse layers, a different pattern. Use full attention at initial context
  lengths; benchmark retrieval and multi-step math before introducing sparsity.
- **Embeddings and biases:** tying [S12] saves Vd parameters and is especially
  relevant where embeddings consume much of a small model (see [S32]). It does
  not save the full vocabulary-projection FLOPs. Choose tied input/output weights.
  Official Llama code [S45] explicitly disables projection biases; this supports
  a simple bias-free design but is not an ablation proving biases hurt accuracy.
  Qvelt follows public component designs, not any model's weights or code.
- **Kernels:** FlashAttention [S36] computes exact attention with IO-aware
  tiling, avoiding full score storage; it is not linear-time attention. PyTorch
  SDPA documentation [S46] makes native GQA support version/backend dependent.
  A CPU path must explicitly map/repeat KV heads rather than assume the native
  GQA switch works everywhere. Reference and optimized paths need parity tests.

## 2. MoE and routing

Switch [S14] uses top-1 routing, capacity limits, and an auxiliary balancing
objective involving token fractions and average router probabilities. Capacity
overflow/drop behavior changes training semantics. DeepSeek-V3 [S29] reports
bias-based auxiliary-loss-free balancing alongside additional sequence-level
balancing; “auxiliary-loss-free” does not mean no balancing machinery.

Sparse activation can expand total capacity for a given active FLOP budget.
It still stores all experts, optimizer states, routing state, and communication
buffers. Small batches can underutilize experts; multi-node dispatch adds a
different systems bottleneck from dense DDP. At 20M–1.5B with no hardware budget,
we cannot justify these costs over a dense control. Reject MoE for version 1,
not forever: revisit with measured dense bottlenecks, actual interconnect, and
matched active/total-parameter and end-to-end-cost comparisons. No cited paper
proves that every small MoE must lose to a dense model.

## 3. Compute, schedules, transfer, optimizers

Chinchilla [S15] studies a particular loss/data/compute regime and finds model
size and tokens should grow approximately together at training-compute optimum.
The familiar ~20 tokens/parameter is a planning reference, not a universal law.
Its scope differs from optimizing lifetime inference cost. LLaMA [S13],
TinyLlama [S35], and SmolLM2 [S30] train relatively small models much longer;
this motivates an optional ~80–100 tokens/parameter Qvelt recipe, **not** a
claim that these much smaller budgets match their quality. Repetition, data
quality, context, and optimizer changes can invalidate transferred estimates.

Cosine decay is a straightforward fixed-budget control [S13]. MiniCPM [S16]
introduces WSD for reusable stable-phase checkpoints and tail decay without
precommitting the total stable duration. SmolLM2 [S30] provides evidence for
staged stable training and a quality-enriched decay phase. Choose WSD as the
extension-friendly default, with explicit cosine ablations at the same tokens,
peak LR, data, seeds, and batch. Do not attribute all annealing gains to the
schedule when the data mixture changes simultaneously.

Batch scaling is not “more GPUs implies linearly larger LR.” The gradient-noise
scale work [S38] motivates measuring the useful batch regime; it does not yield
an optimal Qvelt batch for free. Hold valid-token global batch fixed across rank
counts first, then tune it. muP [S17] offers width-wise hyperparameter transfer
under specific parameterization rules. It is not automatic depth/context/tokenizer
transfer. Start with standard initialization, bounded LR sweeps, and explicit
size-specific settings; defer muP until coordinate/width-transfer checks exist.

AdamW [S18] decouples weight decay from adaptive updates; LLaMA [S13] documents
beta=(0.9,0.95), weight decay 0.1, and clipping 1.0. Adopt that transparent
baseline, not a claim that it is optimal. Lion [S19] stores momentum rather than
two Adam moments and reports competitive results in its studied settings;
different LR/decay needs prevent drop-in comparisons. Muon [S20] provides real
LM scaling evidence, including models around our base/large range, not merely
vision results: orthogonalized matrix updates, weight decay, update-scale
adjustments, and distinct treatment of non-matrix parameters matter. Its reported
compute gains are author results, not a universal multiplier for Qvelt. Keep it
as a serious measured alternative after AdamW correctness, rather than dismissing
it or selecting it without compatible sharding/checkpoint support.

## 4. Data and mathematical learning

FineWeb [S21] ablates filtering/deduplication and educational selection; Lee et
al. [S22] show duplication affects memorization and evaluation. Their gains are
setup-dependent: indiscriminate aggressive deduplication can also remove useful
variety. Preserve equations during extraction, audit filtering by language and
domain, deduplicate across sources, and split by problem family before packing.
SmolLM2 [S30] and OLMo 2 [S42] motivate late high-quality/math-heavy mixtures.
Qvelt's proposed percentages are testable hypotheses, not copied optimum ratios.

Synthetic educational/code data [S39], STaR [S25], and formal-data generation
[S28] show different ways to expand supervision. Teacher fluency is not a
correctness oracle. Retain origin, teacher/version, generation settings, rights,
verifier status, and rejected traces. Reasoning distillation can transfer training
signal from an existing model; it is compatible with original architecture code
but is not independent rediscovery of the teacher's capabilities.

CoT prompting [S23] demonstrates benefits in sufficiently large models in the
studied tasks. It does not establish that a 20M model acquires reasoning simply
by generating long traces. Compare answer-only, concise worked solutions, and
bounded longer traces with identical problem splits and inference budgets.
Process supervision [S24] can localize step errors and outperforms outcome
supervision in its reported experiment, but step labels/reward models cost
resources and can be wrong. Use exact/numeric outcome checks initially; add
process annotations only when quality and rights are defensible.

STaR [S25] includes generating rationales, rationalizing failed examples given
the known answer, and iterating fine-tuning. Our simpler first bootstrapping
step is rejection sampling of answer-verified traces, retaining all failures
for analysis. A correct final answer does not certify every reasoning step.

DeepSeekMath [S26] introduces GRPO, using group-relative rewards instead of a
learned critic. Its paper covers both learned and rule-like supervision contexts;
choosing an exact verifier for Qvelt is a project choice, not a claim that every
reward in that work was exact. RL still needs rollout memory, reference-policy
log-probabilities, policy identity, and careful length/group normalization.
Zero-variance groups and truncation must have specified behavior. DPO [S40]
is a simpler preference objective to implement first when real validated pairs
exist; correctness labels do not manufacture meaningful preferences on their own.

ToRA [S27] interleaves text with tools. Tool-integrated solving is valuable but
must be reported separately from unaided model reasoning, with sandboxed
execution and budgets. DeepSeek-Prover [S28] generates/checks Lean 4 proofs;
kernel-checked proof correctness is relative to the formal statement and imported
axioms, not proof that an autoformalized statement matches the original question.
Lean remains an optional track with a pinned toolchain and separate evaluation.

## 5. Stability and precision

Log per-layer activation/gradient norms, clipping frequency, attention-logit
statistics, optimizer-state finiteness, data-shard identity, LR, and token counts.
Do not silently skip spike batches and call the run reproducible. SmolLM2 [S30]
reports an unresolved spike even after rewinding/skipping suspect data, showing
that a spike need not have one proven data cause.

OLMo 2 [S42] reports QK normalization and logit z-loss stability changes. Define
vocabulary z-loss as lambda * mean(logsumexp(logits)^2); distinguish it from an
MoE router z-loss. Default lambda=0 for the transparent baseline; test 1e-5
and QK-norm as separate, logged ablations if instability appears. Gradient
clipping is not a substitute for locating non-finite states.

Use FP32 CPU correctness and BF16 compute only on supported devices, keeping
FP32 norms, softmax/loss reductions, and Adam states. BF16 range is useful but
does not eliminate overflow/rounding issues. DeepSeek-V3 [S29] documents
fine-grained FP8 scaling and higher-precision sensitive components; this is
substantial engineering evidence, not justification for naive blanket FP8 casts.
Defer FP8 until hardware, kernels, scaling, and accuracy regressions are measured.

## 6. What open small-model reports actually disclose

| Report | Concrete disclosed information | Limits for reproducing it |
| --- | --- | --- |
| MobileLLM [S32], 2024 | Studies sub-billion 125M/350M models, deep/narrow design, embedding sharing, GQA; layer-sharing variant is separate | Favorable author ablations motivate our shape choices, not guaranteed improvements or a reason to share layers immediately |
| MiniCPM [S16], 2024 | 1.2B/2.4B **non-embedding** variants; model/data scaling and WSD; extended variants described | Non-embedding labels must not be compared directly with our total counts; exact datasets/rights are a separate issue |
| TinyLlama [S35], 2024 | 1.1B, Llama-style model/tokenizer, 32 Q heads/4 KV groups, ~3T token training exposure from ~1T source tokens over epochs; FSDP/FlashAttention | Source corpus size and repeated tokens differ; do not call 3T unique data or infer our achievable quality |
| Pythia [S34], 2023 | Fixed ~300B-token ordering across model scales, original/deduplicated Pile variants, checkpoint-oriented research release | Not evidence that every Pile constituent has suitable rights, or that its architecture is the best math choice |
| SmolLM2 [S30], 2025 | Detailed 1.7B Llama-based study, 2048 pretraining context, ~2M-token batch; stages 0–6T, 6–8T, 8–10T, 10–11T; educational/math/code mixture changes and enriched decay | Intro family includes smaller sizes; the detailed 1.7B recipe is not automatically the exact recipe for every size. LR/data changes confound isolated attribution |
| Qwen2.5 [S31], 2024 | Table 1: 0.5B has 24 layers, Q/KV=14/2; 1.5B has 28 layers, 12/2; both tied embeddings, 32K context/8K generation; family report describes 18T pretraining data and >1M SFT examples | Exact raw corpus, small-size-specific token allocation and reproducible full data composition are not disclosed by those aggregate facts |
| Phi-3 [S33], 2024 | Mini is 3.8B trained on 3.3T tokens, Llama-like architecture; filtered web plus synthetic data; 7B small variant uses GQA and alternating dense/block-sparse layers | Mini exceeds our target; exact filtering corpus and synthetic pipeline are not fully reconstructible from the report. Distinguish mini from small |
| OLMo 2 [S42], 2025 | Larger fully open dense models; RMSNorm, changed norm placement, QK-norm, z-loss, late Dolmino mix | Useful stability/curriculum evidence at larger scales; do not attribute its entire recipe to a 125M model or copy its different norm placement accidentally |

SmolLM2's final 1.7B recipe (section 4.1/appendix A) specifically reports
WSD peak LR=5e-4, 2000 warmup steps, AdamW beta=(0.9,0.95), vocabulary=49,152,
tied embeddings and RoPE theta=10,000. Its earlier dataset-ablation runs use
GPT-2 tokenization/cosine and must not be conflated with that final recipe.
For a concrete disclosed mixture, stage 2 uses 75% English web, 20% code and
5% math; the English-web component retains a 60/40 FineWeb-Edu/DCLM ratio.
These are **author-reported** settings, not Qvelt weights or approved data.

“Open weights,” “open training code,” “reproducible source mixture,” and “all
underlying data rights verified” are different properties. None licenses us to
reuse a corpus without its own review.

## Primary bibliography and claim ledger

Every row is **[verified]** at the claim scope shown. Years below refer to the
accessed preprint unless publication timing is explicitly distinguished.

| ID / source title (link to inspected text) | Authors / organization; year | Verified key claim and relevance  Confidence |
| --- | --- | ---  --- |
| [S01 — Attention Is All You Need](https://arxiv.org/pdf/1706.03762v1) | Vaswani et al., Google; 2017 | Scaled dot-product/multi-head attention and Transformer; baseline algebra, not a decoder-only recipe by itself  [verified] |
| [S02 — Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467v1) | Zhang, Sennrich; Edinburgh/Zurich; 2019 | RMS-based normalization without recentering; norm choice  [verified] |
| [S03 — RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v1) | Su et al., Zhuiyi; 2021 | Rotation-based relative positional interaction; Q/K position design  [verified] |
| [S04 — YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/pdf/2309.00071v1) | Peng et al., Nous/EleutherAI collaborators; 2023 | Frequency interpolation plus attention scaling for context extension; future long-context option  [verified] |
| [S05 — Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/pdf/2306.15595v1) | Chen et al., Meta; 2023 | Downscale positions and fine-tune rather than rely on extrapolation; extension requires validation  [verified] |
| [S06 — Query-Key Normalization for Transformers](https://arxiv.org/pdf/2010.04245v1) | Henry et al., Cyndx; 2020 | Head-wise L2 Q/K normalization with learned scale in low-resource translation; distinguish from RMS QK-norm  [verified] |
| [S07 — GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202v1) | Shazeer, Google; 2020 | Gated three-projection FFNs including SwiGLU; FFN formula and budget  [verified] |
| [S08 — Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150v1) | Shazeer, Google; 2019 | Shared KV heads reduce incremental-decoding bandwidth; MQA endpoint  [verified] |
| [S09 — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245v1) | Ainslie et al., Google; 2023 | Intermediate KV head grouping and uptraining results; GQA trade-off  [verified] |
| [S10 — DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/pdf/2405.04434v1) | DeepSeek-AI; 2024 | MLA latent KV compression plus MoE at large scale; alternative, not small-model proof  [verified] |
| [S11 — Mistral 7B](https://arxiv.org/pdf/2310.06825v1) | Jiang et al., Mistral AI; 2023 | GQA and sliding-window attention; long-context efficiency trade-off  [verified] |
| [S12 — Using the Output Embedding to Improve Language Models](https://arxiv.org/pdf/1608.05859v1) | Press, Wolf; Tel Aviv; 2016 preprint, later 2017 publication | Input/output weight tying in language models; parameter savings  [verified] |
| [S13 — LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971v1) | Touvron et al., Meta; 2023 | Pre-RMSNorm, SwiGLU, RoPE, digit splitting, AdamW/cosine and longer small-model training; integrated public precedent  [verified] |
| [S14 — Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961v1) | Fedus, Zoph, Shazeer; Google; 2021 | Top-1 sparse routing, capacity and auxiliary balance; MoE cost/semantics  [verified] |
| [S15 — Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556v1) | Hoffmann et al., DeepMind; 2022 | Approximately joint model/token scaling under a training-compute constraint; budgeting reference  [verified] |
| [S16 — MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://arxiv.org/pdf/2404.06395v1) | Hu et al., Tsinghua/ModelBest; 2024 | WSD and small-model scaling study; extensible training schedule  [verified] |
| [S17 — Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/pdf/2203.03466v1) | Yang, Hu et al., Microsoft/OpenAI; 2022 | muP and width-wise hyperparameter transfer; candidate requiring parameterization tests  [verified] |
| [S18 — Fixing Weight Decay Regularization in Adam](https://arxiv.org/pdf/1711.05101v1) | Loshchilov, Hutter; Freiburg; 2017 | Decoupled decay/AdamW. Accessed v1 title differs from later *Decoupled Weight Decay Regularization* (2019); baseline optimizer  [verified] |
| [S19 — Symbolic Discovery of Optimization Algorithms](https://arxiv.org/pdf/2302.06675v1) | Chen et al., Google/UCLA; 2023 | Lion, sign-momentum optimization and memory/setting trade-offs; alternative  [verified] |
| [S20 — Muon is Scalable for LLM Training](https://arxiv.org/pdf/2502.16982v1) | Liu et al., Moonshot AI/UCLA; 2025 | Matrix orthogonalization, decay and scale adjustments, LM comparisons; serious future optimizer ablation  [verified] |
| [S21 — The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/pdf/2406.17557v1) | Penedo et al., Hugging Face; 2024 | 15T FineWeb and 1.3T educational subset, filter/dedup ablations; data selection evidence  [verified] |
| [S22 — Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499v1) | Lee et al., Google/Penn collaborators; 2021 | Duplicate exposure changes memorization/evaluation; cross-source audit  [verified] |
| [S23 — Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903v1) | Wei et al., Google; 2022 | Worked reasoning prompts help sufficiently large studied models; no guaranteed tiny-model emergence  [verified] |
| [S24 — Let's Verify Step by Step](https://arxiv.org/pdf/2305.20050v1) | Lightman et al., OpenAI; 2023 | Process versus outcome reward supervision and step labels; optional costly supervision track  [verified] |
| [S25 — STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning](https://arxiv.org/pdf/2203.14465v1) | Zelikman, Wu, Goodman; Stanford/Google; 2022 | Iterative rationale generation/rationalization and fine-tuning; bootstrapping with caveats  [verified] |
| [S26 — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300v1) | Shao et al., DeepSeek/Tsinghua/Peking; 2024 | Math data selection and GRPO with group-normalized rewards/no critic; reasoning-RL basis  [verified] |
| [S27 — ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving](https://arxiv.org/pdf/2309.17452v1) | Gou et al., Tsinghua/Microsoft; 2023 | Interleaved text/tool trajectories and learning; separate tool-assisted track  [verified] |
| [S28 — DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://arxiv.org/pdf/2405.14333v1) | Xin et al., DeepSeek and collaborators; 2024 | Synthetic Lean 4 statements/proofs and verification; optional formal reasoning  [verified] |
| [S29 — DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437v1) | DeepSeek-AI; 2024 | MoE balancing, MLA, fine-grained FP8 and mixed-precision machinery; scale-specific systems evidence  [verified] |
| [S30 — SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model](https://arxiv.org/pdf/2502.02737v1) | Ben Allal et al., Hugging Face; 2025 | Staged data-centric 1.7B study and 11T exposure; curriculum and post-training reference  [verified] |
| [S31 — Qwen2.5 Technical Report](https://arxiv.org/pdf/2412.15115v1) | Qwen Team, Alibaba; 2024 | Small-size layer/head/tied-embedding table and family data/post-training disclosure; public comparison  [verified] |
| [S32 — MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://arxiv.org/pdf/2402.14905v1) | Liu et al., Meta; 2024 | Deep/narrow, embedding sharing, GQA and a layer-sharing variant; small-scale design evidence  [verified] |
| [S33 — Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/pdf/2404.14219v1) | Microsoft; 2024 | Mini training data scale and family architectural differences; synthetic/filtered data precedent  [verified] |
| [S34 — Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/pdf/2304.01373v1) | Biderman et al., EleutherAI collaborators; 2023 | Matched-order ~300B-token model suite and checkpoints; reproducibility design  [verified] |
| [S35 — TinyLlama: An Open-Source Small Language Model](https://arxiv.org/pdf/2401.02385v1) | Zhang et al., SUTD; 2024 | 1.1B Llama-style GQA training, repeated source tokens and systems recipe; small-model exposure  [verified] |
| [S36 — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135v1) | Dao et al., Stanford/SUNY Buffalo; 2022 | Exact IO-aware tiled attention; memory-efficient implementation option  [verified] |
| [S37 — On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745v1) | Xiong et al., university/Microsoft collaborators; 2020 | Pre/post-norm gradient analysis; norm-placement rationale  [verified] |
| [S38 — An Empirical Model of Large-Batch Training](https://arxiv.org/pdf/1812.06162v1) | McCandlish et al., OpenAI; 2018 | Noise scale and useful batch-size regime; measure before batch scaling  [verified] |
| [S39 — Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644v1) | Gunasekar et al., Microsoft Research; 2023 | Phi-1 filtered code/textbook-like synthetic training; quality hypothesis, not reusable proprietary pipeline  [verified] |
| [S40 — Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290v1) | Rafailov et al., Stanford; 2023 | Preference loss based on policy/reference log-probabilities; simpler preference baseline  [verified] |
| [S42 — 2 OLMo 2 Furious](https://arxiv.org/pdf/2501.00656v1) | OLMo Team, Allen Institute for AI collaborators; 2025 | Stability modifications and late specialized data mixture; larger-scale precedent  [verified] |
| [S43 — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909v1) | Sennrich, Haddow, Birch; Edinburgh; 2015 preprint | BPE subword segmentation; tokenizer-algorithm foundation, not a math tokenizer comparison  [verified] |
| [S44 — Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Radford et al., OpenAI; 2019 | Byte-level BPE and residual-depth-aware initialization; tokenizer and initialization precedent  [verified] |

Additional **[verified]** implementation/library evidence: [S45 — official Llama
model definitions](https://raw.githubusercontent.com/meta-llama/llama/main/llama/model.py)
(Meta, retrieved 2026-10-31; hash in manifest), bias-free linear layers;
[S46 — PyTorch v2.6.0 SDPA documentation/source](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/nn/functional.py),
backend-specific experimental GQA contract. These are API/design evidence,
not scientific outcome claims or selected dependency pins.
