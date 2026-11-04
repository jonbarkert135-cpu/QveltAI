# Original conversational SFT on CPU

This workflow implements M6.1 for tiny single-process CPU development. It reuses
Qvelt's own decoder, packed optimizer and atomic checkpoint engine. No TRL,
Transformers, model API, external teacher or paid compute is required.

## Reproduce the complete acceptance workflow

Use the pinned environment from README and a new output path:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
make setup lint typecheck
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 make test
uv run --frozen --extra cpu python scripts/sft_smoke.py --directory runs/sft-new --repository . --steps 80
```

Create the `runs` parent first if absent. The script prepares original base-demo
data/tokenizer, trains a three-step tiny base checkpoint, prepares licensed EN/RU
exact-answer conversations, then fine-tunes for80 steps. It also stops an independent
run at26 and resumes to80, comparing weights, Adam, per-rank RNG and consumer state
bit-exactly. Base and SFT are different runs; SFT initializes only base weights and
creates new Adam, counters and schedule. This base is an engineering demo, not a
publicly pretrained language model.

The generated `sft.yaml` is fully resolved and pins actual data, tokenizer and
initial checkpoint identities. The generated `smoke-result.json`, data report,
metrics and manifest-committed checkpoints contain results. The fixed training
probe is an overfit diagnostic; the separate family-heldout masked-LM probe is
small and can reuse blocks. Neither measures mathematical answer accuracy.

## Stop/resume an SFT job

With a generated recipe and a different new output directory:

```bash
uv run --frozen --extra cpu qvelt sft runs/sft-new/sft.yaml --directory runs/sft-manual --repository . --stop-after 26
uv run --frozen --extra cpu qvelt sft runs/sft-new/sft.yaml --directory runs/sft-manual --repository . --resume latest
```

Keep the horizon, objective, source, tokenizer, model, initial checkpoint digest,
software and topology fixed. Paths, worker count and prefetch may change under the
existing identity checks. Resume uses committed SFT state; it does not reload the
base optimizer or reset to base weights. A new run verifies all registered initial
checkpoint files and then loads only `model.safetensors`; arbitrary uploaded
optimizer/pickle files are not an initialization interface. Old-source checkpoints
are not exact-resume compatible with changed executable source.

`qvelt pretrain` requires the pretraining objective and layout1; `qvelt sft` requires
SFT and layout2. Wrong objective/data/command combinations fail. CPU distributed
pretraining retains its existing tests; distributed SFT, CUDA/NCCL/FSDP and large
model performance are not acceptance claims of this runbook.

## Prepare another approved conversational source

Register exact revision, license and permitted training use in DATA_CARD before
reading it. `write_raw_bundle` accepts `RawDocument` records. Each record's `text`
is a JSON document:

```json
{"schema_version":1,"messages":[{"role":"user","content":"Compute 2 + 3."},{"role":"assistant","content":"2 + 3 = 5."}]}
```

Provide a nonempty stable problem `family`, source problem ID and available
language/difficulty/answer metadata. Optional system messages must come first;
subsequent user/assistant turns alternate and finish with assistant. Empty or tool
turns and unknown fields/versions are rejected. Literal control spellings inside
content remain plain text. Arbitrary supplied solution text is retained but not
mathematically verified by the formatter; the demo's exact answers are checked by
the original Fraction/math and SymPy generator.

Call `qvelt.posttrain.data.prepare_sft` with the raw manifest and canonical digest,
the base tokenizer manifest/digest, a new output directory, packing context and `approved_for_training=True` after
recording approval in DATA_CARD.
The raw bundle needs explicit training permission and license records. Preparation
is currently demo-scope only, with no production decontamination approval. It
reuses the tokenizer; train/validation/test are split by whole problem family
across languages. Both train and validation must be nonempty.

Chat layout2 stores BOS/role/content/end-turn/EOS frames. Loss includes assistant
content and end-turn; prompts, role headers, system/user end-turns, outer EOS and
padding are masked. Attention still sees all valid prompt tokens. A conversation
is never split across blocks. If overlength, right truncation reserves masked EOS,
never fabricates a learned end-turn, and rejects cuts outside a nonempty final
assistant response. Reports record truncated counts. Padding and whole-dialog
packing can lower utilization; there is no packing speedup claim.

The recipe uses existing strict YAML/dataclass fields plus `objective: sft`,
`initial_checkpoint` (manifest path) and `initial_sha256` (canonical manifest hash).
Its model architecture and tokenizer must match the initial checkpoint exactly;
sequence length may be smaller than model context. Document attention is required.
LR, processed targets and accumulation normalization count only assistant targets.
The initial checkpoint digest stays in the resume identity; its path can relocate.
