# Tokenizer and data preparation

Scope: Linux/Python3.11–3.13, pinned CPU environment first. The user's machine is
an ASUS TUF F15/Core i5/RTX3060 Laptop/16GB RAM; exact CPU, VRAM, OS and SSD are
unknown. These are CPU data jobs; they do not require CUDA or train a final model.
Use WSL2/Linux if applicable; native Windows artifact writes are not supported.
Do not change driver/CUDA merely to prepare this fixture.

## 1. Offline acceptance from a clean clone

Install uv0.12.23 as in runbook00. From the repository root:

```bash
make setup
make lint
make typecheck
make test
mkdir -p data
uv run --frozen --extra cpu python scripts/data_smoke.py --directory data/offline-01 --repository .
uv run --frozen --extra cpu qvelt data verify data/offline-01/prepared/manifest.json
```

Choose a new output directory each time. The output includes original EN/RU raw
manifests/text/permission/usage files, an authored heldout exclusion snapshot and
registry, fully resolved recipe, prepared tokenizer bundle, source/license copies,
curation report/rejection IDs, compact token/index shards and a top-level
manifest published last. `smoke-result.json` must say replay_bit_exact=true,
saved_consumer_step=3, resumed_batches=7, final_step=10, production_ready=false.
Actual two-worker prefetch runs before the saved cursor is replayed with0 workers.
No network or downloaded model/corpus is used after environment installation.
The100-update model-training fixture is separate: `make smoke`.

Generate a larger original fixture and rerun preparation independently:

```bash
uv run --frozen --extra cpu qvelt data demo --directory data/demo-01 --documents 192 --repository .
uv run --frozen --extra cpu qvelt data prepare data/demo-01/recipe.yaml --directory data/rebuilt-01 --repository .
uv run --frozen --extra cpu qvelt data verify data/rebuilt-01/manifest.json
```

`--documents` is the generated math count **per language**, plus10 curation
canaries per language. Generator difficulty cycles easy/medium/hard: operands up
to20/1000/1000000, combination pool up to10/25/50 and selection k<=10. Eight
families cover sums/fractions, integer/rational linear equations, gcd/lcm,
combinations/permutations. Every gold answer is independently checked with exact
SymPy and stdlib arithmetic. Reward accepts bounded ASCII integer/p/q only;
reasoning traces, executable expressions and ambiguous extraction are rejected.
These are development tasks, not an assessment of general mathematical reasoning.

## 2. Inspect quality, leakage and identities

Read `prepared/report.json` for per-source inputs/retentions, every removal reason,
PII redaction counts, detected languages, split sizes, every matched benchmark,
actual tokenizer and registry hashes, stored-token and valid-target counts.
`rejections.jsonl` stores hashed source/document IDs and reasons, never rejected
raw text or the original identifying metadata. Raw input stores may still contain
sensitive source text: keep permissions restrictive, apply a retention/deletion
policy and do not put them in Git. The authored PII examples use fake contacts.

Cleaning preserves Unicode/whitespace/LaTeX/code apart from explicit redactions.
Size/repeated-line/control-token/phrase quarantine rules are configured in YAML.
Seeded LID is an annotation: short math/code can be wrong; original declared and
inferred language are separate. PII is pattern-based, not full anonymization;
phrase matches are not a universal toxicity classifier. The implemented score is
heuristic. The required learned-classifier development/evaluation protocol is in
`docs/development/data_design.md`; no trained classifier is claimed.

Exact duplicates hash cleaned bytes. MinHash-LSH uses64 permutations,8 bands,
5-token shingles and Jaccard>=.85 by default; it verifies candidates before dropping,
preserves different ordered numeric/operator atoms, and rejects candidate overflow
at1000. Calibrate with labeled pairs per domain. It can miss duplicates and retain
symbolically equivalent problems. Dedup spans all sources before splitting.
Split is a seed-keyed family hash; if no family is given, clean text determines
it. Fractions default .005 validation/.005 test, demo .2/.2 so tiny splits exist.
BPE sees **train only**. The demo1024 vocabulary ceiling is a smoke override;
TokenizerSettings defaults to RFC32768. Small corpora may learn fewer tokens.
HF BPE pair statistics can grow with corpus diversity despite streamed input.

The exclusion registry uses NFC/casefold word/number/operator n-grams (default13)
against every listed benchmark. Questions shorter than13 use complete phrases;
all matching sets are reported. This is lexical exclusion, not a proof against
translations, paraphrases or unknown answer templates. The demo registry protects
original authored questions only. **Do not rename it into a production registry.**

## 3. Approved external acquisition and production gate

Only one external snapshot is currently approved, for evaluation/exclusion:

```bash
uv run --frozen --extra cpu qvelt data download configs/data/gsm8k_test.yaml --directory data/gsm8k-test-01
uv run --frozen --extra cpu qvelt manifest verify data/gsm8k-test-01/raw/manifest.json
```

The working source spec pins OpenAI's immutable revision, original payload and
MIT license hashes. Expect749738 downloaded JSONL bytes/1319 original rows;
only questions are exported. License and `usage.json` are retained; evaluation/
decontamination only, so the training preparer rejects this input even if its
SourceSpec flag is changed. No GSM8K content is used to fit/select the tokenizer.
See DATA_CARD.md for publisher-license scope and caveats. Internet is required
only for this explicit command; CI does not execute it.

Every other D01–D13 research source remains blocked, listed in
`configs/data/candidates.yaml` and `docs/research/data_mix.md`. The catalog is not
a download recipe. Clear actual upstream rights and format first; no scraping,
TB-scale laptop download or fabricated shard URL. The working generic adapter
supports text-only UTF-8 JSONL and the explicit GSM8K question format; Parquet,
WARC, repository license extraction and learned corpus classifiers are later work.

A production DataRecipe must use purpose=production, non-fixture approved source
manifests with explicit training usage policy and a checksum-pinned registry whose
purpose=production. The registry requires all of **gsm8k,math,aime,mmlu,qvelt-math-heldout**, plus all
additional planned required sets. Each entry records name, local raw manifest,
canonical manifest sha256, approved=true, fixture=false. Approval must be supported
by DATA_CARD, not invented by flipping a flag. Put questions and, when licensed,
other evaluation exclusion text in those raw snapshots, never in the training mix.
Missing/unapproved benchmarks stop preparation **before output publication**.
MATH/AIME/MMLU rights/snapshots remain unresolved; production preparation is blocked.

To prepare an approved source, create its real checksum-pinned DownloadSpec with
name/revision/url/sha256/license_id/license_url/license_sha256/approved/use/format/
max_bytes (schema1); retain license obligations. A DataRecipe references acquired
`raw/manifest.json` canonical hashes and the full benchmark registry. Do not reuse
the evaluation-only GSM8K source as a training example. Sources and their stage
weights must match exactly, with explicit zero weights when inactive.

## 4. Packing and deterministic loading

The prepared raw token files are little-endian uint16 if actual V<=65536, uint32
otherwise. Original BOS/text/EOS is stored once per document. A24-byte segment
entry and16-byte block entry describe consecutive input/target pairs. Long windows
overlap one token, add no artificial EOS and retain each original target exactly
once. Blocks concatenate fragments, pad with0 and return int64 inputs/labels/
positions/document IDs, bool valid/loss masks. BOS/PAD are never valid targets.
Default attention permits only earlier tokens of the same document/fragment.
An explicit LoaderConfig attention_mode=causal allows preceding documents, uses
block positions and preserves original IDs in segment_ids; it is a different task.

```python
from pathlib import Path
from qvelt.data.config import load_data_recipe
from qvelt.data.loader import LoaderConfig, PackedDataset, ResumableLoader

root = Path("data/offline-01")
recipe = load_data_recipe(root / "recipe.yaml")
dataset = PackedDataset(root / "prepared/manifest.json")
settings = LoaderConfig(batch_size=2, workers=2, prefetch=2, max_batches=100)
with ResumableLoader(dataset, settings, recipe.mixing) as loader:
    batch = next(loader)
    loader.save_state(root / "cursor-step-1.json")
# Restart in the same software/data/batch/rank/world/seed/schedule configuration.
state = ResumableLoader.read_state(root / "cursor-step-1.json")
with ResumableLoader(dataset, settings, recipe.mixing, state) as loader:
    next_batch = next(loader)
```

For workers>0, put application calls inside `if __name__ == "__main__":` in a
script; spawn imports the script. The built-in CLI/scripts already do this.
A cursor snapshot is immutable: use a new step-specific path. Only a delivered
batch advances the committed cursor, independent of worker prefetch. Changing
workers/prefetch is supported; changing world size/rank/batch/data/seed/attention/
schedule rejects exact resume. Source selection is deterministic weighted sampling;
pretrain/annealing start_step counts **committed loader batches**, not accumulated
optimizer updates. Each source epoch uses a coprime affine bijection, not a uniform
N! permutation. Exhausted-source reuse is intentional; source cursors count it.
Distributed ranks allocate global slots and disjoint source positions before reuse.

The future real-data trainer must atomically save this committed state with model,
optimizer and RNG. This phase proves loader replay, not full packed-training replay.
Read-only lazy memmaps check registered sizes/checksums and reject changed files.
The local backend does not protect against hostile concurrent filesystem writers.

## 5. Measure on the actual machine

```bash
uv run --frozen --extra cpu qvelt data benchmark data/offline-01/prepared/manifest.json --workers 0 2 --warmup 8 --batches 64 --batch-size 8
uv run --frozen --extra cpu python scripts/data_experiments.py --directory data/measure-01 --repository .
```

The second command separately fits both digit policies on768 original math texts
and measures384 new-seed texts in EN/RU. Fertility is BPE tokens/whitespace words,
round-trip errors are counted, and learned tokenizer/corpus hashes are recorded.
Shared template families make this a compression study, not reasoning evaluation.
It then prepares a192-per-language demo and measures0/2 workers after8 warmup
batches,64 measured batches, batch8/context64, one main-process Torch thread.
Reports separate startup/steady-state time and valid-target/capacity throughput.
Tiny fixtures intentionally repeat after exhaustion and can be filesystem-cached.
There is no GPU/model work in the delivery measurement; no GPU-saturation claim.

Disk planning: reserve3GB for a CPU environment (planning allowance, not download
size), plus at least100MB free for this small demo/temporary artifacts. For a real
uint16 corpus the token payload alone is2*T bytes (20GB for10B stored tokens),
plus24 bytes/segment,16/block, document metadata, licenses, tokenizers, raw text,
SQLite indices and temporary snapshots. uint32 doubles token payload. The report's
scratch_snapshot_bytes is a point-in-time sum, not a measured peak and excludes
anonymous temporary buffers. Index disk/RAM/time must be profiled before scale.

The exact cloud CPU evidence and disk totals are in
`docs/development/validation_data.md`. On the unmeasured i5/16GB laptop, budget
**1–5 minutes and100MB dataset workspace** for this demo after setup, and
**2–10 minutes** for the digit/throughput experiment as conservative planning
assumptions, not predictions from a hardware profile. Installation depends on
network/storage. Start with0 workers and smaller batches if RAM is constrained;
compare2 workers using the measured command. Do not attempt billions of tokens
or a32768-vocabulary fit on unreviewed web data on16GB merely because streaming
works on a fixture. No server, paid compute or production training is started.
