# M3.1a: byte-BPE tokenizer library

Design: tokenizers 0.22.2 Apache-2.0; `TokenizerSettings` configures an ordered
iterator-based trainer. All 256 byte symbols and 32 controls count toward the
requested vocabulary ceiling. Small corpora may not fill it. No normalization,
implicit BOS/EOS, padding or truncation is applied. Default pre-tokenization
isolates ASCII digits, then uses ByteLevel without an added prefix space.

Control IDs: pad0, bos1, eos2, system3, user4, assistant5, end_turn6, tool7;
reserved_00 through reserved_23 occupy8–31. Ordinary encode treats these spellings
as text; training rejects them to avoid ambiguity. Explicit chat serialization
is future work. Whole valid UTF-8 strings round-trip, including Cyrillic,
combining marks and LaTeX. Lone surrogates and inputs over1MiB are rejected.
Arbitrary byte-token slices can decode with replacement characters.

`save(Path)` uses M2.2 write-once atomic local storage. `load(Path)` bounds JSON
to 64 MiB, validates the supported pipeline, controls and full byte alphabet.
A standalone JSON file has no external checksum trust anchor. For lineage,
use the completed [M3.1 bundle workflow](tokenizer_workflow.md), which pins raw
data, recipe, code and dependencies and verifies the tokenizer semantics.
The wrapper is typed; tokenizers has no py.typed marker and its internal methods
are outside static checking. Runtime contract tests remain required.

## Actual CPU fixture pilot

Run with `uv run --frozen --extra cpu python` from the repository:

```python
from qvelt.tokenization import TokenizerSettings, train_tokenizer

texts = [
    "Authored fixture: two plus two equals four.",
    "Русский текст: два плюс два равно четыре.",
    r"\frac{1}{2}+\frac{1}{2}=1; 1234567890",
]
for digits in (False, True):
    t = train_tokenizer(texts * 4, TokenizerSettings(384, 2, digits))
    print(digits, t.vocab_size, [len(t.encode(x)) for x in texts])
    assert all(t.decode(t.encode(x)) == x for x in texts)
```

Executed 2026-10-31, Python 3.12.14, CPU. Output:

```text
False 381 [9, 9, 16]
True 371 [9, 9, 26]
```

UTF-8 lengths43/74/37 bytes. Repeated authored fixtures are only software probes;
these counts say nothing about model reasoning or real corpus compression.
Default digit isolation follows the research RFC, with no claimed quality gain.
252 tests passed in 4.59s, including32 tokenizer cases; Ruff/format/mypy pass.
Test coverage includes both policies, controls, non-Latin unseen bytes, malformed
Unicode/JSON, modified normalization, empty corpus, stable training and save/load.
No external corpora, LLM training, browser trace or GPU execution occurred.

## Completed workflow and next scope

YAML/CLI recipes, registered raw-data/tokenizer lineage and independent Python
3.11–3.13 probes are complete for authored fixtures; see
[validation](validation_m3_1.md). M3.2 handles production corpus preparation.
Do not freeze a production 32768-token artifact from toy fixtures. Although
inputs stream and document sizes are bounded, BPE pair statistics grow with
corpus diversity; larger training requires measured memory and approved data.

Ponytail applies in full mode. The implementation reuses strict YAML construction,
atomic IO, typed manifests, observed provenance and the pinned BPE dependency.

## Step B: explicit v3 codec (2026-11-08)

V3 implements the128-control table in [ADR-0053](../../DECISIONS.md#adr-0053-tokenizer-v3-control-block-and-explicit-new-lineage),
including six reserved effort markers and100 future slots. This is a codec,
not tool execution or trained effort support. Ordinary text, including literal
control spellings, encodes to IDs>=128; only explicit control_id calls insert
control IDs. Training retains the existing prohibition on literal control
spellings in corpus text. Byte coverage/whitespace preservation are unchanged.

Select v3 explicitly with TokenizerV3Settings(vocab_size=...). V is mandatory;
there is no v3 production default. The checked request range is384..1048576,
including controls and all256 bytes. This is a ceiling, not a promise that a
small corpus learns that many tokens. JSON remains bounded to64MiB and documents
to1MiB; oversized v3 serialization fails before writing. Full-vocabulary memory,
compression, multilingual quality and production V remain Step E/human decisions.

The existing tokenizer CLI accepts a new recipe schema2 for v3. Recipe schema1
and TokenizerSettings continue selecting the historical v2 trainer; readers still
accept v1. Old recipe serialization/digests and four minimal v1/v2 payload hashes
are checked against values captured on unmodified09446b4/tokenizers0.22.2.
The tokenizer recipe version is independent of the future chat schema version.

Example recipe for the already-approved development fixture ONLY, not production:

```yaml
schema_version: 2
name: v3-development-fixture
training_data_sha256: 832c18b89e947c9953f0b79838b50223642c65c5fb66ed6672859552df42d57b
tokenizer:
  vocab_size: 512
  min_frequency: 2
  isolate_digits: true
```

Use the existing `qvelt tokenizer train`/`verify`/`evaluate` commands with this
recipe and the registered fixture manifest. Never reuse output directories.
Bundles retain source/recipe/provenance links and record128 reserved controls;
the loader rejects relabelling a v3 payload with a legacy recipe. V1/v2 files,
production configs, checkpoints and uv.lock are not migrated or rewritten.

Artifact determinism means equal tokenizer.json SHA-256 for equal ordered corpus,
settings and supported tokenizers build: tested in three independent processes
with different Python hash seeds and Rayon thread counts. Timestamp/provenance-
bearing bundle manifests are not promised identical hashes across new creations.

A real v3 BPE/102720-parameter CPU fixture exercises loss gradients, full/cached
logits and the native Engine's plain-v1 generation within its20M gate. No optimizer
update or useful-model quality claim follows. Legacy chat/SFT/HF-chat formatters
and native snapshot export explicitly reject v3 until their protocols are
implemented/validated. Packed training remains limited by its existing control-
version guards. Step C schema3/tool masks and chat_schema_version reporting fix,
Step D capabilities, actual effort behaviour and production freeze are not done.
