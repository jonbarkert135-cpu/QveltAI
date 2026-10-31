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
