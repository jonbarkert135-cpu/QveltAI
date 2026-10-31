# Qvelt data mix proposal and licensing review

Date: 2026-10-31. Owner: Lead agent. **Proposal, not ingestion approval.**
Intended planning use: train from scratch and potentially release a commercially
usable model/framework; the project license is not selected. No raw corpus,
teacher weights, or training split was downloaded. Only publisher cards/metadata
and research papers were inspected. Sizes below are publisher numbers in their
own units/tokenizers, not measurements with the untrained Qvelt tokenizer.

## License evidence and limitations

`sources.json` records exact card URLs, immutable repository revisions and
SHA-256 hashes. **[verified]** below means the publisher's license statement,
reported size, and relevant provenance passage were read. It does not mean all
upstream rights have been cleared, every shard is downloadable, or the original
content has been independently audited. A license field is not a license audit.

The Apache-2.0 legal text was retrieved and inspected. The authoritative ODC-By
1.0 page and CC-BY-4.0 legal-text endpoint returned HTTP 403; their complete
terms were **not independently verified in this session**. Publisher declarations
remain verified as declarations. Before recommending actual ingestion, obtain
the authoritative texts and validate attribution, redistribution and upstream
rights. Common Crawl's terms were accessible and reserve third-party rights;
access to a crawl is not permission to reproduce arbitrary crawled content.
OpenWebMath explicitly says it does not alter underlying licenses.

**Conditional** means a plausible research candidate with declared rights, but
remaining license/provenance checks block ingestion. **Blocked** means an access
or identifiable license/provenance problem makes it unsuitable for the default
recipe now. No candidate is represented as unconditionally safe or approved.

## Candidate inventory

All snapshot links and full hashes are in `sources.json` under dataset repository
identity. D01–D13 are local catalog IDs, not downloaded data IDs.

| ID / candidate (publisher) | Role; published size at inspected card | Declared license evidence | Concerns / disposition |
| --- | --- | --- | --- |
| D01 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), Hugging Face | English educational web; 1.3T tokens, default version | [verified] ODC-By 1.0 declaration and Common Crawl terms reference | Conditional: obtain legal text, source rights/opt-outs, equation extraction and contamination review; Llama-based quality classifier provenance |
| D02 [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), Hugging Face | Math; FineMath-3+ 34B tokens/21.4M docs; 4+ is a 9.6B-token subset; InfiMM derivatives are separate | [verified] ODC-By 1.0 declaration plus Common Crawl terms | Conditional: web rights, math correctness, benchmark leakage and overlap with D01/D03; do not add subset sizes as unique content |
| D03 [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math), Paster et al. | Mathematical web; 14.7B tokens/6.3M docs | [verified] README License section ODC-By 1.0; nested metadata, despite absent top-level API license | Conditional: original licenses retained; mixed sites, overlap and formatting quality. Card's token count differs from SmolLM2's 12B under its accounting, not necessarily a contradiction |
| D04 [FineWeb2, rus_Cyrl](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), Hugging Face | Russian; card table reports 588,579,493,780 **words**, 699,083,579 documents, 5.82 TB UTF-8 / 1.81 TB disk | [verified] ODC-By 1.0 plus Common Crawl terms | Conditional: these are words, not Qvelt tokens; sample small approved shards, no TB-scale laptop download; language/PII and filtering audit |
| D05 [opc-fineweb-code-corpus](https://huggingface.co/datasets/OpenCoder-LLM/opc-fineweb-code-corpus), OpenCoder/INF | Code-related **web pages**, not a permissively licensed repository-code collection; metadata 100,920,235 examples/~255GB uncompressed | [verified] card metadata says MIT; text says recalled from FineWeb and references 55B code+math-related data jointly, not a verified token count for this split | Blocked default: reconcile MIT packaging assertion with upstream FineWeb/web rights; do not assume all embedded code is MIT |
| D06 [SmolLM-Corpus / python-edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), Hugging Face | Python educational subset; metadata 7,678,448 records, ~0.99GB of listed metadata; exact Qvelt tokens unknown | [verified] aggregate card declares ODC-By; points to The Stack v2 for actual data licensing, not the aggregate wrapper badge | Blocked default code source until file/repository licenses, provenance and upstream access terms are resolved; wrapper license alone is insufficient |
| D07 [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), Wikimedia | Optional EN/RU reference prose; 20231101.en cited as 6.4M samples; RU count not verified | [verified] card lists CC-BY-SA-3.0/GFDL and notes exceptions | Excluded default: attribution/share-alike handling and individual revisions need review; no inference that a code license covers article text |
| D08 [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), Hugging Face | Synthetic educational texts; >30M files/~25B tokens, Mixtral-8x7B outputs | [verified] Apache-2.0 declaration; teacher and seed-source descriptions read | Conditional: seed-content and teacher provenance/terms, factual/math errors and overlap; prefer traceable approved subsets |
| D09 [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), Hugging Face | General SFT; ~1M examples; Smol-Magpie-Ultra component ~400K | [verified] new named components Apache-2.0, **not a uniform license for all imported components** | Conditional component-only review: individual source licenses and Llama teacher terms must be checked. No blanket approval of the mixture |
| D10 [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), AI-MO | Math SFT; ~860K problems/solutions from exams/forums, OCR/translation/reformatting | [verified] Apache-2.0 declaration | Blocked default pending original exam/forum rights, translated derivatives, OCR correctness and contamination review |
| D11 [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), Open-R1 | 220K headline problems, 2–4 DeepSeek-R1 traces; default subset ~94K problems | [verified] Apache-2.0 declaration; most traces Math-Verify, ~12% LLM-judged according to card | Blocked default pending inherited Numina rights and teacher review. Answer checks/LLM judge do not prove every step; 16K traces exceed baseline context |
| D12 [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2), NVIDIA | 14M math problem-solution pairs, GSM8K/MATH-derived/augmented, Llama-3.1-405B solutions | [verified] CC-BY-4.0 declaration, source/teacher passages | Blocked default pending license text, original problem rights, teacher conditions and similarity audit; benchmark-train derivations must not enter held-out families |
| D13 [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata), BigCode | Candidate repository code; card/size not verified here | **[unverified]** card request returned HTTP 401; no license conclusion | Blocked: do not bypass gating or substitute an assumed license; recheck access and per-file permissions before use |

The strongest immediate non-third-party path is project-authored fixtures and
algorithmically generated exact-answer tasks. Those generators do not exist yet;
their code/data license must be selected before release. These are suitable for
software validation, not a substitute for a diverse production pretraining corpus.

## Stage mixtures — hypotheses requiring approved sources

Denominators matter: pretrain/anneal/SFT percentages are **loss-bearing target
tokens after Qvelt tokenization**, preferences are validated pairs, and RL is
prompts. Sources are assigned disjointly after cross-source deduplication. Do not
sample a file as both English web and math, or count a FineMath subset twice.

| Disjoint category | Main pretraining (90% D) | Anneal (10% D) | Candidate mapping |
| --- | ---: | ---: | --- |
| Educational English web | 50% | 25% | D01, approved extracted subset |
| Mathematical exposition/problems | 25% | 45% | D02 primary; D03 optional replacement, not additive duplicates |
| License-audited code | 10% | 10% | No default eligible corpus yet; D05/D06/D13 remain blocked; authored code/generators can supply fixtures only |
| Verified educational synthetic content | 10% | 15% | Approved D08 subsets plus project-authored exact-answer tasks |
| Russian general/educational text | 5% | 5% | Approved D04 subset; tokenizer and language-quality checks |
| **Total** | **100%** | **100%** | Do not run this recipe until every occupied slot has approved data |

If code rights remain unresolved, create a **different named ablation** replacing
code tokens with approved educational text and adjust the manifest; never quietly
fill a blocked slot or claim the original mix ran. With no approved sources,
only local authored fixtures are allowed. Percentages are not source approvals.

FineMath demand for large is 150B*(0.9*0.25+0.1*0.45)=40.5B sampled tokens,
already above the card's 34B 3+ tokens before retokenization/filtering. This
implies repeats or additional approved content; it is not 40.5B unique math.
Its 4+ subset alone would repeat more. Recompute exact source capacities with
the frozen tokenizer, cap repeated problem families, and log unique/seen counts.
The ratio is not justified by the publisher's math benchmarks; it needs ablation.

| Post-training stage / category | Percentage | Candidate / rights gate |
| --- | ---: | --- |
| SFT general instructions | 30% target tokens | Approved D09 **components**, or project-authored instructions |
| SFT concise worked math | 50% target tokens | Authored/approved verified solutions; D10–D12 blocked until resolved |
| SFT code/tool-format instruction | 10% target tokens | Authored executable fixtures or separately licensed pairs; no host execution of generated code |
| SFT Russian instructions/math | 10% target tokens | Authored/approved RU content; no automatic machine-translation rights assumption |
| Preference math / general / RU | 70% / 20% / 10% pairs | Validated pairs from training families only; preserve source/label provenance |
| RL exact arithmetic / bounded algebra / applied word problems / RU variants | 50% / 30% / 15% / 5% prompts | Authored generators and bounded exact verifiers first; use disjoint training families, not benchmark tests |

These stages each sum to 100% within their own denominator; they are not to be
summed across stages. Machine-readable weights are in `data_mix.json`, tested
for totals/ranges. Data scale and repeated epochs depend on the approved corpus,
not a promise that uncreated authored tasks fill a production budget.

## Processing and contamination gates

1. Record authoritative license/terms texts and immutable source revisions;
   resolve ambiguous rights with the user before ingestion. Keep attribution and
   deletion/opt-out provenance. No scraping or site access against terms.
2. Extract text retaining equations, code indentation, LaTeX delimiters and units;
   normalize documented line endings only. Filter PII/secrets, boilerplate,
   gibberish, excessive repetition, and broken math without deleting Russian
   simply for using a different script. Audit false positives by domain/language.
3. Perform exact content hashes and approximate document/substring MinHash
   deduplication [S21,S22]; calibrate thresholds on sampled labeled pairs instead
   of blindly copying a number. Retain an audit trail for removed records.
4. Before training, freeze validation/test identities and problem families.
   Exclude exact/approximate matches, translated variants, answer templates,
   prompt+solution pairs and derivatives of test questions. Training on public
   benchmark training splits is still a contamination risk needing disclosure.
5. Hold out 0.5% of eligible documents/families for data-loss validation using
   deterministic hashed assignment, cap huge sources for affordable validation;
   pin actual IDs. Keep final math tests separate as in `training_rfc.md`.
6. Compute post-filter Qvelt token counts, a stable source ordering and shard
   hashes before mixing. Pack without cross-document loss/attention leakage.
   Record curriculum transitions/repetition and deterministic resume cursors.

No exact/near-match search certifies zero semantic contamination. Maintain a
frozen self-authored out-of-family test in addition to licensed public benchmarks.
Future benchmark approvals belong in the root `DATA_CARD.md` too.
