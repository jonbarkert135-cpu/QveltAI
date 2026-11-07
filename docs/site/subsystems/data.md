# Data and tokenizer

`qvelt.data` approves exact source identities and allowed uses,cleans original
documents,removes duplicates,audits benchmark overlap and writes checksummed
little-endian token shards. Demo approval never upgrades data into production.

The tokenizer is deterministic byte BPE with Unicode round trips and explicit
control IDs. Ordinary user text cannot inject a role token. Tokenizer-v2 adds
reasoning/tool delimiters without reinterpreting legacy artifact IDs. Learned
vocabulary must equal the model's actual embedding vocabulary.

Packed documents have separate attention boundaries and shifted next-token pairs.
Padding supplies no loss;SFT additionally masks user/system targets. Resumable
loaders commit the consumer position rather than an unconsumed prefetch cursor.
Property tests reconstruct every original document pair exactly once across
random context/shard sizes and verify Unicode/control isolation.

Read the [preparation runbook](../reference/runbooks/01_data_preparation.md),
[data card](../reference/DATA_CARD.md) and [source registry](../reference/configs/eval/benchmarks.yaml).
Lexical overlap checks cannot certify unknown prior lineage or semantic cleanliness.
