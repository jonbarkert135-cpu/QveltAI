# Planned exclusion coverage

Production preparation requires `gsm8k`, `math`, `aime`, `mmlu` and
`qvelt-math-heldout` (the frozen1000-problem project final-evaluation contract in
training_rfc.md section6), plus every extra registry required set. The authoritative
baseline IDs live in `qvelt.data.decontamination.MANDATORY_BENCHMARKS`; changing a
benchmark plan requires updating that constant, the registry and licensing record.

GSM8K has an approved evaluation-only question snapshot. The full frozen project
final set, MATH/AIME/MMLU permissions and snapshots remain blocked. Generated
Phase3 template samples/digit-study holdouts are **not** the final project suite.
No execution of a public benchmark or trained-model score is claimed. Optional
Lean/tool-assisted tracks will need their own explicit dataset registration and
rights approval before training promotion; they cannot bypass exclusion coverage.
