# Post-training recipes

`scripts/sft_smoke.py` generates a fully resolved strict `sft.yaml` with actual
base-checkpoint, tokenizer and data manifest digests. These identities cannot be
filled with example hashes. Run `qvelt sft <generated recipe> --directory <new run>`;
see runbooks/04_supervised_finetuning.md for formatting, preparation and replay.

Preferences and reasoning RL remain planned; no configuration advertises an
unimplemented command.
