# posttrain

M6.1 implements strict version1 chat formatting, family-isolated licensed demo
preparation, assistant-only masks and whole-dialog packing. The existing packed
trainer supports pinned Qvelt weights-only SFT initialization and atomic resume.

Use `qvelt sft` with the resolved recipe or `scripts/sft_smoke.py` for the original
exact-answer CPU overfit/replay demonstration. See runbooks/04_supervised_finetuning.md
and docs/development/sft_design.md. No public instruction-following/math accuracy,
distributed SFT, GPU validation, preference optimization or reasoning RL is claimed.
