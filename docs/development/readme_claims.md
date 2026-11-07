# README claim evidence

The README reports framework capabilities and links measured quality numbers only
to RESULTS.md. This table binds its substantive claims to executable evidence.
Historical measurements retain their original source/library versions.

| Claim | Evidence command or test |
| --- | --- |
| Byte-BPE round trips and document-isolated packing | tests/unit/test_tokenization.py;test_hardening_properties.py;test_data_storage_loader.py |
| Original decoder/numerical components | tests/unit/test_model.py;test_model_cache.py;test_model_phase4.py;docs/model.md |
| Atomic pretraining/model/Adam/RNG/consumer replay | python scripts/pretraining_acceptance.py --directory runs/recovery --repository . |
| Assistant masking,full/LoRA SFT,DPO,GRPO | tests/integration/test_safety_data.py;python scripts/posttraining_smoke.py --directory runs/posttraining --repository . |
| Cached inference/private sampling/streaming | python scripts/inference_smoke.py --directory runs/inference --repository . |
| HF,HQQ and actual HTTP/SSE/auth/batching | python scripts/inference_phase8_smoke.py --directory runs/serving --repository . |
| Versioned eval,required lexical audit,verified ledger | qvelt eval-registry --check;qvelt results (receipt paths) --check --readme README.md |
| Weights-only and restricted DCP reload | tests/unit/test_safe_dcp.py;python scripts/security_review.py --directory runs/security |
| Current known security limits | docs/site/security.md;SECURITY.md;profile-specific pip-audit artifacts |
| Strict documentation and real RFC tiny tutorial | make EXTRA_ARGS="--extra docs" docs;python scripts/first_model.py --directory runs/first-model --repository . --steps 6 |
| No demonstrated useful/safe model or hardware acceptance | RESULTS.md negative findings;docs/site/status.md;responsible-release.md |

The tutorial target is a bounded short optimization exercise. It is not a claim
that a useful assistant can be trained in one hour. A lexical audit is not proof
of no semantic contamination;checksums require a trusted expected identity.
