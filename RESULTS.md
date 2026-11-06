# Qvelt results

Generated from verified evaluation receipts;do not hand-edit numbers.

All entries are Qvelt measurements. Limited original/development subsets are not official public benchmark scores.
Wilson intervals are descriptive;template correlation/nonrandom prefixes limit population inference.
Compare only rows with the same comparison key. No scores from external papers are imported.

## mmlu-abstract-algebra / 0f2b85aa567c

[Verified receipt](docs/development/evidence/evaluation-phase7-mmlu-abstract-algebra.json)

- Checkpoint manifest: `9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075`
- Model payload SHA256: `53ad69233d88eeb365a23ef6ae4add04301f1fed4d5323d42278243033c8e31e`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `2a5496d20cf56d2bfb73c95f45a5348f8f6f12f8805ce194cbe4780137375862`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `6f3e4888e08a251adbffcc307d389712f26e5ea6d47e562673c92135c44808a1`

Reproduction command:

```bash
python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075/manifest.json
  sha256: 9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-2a5496d20cf56d2bfb73c95f45a5348f8f6f12f8805ce194cbe4780137375862/manifest.json
  sha256: 2a5496d20cf56d2bfb73c95f45a5348f8f6f12f8805ce194cbe4780137375862
extraction: whole-v1
lm_window: 64
max_cases: 64
max_new_tokens: 8
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 16,
  "choice": {
    "accuracy": 0.25,
    "correct": 4,
    "failed": 5,
    "normalization": "character",
    "normalized_correct": 4,
    "problems": 16,
    "wilson95_descriptive": [
      0.10182067491213048,
      0.49498316535508774
    ]
  },
  "domains": {
    "mmlu-abstract-algebra-development-subset": {
      "accuracy": 0.25,
      "correct": 4,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 16,
      "wilson95_descriptive": [
        0.10182067491213048,
        0.49498316535508774
      ]
    }
  },
  "evaluated_prefix": 16,
  "evaluation_sha256": "ea62724256802f28193a439d2cdff4a8aec90f4084d586ae4977ada820ffe4ba",
  "families": {
    "mmlu-abstract-algebra-test": 16
  },
  "generation": {
    "attempts": 0,
    "average_completion_tokens": null,
    "average_reasoning_tokens": null,
    "failed": 0,
    "failure_modes": {},
    "first_correct": 0,
    "invalid": 0,
    "majority_vote_at_k": {
      "1": {
        "accuracy": null,
        "correct": 0,
        "problems": 0,
        "ties": 0,
        "wilson95_descriptive": null
      }
    },
    "pass_at_1_first": null,
    "pass_at_k": {
      "1": null
    },
    "problems": 0,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {},
    "tokens": 0,
    "truncated": 0,
    "wilson95_descriptive": null
  },
  "languages": {
    "en": {
      "accuracy": 0.25,
      "correct": 4,
      "problems": 16,
      "wilson95_descriptive": [
        0.10182067491213048,
        0.49498316535508774
      ]
    }
  },
  "lm": {
    "documents": 0,
    "failed_documents": 0,
    "nll": 0,
    "perplexity": null,
    "targets": 0
  },
  "official_benchmark": false,
  "problems": 16,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## gsm8k / 102d1c6bda48

[Verified receipt](docs/development/evidence/evaluation-phase7-gsm8k.json)

- Checkpoint manifest: `9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075`
- Model payload SHA256: `53ad69233d88eeb365a23ef6ae4add04301f1fed4d5323d42278243033c8e31e`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `cd14a2ac4d83fce2602be530b13a75a7f25bfc199b21d6697d10bf114aea2921`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `a27ba51a1eb154ff074e959ddeb6c80122790a41893ed3e7d5f1630513582d92`

Reproduction command:

```bash
python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075/manifest.json
  sha256: 9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-cd14a2ac4d83fce2602be530b13a75a7f25bfc199b21d6697d10bf114aea2921/manifest.json
  sha256: cd14a2ac4d83fce2602be530b13a75a7f25bfc199b21d6697d10bf114aea2921
extraction: gsm8k-strict-v1
lm_window: 64
max_cases: 64
max_new_tokens: 8
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 16,
  "choice": {
    "accuracy": null,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 0,
    "wilson95_descriptive": null
  },
  "domains": {
    "gsm8k-development-subset": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 16,
      "wilson95_descriptive": [
        0.0,
        0.19360768053443644
      ]
    }
  },
  "evaluated_prefix": 16,
  "evaluation_sha256": "7805fb4b319bc1cd42d8a41f2b956243265fc93a1ebf47554cd91ffa2635b840",
  "families": {
    "gsm8k-original-test": 16
  },
  "generation": {
    "attempts": 16,
    "average_completion_tokens": 6.0,
    "average_reasoning_tokens": 0.0,
    "failed": 4,
    "failure_modes": {
      "truncation": 16
    },
    "first_correct": 0,
    "invalid": 16,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 16,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.19360768053443644
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0
    },
    "problems": 16,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "context_overflow": 4,
      "token_limit": 12
    },
    "tokens": 96,
    "truncated": 16,
    "wilson95_descriptive": [
      0.0,
      0.19360768053443644
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 16,
      "wilson95_descriptive": [
        0.0,
        0.19360768053443644
      ]
    }
  },
  "lm": {
    "documents": 0,
    "failed_documents": 0,
    "nll": 0,
    "perplexity": null,
    "targets": 0
  },
  "official_benchmark": false,
  "problems": 16,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## original-en-ru-code / 33e81c1e7423

[Verified receipt](docs/development/evidence/evaluation-phase7-sampled.json)

- Checkpoint manifest: `9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075`
- Model payload SHA256: `53ad69233d88eeb365a23ef6ae4add04301f1fed4d5323d42278243033c8e31e`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `e4efda0a7c990691b644f524b4ae86ff36b8a558f6c8a93c2a6c5c4ac3382dd5`

Reproduction command:

```bash
python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075/manifest.json
  sha256: 9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5/manifest.json
  sha256: 5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5
extraction: whole-v1
lm_window: 64
max_cases: 64
max_new_tokens: 8
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
- 4
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 4
sampling: sample
schema_version: 1
seed: 20261106
temperature: 1.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 14,
  "choice": {
    "accuracy": 0.0,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 2,
    "wilson95_descriptive": [
      0.0,
      0.6576197724933468
    ]
  },
  "domains": {
    "original-code": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 2,
      "wilson95_descriptive": [
        0.0,
        0.6576197724933468
      ]
    },
    "original-general": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 2,
      "wilson95_descriptive": [
        0.0,
        0.6576197724933468
      ]
    },
    "original-language": {
      "accuracy": null,
      "correct": 0,
      "lm": {
        "documents": 2,
        "failed_documents": 0,
        "nll": 331.89776611328125,
        "perplexity": 374.9324301437019,
        "targets": 56
      },
      "problems": 0,
      "wilson95_descriptive": null
    },
    "original-math": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 8,
      "wilson95_descriptive": [
        0.0,
        0.3244075648838801
      ]
    }
  },
  "evaluated_prefix": 14,
  "evaluation_sha256": "fc1059d09c8211d1192538d3de9be4714f7632b87d1e1a7765a9e9704b971767",
  "families": {
    "arithmetic-fraction": 2,
    "arithmetic-sum": 2,
    "capital-control": 2,
    "integer-function-control": 2,
    "language-control": 2,
    "linear-integer": 2,
    "linear-rational": 2
  },
  "generation": {
    "attempts": 40,
    "average_completion_tokens": 8.0,
    "average_reasoning_tokens": 0.0,
    "failed": 0,
    "failure_modes": {
      "truncation": 40
    },
    "first_correct": 0,
    "invalid": 40,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 8,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.3244075648838801
        ]
      },
      "4": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 8,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.3244075648838801
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0,
      "4": 0.0
    },
    "problems": 10,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "token_limit": 40
    },
    "tokens": 320,
    "truncated": 40,
    "wilson95_descriptive": [
      2.7755575615628914e-17,
      0.27753279986288915
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 7,
      "wilson95_descriptive": [
        0.0,
        0.3543304350666873
      ]
    },
    "ru": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 5,
      "wilson95_descriptive": [
        2.7755575615628914e-17,
        0.43448246478317465
      ]
    }
  },
  "lm": {
    "documents": 2,
    "failed_documents": 0,
    "nll": 331.89776611328125,
    "perplexity": 374.9324301437019,
    "targets": 56
  },
  "official_benchmark": false,
  "problems": 14,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## original-en-ru-code / 42da483934cd

[Verified receipt](docs/development/evidence/evaluation-phase7-regression-1.json)

- Checkpoint manifest: `f723aee59deb71c5d14b25d4f61e5b4410d00c21cb827fd45b4bfe3eedbcbb17`
- Model payload SHA256: `bef9988df3d915520798941b6494b026efb54f4b07ebf3adf08913cd81d3ec3f`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `b6674c5595a16cb1b75d8b8c53fd268e809888a0e0cb3ecd01f50a462c6305b5`

Reproduction command:

```bash
qvelt evaluate runs/phase7-final/training/regression/step-00000001/evaluation.yaml --directory runs/phase7-final/training/regression/step-00000001-reproduce --repository .
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-f723aee59deb71c5d14b25d4f61e5b4410d00c21cb827fd45b4bfe3eedbcbb17/manifest.json
  sha256: f723aee59deb71c5d14b25d4f61e5b4410d00c21cb827fd45b4bfe3eedbcbb17
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5/manifest.json
  sha256: 5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5
extraction: whole-v1
lm_window: 64
max_cases: 4
max_new_tokens: 4
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 14,
  "choice": {
    "accuracy": null,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 0,
    "wilson95_descriptive": null
  },
  "domains": {
    "original-math": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 4,
      "wilson95_descriptive": [
        0.0,
        0.4898908364545972
      ]
    }
  },
  "evaluated_prefix": 4,
  "evaluation_sha256": "c0d23b40a6e94987f14bae75ee00c67e5e8164d43eb83db597b2b06576ea6704",
  "families": {
    "arithmetic-fraction": 1,
    "arithmetic-sum": 1,
    "linear-integer": 1,
    "linear-rational": 1
  },
  "generation": {
    "attempts": 4,
    "average_completion_tokens": 4.0,
    "average_reasoning_tokens": 0.0,
    "failed": 0,
    "failure_modes": {
      "truncation": 4
    },
    "first_correct": 0,
    "invalid": 4,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 4,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.4898908364545972
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0
    },
    "problems": 4,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "token_limit": 4
    },
    "tokens": 16,
    "truncated": 4,
    "wilson95_descriptive": [
      0.0,
      0.4898908364545972
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 4,
      "wilson95_descriptive": [
        0.0,
        0.4898908364545972
      ]
    }
  },
  "lm": {
    "documents": 0,
    "failed_documents": 0,
    "nll": 0,
    "perplexity": null,
    "targets": 0
  },
  "official_benchmark": false,
  "problems": 4,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## original-en-ru-code / 544228391617

[Verified receipt](docs/development/evidence/evaluation-phase7-final.json)

- Checkpoint manifest: `9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075`
- Model payload SHA256: `53ad69233d88eeb365a23ef6ae4add04301f1fed4d5323d42278243033c8e31e`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `174351b57e5171268a0a53245487a62f66f83b7d5eb6b55a1b08383d9e471f83`

Reproduction command:

```bash
python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075/manifest.json
  sha256: 9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5/manifest.json
  sha256: 5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5
extraction: whole-v1
lm_window: 64
max_cases: 64
max_new_tokens: 8
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 14,
  "choice": {
    "accuracy": 0.0,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 2,
    "wilson95_descriptive": [
      0.0,
      0.6576197724933468
    ]
  },
  "domains": {
    "original-code": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 2,
      "wilson95_descriptive": [
        0.0,
        0.6576197724933468
      ]
    },
    "original-general": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 2,
      "wilson95_descriptive": [
        0.0,
        0.6576197724933468
      ]
    },
    "original-language": {
      "accuracy": null,
      "correct": 0,
      "lm": {
        "documents": 2,
        "failed_documents": 0,
        "nll": 331.89776611328125,
        "perplexity": 374.9324301437019,
        "targets": 56
      },
      "problems": 0,
      "wilson95_descriptive": null
    },
    "original-math": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 8,
      "wilson95_descriptive": [
        0.0,
        0.3244075648838801
      ]
    }
  },
  "evaluated_prefix": 14,
  "evaluation_sha256": "1f8b455e7a6ea7d4eb655e3d39ce6fe2627e511b0598fa5bb4aaeae359536b57",
  "families": {
    "arithmetic-fraction": 2,
    "arithmetic-sum": 2,
    "capital-control": 2,
    "integer-function-control": 2,
    "language-control": 2,
    "linear-integer": 2,
    "linear-rational": 2
  },
  "generation": {
    "attempts": 10,
    "average_completion_tokens": 8.0,
    "average_reasoning_tokens": 0.0,
    "failed": 0,
    "failure_modes": {
      "truncation": 10
    },
    "first_correct": 0,
    "invalid": 10,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 8,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.3244075648838801
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0
    },
    "problems": 10,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "token_limit": 10
    },
    "tokens": 80,
    "truncated": 10,
    "wilson95_descriptive": [
      2.7755575615628914e-17,
      0.27753279986288915
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 7,
      "wilson95_descriptive": [
        0.0,
        0.3543304350666873
      ]
    },
    "ru": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 5,
      "wilson95_descriptive": [
        2.7755575615628914e-17,
        0.43448246478317465
      ]
    }
  },
  "lm": {
    "documents": 2,
    "failed_documents": 0,
    "nll": 331.89776611328125,
    "perplexity": 374.9324301437019,
    "targets": 56
  },
  "official_benchmark": false,
  "problems": 14,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## original-en-ru-code / 878d1d668748

[Verified receipt](docs/development/evidence/evaluation-phase7-regression-2.json)

- Checkpoint manifest: `3cb665fc7b13b3612c62814b7b6302fb75744d472f3919b8a640876c6f1250c3`
- Model payload SHA256: `7f9f90e1138d403489829395121f346c2bfec28a020b56dca00617d21f277c60`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `b6674c5595a16cb1b75d8b8c53fd268e809888a0e0cb3ecd01f50a462c6305b5`

Reproduction command:

```bash
qvelt evaluate runs/phase7-final/training/regression/step-00000002/evaluation.yaml --directory runs/phase7-final/training/regression/step-00000002-reproduce --repository .
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-3cb665fc7b13b3612c62814b7b6302fb75744d472f3919b8a640876c6f1250c3/manifest.json
  sha256: 3cb665fc7b13b3612c62814b7b6302fb75744d472f3919b8a640876c6f1250c3
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5/manifest.json
  sha256: 5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5
extraction: whole-v1
lm_window: 64
max_cases: 4
max_new_tokens: 4
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 14,
  "choice": {
    "accuracy": null,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 0,
    "wilson95_descriptive": null
  },
  "domains": {
    "original-math": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 4,
      "wilson95_descriptive": [
        0.0,
        0.4898908364545972
      ]
    }
  },
  "evaluated_prefix": 4,
  "evaluation_sha256": "bbff41557704466fb88110ac42b2806e4047dd0f6b76013f39c3ae5ffa137a1c",
  "families": {
    "arithmetic-fraction": 1,
    "arithmetic-sum": 1,
    "linear-integer": 1,
    "linear-rational": 1
  },
  "generation": {
    "attempts": 4,
    "average_completion_tokens": 4.0,
    "average_reasoning_tokens": 0.0,
    "failed": 0,
    "failure_modes": {
      "truncation": 4
    },
    "first_correct": 0,
    "invalid": 4,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 4,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.4898908364545972
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0
    },
    "problems": 4,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "token_limit": 4
    },
    "tokens": 16,
    "truncated": 4,
    "wilson95_descriptive": [
      0.0,
      0.4898908364545972
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 4,
      "wilson95_descriptive": [
        0.0,
        0.4898908364545972
      ]
    }
  },
  "lm": {
    "documents": 0,
    "failed_documents": 0,
    "nll": 0,
    "perplexity": null,
    "targets": 0
  },
  "official_benchmark": false,
  "problems": 4,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## arc-challenge / 94606e0fe24a

[Verified receipt](docs/development/evidence/evaluation-phase7-arc-challenge.json)

- Checkpoint manifest: `9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075`
- Model payload SHA256: `53ad69233d88eeb365a23ef6ae4add04301f1fed4d5323d42278243033c8e31e`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `628e0ebdc6c62f130e34c484754720644bec03620e6e02320cd842eef4d4a062`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `ba919b0105a084c4b19ce553365297c31e2fa9fafdc2b2449d45cfe2d9ba734c`

Reproduction command:

```bash
python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075/manifest.json
  sha256: 9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075
choice_normalization: character-without-delimiter
context_limit: 256
dataset:
  path: artifacts/dataset-628e0ebdc6c62f130e34c484754720644bec03620e6e02320cd842eef4d4a062/manifest.json
  sha256: 628e0ebdc6c62f130e34c484754720644bec03620e6e02320cd842eef4d4a062
extraction: whole-v1
lm_window: 64
max_cases: 64
max_new_tokens: 8
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 16,
  "choice": {
    "accuracy": 0.125,
    "correct": 2,
    "failed": 5,
    "normalization": "character-without-delimiter",
    "normalized_correct": 3,
    "problems": 16,
    "wilson95_descriptive": [
      0.03497748774324047,
      0.36022827265758683
    ]
  },
  "domains": {
    "arc-challenge-development-subset": {
      "accuracy": 0.125,
      "correct": 2,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 16,
      "wilson95_descriptive": [
        0.03497748774324047,
        0.36022827265758683
      ]
    }
  },
  "evaluated_prefix": 16,
  "evaluation_sha256": "b1e1cee43f6e511ea201488e7da549fdc38d2dc93e877ef840345f5c639d78ea",
  "families": {
    "arc-challenge-test": 16
  },
  "generation": {
    "attempts": 0,
    "average_completion_tokens": null,
    "average_reasoning_tokens": null,
    "failed": 0,
    "failure_modes": {},
    "first_correct": 0,
    "invalid": 0,
    "majority_vote_at_k": {
      "1": {
        "accuracy": null,
        "correct": 0,
        "problems": 0,
        "ties": 0,
        "wilson95_descriptive": null
      }
    },
    "pass_at_1_first": null,
    "pass_at_k": {
      "1": null
    },
    "problems": 0,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {},
    "tokens": 0,
    "truncated": 0,
    "wilson95_descriptive": null
  },
  "languages": {
    "en": {
      "accuracy": 0.125,
      "correct": 2,
      "problems": 16,
      "wilson95_descriptive": [
        0.03497748774324047,
        0.36022827265758683
      ]
    }
  },
  "lm": {
    "documents": 0,
    "failed_documents": 0,
    "nll": 0,
    "perplexity": null,
    "targets": 0
  },
  "official_benchmark": false,
  "problems": 16,
  "protocols": {
    "choice": "continuation-sum-and-character-without-delimiter-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## original-en-ru-code / aa67e6103544

[Verified receipt](docs/development/evidence/evaluation-phase7-regression-3.json)

- Checkpoint manifest: `9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075`
- Model payload SHA256: `53ad69233d88eeb365a23ef6ae4add04301f1fed4d5323d42278243033c8e31e`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `b6674c5595a16cb1b75d8b8c53fd268e809888a0e0cb3ecd01f50a462c6305b5`

Reproduction command:

```bash
qvelt evaluate runs/phase7-final/training/regression/step-00000003/evaluation.yaml --directory runs/phase7-final/training/regression/step-00000003-reproduce --repository .
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075/manifest.json
  sha256: 9283a7a20cafbcee1789ec868a7746ba65fdb68eda735adf35a011695b73e075
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5/manifest.json
  sha256: 5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5
extraction: whole-v1
lm_window: 64
max_cases: 4
max_new_tokens: 4
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 14,
  "choice": {
    "accuracy": null,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 0,
    "wilson95_descriptive": null
  },
  "domains": {
    "original-math": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 4,
      "wilson95_descriptive": [
        0.0,
        0.4898908364545972
      ]
    }
  },
  "evaluated_prefix": 4,
  "evaluation_sha256": "afd6162b13785d9004ea170a5e605a14911da0ed884dd921c3e42634cbe1258e",
  "families": {
    "arithmetic-fraction": 1,
    "arithmetic-sum": 1,
    "linear-integer": 1,
    "linear-rational": 1
  },
  "generation": {
    "attempts": 4,
    "average_completion_tokens": 4.0,
    "average_reasoning_tokens": 0.0,
    "failed": 0,
    "failure_modes": {
      "truncation": 4
    },
    "first_correct": 0,
    "invalid": 4,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 4,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.4898908364545972
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0
    },
    "problems": 4,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "token_limit": 4
    },
    "tokens": 16,
    "truncated": 4,
    "wilson95_descriptive": [
      0.0,
      0.4898908364545972
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 4,
      "wilson95_descriptive": [
        0.0,
        0.4898908364545972
      ]
    }
  },
  "lm": {
    "documents": 0,
    "failed_documents": 0,
    "nll": 0,
    "perplexity": null,
    "targets": 0
  },
  "official_benchmark": false,
  "problems": 4,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.

## original-en-ru-code / d8a1865714c4

[Verified receipt](docs/development/evidence/evaluation-phase7-base.json)

- Checkpoint manifest: `4179a8d638403ef1fde4eb6e843bba814409f36028b47d8a976bdc57dbaf9c25`
- Model payload SHA256: `b216982e64b25647a1dc8f75d13a2c0c408707dfce6415affe8cf021031eb23a`
- Tokenizer: `b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8`
- Dataset: `5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5`
- Evaluator source: `17fb568ce0a6a1c52dbfba10a743bf07af72cc42fa9c2f2bded9373cbbd3e1b3`
- Comparison key: `174351b57e5171268a0a53245487a62f66f83b7d5eb6b55a1b08383d9e471f83`

Reproduction command:

```bash
python scripts/evaluation_suite_smoke.py --directory runs/phase7-final --repository . --with-harness --external-sources runs/eval-source-cache
```

Exact semantic evaluation config (artifact paths are relocatable content addresses):

```yaml
allow_final: false
approved_for_evaluation: true
audit_jaccard: 0.85
audit_ngram: 13
checkpoint:
  path: artifacts/checkpoint-4179a8d638403ef1fde4eb6e843bba814409f36028b47d8a976bdc57dbaf9c25/manifest.json
  sha256: 4179a8d638403ef1fde4eb6e843bba814409f36028b47d8a976bdc57dbaf9c25
choice_normalization: character
context_limit: 256
dataset:
  path: artifacts/dataset-5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5/manifest.json
  sha256: 5630ce21d3fd966d2063abc8ea9d97767b1a89c6ec445393fb2a403a45468aa5
extraction: whole-v1
lm_window: 64
max_cases: 64
max_new_tokens: 8
numeric_atol: 1.0e-08
numeric_rtol: 1.0e-06
pass_k:
- 1
prompt_format: plain-v1
protocol: qvelt-eval-v2
samples: 1
sampling: greedy
schema_version: 1
seed: 20261106
temperature: 0.0
tokenizer:
  path: artifacts/tokenizer-b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8/manifest.json
  sha256: b36b535c275b6163d505eb73e1aefaf125f5f37259a63f6f02d7a97d8409d5c8
training_sources:
- path: artifacts/training-6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99/manifest.json
  sha256: 6b15af941eb6b1caaad2a9d271807af611c7819a68dc8c3808066bda005e3f99
```

Exact model/runtime config:

```yaml
model:
  context: 256
  ffn: 192
  heads: 4
  hidden: 64
  kv_heads: 2
  layers: 2
  norm_eps: 1.0e-06
  rope_scale: 1.0
  rope_theta: 10000.0
  vocab: 395
name: ci-synthetic-smoke
runtime:
  deterministic: true
  device: cpu
  log_level: INFO
  precision: fp32
  seed: 20261106
schema_version: 1
```

Measured metrics,including failures/intervals:

```json
{
  "available_problems": 14,
  "choice": {
    "accuracy": 0.0,
    "correct": 0,
    "failed": 0,
    "normalization": "character",
    "normalized_correct": 0,
    "problems": 2,
    "wilson95_descriptive": [
      0.0,
      0.6576197724933468
    ]
  },
  "domains": {
    "original-code": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 2,
      "wilson95_descriptive": [
        0.0,
        0.6576197724933468
      ]
    },
    "original-general": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 2,
      "wilson95_descriptive": [
        0.0,
        0.6576197724933468
      ]
    },
    "original-language": {
      "accuracy": null,
      "correct": 0,
      "lm": {
        "documents": 2,
        "failed_documents": 0,
        "nll": 336.35357666015625,
        "perplexity": 405.984048301208,
        "targets": 56
      },
      "problems": 0,
      "wilson95_descriptive": null
    },
    "original-math": {
      "accuracy": 0.0,
      "correct": 0,
      "lm": {
        "documents": 0,
        "failed_documents": 0,
        "nll": 0,
        "perplexity": null,
        "targets": 0
      },
      "problems": 8,
      "wilson95_descriptive": [
        0.0,
        0.3244075648838801
      ]
    }
  },
  "evaluated_prefix": 14,
  "evaluation_sha256": "100dabc068c1221e5ca1e602e9ddc64fb2b24554b0fcf066f6e8346beff923d9",
  "families": {
    "arithmetic-fraction": 2,
    "arithmetic-sum": 2,
    "capital-control": 2,
    "integer-function-control": 2,
    "language-control": 2,
    "linear-integer": 2,
    "linear-rational": 2
  },
  "generation": {
    "attempts": 10,
    "average_completion_tokens": 8.0,
    "average_reasoning_tokens": 0.0,
    "failed": 0,
    "failure_modes": {
      "truncation": 10
    },
    "first_correct": 0,
    "invalid": 10,
    "majority_vote_at_k": {
      "1": {
        "accuracy": 0.0,
        "correct": 0,
        "problems": 8,
        "ties": 0,
        "wilson95_descriptive": [
          0.0,
          0.3244075648838801
        ]
      }
    },
    "pass_at_1_first": 0.0,
    "pass_at_k": {
      "1": 0.0
    },
    "problems": 10,
    "reasoning_length_protocol": "explicit-v2-think-token-span;completion-length-separate",
    "stop_counts": {
      "token_limit": 10
    },
    "tokens": 80,
    "truncated": 10,
    "wilson95_descriptive": [
      2.7755575615628914e-17,
      0.27753279986288915
    ]
  },
  "languages": {
    "en": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 7,
      "wilson95_descriptive": [
        0.0,
        0.3543304350666873
      ]
    },
    "ru": {
      "accuracy": 0.0,
      "correct": 0,
      "problems": 5,
      "wilson95_descriptive": [
        2.7755575615628914e-17,
        0.43448246478317465
      ]
    }
  },
  "lm": {
    "documents": 2,
    "failed_documents": 0,
    "nll": 336.35357666015625,
    "perplexity": 405.984048301208,
    "targets": 56
  },
  "official_benchmark": false,
  "problems": 14,
  "protocols": {
    "choice": "continuation-sum-and-character-mean;first-option-tie",
    "generation": "explicit-control-stop-v1",
    "lm": "rolling-disjoint-targets-v1;BOS-first,previous-token-reset,EOS-last",
    "pass_at_k": "1-C(n-c,k)/C(n,k)"
  },
  "uncertainty": "Wilson95 is descriptive under iid problems;template families are clustered,not population confidence. Sampled pass@k assumes iid attempts conditional on each problem;no across-family interval is claimed."
}
```

Audit completeness: declared training sources only;unknown lineage/semantic overlap is unverified.
