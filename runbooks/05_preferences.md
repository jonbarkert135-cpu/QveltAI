# Original CPU preference optimization

Scope:single-process CPU FP32,original decoder and tokenizer,strict single-turn
pairs and a manifested SFT checkpoint. CUDA/distributed preference training,
reference caches,production data and sampled preference-win claims are unsupported.
This is separate from the available pretraining/SFT trainer. No paid run is needed.

## Reproduce the full engineering check

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export PRE_COMMIT_HOME="$PWD/.cache/pre-commit"
make setup
mkdir -p runs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/preference_smoke.py --directory runs/preferences-demo --repository . --steps 40
```

Use a new directory. The script prepares original data,trains a small engineering
base and30-update SFT checkpoint,creates32 verifier-labeled preference pairs and
runs40 DPO updates. It compares uninterrupted training with stop13/resume40:
all losses,weights,Adam,RNG and application cursor must match exactly. This uses a
127104-parameter reduced decoder with actual BPE445,T128,B2 pairs/update,beta0.1
and LR0.001. It is not the19.34M RFC tiny or a tuned server recipe. No external
teacher,public preference corpus or benchmark is acquired.

The generated dpo.yaml contains actual pair/reference hashes and paths. For a
separate run from the same frozen SFT reference:

```bash
uv run --frozen --extra cpu qvelt dpo runs/preferences-demo/dpo.yaml \
  --directory runs/preferences-branch --repository . --stop-after 10
uv run --frozen --extra cpu qvelt dpo runs/preferences-demo/dpo.yaml \
  --directory runs/preferences-branch --repository . --resume latest
```

Resume requires the same pair/reference hashes,beta,job,source,software,precision
and topology. Paths may be relocated while bytes/identities remain unchanged.
The frozen SFT reference must remain available even after a DPO checkpoint exists.
This command uses strict latest checkpoint validation,not auto-quarantine or a
nonfinite skip policy. Invalid/nonfinite loss,gradients or persistent weights/Adam
fail. Complete checkpoints remain available;interrupted partial writes can still
be quarantined by the shared checkpoint engine on save. No partial gradients resume.

## Prepare a different approved demo source

Register exact source rights/revision and label semantics in DATA_CARD before
preparation. Write a raw bundle through qvelt.data.records.write_raw_bundle with
explicit training permission,license and stable document identifier/family. Each
raw text record is JSON with exactly schema_version=1,prompt,chosen,rejected.
Families split before optimization;duplicate/contradictory prompts or IDs fail.
Use prepare_preferences(source,source_sha,tokenizer_manifest,tokenizer_sha,directory,
context=128,approved_for_training=True). Then construct a DPOJob with the actual
prepared manifest identity and SFT reference manifest/hash;dump_dpo writes YAML.
No template invents a checksum. Prepared bundles are demo-only and retain licenses,
usage policy,raw source identity,tokenizer and explicit token frames.

Both candidates share exactly the same user prefix. Prompt,role markers,padding
and outer EOS are masked;response content and end-turn are included. Oversize
pairs fail rather than truncate the preference decision. Multi-turn/system/tool
formats require a separately reviewed extension. Any parameter/data/reference
change creates a new resolved job/run;do not edit a resumed run's identity.

## Interpret results and costs

DPO uses a per-pair mean of sequence SUM response log probabilities. Long answers
can change the ratio more;there is no token-length averaging. Reference forward
passes are recomputed under no_grad with frozen eval weights;no log-probability
cache exists. Two resident FP32 models cost8P bytes for weights alone;policy
weights/gradients/Adam plus reference total roughly20P before activations/workspace.
No reference optimizer is created. Actual server memory and GPU performance are
unverified;the full-model CPU guard prevents accidental unsupported launches.

Metrics include dpo_loss,reference-relative reward margin,strict positive-margin
relative_accuracy and ties. Initial policy=reference has loss ln(2),margin0 and
all ties;ties are not counted as wins. These are offline pair diagnostics,not
sampled generations,measured KL,mathematical answer accuracy or human preferences.

The fixture chooses verifier-correct answers against gold+1 exact-fraction wrong
answers. It has12 train/16 validation/4 unused test pairs in distinct DPO families.
Cross-stage SFT/base family exposure is not excluded. The observed40-step train
loss improved while validation worsened;retain that negative result and do not
promote the checkpoint as better at reasoning. The tiny LR is an overfit diagnostic,
not the RFC small-model starting hypothesis5e-6/32 pairs. Tuning/public comparison
needs licensed independent data,fixed budgets and more than these16 heldout pairs.

Inspect result.json,metrics-*.jsonl,resolved dpo.yaml and manifested checkpoint
application state. Logs can contain paths;review before attaching in chat. No
external upload,message,teacher call or automatic model promotion is performed.
