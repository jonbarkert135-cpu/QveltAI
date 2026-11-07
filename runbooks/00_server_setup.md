# Prepare a Linux NVIDIA GPU training server

Scope: Ubuntu 24.04 LTS x86-64, NVIDIA GPUs and administrator access. The user's
actual machine is an ASUS TUF F15/RTX3060 Laptop/Core i5/16GB RAM; no server or
exact GPU/driver/VRAM inventory was supplied. These commands are a server recipe,
not evidence they ran on that laptop. Review cluster policy before sudo commands.
The current trainer is a small synthetic engineering path, not full pretraining.

## 1. Inventory and driver

```bash
cat /etc/os-release
uname -m
lspci -nn | grep -i nvidia
free -h
df -h
nvidia-smi
```

On bare-metal Ubuntu with no working NVIDIA driver, install the distribution's
recommended driver and reboot. Secure Boot may require MOK enrollment on reboot;
use the provider/admin driver method on managed servers instead.

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common build-essential git tmux
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

After reconnecting, `nvidia-smi` must list every GPU without errors. The selected
PyTorch2.13.0 CUDA13.0 wheel profile requires a CUDA13-compatible driver and
supported GPU. Check NVIDIA's current driver/runtime compatibility for the actual
server before installation;this updated profile has not passed GPU acceptance.
The earlier CUDA12.8/570.26 recipe was historical evidence,not the current lock.
A driver version alone does not prove the selected GPU/precision path works.

PyTorch wheels supply their CUDA runtime libraries; a system CUDA toolkit/nvcc
is not needed for this repository's current kernels. Do not install a toolkit
just to make `nvidia-smi` display a particular CUDA version (that field is driver
capability, not the Python runtime). For Windows/WSL follow NVIDIA's WSL guide;
these bare-metal driver installation commands are not for inside WSL.

## 2. Checkout, interpreter and locked environment

Use the repository's pinned uv 0.12.23. If uv is not provided, install it in an
isolated tools environment using a distribution Python/pip; do not replace system
Python or disable TLS verification. This example assumes Python3/pip/venv exist.

```bash
sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv "$HOME/.local/share/qvelt-tools"
"$HOME/.local/share/qvelt-tools/bin/python" -m pip install uv==0.12.23
export PATH="$HOME/.local/share/qvelt-tools/bin:$PATH"
git clone https://github.com/jonbarkert135-cpu/QveltAI.git
cd QveltAI
uv python install 3.12.14
make PROFILE=cuda setup
uv lock --check
uv pip check
```

Install the same lock/profile on every training node. CPU validation uses
`make PROFILE=cpu setup`; never select cpu and cuda extras together. Artifacts
must have available host RAM/VRAM/storage; a package install proves no memory fit.

## 3. Verify the actual runtime

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
uv run --frozen --extra cuda qvelt doctor --device cuda
uv run --frozen --extra cuda python - <<'PY'
import torch
print('torch:', torch.__version__, 'built CUDA:', torch.version.cuda)
print('CUDA devices:', torch.cuda.device_count())
print('NCCL compiled:', torch.distributed.is_nccl_available())
assert torch.cuda.is_available()
assert torch.distributed.is_nccl_available()
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_properties(index).total_memory,
              'native BF16:', torch.cuda.is_bf16_supported(including_emulation=False))
print('NCCL version:', torch.cuda.nccl.version())
PY
```

A CUDA runtime build string is not the driver version. Keep actual inventory/diagnostic
output with the run. FP32 is the first numerical reference; only enable bf16 after
the native capability and a GPU training/reference check pass.

## 4. Storage

Choose a fast local filesystem with space for weights, Adam moments, several
checkpoints, tokenized shards and temporary writes; no silent overwrite/reuse.
Example paths below are suggestions, not directories created by the agent.

```bash
mkdir -p "$HOME/qvelt-storage/data" "$HOME/qvelt-storage/tokenizers" "$HOME/qvelt-storage/runs"
export QVELT_RUNS="$HOME/qvelt-storage/runs"
df -h "$QVELT_RUNS"
```

Keep artifacts outside source/under ignored directories. Never store credentials
or raw private examples in logs. Checkpoint publication is tested on Linux local
storage; NFS/object stores/multi-node crash durability are not validated. All ranks
must see the same output path for distributed checkpoints. Verify shared-store
semantics separately before using the sharded writer across nodes.

## 5. CPU and multi-GPU collectives

First check two CPU processes, then choose the actual visible GPU count.
The GPU command below assumes exactly two visible GPUs; change the count to match
the inventory, not a guessed server size. Every rank verifies the exact sum n(n+1)/2.

```bash
uv run --frozen --extra cuda python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/all_reduce.py --device cpu
NCCL_DEBUG=INFO uv run --frozen --extra cuda python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/all_reduce.py --device cuda
```

Every rank verifies the result; rank 0 prints one JSON summary with the expected sum. For two
ranks it is 3. A hang/error is a failure, not readiness. Inspect device visibility,
unique LOCAL_RANK mapping, peer access/interconnect and the NCCL log. On multiple
nodes, allow the selected rendezvous port and NCCL transport connectivity through
cluster policy; do not blindly disable P2P/IB to hide an unexplained failure.

## 6. Short training and checkpoint replay

The shipped smoke uses 117056 parameters, B=2/T=32 and 100 updates (6400 targets on
one process). This is an explicit CI dimension override of the RFC model family,
not a full19M/125M training recipe or a reasoning benchmark.

```bash
make PROFILE=cpu lint
make PROFILE=cpu typecheck
make PROFILE=cpu test
make PROFILE=cpu smoke
```

The CPU smoke compares a 100-step continuous run to stop 37/resume 100, including
exact weights and Adam states. For a new single-GPU pilot, keep a fixed 100-step
horizon and stop after 10 to inspect the checkpoint before continuing:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run --frozen --extra cuda qvelt train configs/train/smoke.yaml --directory "$QVELT_RUNS/gpu-pilot-01" --set experiment.runtime.device=cuda --stop-after 10
uv run --frozen --extra cuda qvelt train configs/train/smoke.yaml --directory "$QVELT_RUNS/gpu-pilot-01" --set experiment.runtime.device=cuda --resume latest
```

Do not change recipe/precision/topology between exact-resume calls. Changing
steps changes the schedule horizon and is intentionally rejected. A committed
checkpoint has manifest.json; incomplete directories are ignored, but corruption
of the latest committed checkpoint fails instead of silently selecting older data.
Optimizer/DCP state is trusted-project state, not an arbitrary uploaded pickle.

Two-GPU DDP/FSDP2 recipes (new output paths; hardware validation required):

```bash
uv run --frozen --extra cuda python -m torch.distributed.run --standalone --nproc-per-node=2 -m qvelt train configs/train/smoke.yaml --directory "$QVELT_RUNS/ddp-pilot-01" --set experiment.runtime.device=cuda --set training.strategy=ddp
uv run --frozen --extra cuda python -m torch.distributed.run --standalone --nproc-per-node=2 -m qvelt train configs/train/smoke.yaml --directory "$QVELT_RUNS/fsdp-pilot-01" --set experiment.runtime.device=cuda --set training.strategy=fsdp
uv run --frozen --extra cuda pytest -m gpu -q
```

FSDP2 shards blocks bottom-up then the root, builds AdamW afterwards and writes
DCP shards. Actual GPU FSDP2 clipping/checkpoint replay still needs those tests;
CPU DTensor/DCP success is not a substitute. No multi-node speed/quality claim.

## 7. tmux and Slurm

For a manual single-node server, start tmux, run the chosen command inside it,
detach with Ctrl-b d and reconnect. Environment activation is not persistent
across a new shell: set PATH/runtime settings or use uv again.

```bash
tmux new-session -s qvelt
# Run one pilot command above inside tmux, then detach.
tmux attach-session -t qvelt
```

For Slurm, this exact template assumes two nodes with four GPUs each and a
partition named gpu. Adapt resource headers to the real cluster before submission.
Install the identical environment/checkout at an accessible path on all nodes
first. This template validates collectives only; checkpoint storage across nodes
requires separate filesystem validation. Save as all_reduce.sbatch:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=qvelt-collective
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
set -euo pipefail
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export QVELT_MASTER_ADDR
QVELT_MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export QVELT_MASTER_PORT=29500
srun bash -c 'exec .venv/bin/python -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4 --node-rank="$SLURM_PROCID" --rdzv-id="$SLURM_JOB_ID" --rdzv-backend=c10d --rdzv-endpoint="$QVELT_MASTER_ADDR:$QVELT_MASTER_PORT" scripts/all_reduce.py --device cuda'
```

Run `sbatch all_reduce.sbatch` from the checkout; observe squeue and the Slurm log.
Use the scheduler's accounting/status, not a submitted job ID, as success evidence.
No server command, Slurm job or paid GPU was launched by this cloud session.

## Sources and support limits

Historical verified text/API sources: NVIDIA CUDA 12.8 GA release notes, Linux driver 570.26
and CUDA 12.x minor-compat 525.60.13 tables (2026-10-31);
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html .
PyTorch 2.9.1 installed torchrun/distributed/FSDP2/DCP/state-dict API sources and
actual CPU runs. Ubuntu/Slurm commands are an assumed-platform operator recipe,
not executed server validation. Consult the actual provider/cluster documentation
for driver ownership, partitions, fabric and filesystem policy.

Current Phase9 profile is Torch2.13.0/CUDA13.0. Its CPU tests and restricted DCP
reload pass;the actual intended GPU/driver/NCCL/FSDP path is unrun. Historical
CUDA12.8 driver numbers above are not a current compatibility guarantee.
