"""Two-process numerical oracle plus DCP state and rank RNG round trip."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict_loader import load as distributed_load
from torch.distributed.checkpoint.state_dict_saver import save as distributed_save
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel

from qvelt.config.training import load_training_job
from qvelt.data.synthetic import synthetic_batch
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss
from qvelt.train.optim import make_optimizer
from qvelt.train.smoke import _equal
from qvelt.utils.checkpoints import (
    initialize_training_run,
    load_checkpoint,
    save_checkpoint,
)
from qvelt.utils.distributed import distributed_context
from qvelt.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    job = load_training_job(root / "configs/train/smoke.yaml")
    job = replace(
        job,
        training=replace(
            job.training,
            steps=4,
            batch_size=1,
            sequence_length=4,
            warmup_steps=0,
            decay_steps=0,
            strategy="ddp",
        ),
    )
    torch.set_num_threads(1)
    with distributed_context(args.device) as context:
        mesh = init_device_mesh("cpu", (context.world_size,))
        full = torch.arange(16, dtype=torch.float32)
        original_shard = distribute_tensor(full, mesh, [Shard(0)])
        target_shard = distribute_tensor(torch.zeros_like(full), mesh, [Shard(0)])
        distributed_save(
            {"weight": original_shard},
            checkpoint_id=args.directory.parent / "dtensor-probe",
        )
        distributed_load(
            {"weight": target_shard},
            checkpoint_id=args.directory.parent / "dtensor-probe",
        )
        torch.testing.assert_close(
            target_shard.to_local(),
            full.chunk(context.world_size)[context.rank],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(target_shard.full_tensor(), full, rtol=0, atol=0)
        seed_everything(job.experiment.runtime)
        base = Decoder(job.experiment.model).to(context.device)
        original = copy.deepcopy(base.state_dict())
        model = DistributedDataParallel(base)
        optimizer = make_optimizer(model, job.training)
        batch = synthetic_batch(
            0,
            1,
            4,
            288,
            rank=context.rank,
            world_size=context.world_size,
            device=context.device,
        )
        mask = torch.tensor(
            [
                [True, True, True, True]
                if context.rank == 0
                else [False, True, False, True]
            ],
            device=context.device,
        )
        local, n = causal_loss(model(batch.inputs), batch.targets, mask)
        torch.autograd.backward(local * (context.world_size / context.sum(n)))
        oracle = Decoder(job.experiment.model).to(context.device)
        oracle.load_state_dict(original)
        global_batch = synthetic_batch(0, 2, 4, 288, device=context.device)
        total, count = causal_loss(
            oracle(global_batch.inputs),
            global_batch.targets,
            torch.tensor(
                [[True] * 4, [False, True, False, True]], device=context.device
            ),
        )
        torch.autograd.backward(total / count)
        for actual, expected in zip(
            base.parameters(), oracle.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=3e-7)
        # Now perform one ordinary, fully valid synthetic update for checkpointing.
        optimizer.zero_grad(set_to_none=True)
        local, n = causal_loss(model(batch.inputs), batch.targets)
        torch.autograd.backward(local * (context.world_size / context.sum(n)))
        optimizer.step()
        initialize_training_run(job, args.directory, root, context)
        save_checkpoint(model, optimizer, job, args.directory, 1, context, sharded=True)
        expected_weights = copy.deepcopy(base.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_random = torch.rand(4)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.add_(1)
        optimizer.state.clear()
        torch.rand(17)
        assert load_checkpoint(model, optimizer, job, args.directory, context) == 1
        _equal(base.state_dict(), expected_weights)
        _equal(optimizer.state_dict(), expected_optimizer)
        torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
        if context.rank == 0:
            (args.directory / "distributed-test-result.json").write_text(
                json.dumps(
                    {
                        "world_size": 2,
                        "cpu_dtensor_shards_equal": True,
                        "uneven_mask_gradient_oracle": True,
                        "dcp_weights_optimizer_rng_equal": True,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
