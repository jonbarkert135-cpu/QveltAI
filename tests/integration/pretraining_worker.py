"""Actual two-rank uneven accumulation and coordinated rank-local failure oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from qvelt.config import ModelConfig
from qvelt.model.decoder import Decoder
from qvelt.train.accumulation import accumulate_gradients
from qvelt.utils.distributed import distributed_context


def sample(rank: int, micro: int) -> dict[str, Tensor]:
    tokens = torch.tensor([[32 + rank * 8 + micro, 33, 34, 35]])
    masks = {
        (0, 0): [True, True, True, False],
        (0, 1): [False] * 4,
        (1, 0): [True, False, False, False],
        (1, 1): [True, True, False, False],
    }
    return {
        "input_ids": tokens,
        "labels": tokens + 1,
        "positions": torch.arange(4)[None],
        "document_ids": torch.zeros_like(tokens),
        "valid_mask": torch.ones_like(tokens, dtype=torch.bool),
        "loss_mask": torch.tensor([masks[(rank, micro)]]),
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(19)
    with distributed_context("cpu") as context:
        config = ModelConfig(
            layers=1, hidden=16, heads=2, kv_heads=1, ffn=32, vocab=288, context=4
        )
        model = Decoder(config)
        oracle = Decoder(config)
        oracle.load_state_dict(model.state_dict())
        ddp = DistributedDataParallel(model, broadcast_buffers=False)
        local = iter([sample(context.rank, micro) for micro in range(2)])
        stats = accumulate_gradients(ddp, lambda: next(local), 2, context)
        all_batches = [sample(rank, micro) for rank in range(2) for micro in range(2)]
        merged = {
            name: torch.cat([value[name] for value in all_batches])
            for name in all_batches[0]
        }
        logits = oracle(
            merged["input_ids"],
            merged["positions"],
            merged["document_ids"],
            merged["valid_mask"],
        )
        selected = merged["loss_mask"]
        expected = F.cross_entropy(logits[selected], merged["labels"][selected])
        torch.autograd.backward(expected)
        for a, b in zip(model.parameters(), oracle.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-5, atol=5e-6)
        assert stats.targets == 6
        assert abs(stats.loss - float(expected.detach())) < 1e-5

        def fail_one_rank() -> None:
            if context.rank == 1:
                raise OSError("Deliberate rank-local failure")

        failed = False
        try:
            context.collective(fail_one_rank, "Coordinated failure probe")
        except RuntimeError:
            failed = True
        assert int(context.sum(torch.tensor(int(failed))).item()) == context.world_size
        if context.rank == 0:
            Path(sys.argv[1]).write_text(
                json.dumps(
                    {
                        "uneven_window_gradients": True,
                        "global_targets": stats.targets,
                        "all_ranks_failed_together": True,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
