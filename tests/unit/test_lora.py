"""Dense value/gradient references, frozen base and merged decoder checks."""

import pytest
import torch
from torch import nn

from qvelt.config import ModelConfig
from qvelt.model.decoder import Decoder
from qvelt.posttrain.lora import LoRALinear, attach_lora, merge_lora


def test_lora_dense_values_gradients_and_initialization() -> None:
    torch.manual_seed(7)
    base = nn.Linear(5, 4, bias=False).double()
    layer = LoRALinear(base, 2, 3)
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.equal(layer(x), base(x))
    with torch.no_grad():
        layer.b.normal_()
    actual = layer(x)
    reference = x @ (base.weight.detach() + 1.5 * (layer.b @ layer.a)).T
    torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-13)
    for a, b in zip(
        torch.autograd.grad(
            actual.square().sum(), (x, layer.a, layer.b), retain_graph=True
        ),
        torch.autograd.grad(reference.square().sum(), (x, layer.a, layer.b)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert not layer.weight.requires_grad and base.weight.grad is None


def test_adapter_only_updates_and_merge_parity() -> None:
    torch.manual_seed(19)
    model = Decoder(
        ModelConfig(
            vocab=288, context=32, layers=1, hidden=16, heads=2, kv_heads=1, ffn=48
        )
    )
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    count = attach_lora(model, 2, 4)
    assert count == 2 * (
        16 + 16 + 16 + 8 + 16 + 8 + 16 + 16 + 16 + 48 + 16 + 48 + 48 + 16
    )
    ids = torch.tensor([[1, 4, 40, 6, 5, 41]])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    for _ in range(3):
        optimizer.zero_grad()
        model(ids).square().mean().backward()
        optimizer.step()
    for name, value in initial.items():
        assert torch.equal(model.state_dict()[name], value)
    assert any(
        bool(module.b.abs().sum())
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )
    expected = model.eval()(ids).detach()
    merge_lora(model)
    assert set(model.state_dict()) == set(initial)
    torch.testing.assert_close(model(ids), expected, rtol=2e-5, atol=1e-6)
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(5, 4), 2, 4)
