"""Real v3 BPE/decoder/cache/CPU engine; chat and production export stay gated."""

from pathlib import Path

import pytest
import torch

from qvelt.config import ExperimentConfig, ModelConfig
from qvelt.eval.config import ArtifactRef
from qvelt.eval.runner import prompt_ids
from qvelt.inference.artifacts import export_snapshot
from qvelt.inference.chat import chat_prefix
from qvelt.inference.config import EngineLimits, GenerationOptions, TextRequest
from qvelt.inference.generation import GenerationEngine
from qvelt.inference.huggingface import chat_template
from qvelt.model.decoder import Decoder
from qvelt.model.loss import causal_loss
from qvelt.posttrain.chat import Message, encode_chat
from qvelt.tokenizer.core import QveltTokenizer, TokenizerV3Settings, train_tokenizer
from qvelt.tokenizer.recipe import TokenizerRecipe, load_tokenizer_recipe
from qvelt.tokenizer.workflow import iter_documents, train_tokenizer_bundle
from qvelt.utils.manifests import manifest_digest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/tokenizer"
DATA_SHA = load_tokenizer_recipe(
    ROOT / "configs/tokenizer_smoke.yaml"
).training_data_sha256


@pytest.fixture()
def engine(tmp_path: Path) -> GenerationEngine:
    tokenizer = train_tokenizer(
        iter_documents(FIXTURES / "train.manifest.json", DATA_SHA),
        TokenizerV3Settings(512),
    )
    tokenizer.save(tmp_path / "v3.json")
    tokenizer = QveltTokenizer.load(tmp_path / "v3.json")
    with torch.random.fork_rng():
        torch.manual_seed(20261108)
        model = Decoder(
            ModelConfig(
                layers=2,
                hidden=64,
                heads=4,
                kv_heads=1,
                ffn=128,
                vocab=tokenizer.vocab_size,
                context=128,
            )
        ).eval()
    return GenerationEngine(model, tokenizer, EngineLimits(max_context=128))


def test_real_tokenizer_cpu_model_cache_gradient_and_generation(
    engine: GenerationEngine,
) -> None:
    model, tokenizer = engine.model, engine.tokenizer
    assert tokenizer.control_version == 3
    actual = sum(p.numel() for p in model.parameters())
    assert actual == model.config.parameter_count <= 20_000_000
    text = "Привіт! Привет! JSON 123"
    frame = [1, *tokenizer.encode(text), 2]
    assert all(t >= 128 for t in frame[1:-1])
    ids = torch.tensor([frame])
    with torch.no_grad():
        full = model(ids)
        first, cache = model.forward_cached(ids[:, :3])
        rest, _ = model.forward_cached(ids[:, 3:], cache)
    torch.testing.assert_close(torch.cat((first, rest), 1), full, atol=2e-6, rtol=2e-5)
    model.train()
    loss, count = causal_loss(model(ids[:, :-1]), ids[:, 1:])
    torch.autograd.backward(loss / count)
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    model.zero_grad(set_to_none=True)
    model.eval()
    # Gradients changed, but no parameter update took place; a fresh engine owns the cache.
    engine = GenerationEngine(model, tokenizer, EngineLimits(max_context=128))
    requests = engine.encode_requests(
        (
            TextRequest(
                "v3-plain",
                text,
                GenerationOptions(max_new_tokens=4, stop_tokens=(), context_limit=128),
            ),
        ),
        prompt_format="plain-v1",
    )
    left, right = engine.generate(requests)[0], engine.generate(requests)[0]
    assert left.token_ids == right.token_ids and len(left.token_ids) == 4
    assert tokenizer.decode(left.token_ids) == left.text


def test_v3_cannot_silently_use_legacy_chat(engine: GenerationEngine) -> None:
    tokenizer = engine.tokenizer
    messages = (Message("user", "Question"), Message("assistant", "Answer"))
    with pytest.raises(ValueError, match="schema3"):
        encode_chat(messages, tokenizer, 128)
    with pytest.raises(ValueError, match="schema3"):
        chat_prefix(messages[:1], tokenizer)
    with pytest.raises(ValueError, match="schema3"):
        chat_template(tokenizer.control_tokens, 3)
    with pytest.raises(ValueError, match="schema3"):
        prompt_ids(tokenizer, "Question", "chat-v1")
    with pytest.raises(ValueError, match="schema3"):
        engine.encode_requests(
            (TextRequest("chat", "Question"),), prompt_format="chat-v1"
        )


def test_v3_snapshot_export_rejects_before_output(
    engine: GenerationEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate export's tokenizer boundary; checkpoint verification is covered separately.
    path = tmp_path / "tokenizer"
    manifest = train_tokenizer_bundle(
        TokenizerRecipe(DATA_SHA, schema_version=2, tokenizer=TokenizerV3Settings(512)),
        FIXTURES / "train.manifest.json",
        path,
        ROOT,
    )
    monkeypatch.setattr(
        "qvelt.inference.artifacts.load_checkpoint_model",
        lambda *args: (
            engine.model,
            ExperimentConfig(model=engine.model.config),
            manifest,
        ),
    )
    with pytest.raises(ValueError, match="v3 snapshots"):
        export_snapshot(
            ArtifactRef("not-loaded", "a" * 64),
            ArtifactRef(str(path / "manifest.json"), manifest_digest(manifest)),
            tmp_path / "must-not-exist",
            ROOT,
        )
    assert not (tmp_path / "must-not-exist").exists()
