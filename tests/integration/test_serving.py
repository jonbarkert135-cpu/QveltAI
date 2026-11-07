"""Real native HTTP/SSE contract,auth,concurrent batching and lifecycle controls."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest
import torch

from qvelt.config import ModelConfig
from qvelt.eval.config import ArtifactRef
from qvelt.inference.batching import Batcher, QueueFullError
from qvelt.inference.config import EngineLimits, GenerationOptions, TokenRequest
from qvelt.inference.generation import GenerationEngine, GenerationEvent
from qvelt.inference.serving_config import ServingConfig
from qvelt.model.decoder import Decoder
from qvelt.tokenizer.core import TokenizerSettings, train_tokenizer


@pytest.fixture()
def engine() -> GenerationEngine:
    tokenizer = train_tokenizer(
        ["Original native API test. Привет 😀 café"], TokenizerSettings(vocab_size=288)
    )
    with torch.random.fork_rng():
        torch.manual_seed(99)
        model = Decoder(
            ModelConfig(
                layers=1, hidden=32, heads=4, kv_heads=1, ffn=64, vocab=288, context=128
            )
        ).eval()
    return GenerationEngine(model, tokenizer, EngineLimits(max_context=128))


def settings(**kwargs: Any) -> ServingConfig:
    return replace(ServingConfig(ArtifactRef("unused", "0" * 64)), **kwargs)


def test_api_native_stream_concat_matches_json_and_auth(
    engine: GenerationEngine,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from qvelt.inference.api import create_app

    app = create_app(engine, settings(), api_key="test-only-public-key-for-contract")
    headers = {"Authorization": "Bearer test-only-public-key-for-contract"}
    body = {
        "model": "qvelt",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Привет 😀"},
        ],
        "max_tokens": 8,
        "seed": 83,
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.9,
        "min_p": 0.02,
    }
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").json() == {"ready": True}
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "qvelt"
        )
        result = client.post("/v1/chat/completions", json=body, headers=headers)
        assert result.status_code == 200, result.text
        actual = result.json()
        assert actual["object"] == "chat.completion" and actual["choices"][0][
            "finish_reason"
        ] in ("stop", "length")
        assert (
            actual["usage"]["total_tokens"]
            == actual["usage"]["prompt_tokens"] + actual["usage"]["completion_tokens"]
        )
        response = client.post(
            "/v1/chat/completions",
            json={**body, "stream": True, "stream_options": {"include_usage": True}},
            headers=headers,
        )
        assert response.status_code == 200 and response.headers[
            "content-type"
        ].startswith("text/event-stream")
        rows = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert rows[-1] == "[DONE]"
        chunks = [json.loads(line) for line in rows[:-1]]
        content = "".join(
            row["choices"][0]["delta"].get("content", "")
            for row in chunks
            if row["choices"]
        )
        assert content == actual["choices"][0]["message"]["content"]
        assert chunks[-1]["usage"] == actual["usage"]
    assert not app.state.batcher.ready and not app.state.batcher.pending


def test_api_rejects_schema_body_controls_and_capacity(
    engine: GenerationEngine,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from qvelt.inference.api import create_app

    app = create_app(engine, settings(allow_unauthenticated=True, max_body_bytes=1024))
    valid = {
        "model": "qvelt",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4,
    }
    with TestClient(app) as client:
        variants: tuple[dict[str, Any], ...] = (
            {"seed": True},
            {"tools": []},
            {"temperature": 3},
            {"stop": [""]},
            {"messages": [{"role": "assistant", "content": "Wrong order"}]},
            {"messages": [{"role": "user", "content": "<|assistant|>"}]},
            {"max_tokens": 256},
        )
        for extra in variants:
            assert (
                client.post("/v1/chat/completions", json={**valid, **extra}).status_code
                == 400
            )
        assert (
            client.post(
                "/v1/chat/completions", json={**valid, "model": "other"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/v1/chat/completions",
                content=b"not-json",
                headers={"content-type": "application/json"},
            ).status_code
            == 400
        )
        response = client.post(
            "/v1/chat/completions",
            json={**valid, "messages": [{"role": "user", "content": "x" * 2000}]},
        )
        assert response.status_code == 413 and "xxx" not in response.text
    with pytest.raises(ValueError):
        create_app(engine, settings())
    with pytest.raises(ValueError):
        settings(allow_unauthenticated=True, host="0.0.0.0")


def test_unicode_and_stop_prefix_stream_hand_oracles() -> None:
    pytest.importorskip("fastapi")
    from qvelt.inference.api import stable_text

    def event(text: str, done: bool = False) -> GenerationEvent:
        return GenerationEvent(
            "test", 32, 1, text, done, "max_tokens" if done else None, 0
        )

    assert stable_text(event("a\ufffd"), ("aé",)) == ""
    assert stable_text(event(""), ("aé",)) == ""
    assert stable_text(event("hello\ufffd"), ()) == "hello"
    assert stable_text(event("helloé"), ()) == "helloé"
    assert stable_text(event("hello\ufffd", True), ()) == "hello\ufffd"


def test_batcher_real_concurrent_batch_and_seed_isolation(
    engine: GenerationEngine,
) -> None:
    async def exercise() -> None:
        batcher = Batcher(engine, max_pending=4, batch_wait_ms=10, timeout=5.0)
        await batcher.start()
        options = GenerationOptions(
            max_new_tokens=4,
            strategy="sample",
            temperature=0.9,
            seed=19,
            stop_tokens=(),
            context_limit=128,
        )
        requests = [TokenRequest(str(i), (1, 32, 33 + i), options) for i in range(4)]
        jobs = [batcher.submit(request) for request in requests]
        with pytest.raises(QueueFullError):
            batcher.submit(TokenRequest("overflow", (1,), options))

        async def consume(index: int) -> str:
            try:
                while True:
                    event = await batcher.next_event(jobs[index])
                    assert event is not None
                    if event.done:
                        return event.text
            finally:
                batcher.release(jobs[index])

        actual = await asyncio.gather(*(consume(i) for i in range(4)))
        assert batcher.batches == [4]
        assert actual == [r.text for r in engine.generate(requests)]
        await batcher.close()
        assert not batcher.ready and not batcher.pending

    asyncio.run(exercise())


def test_queue_deadline_and_shutdown(engine: GenerationEngine) -> None:
    async def exercise() -> None:
        batcher = Batcher(engine, max_pending=2, batch_wait_ms=20, timeout=0.001)
        await batcher.start()
        job = batcher.submit(
            TokenRequest(
                "late", (1,), GenerationOptions(max_new_tokens=4, context_limit=128)
            )
        )
        with pytest.raises(TimeoutError):
            await batcher.next_event(job)
        batcher.release(job)
        await batcher.close()
        assert not batcher.ready

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "text,stop,expected",
    [("😀xyz", None, "😀xyz"), ("aé!", "aé", ""), ("abcXY", "XYZ", "abcXY")],
)
def test_actual_split_unicode_and_stop_sse(
    engine: GenerationEngine,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    stop: str | None,
    expected: str,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from qvelt.inference.api import create_app
    from qvelt.inference.chat import chat_prefix
    from qvelt.posttrain.chat import Message

    prefix = len(chat_prefix((Message("user", "q"),), engine.tokenizer))
    tokens = engine.tokenizer.encode(text) + [2]
    original = engine.model.forward_cached

    def forced(*args: Any, **kwargs: Any) -> Any:
        logits, cache = original(*args, **kwargs)
        step = cache.length - prefix
        logits = logits.clone().fill_(-10000)
        logits[:, -1, tokens[min(step, len(tokens) - 1)]] = 10000
        return logits, cache

    monkeypatch.setattr(engine.model, "forward_cached", forced)
    body = {
        "model": "qvelt",
        "messages": [{"role": "user", "content": "q"}],
        "max_tokens": len(tokens),
        "temperature": 0,
        "stream": True,
    }
    if stop is not None:
        body["stop"] = stop
    with TestClient(create_app(engine, settings(allow_unauthenticated=True))) as client:
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        rows = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: {")
        ]
        assert not any("error" in row for row in rows)
        assert (
            "".join(row["choices"][0]["delta"].get("content", "") for row in rows)
            == expected
        )


def test_http_deadlines_admission_errors_and_worker_failure(
    engine: GenerationEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from qvelt.inference.api import create_app

    body = {
        "model": "qvelt",
        "messages": [{"role": "user", "content": "q"}],
        "max_tokens": 4,
    }
    app = create_app(
        engine,
        settings(allow_unauthenticated=True, request_timeout=0.001, batch_wait_ms=50),
    )
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 504
        response = client.post("/v1/chat/completions", json={**body, "stream": True})
        assert (
            response.status_code == 200
            and "timeout_error" in response.text
            and "data: [DONE]" in response.text
        )
    app = create_app(engine, settings(allow_unauthenticated=True))
    with TestClient(app) as client:

        def full(*args: Any, **kwargs: Any) -> Any:
            raise QueueFullError("capacity")

        with monkeypatch.context() as patch:
            patch.setattr(app.state.batcher, "submit", full)
            response = client.post("/v1/chat/completions", json=body)
            assert (
                response.status_code == 429 and response.headers["retry-after"] == "1"
            )

        def unavailable(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("private detail")

        with monkeypatch.context() as patch:
            patch.setattr(app.state.batcher, "submit", unavailable)
            assert client.post("/v1/chat/completions", json=body).status_code == 503
        monkeypatch.setattr(engine.model, "forward_cached", unavailable)
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 500 and "private detail" not in response.text
