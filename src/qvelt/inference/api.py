"""Optional single-worker OpenAI chat HTTP/SSE adapter with bounded native batching."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from qvelt.inference.batching import Batcher, QueueFullError
from qvelt.inference.chat import chat_prefix
from qvelt.inference.config import GenerationOptions, TokenRequest
from qvelt.inference.engine import Engine
from qvelt.inference.generation import GenerationEvent, display_text
from qvelt.inference.serving_config import ServingConfig
from qvelt.posttrain.chat import Message


class ChatMessage(BaseModel):  # type: ignore[misc]  # Optional SDK import boundary.
    model_config = ConfigDict(extra="forbid", strict=True)
    role: str
    content: str = Field(min_length=1, max_length=8192)
    reasoning: str | None = Field(default=None, min_length=1, max_length=8192)


class ChatRequest(BaseModel):  # type: ignore[misc]  # Optional SDK import boundary.
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    stream: bool = False
    max_tokens: int = Field(default=32, ge=1, le=256)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=262144)
    min_p: float = Field(default=0.0, ge=0, le=1)
    seed: int = Field(default=20261107, ge=0, lt=2**32)
    stop: str | list[str] | None = None
    repetition_penalty: float = Field(default=1.0, ge=0.01, le=100)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    n: int = Field(default=1, ge=1, le=1)
    stream_options: dict[str, bool] | None = None


def error(
    status: int, message: str, kind: str = "invalid_request_error"
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind, "param": None, "code": kind}},
        status_code=status,
    )


class RequestBoundary:
    """Authenticate before reading a body;bound bytes before JSON/Pydantic parsing."""

    def __init__(self, app: Any, *, key: str | None, max_body_bytes: int) -> None:
        self.app, self.key, self.maximum = app, key, max_body_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"].startswith("/v1/") and self.key is not None:
            headers = dict(scope.get("headers", []))
            actual = headers.get(b"authorization", b"")
            expected = ("Bearer " + self.key).encode()
            if not hmac.compare_digest(
                hashlib.sha256(actual).digest(), hashlib.sha256(expected).digest()
            ):
                await error(401, "Invalid bearer token", "authentication_error")(
                    scope, receive, send
                )
                return
        if scope["method"] == "POST":
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.maximum:
                    await error(413, "Request body exceeds byte limit")(
                        scope, receive, send
                    )
                    return
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay() -> dict[str, Any]:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            await self.app(scope, replay, send)
        else:
            await self.app(scope, receive, send)


def stable_text(event: GenerationEvent, stops: tuple[str, ...]) -> str:
    """Byte-BPE can revise trailing replacement chars;hold partial Unicode/stop prefixes."""
    return (
        event.text
        if event.done
        else display_text(event.text.rstrip("\ufffd"), stops, final=False)[0]
    )


def _finish(event: GenerationEvent) -> str:
    if event.reason in ("stop_token", "stop_string"):
        return "stop"
    if event.reason in ("max_tokens", "output_limit"):
        return "length"
    if event.reason in ("timeout", "cancelled"):
        raise TimeoutError("Request deadline/cancellation")
    raise RuntimeError("Generation failed")


def _sse(value: object) -> str:
    return "data: " + json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n\n"


def create_app(
    engine: Engine, settings: ServingConfig, *, api_key: str | None = None
) -> FastAPI:
    if api_key is None and not settings.allow_unauthenticated:
        raise ValueError(
            "Serving requires configured bearer auth or explicit loopback-only opt-out"
        )
    if api_key is not None and (
        not 16 <= len(api_key) <= 512 or any(ord(c) < 33 for c in api_key)
    ):
        raise ValueError("Invalid serving key configuration")
    batcher = Batcher(
        engine,
        max_pending=settings.max_pending,
        batch_wait_ms=settings.batch_wait_ms,
        timeout=settings.request_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await batcher.start()
        try:
            yield
        finally:
            await batcher.close()
            # Uvicorn re-raises SIGTERM after lifespan cleanup;emit evidence here.
            app.state.shutdown_summary = {
                "stopped": True,
                "pending_after_shutdown": len(batcher.pending),
                "batches": batcher.batches,
                "cancelled_or_expired": batcher.cancelled_or_expired,
            }
            print(json.dumps(app.state.shutdown_summary), flush=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        RequestBoundary, key=api_key, max_body_bytes=settings.max_body_bytes
    )
    app.state.batcher = batcher

    @app.exception_handler(RequestValidationError)  # type: ignore[misc]  # Optional FastAPI decorator boundary.
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error(400, "Invalid chat request schema")

    @app.get("/healthz")  # type: ignore[misc]  # Optional FastAPI decorator boundary.
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")  # type: ignore[misc]  # Optional FastAPI decorator boundary.
    async def ready() -> JSONResponse:
        return JSONResponse(
            {"ready": batcher.ready}, status_code=200 if batcher.ready else 503
        )

    @app.get("/v1/models")  # type: ignore[misc]  # Optional FastAPI decorator boundary.
    async def models() -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "qvelt",
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")  # type: ignore[misc]  # Optional FastAPI decorator boundary.
    async def completions(body: ChatRequest, request: Request) -> Any:
        if body.model != settings.model_name:
            return error(404, "Unknown model", "model_not_found")
        if body.stream_options is not None and (
            set(body.stream_options) != {"include_usage"} or not body.stream
        ):
            return error(400, "Unsupported stream options")
        try:
            stops = (
                ()
                if body.stop is None
                else (body.stop,)
                if isinstance(body.stop, str)
                else tuple(body.stop)
            )
            options = GenerationOptions(
                max_new_tokens=body.max_completion_tokens or body.max_tokens,
                strategy="greedy" if body.temperature == 0 else "sample",
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                min_p=body.min_p,
                seed=body.seed,
                repetition_penalty=body.repetition_penalty,
                frequency_penalty=body.frequency_penalty,
                presence_penalty=body.presence_penalty,
                stop_strings=stops,
                context_limit=engine.limits.max_context,
                max_seconds=settings.request_timeout,
            )
            tokens = chat_prefix(
                tuple(Message(m.role, m.content, m.reasoning) for m in body.messages),
                engine.tokenizer,
            )
            ident = "chatcmpl-" + uuid.uuid4().hex
            job = batcher.submit(TokenRequest(ident, tokens, options))
        except QueueFullError:
            response = error(429, "Request capacity exhausted", "rate_limit_error")
            response.headers["Retry-After"] = "1"
            return response
        except ValueError:
            return error(400, "Invalid chat framing or generation capacity")
        except RuntimeError:
            return error(503, "Generation worker unavailable", "server_error")
        created = int(time.time())
        envelope = {
            "id": ident,
            "created": created,
            "model": settings.model_name,
            "system_fingerprint": "qvelt-native-cpu-v1",
        }

        def usage(event: GenerationEvent) -> dict[str, int]:
            return {
                "prompt_tokens": len(tokens),
                "completion_tokens": event.generated_tokens,
                "total_tokens": len(tokens) + event.generated_tokens,
            }

        async def next_event() -> GenerationEvent:
            if await request.is_disconnected():
                raise TimeoutError("Client disconnected")
            event = await batcher.next_event(job)
            if event is None:
                raise RuntimeError("Generation ended without a terminal event")
            return event

        if not body.stream:
            try:
                while True:
                    event = await next_event()
                    if event.done:
                        reason = _finish(event)
                        return JSONResponse(
                            {
                                **envelope,
                                "object": "chat.completion",
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": event.text,
                                        },
                                        "finish_reason": reason,
                                        "logprobs": None,
                                    }
                                ],
                                "usage": usage(event),
                            }
                        )
            except TimeoutError:
                return error(504, "Request deadline exceeded", "timeout_error")
            except RuntimeError:
                return error(500, "Generation failed", "server_error")
            finally:
                batcher.release(job)

        async def stream() -> AsyncIterator[str]:
            previous = ""
            try:
                yield _sse(
                    {
                        **envelope,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                while True:
                    event = await next_event()
                    text = stable_text(event, stops)
                    if not text.startswith(previous):
                        raise RuntimeError("Nonmonotonic decoded stream")
                    delta = text[len(previous) :]
                    previous = text
                    if delta:
                        yield _sse(
                            {
                                **envelope,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    if event.done:
                        reason = _finish(event)
                        yield _sse(
                            {
                                **envelope,
                                "object": "chat.completion.chunk",
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": reason}
                                ],
                            }
                        )
                        if body.stream_options and body.stream_options["include_usage"]:
                            yield _sse(
                                {
                                    **envelope,
                                    "object": "chat.completion.chunk",
                                    "choices": [],
                                    "usage": usage(event),
                                }
                            )
                        break
            except TimeoutError:
                yield _sse(
                    {
                        "error": {
                            "message": "Request deadline exceeded",
                            "type": "timeout_error",
                            "code": "timeout_error",
                        }
                    }
                )
            except RuntimeError:
                yield _sse(
                    {
                        "error": {
                            "message": "Generation failed",
                            "type": "server_error",
                            "code": "server_error",
                        }
                    }
                )
            finally:
                batcher.release(job)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
