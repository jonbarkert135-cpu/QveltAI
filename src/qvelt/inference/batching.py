"""One CPU model worker, bounded microbatches and cooperative per-request deadlines."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field

from qvelt.inference.config import TokenRequest
from qvelt.inference.engine import Engine
from qvelt.inference.generation import GenerationEvent


class QueueFullError(RuntimeError):
    pass


@dataclass
class Pending:
    request: TokenRequest
    deadline: float
    events: asyncio.Queue[GenerationEvent | Exception | None]
    cancelled: threading.Event = field(default_factory=threading.Event)


class Batcher:
    def __init__(
        self, engine: Engine, *, max_pending: int, batch_wait_ms: int, timeout: float
    ) -> None:
        if (
            type(max_pending) is not int
            or not 1 <= max_pending <= 64
            or type(batch_wait_ms) is not int
            or not 0 <= batch_wait_ms <= 100
            or type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= 300
        ):
            raise ValueError("Invalid bounded batching policy")
        self.engine = engine
        self.max_pending, self.batch_wait, self.timeout = (
            max_pending,
            batch_wait_ms / 1000,
            timeout,
        )
        self.queue: asyncio.Queue[Pending | None] = asyncio.Queue(max_pending)
        self.pending: dict[str, Pending] = {}
        self.task: asyncio.Task[None] | None = None
        self.closed = False
        self.batches: list[int] = []
        self.cancelled_or_expired = 0

    @property
    def ready(self) -> bool:
        return not self.closed and self.task is not None and not self.task.done()

    async def start(self) -> None:
        if self.task is not None:
            raise RuntimeError("Batcher already started")
        self.task = asyncio.create_task(self._run())

    def submit(self, request: TokenRequest) -> Pending:
        if not self.ready:
            raise RuntimeError("Worker unavailable")
        self.engine.check_batch([request])
        if len(self.pending) >= self.max_pending or self.queue.full():
            raise QueueFullError("Request capacity exhausted")
        if request.identifier in self.pending:
            raise ValueError("Duplicate request identifier")
        job = Pending(
            request,
            time.monotonic() + self.timeout,
            asyncio.Queue(request.options.max_new_tokens + 3),
        )
        self.pending[request.identifier] = job
        self.queue.put_nowait(job)
        return job

    def release(self, job: Pending) -> None:
        job.cancelled.set()
        self.pending.pop(job.request.identifier, None)

    async def next_event(self, job: Pending) -> GenerationEvent | None:
        remaining = job.deadline - time.monotonic()
        try:
            if remaining <= 0:
                raise TimeoutError
            event = await asyncio.wait_for(job.events.get(), remaining)
        except TimeoutError:
            job.cancelled.set()
            raise TimeoutError("Request deadline exceeded") from None
        if isinstance(event, Exception):
            raise event
        return event

    def _generate(self, batch: list[Pending], loop: asyncio.AbstractEventLoop) -> None:
        jobs = {job.request.identifier: job for job in batch}

        def cancelled(identifier: str) -> bool:
            job = jobs[identifier]
            return job.cancelled.is_set() or time.monotonic() >= job.deadline

        try:
            for event in self.engine.stream(
                tuple(job.request for job in batch), cancelled=cancelled
            ):
                if event.reason in ("cancelled", "timeout"):
                    self.cancelled_or_expired += 1
                loop.call_soon_threadsafe(
                    jobs[event.identifier].events.put_nowait, event
                )
        except Exception:
            for job in batch:
                loop.call_soon_threadsafe(
                    job.events.put_nowait, RuntimeError("Generation worker failed")
                )
        finally:
            for job in batch:
                loop.call_soon_threadsafe(job.events.put_nowait, None)

    async def _run(self) -> None:
        carry: Pending | None = None
        while not self.closed:
            first = carry if carry is not None else await self.queue.get()
            carry = None
            if first is None:
                break
            if first.cancelled.is_set():
                self.cancelled_or_expired += 1
                continue
            if self.batch_wait:
                await asyncio.sleep(self.batch_wait)
            batch = [first]
            while len(batch) < self.engine.limits.max_batch:
                try:
                    candidate = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if candidate is None:
                    self.closed = True
                    break
                if candidate.cancelled.is_set():
                    self.cancelled_or_expired += 1
                    continue
                try:
                    self.engine.check_batch(
                        [job.request for job in batch] + [candidate.request]
                    )
                except ValueError:
                    carry = candidate
                    break
                batch.append(candidate)
            self.batches.append(len(batch))
            # Retain bounded operational counters,not an unbounded request history.
            del self.batches[:-64]
            await asyncio.to_thread(self._generate, batch, asyncio.get_running_loop())

    async def close(self) -> None:
        self.closed = True
        for job in self.pending.values():
            job.cancelled.set()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(None)
        if self.task is not None:
            await self.task
        self.pending.clear()
