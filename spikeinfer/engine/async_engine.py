"""Asyncio wrapper around :class:`LLMEngine`, for the HTTP server.

The engine is single-threaded and synchronous by design -- one CUDA stream, one
step at a time. This wraps it in a background task that keeps stepping while
requests exist and fans each step's outputs into per-request asyncio queues, so
many concurrent HTTP connections share one continuously-batched engine.

The engine step runs in a thread executor, not on the event loop, so a step
never blocks the streaming of tokens produced by the previous one.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..config import EngineConfig
from ..sampling_params import SamplingParams
from .llm_engine import LLMEngine
from .sequence import RequestOutput


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._streams: dict[str, asyncio.Queue] = {}
        self._loop_task: asyncio.Task | None = None
        self._request_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._errored: BaseException | None = None

    @classmethod
    def from_engine_config(cls, config: EngineConfig) -> AsyncLLMEngine:
        return cls(LLMEngine(config))

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.get_running_loop().create_task(self._run())

    async def shutdown(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    # -- request API ------------------------------------------------------

    async def generate(
        self,
        prompt: str | None,
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: list[int] | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Yield a :class:`RequestOutput` per engine step until finished."""
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._streams[request_id] = queue
            self.engine.add_request(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
            )
        self.start()
        self._request_event.set()

        try:
            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            async with self._lock:
                self._streams.pop(request_id, None)
                if not self.engine.has_unfinished_requests():
                    pass

    async def abort(self, request_id: str) -> None:
        async with self._lock:
            self.engine.abort_request(request_id)
            queue = self._streams.pop(request_id, None)
        if queue is not None:
            await queue.put(asyncio.CancelledError())

    # -- background loop --------------------------------------------------

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.engine.has_unfinished_requests():
                self._request_event.clear()
                await self._request_event.wait()
                continue
            try:
                outputs = await loop.run_in_executor(None, self.engine.step)
            except BaseException as exc:  # surface to every waiting client
                async with self._lock:
                    for queue in self._streams.values():
                        queue.put_nowait(exc)
                    self._streams.clear()
                self._errored = exc
                raise
            for output in outputs:
                queue = self._streams.get(output.request_id)
                if queue is not None:
                    queue.put_nowait(output)
            # Yield to the loop so streaming clients drain between steps.
            await asyncio.sleep(0)

    # -- introspection ----------------------------------------------------

    def describe(self) -> dict:
        return self.engine.describe()

    @property
    def stats(self):
        return self.engine.stats
