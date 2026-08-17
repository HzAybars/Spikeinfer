"""OpenAI-compatible HTTP server.

    spikeinfer serve ./qwen-spiking --port 8000

    curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"spiking-qwen","messages":[{"role":"user","content":"hi"}]}'

One :class:`AsyncLLMEngine` backs every connection, so concurrent requests are
continuously batched rather than queued behind one another.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from ...config import EngineConfig
from ...engine.async_engine import AsyncLLMEngine
from ...engine.sequence import RequestOutput
from .protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessageResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
    DeltaMessage,
    DetokenizeRequest,
    DetokenizeResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    TokenizeRequest,
    TokenizeResponse,
    UsageInfo,
    _request_id,
)

DONE = "data: [DONE]\n\n"


def build_app(
    engine: AsyncLLMEngine,
    served_model_name: str = "spiking-qwen",
    api_key: str | None = None,
    allow_origins: list[str] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="spikeinfer",
        description="Inference server for spiking (LIF) transformers",
        version=_version(),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine
    app.state.model_name = served_model_name

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1"):
            header = request.headers.get("Authorization", "")
            if header != f"Bearer {api_key}":
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(
                        message="invalid API key", type="authentication_error", code=401
                    ).model_dump(),
                )
        return await call_next(request)

    # -- health and introspection ----------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        info = engine.describe()
        return ModelList(
            data=[
                ModelCard(
                    id=served_model_name,
                    root=info["model"],
                    max_model_len=info["max_model_len"],
                )
            ]
        )

    @app.get("/stats")
    async def stats():
        return {**engine.describe(), **_stats_dict(engine)}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return engine.stats.to_prometheus(served_model_name)

    @app.post("/tokenize")
    async def tokenize(request: TokenizeRequest):
        tokenizer = _tokenizer(engine)
        ids = tokenizer.encode(request.prompt)
        return TokenizeResponse(
            tokens=ids, count=len(ids), max_model_len=engine.describe()["max_model_len"]
        )

    @app.post("/detokenize")
    async def detokenize(request: DetokenizeRequest):
        return DetokenizeResponse(prompt=_tokenizer(engine).decode(request.tokens))

    # -- completions ------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt, token_ids = _resolve_prompt(engine, request.prompt)
        params = request.to_sampling_params(logprobs=request.logprobs)
        request_id = _request_id("cmpl")
        stream = engine.generate(prompt, params, request_id, prompt_token_ids=token_ids)

        if request.stream:
            return StreamingResponse(
                _stream_completions(stream, request_id, served_model_name, request.echo, prompt),
                media_type="text/event-stream",
            )

        final = await _drain(stream)
        prefix = prompt if request.echo and prompt else ""
        return CompletionResponse(
            id=request_id,
            model=served_model_name,
            choices=[
                CompletionChoice(
                    index=out.index,
                    text=prefix + out.text,
                    finish_reason=out.finish_reason,
                    stop_reason=out.stop_reason,
                )
                for out in final.outputs
            ],
            usage=_usage(final),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        prompt = _apply_chat_template(engine, request)
        params = request.to_sampling_params(logprobs=request.resolved_logprobs())
        request_id = _request_id("chatcmpl")
        stream = engine.generate(prompt, params, request_id)

        if request.stream:
            return StreamingResponse(
                _stream_chat(stream, request_id, served_model_name),
                media_type="text/event-stream",
            )

        final = await _drain(stream)
        return ChatCompletionResponse(
            id=request_id,
            model=served_model_name,
            choices=[
                ChatCompletionChoice(
                    index=out.index,
                    message=ChatMessageResponse(content=out.text),
                    finish_reason=out.finish_reason,
                    stop_reason=out.stop_reason,
                )
                for out in final.outputs
            ],
            usage=_usage(final),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content=ErrorResponse(message=str(exc)).model_dump())

    return app


# -- streaming helpers ----------------------------------------------------


async def _stream_completions(
    stream: AsyncIterator[RequestOutput],
    request_id: str,
    model: str,
    echo: bool,
    prompt: str | None,
) -> AsyncIterator[str]:
    first = True
    final: RequestOutput | None = None
    async for output in stream:
        final = output
        for out in output.outputs:
            text = out.delta
            if first and echo and prompt:
                text = prompt + text
            if not text and out.finish_reason is None:
                continue
            chunk = CompletionStreamResponse(
                id=request_id,
                model=model,
                choices=[
                    CompletionChoice(
                        index=out.index,
                        text=text,
                        finish_reason=out.finish_reason,
                        stop_reason=out.stop_reason,
                    )
                ],
            )
            first = False
            yield f"data: {chunk.model_dump_json()}\n\n"
    if final is not None:
        usage = CompletionStreamResponse(
            id=request_id, model=model, choices=[], usage=_usage(final)
        )
        yield f"data: {usage.model_dump_json()}\n\n"
    yield DONE


async def _stream_chat(
    stream: AsyncIterator[RequestOutput], request_id: str, model: str
) -> AsyncIterator[str]:
    opened = False
    final: RequestOutput | None = None
    async for output in stream:
        final = output
        if not opened:
            opened = True
            head = ChatCompletionStreamResponse(
                id=request_id,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(index=i, delta=DeltaMessage(role="assistant"))
                    for i in range(len(output.outputs))
                ],
            )
            yield f"data: {head.model_dump_json()}\n\n"
        for out in output.outputs:
            if not out.delta and out.finish_reason is None:
                continue
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=out.index,
                        delta=DeltaMessage(content=out.delta or ""),
                        finish_reason=out.finish_reason,
                        stop_reason=out.stop_reason,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
    if final is not None:
        usage = ChatCompletionStreamResponse(
            id=request_id, model=model, choices=[], usage=_usage(final)
        )
        yield f"data: {usage.model_dump_json()}\n\n"
    yield DONE


async def _drain(stream: AsyncIterator[RequestOutput]) -> RequestOutput:
    final: RequestOutput | None = None
    async for output in stream:
        final = output
    if final is None:
        raise HTTPException(status_code=500, detail="engine produced no output")
    return final


# -- plumbing -------------------------------------------------------------


def _tokenizer(engine: AsyncLLMEngine):
    tokenizer = engine.tokenizer
    if tokenizer is None:
        raise HTTPException(status_code=400, detail="no tokenizer available for this model")
    return tokenizer


def _resolve_prompt(engine: AsyncLLMEngine, prompt) -> tuple[str | None, list[int] | None]:
    """OpenAI allows a string, a token-id list, or a batch of either."""
    if isinstance(prompt, str):
        return prompt, None
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return None, list(prompt)
    if isinstance(prompt, list) and len(prompt) == 1:
        return _resolve_prompt(engine, prompt[0])
    raise HTTPException(
        status_code=400,
        detail="batched prompts are not supported; send one request per prompt",
    )


def _apply_chat_template(engine: AsyncLLMEngine, request: ChatCompletionRequest) -> str:
    tokenizer = _tokenizer(engine)
    messages = [
        {"role": m.role, "content": m.text(), **({"name": m.name} if m.name else {})}
        for m in request.messages
    ]
    if getattr(tokenizer, "chat_template", None) is None and request.chat_template is None:
        # Base models have no template; fall back to a plain transcript rather
        # than refusing, since these checkpoints are usually base models.
        body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return body + "\nassistant:"
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(request.add_generation_prompt),
        chat_template=request.chat_template,
    )


def _usage(output: RequestOutput) -> UsageInfo:
    prompt_tokens = len(output.prompt_token_ids)
    completion_tokens = sum(len(o.token_ids) for o in output.outputs)
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _stats_dict(engine: AsyncLLMEngine) -> dict:
    stats = engine.stats
    return {
        "running": stats.num_running,
        "waiting": stats.num_waiting,
        "cache_usage": round(stats.cache_usage, 4),
        "generation_tokens": stats.generation_tokens,
        "prompt_tokens": stats.prompt_tokens,
        "tokens_per_second": round(stats.throughput, 2),
        "avg_ttft_ms": round(stats.avg_ttft_ms, 2),
        "avg_itl_ms": round(stats.avg_itl_ms, 3),
        "preemptions": stats.preemptions,
        "uptime_s": round(stats.uptime, 1),
    }


def _version() -> str:
    from ... import __version__

    return __version__


def run_server(
    config: EngineConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
    served_model_name: str = "spiking-qwen",
    api_key: str | None = None,
    log_level: str = "info",
) -> None:
    """Build the engine, then hand the app to uvicorn."""
    import uvicorn

    engine = AsyncLLMEngine.from_engine_config(config)
    info = engine.describe()
    print(json.dumps(info, indent=2))
    print(f"\nspikeinfer serving '{served_model_name}' on http://{host}:{port}")
    app = build_app(engine, served_model_name=served_model_name, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
