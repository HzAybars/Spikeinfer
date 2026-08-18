"""OpenAI API surface.

Compatibility is the feature, so these tests assert on the wire format: the
field names, the SSE framing, the ``[DONE]`` sentinel, and the usage block --
the things a client library breaks on. A character-level stub tokenizer stands
in for a real one so the whole API can be exercised without any tokenizer files.
"""
from __future__ import annotations

import json

import pytest
import torch

from conftest import CUDA_AVAILABLE, build_fast_model, small_config

TEST_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


class CharTokenizer:
    """Character-level tokenizer over the tiny model's 512-token vocabulary."""

    eos_token_id = 511
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 500) + 1 for c in text] or [1]

    def decode(self, ids) -> str:
        return "".join(self._char(i) for i in ids)

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self._char(i) for i in ids]

    @staticmethod
    def _char(token_id: int) -> str:
        return chr(max(0, token_id - 1))

    def convert_tokens_to_string(self, tokens) -> str:
        return "".join(tokens)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from spikeinfer.config import (
        CacheConfig,
        EngineConfig,
        GraphConfig,
        ModelConfig,
        SchedulerConfig,
    )
    from spikeinfer.engine.async_engine import AsyncLLMEngine
    from spikeinfer.engine.llm_engine import LLMEngine
    from spikeinfer.entrypoints.openai.server import build_app
    from spikeinfer.loader import save_model

    cfg = small_config()
    path = tmp_path_factory.mktemp("server_model")
    save_model(build_fast_model(cfg, "cpu", seed=1), cfg, path)

    config = EngineConfig(
        model=ModelConfig(model=str(path), dtype="float32"),
        cache=CacheConfig(block_size=16, num_gpu_blocks_override=64),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128, max_model_len=64),
        graph=GraphConfig(enabled=False),
        device=TEST_DEVICE,
    )
    engine = AsyncLLMEngine(LLMEngine(config, tokenizer=CharTokenizer()))
    app = build_app(engine, served_model_name="test-model")
    with TestClient(app) as test_client:
        yield test_client
    del engine
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()


def sse_events(text: str) -> list[dict]:
    """Parse an SSE body into its JSON payloads, dropping the sentinel."""
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


# -- introspection --------------------------------------------------------


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "test-model"
    assert body["data"][0]["max_model_len"] == 64


def test_metrics_are_prometheus_text(client):
    text = client.get("/metrics").text
    assert "# TYPE spikeinfer_generation_tokens_total counter" in text
    assert 'spikeinfer_gpu_cache_blocks{model="test-model"}' in text


def test_stats_endpoint(client):
    body = client.get("/stats").json()
    assert body["block_size"] == 16
    assert "tokens_per_second" in body
    assert "kv_capacity_tokens" in body


def test_tokenize_and_detokenize(client):
    tokens = client.post("/tokenize", json={"prompt": "abc"}).json()
    assert tokens["count"] == 3
    assert tokens["max_model_len"] == 64

    text = client.post("/detokenize", json={"tokens": tokens["tokens"]}).json()
    assert text["prompt"] == "abc"


# -- completions ----------------------------------------------------------


def test_completion_returns_the_openai_shape(client):
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "hello", "max_tokens": 5, "temperature": 0},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert len(body["choices"]) == 1
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 5


def test_completion_accepts_token_ids(client):
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": [4, 8, 15], "max_tokens": 3, "temperature": 0},
    )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 3


def test_completion_echo(client):
    body = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "abc", "max_tokens": 2,
              "temperature": 0, "echo": True},
    ).json()
    assert body["choices"][0]["text"].startswith("abc")


def test_completion_n(client):
    body = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "hi", "max_tokens": 3, "n": 3,
              "temperature": 1.0},
    ).json()
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]
    assert body["usage"]["completion_tokens"] == 9


def test_completion_streaming_framing(client):
    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "test-model", "prompt": "hi", "max_tokens": 4, "temperature": 0,
              "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert body.rstrip().endswith("data: [DONE]")
    events = sse_events(body)
    assert events, "no chunks were streamed"
    assert all(e["object"] == "text_completion" for e in events)
    assert events[-1]["usage"]["completion_tokens"] == 4
    assert any(e["choices"] and e["choices"][0]["finish_reason"] == "length" for e in events)


def test_streamed_text_matches_the_non_streamed_text(client):
    payload = {"model": "test-model", "prompt": "stream me", "max_tokens": 6, "temperature": 0}
    whole = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    with client.stream("POST", "/v1/completions", json={**payload, "stream": True}) as response:
        body = "".join(response.iter_text())
    streamed = "".join(
        e["choices"][0]["text"] for e in sse_events(body) if e["choices"]
    )
    assert streamed == whole


# -- chat -----------------------------------------------------------------


def test_chat_completion_shape(client):
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0,
        },
    ).json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["completion_tokens"] == 4


def test_chat_accepts_content_parts(client):
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello there"}]}
            ],
            "max_tokens": 2,
            "temperature": 0,
        },
    ).json()
    assert body["usage"]["prompt_tokens"] > 0


def test_chat_streaming_opens_with_a_role_chunk(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "temperature": 0,
            "stream": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    events = sse_events(body)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert body.rstrip().endswith("data: [DONE]")


# -- errors and auth ------------------------------------------------------


def test_unknown_sampling_field_is_rejected(client):
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": "hi", "temperatur": 0.5},
    )
    assert response.status_code == 422, "a typo'd parameter must not be silently ignored"


def test_prompt_over_the_model_limit_is_a_client_error(client):
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": list(range(1, 200)), "max_tokens": 2},
    )
    assert response.status_code == 400
    assert "model limit" in response.json()["message"]


def test_api_key_is_enforced_when_set(tmp_path_factory):
    from spikeinfer.config import (
        CacheConfig,
        EngineConfig,
        GraphConfig,
        ModelConfig,
        SchedulerConfig,
    )
    from spikeinfer.engine.async_engine import AsyncLLMEngine
    from spikeinfer.engine.llm_engine import LLMEngine
    from spikeinfer.entrypoints.openai.server import build_app
    from spikeinfer.loader import save_model

    cfg = small_config()
    path = tmp_path_factory.mktemp("auth_model")
    save_model(build_fast_model(cfg, "cpu", seed=2), cfg, path)
    engine = AsyncLLMEngine(
        LLMEngine(
            EngineConfig(
                model=ModelConfig(model=str(path), dtype="float32"),
                cache=CacheConfig(block_size=16, num_gpu_blocks_override=32),
                scheduler=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64,
                                          max_model_len=64),
                graph=GraphConfig(enabled=False),
                device=TEST_DEVICE,
            ),
            tokenizer=CharTokenizer(),
        )
    )
    with TestClient(build_app(engine, api_key="secret")) as client:
        assert client.get("/health").status_code == 200, "/health must stay unauthenticated"

        payload = {"model": "spiking-qwen", "prompt": "hi", "max_tokens": 1}
        assert client.post("/v1/completions", json=payload).status_code == 401
        ok = client.post(
            "/v1/completions", json=payload, headers={"Authorization": "Bearer secret"}
        )
        assert ok.status_code == 200
    del engine
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
