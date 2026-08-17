"""OpenAI API request/response schemas.

Field names and defaults follow the OpenAI REST API so existing clients (the
``openai`` SDK, LangChain, curl snippets from the docs) work unmodified. Fields
OpenAI does not define but vLLM popularised -- ``top_k``, ``min_p``,
``repetition_penalty``, ``ignore_eos``, ``stop_token_ids`` -- are accepted as
extras; anything unknown is rejected rather than silently ignored, because a
typo'd sampling parameter that quietly does nothing is worse than an error.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...sampling_params import SamplingParams


def _request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingMixin(_Base):
    """Sampling fields shared by the completion and chat endpoints."""

    max_tokens: int | None = 64
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    seed: int | None = None
    logit_bias: dict[str, float] | None = None
    stream: bool | None = False
    user: str | None = None

    # Extensions.
    top_k: int | None = -1
    min_p: float | None = 0.0
    repetition_penalty: float | None = 1.0
    ignore_eos: bool | None = False
    min_tokens: int | None = 0
    stop_token_ids: list[int] | None = None

    def to_sampling_params(self, logprobs: int | None = None) -> SamplingParams:
        return SamplingParams.from_openai(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            n=self.n,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            logprobs=logprobs,
            seed=self.seed,
            logit_bias=self.logit_bias,
            ignore_eos=self.ignore_eos,
            min_tokens=self.min_tokens,
            stop_token_ids=self.stop_token_ids,
        )


class CompletionRequest(SamplingMixin):
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    echo: bool | None = False
    logprobs: int | None = None
    suffix: str | None = None
    best_of: int | None = None
    stream_options: dict[str, Any] | None = None


class ChatMessage(_Base):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def text(self) -> str:
        """Flatten OpenAI's content-parts form down to plain text."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if isinstance(part, dict))


class ChatCompletionRequest(SamplingMixin):
    model: str
    messages: list[ChatMessage]
    logprobs: bool | None = False
    top_logprobs: int | None = None
    add_generation_prompt: bool | None = True
    chat_template: str | None = None
    stream_options: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    def resolved_logprobs(self) -> int | None:
        if not self.logprobs:
            return None
        return self.top_logprobs if self.top_logprobs is not None else 0


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LogProbs(BaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: LogProbs | None = None
    finish_reason: str | None = None
    stop_reason: str | int | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _request_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None


class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessageResponse
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None
    stop_reason: str | int | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _request_id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None
    stop_reason: str | int | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]
    usage: UsageInfo | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "spikeinfer"
    root: str | None = None
    max_model_len: int | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str = "invalid_request_error"
    code: int = 400


class TokenizeRequest(_Base):
    model: str | None = None
    prompt: str
    add_special_tokens: bool | None = True


class TokenizeResponse(BaseModel):
    tokens: list[int]
    count: int
    max_model_len: int


class DetokenizeRequest(_Base):
    model: str | None = None
    tokens: list[int]


class DetokenizeResponse(BaseModel):
    prompt: str
