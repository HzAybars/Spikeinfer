"""Per-request sampling configuration.

Deliberately mirrors the vLLM / OpenAI field names, so a client written against
either works here unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

_GREEDY_TEMP = 1e-5


@dataclass
class SamplingParams:
    n: int = 1
    """Completions to return per prompt. Implemented by forking the request."""

    max_tokens: int = 64
    min_tokens: int = 0
    """Suppress EOS until this many tokens have been generated."""

    temperature: float = 1.0
    """0 (or near) means greedy: the sampler takes the argmax path."""

    top_p: float = 1.0
    top_k: int = -1
    """-1 disables. Values >= vocab_size are equivalent to disabled."""

    min_p: float = 0.0
    """Drop tokens below ``min_p * p_max``. Applied before top_k/top_p."""

    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False

    logprobs: int | None = None
    """Return this many top logprobs per generated token (OpenAI semantics)."""

    seed: int | None = None
    logit_bias: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        self.stop = list(self.stop)
        self.stop_token_ids = list(self.stop_token_ids)
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be >= 0")
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")

    @property
    def is_greedy(self) -> bool:
        return self.temperature < _GREEDY_TEMP

    @property
    def needs_penalties(self) -> bool:
        return (
            self.repetition_penalty != 1.0
            or self.presence_penalty != 0.0
            or self.frequency_penalty != 0.0
        )

    def clone(self) -> SamplingParams:
        return SamplingParams(
            n=1,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            stop=list(self.stop),
            stop_token_ids=list(self.stop_token_ids),
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            logprobs=self.logprobs,
            seed=self.seed,
            logit_bias=dict(self.logit_bias) if self.logit_bias else None,
        )

    @classmethod
    def from_openai(
        cls,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        n: int | None = None,
        stop: Sequence[str] | str | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
        logprobs: int | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | dict[int, float] | None = None,
        ignore_eos: bool | None = None,
        min_tokens: int | None = None,
        stop_token_ids: Sequence[int] | None = None,
    ) -> SamplingParams:
        """Build from OpenAI-style fields, tolerating ``None`` for every one."""
        bias = None
        if logit_bias:
            bias = {int(k): float(v) for k, v in logit_bias.items()}
        return cls(
            n=n if n is not None else 1,
            max_tokens=max_tokens if max_tokens is not None else 64,
            min_tokens=min_tokens or 0,
            temperature=temperature if temperature is not None else 1.0,
            top_p=top_p if top_p is not None else 1.0,
            top_k=top_k if top_k is not None else -1,
            min_p=min_p or 0.0,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else 1.0,
            presence_penalty=presence_penalty or 0.0,
            frequency_penalty=frequency_penalty or 0.0,
            stop=list(stop) if isinstance(stop, (list, tuple)) else ([stop] if stop else []),
            stop_token_ids=list(stop_token_ids or []),
            ignore_eos=bool(ignore_eos),
            logprobs=logprobs,
            seed=seed,
            logit_bias=bias,
        )
