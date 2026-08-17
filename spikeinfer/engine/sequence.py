"""Request and sequence state.

A user request becomes one :class:`Sequence` per requested completion (``n``).
The scheduler, block manager, model runner and sampler all key off these
objects; nothing else carries per-request state.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from ..sampling_params import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_STOPPED = enum.auto()  # EOS or a stop condition
    FINISHED_LENGTH = enum.auto()  # hit max_tokens or max_model_len
    FINISHED_ABORTED = enum.auto()

    @property
    def is_finished(self) -> bool:
        return self.name.startswith("FINISHED")

    @property
    def finish_reason(self) -> str | None:
        if self is SequenceStatus.FINISHED_STOPPED:
            return "stop"
        if self is SequenceStatus.FINISHED_LENGTH:
            return "length"
        if self is SequenceStatus.FINISHED_ABORTED:
            return "abort"
        return None


@dataclass
class Sequence:
    """One in-flight completion."""

    seq_id: str
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams

    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    finish_time: float | None = None

    num_computed_tokens: int = 0
    """Tokens whose KV is already in the cache. Advances by the chunk size on
    every scheduled step, which is what makes chunked prefill work."""

    block_table: list[int] = field(default_factory=list)
    """Physical block ids, logical order. Owned by the block manager."""

    output_text: str = ""
    """Detokenized so far. Written by the detokenizer, read for stop strings."""

    streamed_offset: int = 0
    """How much of ``output_text`` the client has already been sent."""

    # Incremental-detokenization bookkeeping (see engine/detokenizer.py).
    prefix_offset: int = 0
    read_offset: int = 0
    tokens: list[str] = field(default_factory=list)

    stop_reason: str | int | None = None
    cumulative_logprob: float = 0.0
    logprobs: list[dict[int, float]] = field(default_factory=list)

    # Counts for the frequency/presence penalties, kept incrementally so the
    # sampler never has to re-scan the output on every step.
    token_counts: dict[int, int] = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def is_prefilling(self) -> bool:
        """True until every prompt token's KV has been computed."""
        return self.num_computed_tokens < self.prompt_len

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.total_len - self.num_computed_tokens

    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def tokens_to_compute(self, budget: int) -> int:
        """How many tokens this sequence would contribute to a step of ``budget``."""
        return min(self.num_uncomputed_tokens, budget)

    def slice_to_compute(self, count: int) -> list[int]:
        start = self.num_computed_tokens
        return self.token_ids()[start : start + count]

    def append_token(self, token_id: int, logprobs: dict[int, float] | None = None) -> None:
        self.output_token_ids.append(token_id)
        self.token_counts[token_id] = self.token_counts.get(token_id, 0) + 1
        if logprobs is not None:
            self.logprobs.append(logprobs)
            self.cumulative_logprob += logprobs.get(token_id, 0.0)
        if self.first_token_time is None:
            self.first_token_time = time.monotonic()

    def reset_for_recompute(self) -> None:
        """Preemption: drop the cached KV, keep the tokens already produced.

        The generated tokens stay in ``output_token_ids`` and are re-prefilled
        along with the prompt, so preemption is invisible in the output.
        """
        self.num_computed_tokens = 0
        self.block_table = []
        self.status = SequenceStatus.WAITING

    def finish(self, status: SequenceStatus, stop_reason: str | int | None = None) -> None:
        self.status = status
        self.stop_reason = stop_reason
        self.finish_time = time.monotonic()


@dataclass
class CompletionOutput:
    index: int
    text: str
    """Everything generated so far, cumulative."""
    token_ids: list[int]
    cumulative_logprob: float
    logprobs: list[dict[int, float]] | None
    finish_reason: str | None
    stop_reason: str | int | None = None
    delta: str = ""
    """New text since the previous engine step -- what a stream should emit."""
    delta_token_ids: list[int] = field(default_factory=list)


@dataclass
class RequestOutput:
    """What the engine hands back for a request, streaming or not."""

    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
    metrics: RequestMetrics | None = None

    @property
    def text(self) -> str:
        return self.outputs[0].text if self.outputs else ""


@dataclass
class RequestMetrics:
    arrival_time: float
    first_token_time: float | None
    finish_time: float | None
    prompt_tokens: int
    generated_tokens: int

    @property
    def time_to_first_token(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def total_time(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time
