"""Continuous-batching scheduler.

Every engine step asks the scheduler for a batch. The policy is FCFS with
chunked prefill, in the shape vLLM's V1 scheduler settled on:

* running sequences are considered first, so an admitted request keeps making
  progress and latency stays predictable;
* a sequence contributes ``min(uncomputed, remaining_budget)`` tokens, which
  makes prefill and decode the same code path -- a decode is just a one-token
  chunk that happens to reach the end of the sequence;
* if the cache is full, the newest running sequences are preempted (their blocks
  freed, their tokens re-queued for recompute) until the batch fits.

Preemption recomputes rather than swapping to host memory. With a packed spike
cache a block is ~96 KB, so refilling one is cheaper than moving it over PCIe.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..config import SchedulerConfig
from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSequence:
    seq: Sequence
    num_tokens: int
    """Tokens computed for this sequence in this step."""

    is_decode: bool
    """One token that completes the sequence -- eligible for the paged kernel
    and for CUDA graph replay."""

    will_sample: bool
    """Whether this step produces a new token for this sequence."""


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    ignored: list[Sequence] = field(default_factory=list)
    """Sequences dropped because they can never fit in the cache at all."""
    num_batched_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def decodes(self) -> list[ScheduledSequence]:
        return [s for s in self.scheduled if s.is_decode]

    @property
    def prefills(self) -> list[ScheduledSequence]:
        return [s for s in self.scheduled if not s.is_decode]


class Scheduler:
    def __init__(self, config: SchedulerConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    # -- queue management -------------------------------------------------

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, request_id: str) -> list[Sequence]:
        aborted = []
        for seq in list(self.waiting):
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                seq.finish(SequenceStatus.FINISHED_ABORTED)
                aborted.append(seq)
        for seq in list(self.running):
            if seq.request_id == request_id:
                self.running.remove(seq)
                self.block_manager.free(seq)
                seq.finish(SequenceStatus.FINISHED_ABORTED)
                aborted.append(seq)
        return aborted

    def free_finished(self) -> list[Sequence]:
        """Release blocks for sequences that finished during the last step."""
        finished = [s for s in self.running if s.status.is_finished]
        for seq in finished:
            self.running.remove(seq)
            self.block_manager.free(seq)
        return finished

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_pending(self) -> int:
        return len(self.waiting) + len(self.running)

    # -- the step ---------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        max_len = self.config.max_model_len
        preempted: set[str] = set()

        # 1. Running sequences, oldest first.
        for seq in list(self.running):
            if budget <= 0 or len(out.scheduled) >= self.config.max_num_seqs:
                break
            if seq.seq_id in preempted:
                # Freed to make room for an older sequence earlier in this same
                # step. Re-admitting it now would just undo that.
                continue
            num = self._chunk_size(seq, budget)
            if num == 0:
                continue
            if self._never_fits(seq, num):
                self._drop(seq, out)
                continue
            while not self.block_manager.allocate(seq, num):
                victim = self._pick_victim(exclude=seq)
                if victim is None:
                    # Nothing left to give up: this sequence yields instead and
                    # retries in a later, smaller batch.
                    self._preempt(seq, out)
                    preempted.add(seq.seq_id)
                    num = 0
                    break
                self._preempt(victim, out)
                preempted.add(victim.seq_id)
            if num == 0:
                continue
            out.scheduled.append(self._make(seq, num, max_len))
            budget -= num

        # 2. Admit waiting sequences into the leftover budget.
        while self.waiting and budget > 0 and len(out.scheduled) < self.config.max_num_seqs:
            seq = self.waiting[0]
            if seq.seq_id in preempted:
                # Preempted sequences are pushed to the front of the queue, so
                # hitting one means the cache is full: stop admitting.
                break
            num = self._chunk_size(seq, budget)
            if num == 0:
                # Prompt longer than the whole budget and chunking is off.
                break
            if self._never_fits(seq, num):
                self.waiting.popleft()
                self._drop(seq, out)
                continue
            if not self.block_manager.can_allocate(seq, num):
                break
            self.waiting.popleft()
            self.block_manager.allocate(seq, num)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.scheduled.append(self._make(seq, num, max_len))
            budget -= num

        out.num_batched_tokens = sum(s.num_tokens for s in out.scheduled)
        # Decodes first: the CUDA-graph path replays a contiguous decode prefix.
        out.scheduled.sort(key=lambda s: not s.is_decode)
        return out

    # -- helpers ----------------------------------------------------------

    def _chunk_size(self, seq: Sequence, budget: int) -> int:
        uncomputed = seq.num_uncomputed_tokens
        if uncomputed <= 0:
            return 0
        if not self.config.enable_chunked_prefill and uncomputed > budget:
            return 0
        return min(uncomputed, budget)

    def _make(self, seq: Sequence, num: int, max_len: int) -> ScheduledSequence:
        will_sample = seq.num_computed_tokens + num >= seq.total_len
        return ScheduledSequence(
            seq=seq,
            num_tokens=num,
            is_decode=(num == 1 and will_sample),
            will_sample=will_sample,
        )

    def _never_fits(self, seq: Sequence, num: int) -> bool:
        """True when even an empty cache could not hold this sequence.

        Without this check the sequence would preempt everything, still fail,
        preempt itself, and be rescheduled forever -- a livelock rather than an
        error. It is only reachable through a hand-set block count, since the
        cache is normally sized from ``max_model_len``.
        """
        needed = self.block_manager.blocks_for(seq.num_computed_tokens + num)
        return needed > self.block_manager.num_blocks

    def _drop(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.block_manager.free(seq)
        if seq in self.running:
            self.running.remove(seq)
        seq.finish(SequenceStatus.FINISHED_ABORTED, stop_reason="kv_cache_too_small")
        out.ignored.append(seq)

    def _pick_victim(self, exclude: Sequence) -> Sequence | None:
        """Newest running sequence that owns blocks and is not ``exclude``."""
        for seq in reversed(self.running):
            if seq is not exclude and seq.block_table:
                return seq
        return None

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.block_manager.free(seq)
        seq.reset_for_recompute()
        if seq in self.running:
            self.running.remove(seq)
        self.waiting.appendleft(seq)
        out.preempted.append(seq)
        self.num_preemptions += 1
        # Anything already scheduled for the victim this step is void.
        out.scheduled = [s for s in out.scheduled if s.seq is not seq]
