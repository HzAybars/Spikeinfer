"""Scheduling policy, without a GPU in sight.

The scheduler is pure Python over sequence bookkeeping, so it can be tested
exhaustively and fast. The behaviours that matter are the ones that go wrong
under load: the token budget must be respected, a long prompt must not starve
decodes, preemption must free real blocks, and a preempted sequence must come
back without losing the tokens it already produced.
"""
from __future__ import annotations

from conftest import make_sequence
from spikeinfer.config import SchedulerConfig
from spikeinfer.engine.block_manager import BlockManager
from spikeinfer.engine.scheduler import Scheduler
from spikeinfer.engine.sequence import SequenceStatus


def build(num_blocks=64, block_size=16, **config):
    settings = dict(
        max_num_seqs=8, max_num_batched_tokens=64, max_model_len=256,
        enable_chunked_prefill=True,
    )
    settings.update(config)
    manager = BlockManager(num_blocks, block_size)
    return Scheduler(SchedulerConfig(**settings), manager), manager


def test_nothing_scheduled_when_idle():
    scheduler, _ = build()
    assert scheduler.schedule().is_empty
    assert not scheduler.has_work


def test_a_waiting_prompt_is_admitted_as_one_prefill():
    scheduler, _ = build()
    seq = make_sequence(prompt_len=20)
    scheduler.add(seq)

    out = scheduler.schedule()
    assert len(out.scheduled) == 1
    item = out.scheduled[0]
    assert item.num_tokens == 20
    assert not item.is_decode
    assert item.will_sample, "a full prefill produces the first token"
    assert seq.status is SequenceStatus.RUNNING


def test_prefill_then_decode():
    scheduler, _ = build()
    seq = make_sequence(prompt_len=5)
    scheduler.add(seq)

    first = scheduler.schedule().scheduled[0]
    seq.num_computed_tokens += first.num_tokens
    seq.append_token(42)

    second = scheduler.schedule().scheduled[0]
    assert second.num_tokens == 1
    assert second.is_decode, "a one-token step that reaches the end is a decode"
    assert second.will_sample


def test_chunked_prefill_splits_a_long_prompt():
    scheduler, _ = build(max_num_batched_tokens=32)
    seq = make_sequence(prompt_len=80)
    scheduler.add(seq)

    chunks = []
    for _ in range(3):
        item = scheduler.schedule().scheduled[0]
        chunks.append(item.num_tokens)
        seq.num_computed_tokens += item.num_tokens
        assert item.will_sample == (seq.num_computed_tokens == seq.total_len)
    assert chunks == [32, 32, 16]
    assert not any_is_decode_before_the_end(chunks)


def any_is_decode_before_the_end(chunks):
    return any(c == 1 for c in chunks[:-1])


def test_chunking_off_defers_a_prompt_that_does_not_fit():
    scheduler, _ = build(
        max_num_batched_tokens=32, max_model_len=32, enable_chunked_prefill=False
    )
    scheduler.add(make_sequence(prompt_len=40))
    assert scheduler.schedule().is_empty, "an oversized prompt must not be chunked"


def test_token_budget_is_respected():
    scheduler, _ = build(max_num_batched_tokens=40)
    for i in range(4):
        scheduler.add(make_sequence(prompt_len=15, seq_id=f"s{i}"))

    out = scheduler.schedule()
    assert out.num_batched_tokens <= 40
    assert sum(s.num_tokens for s in out.scheduled) == out.num_batched_tokens


def test_sequence_count_is_capped():
    scheduler, _ = build(max_num_seqs=3, max_num_batched_tokens=1024)
    for i in range(10):
        scheduler.add(make_sequence(prompt_len=4, seq_id=f"s{i}"))
    assert len(scheduler.schedule().scheduled) == 3


def test_decodes_are_ordered_before_prefills():
    """The model runner puts decodes in a contiguous prefix so a CUDA graph
    can replay them; the scheduler is what guarantees that ordering."""
    scheduler, _ = build(max_num_batched_tokens=1024)
    decoding = make_sequence(prompt_len=4, seq_id="decoding")
    scheduler.add(decoding)
    item = scheduler.schedule().scheduled[0]
    decoding.num_computed_tokens += item.num_tokens
    decoding.append_token(7)

    scheduler.add(make_sequence(prompt_len=30, seq_id="fresh"))
    out = scheduler.schedule()
    kinds = [s.is_decode for s in out.scheduled]
    assert kinds == sorted(kinds, reverse=True), "a prefill was placed before a decode"


def test_preemption_frees_blocks_and_requeues():
    scheduler, manager = build(num_blocks=4, block_size=16, max_num_batched_tokens=1024)
    first = make_sequence(prompt_len=32, seq_id="first")
    second = make_sequence(prompt_len=32, seq_id="second")
    scheduler.add(first)
    scheduler.add(second)

    out = scheduler.schedule()  # 2 blocks each, pool exhausted
    for item in out.scheduled:
        item.seq.num_computed_tokens += item.num_tokens
        item.seq.append_token(1)
    assert manager.num_free_blocks == 0

    # Both now need a 3rd block for token 33; only preemption can supply it.
    for seq in (first, second):
        seq.num_computed_tokens = 32
    out = scheduler.schedule()

    assert out.preempted, "expected a preemption when the pool is exhausted"
    victim = out.preempted[0]
    assert victim.block_table == [], "a preempted sequence must give its blocks back"
    assert victim.status is SequenceStatus.WAITING
    assert victim in scheduler.waiting
    assert scheduler.num_preemptions == len(out.preempted)
    scheduled_ids = {s.seq.seq_id for s in out.scheduled}
    assert victim.seq_id not in scheduled_ids, (
        "a sequence preempted this step must not be re-admitted in the same step"
    )


def test_preempted_sequence_keeps_its_generated_tokens():
    scheduler, manager = build(num_blocks=2, block_size=16)
    seq = make_sequence(prompt_len=16)
    scheduler.add(seq)
    item = scheduler.schedule().scheduled[0]
    seq.num_computed_tokens += item.num_tokens
    for token in (11, 12, 13):
        seq.append_token(token)

    seq.reset_for_recompute()
    assert seq.output_token_ids == [11, 12, 13], "recompute must not drop output"
    assert seq.num_computed_tokens == 0
    assert seq.num_uncomputed_tokens == 19, "prompt plus generated tokens are recomputed"


def test_finished_sequences_release_their_blocks():
    scheduler, manager = build(num_blocks=8, block_size=16)
    seq = make_sequence(prompt_len=16)
    scheduler.add(seq)
    scheduler.schedule()
    assert manager.num_free_blocks == 7

    seq.finish(SequenceStatus.FINISHED_STOPPED)
    freed = scheduler.free_finished()
    assert freed == [seq]
    assert manager.num_free_blocks == 8
    assert seq not in scheduler.running


def test_abort_removes_from_both_queues():
    scheduler, manager = build()
    running = make_sequence(prompt_len=8, seq_id="r", request_id="req")
    waiting = make_sequence(prompt_len=8, seq_id="w", request_id="req")
    scheduler.add(running)
    scheduler.schedule()
    scheduler.add(waiting)

    aborted = scheduler.abort("req")
    assert len(aborted) == 2
    assert all(s.status is SequenceStatus.FINISHED_ABORTED for s in aborted)
    assert not scheduler.running and not scheduler.waiting
    assert manager.num_free_blocks == manager.num_blocks


def test_admission_stops_when_blocks_run_out():
    scheduler, manager = build(num_blocks=2, block_size=16, max_num_batched_tokens=1024)
    for i in range(5):
        scheduler.add(make_sequence(prompt_len=32, seq_id=f"s{i}"))

    out = scheduler.schedule()
    assert len(out.scheduled) == 1, "admitted more sequences than the cache can hold"
    assert len(scheduler.waiting) == 4
