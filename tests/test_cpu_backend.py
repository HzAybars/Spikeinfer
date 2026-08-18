"""The CPU decode path against the reference oracle.

``spike_attention_cpu`` replaces the per-sequence Python loop in
``paged_spike_attn_decode_ref`` with a batched, memory-bounded one. It must not
change a single answer while doing so, so every case here compares the two
directly on the same inputs.

Scores are integers in [0, head_dim] whichever way they are computed, but the
softmax and the p.v accumulation are floating point and the two implementations
reach them by different routes (grouped SDPA vs one SDPA per sequence), so the
comparison is a tight tolerance rather than bit equality. The *scores*
themselves are checked exactly, separately, since that is where a packing or
block-table bug would show up.
"""
from __future__ import annotations

import pytest
import torch

from spikeinfer.kernels.packing import pack_spikes, unpack_spikes
from spikeinfer.kernels.spike_attention import paged_spike_attn_decode_ref
from spikeinfer.kernels.spike_attention_cpu import (
    _slot_matrix,
    paged_spike_attn_decode_cpu,
)

TOL = dict(rtol=1e-5, atol=1e-6)


def build_case(
    n_seqs, seq_lens, head_dim=64, block_size=16, num_q_heads=4, num_kv_heads=2, T=4, seed=0
):
    """A paged cache with blocks assigned in scrambled order, as the allocator does."""
    torch.manual_seed(seed)
    max_blocks = max((n + block_size - 1) // block_size for n in seq_lens)
    num_blocks = n_seqs * max_blocks + 3
    num_slots = num_blocks * block_size

    # Random spikes everywhere, including the blocks nobody owns -- reading one
    # by accident must change the answer, which is what makes this a real check.
    k_cache = pack_spikes(
        (torch.rand(T, num_kv_heads, num_slots, head_dim) < 0.3).to(torch.float32)
    )
    v_cache = pack_spikes(
        (torch.rand(T, num_kv_heads, num_slots, head_dim) < 0.3).to(torch.float32)
    )
    q = (torch.rand(T, n_seqs, num_q_heads, head_dim) < 0.3).to(torch.float32)
    q_packed = pack_spikes(q)

    order = torch.randperm(num_blocks - 1)
    tables = torch.zeros((n_seqs, max_blocks), dtype=torch.int32)
    cursor = 0
    for i, length in enumerate(seq_lens):
        needed = (length + block_size - 1) // block_size
        tables[i, :needed] = order[cursor : cursor + needed].to(torch.int32)
        tables[i, needed:] = num_blocks - 1  # pad block, as the runner does
        cursor += needed

    return dict(
        q_packed=q_packed,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=tables,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        head_dim=head_dim,
        block_size=block_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def compare(case):
    expected = paged_spike_attn_decode_ref(**case, dtype=torch.float32)
    actual = paged_spike_attn_decode_cpu(**case, dtype=torch.float32)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, **TOL)
    return actual


@pytest.mark.parametrize(
    "seq_lens",
    [
        [1],  # a sequence one token long
        [16],  # exactly one block
        [17],  # one token into the second block
        [1, 5, 33, 64],  # ragged, the ordinary decode batch
        [100] * 8,  # uniform and longer than one block
    ],
    ids=["single", "one-block", "block-boundary", "ragged", "uniform"],
)
def test_matches_reference(seq_lens):
    compare(build_case(len(seq_lens), seq_lens))


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(2, 2), (4, 2), (8, 1)])
def test_gqa_ratios(num_q_heads, num_kv_heads):
    compare(
        build_case(
            4, [7, 20, 40, 3], num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
        )
    )


@pytest.mark.parametrize("head_dim", [32, 64, 96, 128])
def test_head_dims(head_dim):
    compare(build_case(3, [10, 33, 50], head_dim=head_dim))


def test_query_that_fires_no_spikes():
    """An all-zero query scores 0 against every key, so attention is a plain
    mean over the context. A popcount or masking bug turns that into a NaN."""
    case = build_case(3, [5, 19, 40])
    case["q_packed"] = torch.zeros_like(case["q_packed"])
    actual = compare(case)
    assert torch.isfinite(actual).all()


def test_reads_only_blocks_the_sequence_owns():
    """Overwriting every unowned block must not move the answer."""
    case = build_case(4, [9, 25, 41, 3])
    before = paged_spike_attn_decode_cpu(**case, dtype=torch.float32)

    owned = set()
    for i, length in enumerate(case["seq_lens"].tolist()):
        needed = (length + case["block_size"] - 1) // case["block_size"]
        owned.update(case["block_tables"][i, :needed].tolist())

    bs = case["block_size"]
    for block in range(case["k_cache"].shape[2] // bs):
        if block not in owned:
            case["k_cache"][:, :, block * bs : (block + 1) * bs] = -1  # every bit set
            case["v_cache"][:, :, block * bs : (block + 1) * bs] = -1

    after = paged_spike_attn_decode_cpu(**case, dtype=torch.float32)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_padding_tail_of_the_last_block_is_ignored():
    """A sequence that does not fill its final block must not see the slots past
    its length, even though they are inside a block it owns."""
    case = build_case(2, [5, 21])  # neither is a multiple of block_size=16
    before = paged_spike_attn_decode_cpu(**case, dtype=torch.float32)

    bs = case["block_size"]
    for i, length in enumerate(case["seq_lens"].tolist()):
        last = case["block_tables"][i, (length - 1) // bs].item()
        tail = last * bs + (length % bs)
        case["k_cache"][:, :, tail : (last + 1) * bs] = -1
        case["v_cache"][:, :, tail : (last + 1) * bs] = -1

    after = paged_spike_attn_decode_cpu(**case, dtype=torch.float32)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_grouping_does_not_change_the_answer(monkeypatch):
    """The gather budget only controls how many sequences share one SDPA call."""
    case = build_case(6, [12, 40, 5, 77, 33, 1])
    whole = paged_spike_attn_decode_cpu(**case, dtype=torch.float32)
    monkeypatch.setenv("SPIKEINFER_CPU_GATHER_BUDGET_MB", "0.001")  # forces one per group
    torch.testing.assert_close(
        paged_spike_attn_decode_cpu(**case, dtype=torch.float32), whole, **TOL
    )


def test_scores_are_exact_integers():
    """The popcount identity the whole design rests on: with q and k binary,
    q.k is an exact integer, so the packed and unpacked routes must agree
    bit-for-bit before any softmax is involved."""
    torch.manual_seed(3)
    head_dim = 64
    q = (torch.rand(4, 3, 8, head_dim) < 0.4).to(torch.float32)
    k = (torch.rand(4, 3, 20, head_dim) < 0.4).to(torch.float32)
    from_dense = torch.matmul(q, k.transpose(-1, -2))

    packed_q, packed_k = pack_spikes(q), pack_spikes(k)
    round_tripped = torch.matmul(
        unpack_spikes(packed_q, head_dim, torch.float32),
        unpack_spikes(packed_k, head_dim, torch.float32).transpose(-1, -2),
    )
    assert torch.equal(from_dense, round_tripped)
    assert from_dense.max() <= head_dim


def test_slot_matrix_walks_the_block_table():
    tables = torch.tensor([[3, 1, 7], [5, 2, 0]], dtype=torch.int32)
    slots = _slot_matrix(tables, length=10, block_size=4)
    assert slots.tolist() == [
        [12, 13, 14, 15, 4, 5, 6, 7, 28, 29],
        [20, 21, 22, 23, 8, 9, 10, 11, 0, 1],
    ]
