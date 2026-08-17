"""The paged popcount attention kernel against an SDPA oracle.

The kernel replaces ``softmax(q k^T / sqrt(d)) v`` with bit operations, so the
oracle is the same computation on *unpacked* spikes through
``scaled_dot_product_attention``. Agreement is checked to a tight relative
bound rather than bit-exactly: the scores are exact (integer popcounts), but the
value accumulation is a flash-style online softmax whose summation order differs
from SDPA's.

The cases that would hide a paging bug are the ones worth writing: physical
blocks assigned out of order, sequences whose length is not a multiple of the
block size, and a batch where every sequence has a different length.
"""
from __future__ import annotations

import pytest
import torch

from conftest import requires_cuda
from spikeinfer.kernels.packing import pack_spikes
from spikeinfer.kernels.spike_attention import (
    gather_kv_unpacked,
    paged_spike_attn_decode,
    paged_spike_attn_decode_ref,
)

HEAD_DIM = 64
REL_TOL = 1e-5


def _random_cache(T, kv_heads, num_slots, device, density=0.4, seed=0):
    torch.manual_seed(seed)
    return pack_spikes(
        (torch.rand(T, kv_heads, num_slots, HEAD_DIM, device=device) > (1 - density)).float()
    )


def _shuffled_tables(n_seqs, max_blocks, num_blocks, device, seed=0):
    """Physical blocks in a deliberately scrambled order.

    A kernel that assumed block ``i`` of a sequence lives at physical block ``i``
    would pass with an identity table and fail here.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=generator)[: n_seqs * max_blocks]
    return perm.reshape(n_seqs, max_blocks).to(device=device, dtype=torch.int32)


def _run(seq_lens, block_size, T=4, q_heads=14, kv_heads=2, device="cuda", seed=0):
    n_seqs = len(seq_lens)
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    num_blocks = max(n_seqs * max_blocks, 8)

    k_cache = _random_cache(T, kv_heads, num_blocks * block_size, device, seed=seed)
    v_cache = _random_cache(T, kv_heads, num_blocks * block_size, device, seed=seed + 1)
    tables = _shuffled_tables(n_seqs, max_blocks, num_blocks, device, seed=seed)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    torch.manual_seed(seed + 2)
    q = (torch.rand(T, n_seqs, q_heads, HEAD_DIM, device=device) > 0.6).float()
    q_packed = pack_spikes(q)

    kwargs = dict(
        head_dim=HEAD_DIM,
        block_size=block_size,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
    )
    got = paged_spike_attn_decode(q_packed, k_cache, v_cache, tables, lens, **kwargs)
    want = paged_spike_attn_decode_ref(q_packed, k_cache, v_cache, tables, lens, **kwargs)
    return got, want


def _assert_close(got, want, context=""):
    assert torch.isfinite(got).all(), f"non-finite output {context}"
    scale = want.abs().max().clamp(min=1e-6)
    err = (got - want).abs().max() / scale
    assert err < REL_TOL, f"relative error {err:.2e} exceeds {REL_TOL:.0e} {context}"


@requires_cuda
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [1],  # a one-token sequence: attends only to itself
        [7],  # shorter than any block
        [32],  # exactly one block at the common size
        [33],  # one past a block boundary
        [128],
        [1, 5, 40, 97],  # ragged batch, the real serving case
        [64, 64, 64, 64, 64, 64, 64, 64],
    ],
)
def test_matches_sdpa_oracle(block_size, seq_lens, device):
    got, want = _run(seq_lens, block_size, device=device)
    _assert_close(got, want, f"(block_size={block_size}, lens={seq_lens})")


@requires_cuda
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_across_timesteps(T, device):
    got, want = _run([5, 61], block_size=32, T=T, device=device)
    _assert_close(got, want, f"(T={T})")


@requires_cuda
@pytest.mark.parametrize("q_heads,kv_heads", [(14, 2), (4, 4), (8, 1), (16, 8)])
def test_grouped_query_attention(q_heads, kv_heads, device):
    """Query head h must read KV head ``h // (q_heads // kv_heads)``."""
    got, want = _run([48], 32, q_heads=q_heads, kv_heads=kv_heads, device=device)
    _assert_close(got, want, f"(GQA {q_heads}/{kv_heads})")


@requires_cuda
def test_all_zero_query_is_finite(device):
    """A token that fires no spikes gives every key the same score.

    scores are all 0, so softmax is uniform -- the output must be the mean of
    the values, not a NaN from an all -inf row.
    """
    T, n_seqs, q_heads, kv_heads, block_size = 4, 2, 14, 2, 32
    k_cache = _random_cache(T, kv_heads, 8 * block_size, device)
    v_cache = _random_cache(T, kv_heads, 8 * block_size, device, seed=3)
    tables = _shuffled_tables(n_seqs, 2, 8, device)
    lens = torch.tensor([40, 17], dtype=torch.int32, device=device)
    q_packed = pack_spikes(torch.zeros(T, n_seqs, q_heads, HEAD_DIM, device=device))

    kwargs = dict(head_dim=HEAD_DIM, block_size=block_size, num_q_heads=q_heads,
                  num_kv_heads=kv_heads)
    got = paged_spike_attn_decode(q_packed, k_cache, v_cache, tables, lens, **kwargs)
    want = paged_spike_attn_decode_ref(q_packed, k_cache, v_cache, tables, lens, **kwargs)
    assert torch.isfinite(got).all()
    _assert_close(got, want, "(zero query)")


@requires_cuda
def test_only_the_sequence_own_blocks_are_read(device):
    """Rewriting blocks a sequence does not own must not change its output."""
    T, q_heads, kv_heads, block_size = 4, 14, 2, 32
    num_blocks = 16
    k_cache = _random_cache(T, kv_heads, num_blocks * block_size, device)
    v_cache = _random_cache(T, kv_heads, num_blocks * block_size, device, seed=5)
    table = torch.tensor([[3, 9]], dtype=torch.int32, device=device)
    lens = torch.tensor([50], dtype=torch.int32, device=device)
    q_packed = pack_spikes(
        (torch.rand(T, 1, q_heads, HEAD_DIM, device=device) > 0.6).float()
    )

    kwargs = dict(head_dim=HEAD_DIM, block_size=block_size, num_q_heads=q_heads,
                  num_kv_heads=kv_heads)
    before = paged_spike_attn_decode(q_packed, k_cache, v_cache, table, lens, **kwargs).clone()

    for block in range(num_blocks):
        if block not in (3, 9):
            k_cache[:, :, block * block_size : (block + 1) * block_size] = -1
            v_cache[:, :, block * block_size : (block + 1) * block_size] = -1
    after = paged_spike_attn_decode(q_packed, k_cache, v_cache, table, lens, **kwargs)
    assert torch.equal(before, after), "attention read outside the sequence's blocks"


@requires_cuda
def test_padding_tail_of_the_last_block_is_ignored(device):
    """seq_len 33 with block_size 32 must ignore slots 34..64 of block two."""
    T, q_heads, kv_heads, block_size = 4, 14, 2, 32
    k_cache = _random_cache(T, kv_heads, 4 * block_size, device)
    v_cache = _random_cache(T, kv_heads, 4 * block_size, device, seed=7)
    table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    lens = torch.tensor([33], dtype=torch.int32, device=device)
    q_packed = pack_spikes((torch.rand(T, 1, q_heads, HEAD_DIM, device=device) > 0.6).float())
    kwargs = dict(head_dim=HEAD_DIM, block_size=block_size, num_q_heads=q_heads,
                  num_kv_heads=kv_heads)

    before = paged_spike_attn_decode(q_packed, k_cache, v_cache, table, lens, **kwargs).clone()
    k_cache[:, :, block_size + 1 :] = -1  # everything past position 33
    v_cache[:, :, block_size + 1 :] = -1
    after = paged_spike_attn_decode(q_packed, k_cache, v_cache, table, lens, **kwargs)
    assert torch.equal(before, after), "attention read past seq_len"


@requires_cuda
def test_gather_kv_unpacked_reassembles_the_sequence(device):
    """The prefill path's reader must invert the paged writes exactly."""
    T, kv_heads, block_size = 4, 2, 16
    num_blocks = 8
    dense = (torch.rand(T, kv_heads, num_blocks * block_size, HEAD_DIM, device=device) > 0.5)
    cache = pack_spikes(dense.float())
    table = torch.tensor([5, 2, 7], dtype=torch.int32, device=device)
    seq_len = 40  # 2 full blocks + 8

    got = gather_kv_unpacked(cache, table, seq_len, HEAD_DIM, block_size, torch.float32)
    expected = torch.cat(
        [dense[:, :, b * block_size : (b + 1) * block_size] for b in (5, 2, 7)], dim=2
    )[:, :, :seq_len].float()
    assert torch.equal(got, expected)


def test_reference_runs_without_a_gpu(device):
    """The torch fallback is what makes the package importable without Triton."""
    T, n_seqs, q_heads, kv_heads, block_size = 2, 2, 4, 2, 16
    k_cache = _random_cache(T, kv_heads, 4 * block_size, device)
    v_cache = _random_cache(T, kv_heads, 4 * block_size, device, seed=11)
    tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    lens = torch.tensor([20, 9], dtype=torch.int32, device=device)
    q_packed = pack_spikes((torch.rand(T, n_seqs, q_heads, HEAD_DIM, device=device) > 0.5).float())

    out = paged_spike_attn_decode_ref(
        q_packed, k_cache, v_cache, tables, lens,
        head_dim=HEAD_DIM, block_size=block_size, num_q_heads=q_heads, num_kv_heads=kv_heads,
    )
    assert out.shape == (T, n_seqs, q_heads, HEAD_DIM)
    assert torch.isfinite(out).all()
