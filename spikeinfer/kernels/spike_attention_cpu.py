"""Paged spike attention for the CPU backend.

The Triton kernel next door wins by never materialising a dense key: it reads
packed words and scores them with a SWAR popcount, which on a GPU is 32x less
HBM traffic for the same answer. That reasoning does not transfer to a CPU.
Measured on this machine (8 cores, torch 2.6, head_dim 64, T=4, GQA 7:1), the
popcount form loses to unpacking and letting BLAS do a GEMM, at every shape
tried::

    T,N,H_kv,n_rep,keys    popcount     unpack+GEMM
    4, 1, 2, 7,    128      0.167 ms      0.071 ms
    4, 1, 2, 7,   4096      0.735 ms      0.342 ms
    4,32, 2, 7,    512      2.766 ms      1.375 ms

Both produce bit-identical scores -- they are integers in [0, head_dim] either
way -- so this is purely a speed choice. A vectorised popcount costs six passes
over the packed tensor in separate torch kernels, which is memory-bound on a
CPU, while the GEMM it replaces is one call into a threaded, vectorised BLAS.
The cache stays bit-packed regardless; that is a *storage* decision and its
32x memory win is untouched. Only the compute unpacks, one chunk at a time.

So what is actually wrong with ``paged_spike_attn_decode_ref``, which also
unpacks and also calls SDPA? Three things, all structural rather than
arithmetic:

* it loops over sequences in Python, so a 32-way decode batch is 32 separate
  SDPA calls plus 64 gathers;
* it calls ``repeat_interleave(n_rep)`` on K and V, materialising a 7-fold copy
  of both, where ``enable_gqa`` broadcasts instead;
* it gathers each sequence's *entire* context, so peak memory grows with the
  longest sequence times the batch with nothing bounding it.

This module fixes those three and keeps the arithmetic identical. Measured
against ``_ref`` on the same inputs (8 cores, block_size 32, head_dim 64, T=4,
GQA 4:2; ``*`` marks a ragged batch):

    decode batch x context     _ref        this     speedup
     1 x  128                 0.210 ms   0.251 ms     0.84x
     1 x  512                 0.357 ms   0.417 ms     0.86x
     4 x  256                 1.107 ms   0.701 ms     1.58x
     8 x  128                 1.901 ms   0.702 ms     2.71x
     8 x   99*                1.327 ms   0.645 ms     2.06x
    32 x  256                 9.116 ms   4.522 ms     2.02x
    32 x 1024                19.792 ms  18.591 ms     1.06x
    64 x  512                24.419 ms  18.627 ms     1.31x

Batch 1 is genuinely a little slower -- grouping, slot-matrix construction and
the output copy are fixed costs that a one-sequence loop does not pay, about
40 us per call. That is ~1 ms per token across 24 layers, against a CPU decode
step measured in tens of milliseconds where the projections, not attention,
dominate. It was not worth a second code path to recover.

It remains the correctness oracle's twin, not its replacement: ``_ref`` stays in
``spike_attention.py`` and ``tests/test_cpu_backend.py`` checks the two against
each other on scrambled block tables, ragged lengths and every GQA ratio.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .packing import unpack_spikes

_DEFAULT_GATHER_BUDGET = 16 * 2**20
"""Bytes of unpacked K+V the gather may hold at once. Sequences are processed in
groups sized to fit; a group is never smaller than one sequence, so a single
very long context is still served (it just gets the whole budget to itself).

16 MiB is measured, not guessed: it is where a group stops fitting in cache. A
32x1024 decode batch takes 22.9 ms here and 47.5 ms at a 256 MiB budget, which
is worse than the per-sequence loop this module replaces. Bigger is not better.
Override with ``SPIKEINFER_CPU_GATHER_BUDGET_MB``."""


def _gather_budget() -> int:
    raw = os.environ.get("SPIKEINFER_CPU_GATHER_BUDGET_MB")
    if not raw:
        return _DEFAULT_GATHER_BUDGET
    try:
        return max(1, int(float(raw) * 2**20))
    except ValueError:  # pragma: no cover - user typo should not be fatal
        return _DEFAULT_GATHER_BUDGET


def _slot_matrix(
    block_tables: torch.Tensor, length: int, block_size: int
) -> torch.Tensor:
    """``[n, max_blocks]`` block table -> ``[n, length]`` physical slot indices.

    The same logical->physical walk ``gather_kv_unpacked`` does for one
    sequence, done for a whole group at once.
    """
    n_blocks = (length + block_size - 1) // block_size
    within = torch.arange(block_size, device=block_tables.device, dtype=torch.int64)
    slots = block_tables[:, :n_blocks].to(torch.int64) * block_size
    return (slots[:, :, None] + within[None, None, :]).reshape(block_tables.shape[0], -1)[
        :, :length
    ]


def paged_spike_attn_decode_cpu(
    q_packed: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    head_dim: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    out: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """One decode step against the paged spike cache, on CPU.

    Signature matches :func:`~spikeinfer.kernels.spike_attention.paged_spike_attn_decode`
    exactly so it can be dropped in behind the same dispatch.
    """
    T, n_seqs, n_heads, _ = q_packed.shape
    if out is None:
        out = torch.empty((T, n_seqs, n_heads, head_dim), device=q_packed.device, dtype=dtype)

    lens = seq_lens.to(torch.int64).tolist()
    # Sequences are grouped by position, not sorted by length: sorting would
    # pack the padding tighter but costs a permutation of the output, and decode
    # batches are already length-homogeneous in practice (they advance together).
    per_seq_bytes = T * num_kv_heads * head_dim * 4 * 2
    budget = _gather_budget()

    start = 0
    while start < n_seqs:
        stop = start + 1
        longest = lens[start]
        while stop < n_seqs:
            candidate = max(longest, lens[stop])
            if (stop + 1 - start) * candidate * per_seq_bytes > budget:
                break
            longest = candidate
            stop += 1
        _attend_group(
            q_packed[:, start:stop],
            k_cache,
            v_cache,
            block_tables[start:stop],
            lens[start:stop],
            longest,
            head_dim,
            block_size,
            num_q_heads,
            num_kv_heads,
            out[:, start:stop],
        )
        start = stop
    return out


def _attend_group(
    q_packed: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    lens: list[int],
    length: int,
    head_dim: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None:
    """Attend one group of sequences, padded to a common ``length``."""
    T, n, n_heads, _ = q_packed.shape
    device = q_packed.device

    slots = _slot_matrix(block_tables, length, block_size).reshape(-1)

    def gather(cache: torch.Tensor) -> torch.Tensor:
        # [T, H_kv, n*length, W] -> [T*n, H_kv, length, head_dim].
        # The transpose that puts n next to T happens on the *packed* words,
        # which are 32x smaller than the spikes they encode; unpacking after
        # the permute rather than before saves copying the dense tensor twice.
        words = cache.index_select(2, slots).view(T, num_kv_heads, n, length, -1)
        words = words.permute(0, 2, 1, 3, 4).reshape(T * n, num_kv_heads, length, -1)
        return unpack_spikes(words, head_dim, torch.float32)

    q = unpack_spikes(q_packed, head_dim, torch.float32)  # [T, n, H_q, head_dim]
    q = q.reshape(T * n, n_heads, 1, head_dim)

    # Positions past a sequence's length address whatever the block table's
    # padding points at; -inf makes their contribution exactly zero. Row 0 is
    # always valid (seq_len >= 1), so no row is fully masked and SDPA cannot
    # produce a NaN here.
    #
    # A group whose sequences are all the same length has no padding to mask,
    # and building the mask anyway measurably costs more than the attention at
    # small batches -- a single-sequence decode is always this case, and a
    # steady-state decode batch usually is too, since sequences advance in
    # lockstep. Skipping it is what keeps this path from losing to `_ref` at
    # batch 1.
    mask = None
    if any(length != other for other in lens):
        valid = torch.arange(length, device=device)[None, :] < torch.tensor(
            lens, device=device
        )[:, None]
        mask = torch.zeros((n, length), dtype=torch.float32, device=device)
        mask.masked_fill_(~valid, float("-inf"))
        mask = (
            mask[None, :, None, None, :]
            .expand(T, n, 1, 1, length)
            .reshape(T * n, 1, 1, length)
        )

    context = F.scaled_dot_product_attention(
        q,
        gather(k_cache),
        gather(v_cache),
        attn_mask=mask,
        enable_gqa=num_q_heads != num_kv_heads,
    )  # [T*n, H_q, 1, head_dim]

    out.copy_(context.view(T, n, n_heads, head_dim).to(out.dtype))
