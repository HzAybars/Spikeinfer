"""Paged attention over bit-packed spike keys and values (Triton).

This kernel is the reason a spiking model can be served differently from a dense
one. After the LIF, ``q``, ``k`` and ``v`` are binary. That collapses the two
attention matmuls into integer bit ops:

    q . k  =  popcount(q_bits & k_bits)          exact, 0..head_dim
    p . v  =  sum of p[j] where bit d of v[j] is set

The first is exact by construction -- no floating-point contraction order to
worry about, no tensor cores, no accumulation error. The second is an ordinary
fp32 accumulation, done with a flash-style online softmax so nothing of size
``seq_len`` is ever materialised.

Because the operands are packed (``kernels/packing.py``), the kernel reads 32x
fewer bytes than an fp32 cache would. Decode attention is purely
bandwidth-bound, so that is close to a 32x speedup on the attention read at long
context -- and it is what lets the cache be paged at a sane block count.

Scope: **decode only** (one query position per program). Prefill goes through
:func:`gather_kv_unpacked` + SDPA, which is compute-bound and already saturates
tensor cores; a packed prefill kernel would trade tensor cores for CUDA-core
popcounts and lose. See ``docs/architecture.md``.

There is no ``triton.autotune`` here, for the same reason as the LIF kernel:
autotune launches benchmark kernels, which is illegal during CUDA graph capture.
Block sizes come from a fixed heuristic.
"""
from __future__ import annotations

import math

import torch

from .packing import BITS_PER_WORD, packed_width, unpack_spikes

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - installs without triton
    _HAS_TRITON = False


if _HAS_TRITON:

    # A @triton.jit body may only close over globals that are already
    # tl.constexpr, so the packing layout constant is re-declared as one.
    _BITS = tl.constexpr(BITS_PER_WORD)

    @triton.jit
    def _popc(v):
        """Population count of an int32 tile, SWAR.

        Triton 3.7 exposes no ``popc`` intrinsic, so this is the classic
        divide-and-conquer form. It is exact on negative inputs too: every
        intermediate mask (0x55.., 0x33.., 0x0F..) has bit 31 clear, so the
        sign extension an arithmetic ``>>`` introduces is masked away before it
        can pollute a count.
        """
        v = v - ((v >> 1) & 0x55555555)
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
        v = (v + (v >> 4)) & 0x0F0F0F0F
        return ((v * 0x01010101) >> 24) & 0x3F

    @triton.jit
    def _paged_spike_attn_decode_kernel(
        Q_ptr,  # int32 [T, N_DEC, H_Q, PACK_W]      packed query spikes
        K_ptr,  # int32 [T, H_KV, NUM_SLOTS, PACK_W] packed key cache
        V_ptr,  # int32 [T, H_KV, NUM_SLOTS, PACK_W] packed value cache
        Out_ptr,  # dtype [T, N_DEC, H_Q, HEAD_DIM]
        BlockTables,  # int32 [N_DEC, MAX_BLOCKS]
        SeqLens,  # int32 [N_DEC]
        scale,
        stride_qt, stride_qn, stride_qh, stride_qw,
        stride_kt, stride_kh, stride_ks, stride_kw,
        stride_vt, stride_vh, stride_vs, stride_vw,
        stride_ot, stride_on, stride_oh, stride_od,
        stride_btn, stride_btb,
        N_REP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PACK_W: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        pid_n = tl.program_id(0)  # sequence in the decode batch
        pid_h = tl.program_id(1)  # query head
        pid_t = tl.program_id(2)  # spiking timestep

        kv_head = pid_h // N_REP
        seq_len = tl.load(SeqLens + pid_n)

        offs_s = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_DIM)
        # Channel d lives in word d//32 at bit d%32 -- see kernels/packing.py.
        word_of_d = offs_d // _BITS
        bit_of_d = offs_d % _BITS

        q_base = Q_ptr + pid_t * stride_qt + pid_n * stride_qn + pid_h * stride_qh
        k_base = K_ptr + pid_t * stride_kt + kv_head * stride_kh
        v_base = V_ptr + pid_t * stride_vt + kv_head * stride_vh

        m_i = float("-inf")  # running max of the scores
        l_i = 0.0  # running softmax denominator
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for blk in range(0, num_blocks):
            phys = tl.load(BlockTables + pid_n * stride_btn + blk * stride_btb)
            slots = phys * BLOCK_SIZE + offs_s
            valid = (blk * BLOCK_SIZE + offs_s) < seq_len

            # ---- scores = popcount(q & k) -------------------------------
            counts = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
            for w in tl.static_range(PACK_W):
                qw = tl.load(q_base + w * stride_qw)
                kw = tl.load(
                    k_base + slots * stride_ks + w * stride_kw, mask=valid, other=0
                )
                counts += _popc(qw & kw)

            scores = counts.to(tl.float32) * scale
            scores = tl.where(valid, scores, float("-inf"))

            # ---- online softmax ----------------------------------------
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(valid, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            # ---- acc += p @ v, with v read one bit at a time ------------
            vw = tl.load(
                v_base + slots[:, None] * stride_vs + word_of_d[None, :] * stride_vw,
                mask=valid[:, None],
                other=0,
            )
            v_bits = ((vw >> bit_of_d[None, :]) & 1).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v_bits, axis=0)
            m_i = m_new

        out = acc / l_i
        out_ptr = (
            Out_ptr
            + pid_t * stride_ot
            + pid_n * stride_on
            + pid_h * stride_oh
            + offs_d * stride_od
        )
        tl.store(out_ptr, out.to(Out_ptr.dtype.element_ty))


def paged_spike_attn_decode(
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
    """One decode step of attention against the paged spike cache.

    Args:
        q_packed: ``int32 [T, N, H_q, W]`` packed query spikes, one query
            position per sequence (``W = ceil(head_dim/32)``).
        k_cache, v_cache: ``int32 [T, H_kv, num_slots, W]`` paged caches.
        block_tables: ``int32 [N, max_blocks]`` logical->physical block map.
        seq_lens: ``int32 [N]`` cached length per sequence, *including* the
            token being decoded (write the KV before calling this).
        out: optional preallocated ``[T, N, H_q, head_dim]`` output, required
            for CUDA graph capture so the buffer address is stable.

    Returns:
        ``[T, N, H_q, head_dim]`` attention output, ready to be viewed as
        ``[T, 1, N, H_q*head_dim]`` for ``o_proj``.
    """
    T, n_seqs, n_heads, words = q_packed.shape
    if words != packed_width(head_dim):
        raise ValueError(f"q has {words} words, expected {packed_width(head_dim)}")
    if n_heads != num_q_heads:
        raise ValueError(f"q has {n_heads} heads, expected {num_q_heads}")

    if out is None:
        out = torch.empty((T, n_seqs, n_heads, head_dim), device=q_packed.device, dtype=dtype)

    if not _HAS_TRITON or not q_packed.is_cuda:
        ref = paged_spike_attn_decode_ref(
            q_packed, k_cache, v_cache, block_tables, seq_lens,
            head_dim, block_size, num_q_heads, num_kv_heads, dtype=out.dtype,
        )
        out.copy_(ref)
        return out

    grid = (n_seqs, num_q_heads, T)
    _paged_spike_attn_decode_kernel[grid](
        q_packed, k_cache, v_cache, out,
        block_tables, seq_lens,
        1.0 / math.sqrt(head_dim),
        q_packed.stride(0), q_packed.stride(1), q_packed.stride(2), q_packed.stride(3),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        N_REP=num_q_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        PACK_W=words,
        HEAD_DIM=head_dim,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def gather_kv_unpacked(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read one sequence's cached spikes back as a dense ``[T, H_kv, S, D]``.

    Used by the prefill path, which hands the result to SDPA. Unpacking is a
    shift-and-mask over the packed words, so it costs one pass over 1/32 of the
    dense bytes plus the write of the dense tensor.
    """
    n_blocks = (seq_len + block_size - 1) // block_size
    slots = (
        block_table[:n_blocks, None].to(torch.int64) * block_size
        + torch.arange(block_size, device=cache.device)[None, :]
    ).reshape(-1)[:seq_len]
    words = cache.index_select(2, slots)  # [T, H_kv, S, W]
    return unpack_spikes(words, head_dim, dtype)


def paged_spike_attn_decode_ref(
    q_packed: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    head_dim: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pure-PyTorch equivalent of :func:`paged_spike_attn_decode`.

    The correctness oracle for the Triton kernel, and the CPU fallback. Runs one
    sequence at a time through ``scaled_dot_product_attention`` on unpacked
    spikes, which is exactly what the eager engine used to do.
    """
    import torch.nn.functional as F

    T = q_packed.shape[0]
    n_seqs = q_packed.shape[1]
    n_rep = num_q_heads // num_kv_heads
    out = torch.empty((T, n_seqs, num_q_heads, head_dim), device=q_packed.device, dtype=dtype)

    for i in range(n_seqs):
        s_len = int(seq_lens[i])
        q = unpack_spikes(q_packed[:, i], head_dim, torch.float32)  # [T, H_q, D]
        k = gather_kv_unpacked(k_cache, block_tables[i], s_len, head_dim, block_size, torch.float32)
        v = gather_kv_unpacked(v_cache, block_tables[i], s_len, head_dim, block_size, torch.float32)
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        o = F.scaled_dot_product_attention(q.unsqueeze(2), k, v)  # [T, H_q, 1, D]
        out[:, i] = o.squeeze(2).to(dtype)
    return out
