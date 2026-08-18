"""Attention against the paged spike cache.

One entry point, :func:`paged_attention`, used by every step the engine takes.
It does three things in order:

1. packs this step's ``k``/``v`` spikes and scatters them into the paged cache
   at the slots the block manager assigned;
2. runs the decode tokens through the packed Triton kernel
   (``kernels/spike_attention.py``) -- one query position each, attending to
   their full cached prefix;
3. runs the prefill chunks through SDPA on unpacked spikes, one sequence at a
   time, with a causal mask offset by however much of that sequence is already
   cached (which is what makes chunked prefill work).

The split is on purpose. Decode is bandwidth-bound, so reading 1-bit keys wins
big. Prefill is compute-bound and SDPA already reaches the tensor cores; a
packed prefill kernel would trade those for CUDA-core popcounts and lose.

Token order in the batch is **decodes first, then prefills**, so the decode
region is a contiguous prefix that a CUDA graph can replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .kernels.packing import pack_spikes
from .kernels.spike_attention import gather_kv_unpacked, paged_spike_attn_decode


@dataclass
class AttentionMetadata:
    """Everything attention needs that is not a weight or an activation."""

    slot_mapping: torch.Tensor
    """int64 ``[n_tok]`` -- physical cache slot for each token in the batch."""

    block_size: int
    num_decode_tokens: int = 0

    # Decode region: rows 0..num_decode_tokens of the batch.
    block_tables: torch.Tensor | None = None
    """int32 ``[n_dec, max_blocks]``."""
    seq_lens: torch.Tensor | None = None
    """int32 ``[n_dec]`` -- cached length including the token being decoded."""

    # Prefill region: one entry per chunk, in batch order.
    prefill_query_starts: list[int] = field(default_factory=list)
    prefill_query_lens: list[int] = field(default_factory=list)
    prefill_context_lens: list[int] = field(default_factory=list)
    prefill_block_tables: list[torch.Tensor] = field(default_factory=list)

    _mirrors: dict = field(default_factory=dict, repr=False, compare=False)
    """Per-device copies, built on demand. See :meth:`on`."""

    @property
    def num_prefills(self) -> int:
        return len(self.prefill_query_lens)

    def on(self, device: torch.device) -> AttentionMetadata:
        """This metadata, with its tensors on ``device``.

        A hybrid placement runs some layers on the GPU and some on the CPU, and
        each layer's cache slabs live where it computes -- so ``index_copy_``
        and the decode kernel need indices on the matching device. Copies are
        memoised per step because the metadata is rebuilt every step anyway;
        the cache lives on the object and dies with it.

        Returns ``self`` when the device already matches, which is every call in
        the ordinary single-device case, so this costs a comparison there.
        """
        if self.slot_mapping.device == device:
            return self
        key = str(device)
        mirror = self._mirrors.get(key)
        if mirror is None:
            mirror = AttentionMetadata(
                slot_mapping=self.slot_mapping.to(device, non_blocking=True),
                block_size=self.block_size,
                num_decode_tokens=self.num_decode_tokens,
                block_tables=None
                if self.block_tables is None
                else self.block_tables.to(device, non_blocking=True),
                seq_lens=None
                if self.seq_lens is None
                else self.seq_lens.to(device, non_blocking=True),
                prefill_query_starts=self.prefill_query_starts,
                prefill_query_lens=self.prefill_query_lens,
                prefill_context_lens=self.prefill_context_lens,
                prefill_block_tables=[t.to(device, non_blocking=True) for t in self.prefill_block_tables],
            )
            self._mirrors[key] = mirror
        return mirror


def _causal_offset_mask(
    query_len: int, total_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    """Mask for a chunk of ``query_len`` queries sitting at offset ``ctx``.

    ``None`` when the chunk covers the whole sequence and SDPA's own
    ``is_causal`` is equivalent (the common, uncached-prefill case).
    """
    ctx = total_len - query_len
    mask = torch.full((query_len, total_len), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=ctx + 1)


def paged_attention(
    q_spk: torch.Tensor,
    k_spk: torch.Tensor,
    v_spk: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttentionMetadata,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Spike attention over the paged cache.

    Args:
        q_spk, k_spk, v_spk: ``[T, 1, H, n_tok, head_dim]`` binary spikes. The
            batch axis is 1 because sequences are flattened into ``n_tok``;
            which tokens belong to which sequence lives in ``meta``.
        k_cache, v_cache: ``int32 [T, H_kv, num_slots, W]`` for this layer.

    Returns:
        ``[T, 1, n_tok, H_q * head_dim]``, ready for ``o_proj``.
    """
    T = q_spk.shape[0]
    n_tok = q_spk.shape[3]
    dtype, device = q_spk.dtype, q_spk.device

    # 1. Persist this step's keys and values, packed.
    k_cache.index_copy_(2, meta.slot_mapping, pack_spikes(k_spk.squeeze(1)))
    v_cache.index_copy_(2, meta.slot_mapping, pack_spikes(v_spk.squeeze(1)))

    out = torch.empty((T, n_tok, num_q_heads, head_dim), device=device, dtype=dtype)

    # 2. Decodes: packed popcount kernel.
    n_dec = meta.num_decode_tokens
    if n_dec:
        q_dec = q_spk[:, 0, :, :n_dec, :].permute(0, 2, 1, 3)  # [T, n_dec, H_q, D]
        paged_spike_attn_decode(
            pack_spikes(q_dec),
            k_cache,
            v_cache,
            meta.block_tables,
            meta.seq_lens,
            head_dim=head_dim,
            block_size=meta.block_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            out=out[:, :n_dec],
        )

    # 3. Prefill chunks: unpack the sequence's cache, then SDPA.
    for i in range(meta.num_prefills):
        start = meta.prefill_query_starts[i]
        q_len = meta.prefill_query_lens[i]
        total = meta.prefill_context_lens[i] + q_len
        table = meta.prefill_block_tables[i]

        k_all = gather_kv_unpacked(k_cache, table, total, head_dim, meta.block_size, dtype)
        v_all = gather_kv_unpacked(v_cache, table, total, head_dim, meta.block_size, dtype)
        q = q_spk[:, 0, :, start : start + q_len, :]  # [T, H_q, q_len, D]

        o = F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            attn_mask=_causal_offset_mask(q_len, total, device, dtype),
            enable_gqa=num_q_heads != num_kv_heads,
        )
        out[:, start : start + q_len] = o.permute(0, 2, 1, 3)

    return out.view(T, 1, n_tok, num_q_heads * head_dim)
