"""Bit-packing for spike tensors.

Every tensor this engine caches is binary: ``lif_multistep`` emits 0.0/1.0, so a
head_dim=64 key costs 256 bytes in fp32 and 8 bytes packed. The KV cache is the
dominant memory consumer in any LLM server, and a spiking model makes it *worse*
than a dense one -- the extra T axis multiplies it by the timestep count. Packing
inverts that: at T=4 the packed spike cache is 4x *smaller* than the fp16 KV
cache of the dense Qwen2.5-0.5B this model was converted from.

    fp32 spikes   98.3 KB/token   (2 * 24 layers * T4 * 2 kv heads * 64 * 4B)
    bf16 spikes   49.2 KB/token
    packed bits    3.1 KB/token   <- this module
    dense fp16     12.3 KB/token  (reference point: no T axis, no spikes)

The layout convention, which ``kernels/spike_attention.py`` and every unpack path
must agree on:

    bit i of word w  <->  channel  w * 32 + i

i.e. little-endian within a word, words ascending along the channel axis. Words
are ``int32`` because that is what Triton's bitwise ops and the SWAR popcount
operate on natively; the sign bit is used as a value bit, so word values are
reinterpreted, never compared numerically.

Nothing here is a Triton kernel. Packing is a pure-tensor bit shuffle that runs
at memory bandwidth in eager PyTorch, and keeping it in torch means it works on
CPU (tests, no-GPU installs) with the same code path.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - installs without triton
    _HAS_TRITON = False

BITS_PER_WORD = 32
_UINT32 = 1 << 32
_INT32_MAX = 1 << 31


def packed_width(num_channels: int) -> int:
    """Words needed to hold ``num_channels`` spikes. head_dim=64 -> 2."""
    return (num_channels + BITS_PER_WORD - 1) // BITS_PER_WORD


def _shifts(device: torch.device, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    return torch.arange(BITS_PER_WORD, device=device, dtype=dtype)


if _HAS_TRITON:

    @triton.jit
    def _pack_kernel(
        X_ptr,
        Out_ptr,
        n_rows,
        C,
        stride_xr,
        stride_or,
        W: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        row_ok = rows < n_rows
        bits = tl.arange(0, 32)

        for w in tl.static_range(W):
            cols = w * 32 + bits
            mask = row_ok[:, None] & (cols[None, :] < C)
            x = tl.load(X_ptr + rows[:, None] * stride_xr + cols[None, :], mask=mask, other=0)
            # Shifting a 1 into bit 31 overflows int32 into the sign bit; the
            # sum wraps in two's complement, which is exactly the bit pattern
            # wanted. At most one term can set bit 31, so nothing is lost.
            word = tl.sum((x != 0).to(tl.int32) << bits[None, :], axis=1)
            tl.store(Out_ptr + rows * stride_or + w, word, mask=row_ok)


def pack_spikes(x: torch.Tensor) -> torch.Tensor:
    """``[..., C]`` of 0/1 values -> ``int32 [..., ceil(C/32)]``.

    Accepts any dtype: anything nonzero packs as a 1, so fp32/bf16 spike tensors
    and a ``uint8`` cache both work. ``C`` need not be a multiple of 32; the tail
    of the last word is zero-filled.

    On CUDA this is one Triton launch. That matters more than it looks: the
    decode path packs q, k and v for all 24 layers on every step, and the
    tensor-op version costs about seven elementwise kernels each -- roughly 500
    extra kernels per token, which shows up even inside a CUDA graph.
    """
    if _HAS_TRITON and x.is_cuda:
        return _pack_spikes_triton(x)
    channels = x.shape[-1]
    words = packed_width(channels)
    pad = words * BITS_PER_WORD - channels

    bits = (x != 0).to(torch.int64)
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.reshape(*bits.shape[:-1], words, BITS_PER_WORD)

    acc = (bits << _shifts(x.device)).sum(dim=-1)
    # Values >= 2^31 must wrap to negative int32 rather than saturate; do the
    # wrap explicitly instead of relying on int64->int32 cast behaviour.
    acc = torch.where(acc >= _INT32_MAX, acc - _UINT32, acc)
    return acc.to(torch.int32)


def _pack_spikes_triton(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[-1]
    words = packed_width(channels)
    flat = x.contiguous().reshape(-1, channels)
    out = torch.empty((flat.shape[0], words), dtype=torch.int32, device=x.device)

    rows = flat.shape[0]
    block_r = min(256, max(1, 1 << (rows - 1).bit_length())) if rows > 1 else 1
    _pack_kernel[(triton.cdiv(rows, block_r),)](
        flat,
        out,
        rows,
        channels,
        flat.stride(0),
        out.stride(0),
        W=words,
        BLOCK_R=block_r,
        num_warps=4,
    )
    return out.view(*x.shape[:-1], words)


def unpack_spikes(
    words: torch.Tensor, num_channels: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``int32 [..., W]`` -> ``[..., num_channels]`` of 0/1 in ``dtype``.

    Inverse of :func:`pack_spikes`. The ``& 1`` after the shift is what makes the
    sign bit safe: ``>>`` on a negative int32 sign-extends, and masking discards
    the extension.
    """
    expected = packed_width(num_channels)
    if words.shape[-1] != expected:
        raise ValueError(
            f"expected {expected} words for {num_channels} channels, got {words.shape[-1]}"
        )
    bits = (words.unsqueeze(-1) >> _shifts(words.device, torch.int32)) & 1
    bits = bits.reshape(*words.shape[:-1], expected * BITS_PER_WORD)
    return bits[..., :num_channels].to(dtype)


def packed_bytes_per_token(
    num_layers: int, num_timesteps: int, num_kv_heads: int, head_dim: int
) -> int:
    """KV bytes one cached token costs, both K and V, across the whole model."""
    return 2 * num_layers * num_timesteps * num_kv_heads * packed_width(head_dim) * 4
