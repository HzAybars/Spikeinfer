"""Bit-packing must be exactly invertible.

Everything downstream assumes it: the KV cache stores only the packed form, so a
single lost or misplaced bit is a silently wrong key that attention will happily
score. These tests are therefore all ``torch.equal``, never a tolerance.

The awkward cases are the ones that carry bugs: a channel count that is not a
multiple of 32 (tail padding), and words whose bit 31 is set (int32 sign bit --
a naive ``>>`` sign-extends and a naive int64 cast can saturate).
"""
from __future__ import annotations

import pytest
import torch

from conftest import requires_cuda
from spikeinfer.kernels.packing import (
    BITS_PER_WORD,
    pack_spikes,
    packed_bytes_per_token,
    packed_width,
    unpack_spikes,
)


@pytest.mark.parametrize("channels", [1, 31, 32, 33, 63, 64, 128, 4864])
def test_packed_width(channels):
    assert packed_width(channels) == (channels + 31) // 32


@pytest.mark.parametrize("channels", [1, 31, 32, 33, 63, 64, 128, 256, 4864])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
def test_roundtrip_is_exact(channels, dtype, device):
    torch.manual_seed(channels)
    spikes = (torch.rand(3, 5, channels, device=device) > 0.5).to(dtype)
    words = pack_spikes(spikes)

    assert words.dtype == torch.int32
    assert words.shape == (3, 5, packed_width(channels))

    back = unpack_spikes(words, channels, dtype)
    assert torch.equal(spikes, back), f"roundtrip lost bits at C={channels}, {dtype}"


def test_all_ones_sets_the_sign_bit(device):
    """A fully firing 32-channel group is 0xFFFFFFFF, i.e. -1 as int32."""
    spikes = torch.ones(1, 32, device=device)
    words = pack_spikes(spikes)
    assert words.item() == -1
    assert torch.equal(unpack_spikes(words, 32, torch.float32), spikes)


def test_bit_order_is_little_endian_within_a_word(device):
    """Channel ``c`` must live at bit ``c % 32`` of word ``c // 32``."""
    for channel in (0, 1, 31, 32, 63):
        spikes = torch.zeros(1, 64, device=device)
        spikes[0, channel] = 1.0
        words = pack_spikes(spikes)[0]
        expected_word, expected_bit = channel // BITS_PER_WORD, channel % BITS_PER_WORD
        for w in range(2):
            value = int(words[w]) & 0xFFFFFFFF
            want = (1 << expected_bit) if w == expected_word else 0
            assert value == want, f"channel {channel} landed wrong: word {w} = {value:#x}"


def test_tail_of_a_partial_word_is_zero(device):
    """C=33 uses two words; the 31 unused bits of the second must be clear."""
    spikes = torch.ones(1, 33, device=device)
    words = pack_spikes(spikes)
    assert int(words[0, 1]) == 1, "only channel 32 should be set in the tail word"


def test_nonzero_values_count_as_spikes(device):
    """Packing keys off ``!= 0``, so a graded spike tensor still packs."""
    graded = torch.tensor([[0.0, 0.5, 0.0, 2.0]], device=device)
    words = pack_spikes(graded)
    assert torch.equal(
        unpack_spikes(words, 4, torch.float32),
        torch.tensor([[0.0, 1.0, 0.0, 1.0]], device=device),
    )


def test_unpack_rejects_the_wrong_width(device):
    words = pack_spikes(torch.ones(1, 64, device=device))
    with pytest.raises(ValueError, match="expected 4 words"):
        unpack_spikes(words, 128, torch.float32)


def test_packed_cache_is_smaller_than_the_dense_model_it_came_from():
    """The headline claim, as an assertion: T=4 packed spikes beat dense fp16.

    Qwen2.5-0.5B: 24 layers, 2 KV heads, head_dim 64. Dense fp16 KV is
    2 * 24 * 2 * 64 * 2 bytes = 12288 per token.
    """
    packed = packed_bytes_per_token(num_layers=24, num_timesteps=4, num_kv_heads=2, head_dim=64)
    dense_fp16 = 2 * 24 * 2 * 64 * 2
    assert packed == 3072
    assert packed * 4 == dense_fp16


@requires_cuda
def test_pack_handles_noncontiguous_input(device):
    """q/k/v arrive as permuted views; packing must not silently transpose."""
    spikes = (torch.rand(2, 4, 8, 64, device=device) > 0.5).float()
    view = spikes.permute(0, 2, 1, 3)
    assert not view.is_contiguous()
    assert torch.equal(unpack_spikes(pack_spikes(view), 64, torch.float32), view)
