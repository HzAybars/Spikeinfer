"""The kernels on their own, with no engine and no model.

``lif_multistep`` runs a whole T-step LIF recurrence in one launch, and
``pack_spikes`` turns its binary output into one bit per spike. Neither knows
anything about Qwen2 or about serving -- they take timesteps, channels and a
tensor, so they are reusable against any spiking model.

    python examples/lif_kernel.py
"""
from __future__ import annotations

import torch

from spikeinfer.kernels import lif_multistep, pack_spikes, packed_width, unpack_spikes

T, BATCH, CHANNELS = 4, 2, 512


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("no CUDA device: running the pure-PyTorch fallback\n")

    torch.manual_seed(0)
    current = torch.randn(T, BATCH, CHANNELS, device=device)
    beta = torch.rand(CHANNELS, device=device)
    threshold = torch.rand(CHANNELS, device=device) * 0.5 + 0.1

    spikes = lif_multistep(current, beta, threshold)
    assert spikes.shape == current.shape
    assert set(spikes.unique().tolist()) <= {0.0, 1.0}, "LIF output must be binary"
    print(f"input current : {tuple(current.shape)} {current.dtype}")
    print(f"spikes        : {tuple(spikes.shape)}, {100 * spikes.mean().item():.1f}% firing")

    words = pack_spikes(spikes)
    print(f"packed        : {tuple(words.shape)} int32 ({packed_width(CHANNELS)} words/neuron)")

    recovered = unpack_spikes(words, CHANNELS, torch.float32)
    assert torch.equal(spikes, recovered), "packing must be exactly invertible"
    print("roundtrip     : exact")

    print(
        f"memory        : {spikes.nbytes:,} B -> {words.nbytes:,} B "
        f"({spikes.nbytes / words.nbytes:.0f}x smaller)"
    )


if __name__ == "__main__":
    main()
