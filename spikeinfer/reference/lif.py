"""Leaky Integrate-and-Fire spiking neuron wrapper used throughout SpikingQwen.

Wraps snntorch.Leaky with per-channel learnable decay (beta) and threshold,
matching the LIF used in Spikformer / Spike-driven Transformer style
spiking-transformer conversions. See DESIGN.md for placement rationale.
"""
import torch
import torch.nn as nn
from snntorch import Leaky, surrogate


class LIFNeuron(nn.Module):
    """Per-channel LIF neuron. Stateless module; membrane potential is
    threaded through forward() explicitly by the caller so a single
    instance can be unrolled over T timesteps with externally-owned state
    (needed since batch/seq shape varies per call, unlike snntorch's
    init_hidden convenience mode)."""

    def __init__(self, num_channels: int, beta_init: float = 0.9, threshold_init: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        beta = torch.full((num_channels,), beta_init)
        threshold = torch.full((num_channels,), threshold_init)
        self.lif = Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract",
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None = None):
        """x: [..., num_channels] real-valued input current for this timestep.
        mem: previous membrane potential, same shape as x, or None to init zeros.
        Returns (spike, new_mem), both same shape as x."""
        if mem is None:
            mem = torch.zeros_like(x)
        spike, new_mem = self.lif(x, mem)
        return spike, new_mem
