"""Tests for the fused multi-timestep LIF Triton kernel (``lif_multistep``)
and its pure-PyTorch fallback (``lif_multistep_ref``).

Ground truth is ``spikeinfer.reference.lif.LIFNeuron`` (a thin wrapper around
``snntorch.Leaky``, configured exactly as the model uses it), stepped
manually one timestep at a time:

    mem = None
    for t in range(T):
        spk, mem = neuron(x[t], mem)

Both ``lif_multistep`` (Triton, GPU) and ``lif_multistep_ref`` (pure torch)
must reproduce that recurrence bit-for-bit (``torch.equal``) -- see the
"three load-bearing details" note at the top of
``spikeinfer/kernels/lif_triton.py``.
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_ref_model, requires_cuda, small_config
from spikeinfer.kernels import lif_multistep, lif_multistep_ref, validate_thresholds
from spikeinfer.reference.lif import LIFNeuron

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rand_beta(n, device):
    """Beta sampled outside [0, 1] on both sides, to exercise the internal
    ``clamp(beta, 0, 1)`` that both the kernel and snntorch apply per step."""
    return torch.rand(n, device=device) * 4.0 - 1.5  # range [-1.5, 2.5]


def _rand_threshold(n, device):
    return torch.rand(n, device=device) * 0.95 + 0.05  # range (0.05, 1.0)


def _rand_input(shape, device, scale=2.0):
    return torch.randn(shape, device=device) * scale


def _make_neuron(n, beta, threshold, device):
    neuron = LIFNeuron(n).to(device)
    neuron.lif.beta.data = beta.clone()
    neuron.lif.threshold.data = threshold.clone()
    return neuron


@torch.no_grad()
def _snntorch_ground_truth(x, beta, threshold):
    """Manual T-step loop over LIFNeuron -- the golden reference."""
    T = x.shape[0]
    neuron = _make_neuron(x.shape[-1], beta, threshold, x.device)
    mem = None
    spikes = []
    for t in range(T):
        spk, mem = neuron(x[t], mem)
        spikes.append(spk)
    return torch.stack(spikes, dim=0), mem


# ---------------------------------------------------------------------------
# main sweep: T x N x M, bit-exact vs snntorch
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("M", [128, 1792])
@pytest.mark.parametrize("N", [64, 4864, 63])
@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_lif_multistep_bit_exact_vs_snntorch(T, N, M, device):
    torch.manual_seed(1000 * T + 10 * N + M)
    beta = _rand_beta(N, device)
    threshold = _rand_threshold(N, device)
    x = _rand_input((T, M, N), device)

    ref_spikes, ref_mem = _snntorch_ground_truth(x, beta, threshold)
    got_spikes, got_mem = lif_multistep(x, beta, threshold, return_membrane=True)

    n_diff = (got_spikes != ref_spikes).sum().item()
    assert torch.equal(got_spikes, ref_spikes), (
        f"spike mismatch T={T} N={N} M={M}: {n_diff}/{ref_spikes.numel()} elements differ"
    )
    max_mem_diff = (got_mem - ref_mem).abs().max().item()
    assert torch.equal(got_mem, ref_mem), (
        f"membrane mismatch T={T} N={N} M={M}: max abs diff {max_mem_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# lif_multistep_ref matches snntorch bit-exactly (pure torch, CPU or CUDA)
# ---------------------------------------------------------------------------

def test_lif_multistep_ref_bit_exact_vs_snntorch(device):
    torch.manual_seed(123)
    T, M, N = 4, 128, 64
    beta = _rand_beta(N, device)
    threshold = _rand_threshold(N, device)
    x = _rand_input((T, M, N), device)

    ref_spikes, ref_mem = _snntorch_ground_truth(x, beta, threshold)
    got_spikes, got_mem = lif_multistep_ref(x, beta, threshold)

    n_diff = (got_spikes != ref_spikes).sum().item()
    assert torch.equal(got_spikes, ref_spikes), (
        f"lif_multistep_ref spike mismatch: {n_diff}/{ref_spikes.numel()} elements differ"
    )
    max_mem_diff = (got_mem - ref_mem).abs().max().item()
    assert torch.equal(got_mem, ref_mem), (
        f"lif_multistep_ref membrane mismatch: max abs diff {max_mem_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# 5-D input [T, B, H, S, D]
# ---------------------------------------------------------------------------

@requires_cuda
def test_lif_multistep_5d_input_bit_exact(device):
    torch.manual_seed(42)
    T, B, H, S, D = 4, 2, 3, 5, 64
    beta = _rand_beta(D, device)
    threshold = _rand_threshold(D, device)
    x = _rand_input((T, B, H, S, D), device)

    ref_spikes, ref_mem = _snntorch_ground_truth(x, beta, threshold)
    got_spikes, got_mem = lif_multistep(x, beta, threshold, return_membrane=True)

    n_diff = (got_spikes != ref_spikes).sum().item()
    assert torch.equal(got_spikes, ref_spikes), (
        f"5-D spike mismatch: {n_diff}/{ref_spikes.numel()} elements differ, shape={tuple(x.shape)}"
    )
    assert torch.equal(got_mem, ref_mem), (
        f"5-D membrane mismatch: max abs diff {(got_mem - ref_mem).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# non-contiguous (transposed) input
# ---------------------------------------------------------------------------

@requires_cuda
def test_lif_multistep_noncontiguous_input_bit_exact(device):
    torch.manual_seed(7)
    T, A, N = 4, 37, 64
    beta = _rand_beta(N, device)
    threshold = _rand_threshold(N, device)
    x_base = _rand_input((T, N, A), device)
    x = x_base.transpose(-1, -2)  # [T, A, N], non-contiguous
    assert not x.is_contiguous(), "test setup bug: input should be non-contiguous"

    ref_spikes, ref_mem = _snntorch_ground_truth(x, beta, threshold)
    got_spikes, got_mem = lif_multistep(x, beta, threshold, return_membrane=True)

    n_diff = (got_spikes != ref_spikes).sum().item()
    assert torch.equal(got_spikes, ref_spikes), (
        f"non-contiguous spike mismatch: {n_diff}/{ref_spikes.numel()} elements differ"
    )
    assert torch.equal(got_mem, ref_mem), (
        f"non-contiguous membrane mismatch: max abs diff {(got_mem - ref_mem).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# out_dtype=torch.uint8
# ---------------------------------------------------------------------------

@requires_cuda
def test_lif_multistep_uint8_out_dtype_bit_exact(device):
    torch.manual_seed(11)
    T, M, N = 4, 128, 64
    beta = _rand_beta(N, device)
    threshold = _rand_threshold(N, device)
    x = _rand_input((T, M, N), device)

    ref_spikes, _ = _snntorch_ground_truth(x, beta, threshold)
    got_spikes = lif_multistep(x, beta, threshold, out_dtype=torch.uint8)

    assert got_spikes.dtype == torch.uint8, f"expected uint8 output, got {got_spikes.dtype}"
    ref_u8 = ref_spikes.to(torch.uint8)
    n_diff = (got_spikes != ref_u8).sum().item()
    assert torch.equal(got_spikes, ref_u8), (
        f"uint8 spike mismatch: {n_diff}/{ref_u8.numel()} elements differ"
    )


# ---------------------------------------------------------------------------
# boundary-stress: membrane hovers right at threshold -- catches FMA fusion
# ---------------------------------------------------------------------------

@requires_cuda
def test_lif_multistep_threshold_boundary_stress(device):
    torch.manual_seed(0)
    T, M, N = 8, 256, 64
    beta = torch.full((N,), 0.9, device=device)
    threshold = torch.full((N,), 1.0, device=device)
    x = torch.ones(T, M, N, device=device) * 1.0 + torch.randn(T, M, N, device=device) * 1e-6

    ref_spikes, ref_mem = _snntorch_ground_truth(x, beta, threshold)
    got_spikes, got_mem = lif_multistep(x, beta, threshold, return_membrane=True)

    n_diff = (got_spikes != ref_spikes).sum().item()
    assert torch.equal(got_spikes, ref_spikes), (
        f"boundary-stress spike mismatch: {n_diff}/{ref_spikes.numel()} elements differ -- "
        f"possible FMA-contraction regression (enable_fp_fusion should be False)"
    )
    assert torch.equal(got_mem, ref_mem), (
        f"boundary-stress membrane mismatch: max abs diff {(got_mem - ref_mem).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# validate_thresholds
# ---------------------------------------------------------------------------

def test_validate_thresholds_ok_on_randomized_model(device):
    cfg = small_config()
    model = build_ref_model(cfg, device, seed=0)
    assert validate_thresholds(model) is True


def test_validate_thresholds_raises_on_nonpositive_threshold(device):
    cfg = small_config()
    model = build_ref_model(cfg, device, seed=0)

    found = False
    for m in model.modules():
        if isinstance(m, LIFNeuron):
            m.lif.threshold.data[0] = 0.0
            found = True
            break
    assert found, "test setup bug: no LIFNeuron found in model"

    with pytest.raises(ValueError):
        validate_thresholds(model)
