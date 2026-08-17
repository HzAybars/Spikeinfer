"""Fused multi-timestep LIF kernel (Triton).

Replaces the ~10 elementwise kernels + 4 full-size temporaries that
``snntorch.Leaky.forward`` issues *per timestep, per neuron site* with a single
launch that runs the whole T-step recurrence with the membrane potential held
in registers.

Semantics are bit-exact against snntorch's ``Leaky`` configured as this project
uses it (``reset_mechanism="subtract"``, ``reset_delay=True``,
``graded_spikes_factor=1.0``). The recurrence, read straight off
``snntorch/_neurons/leaky.py:209-220``:

    r_t = H(U_{t-1} - theta)                    # from the PREVIOUS membrane
    U_t = clamp(beta, 0, 1) * U_{t-1} + I_t - r_t * theta
    S_t = H(U_t - theta)

Three details are load-bearing; get any of them wrong and the model degrades
silently rather than failing:

1. ``r_t`` is computed from ``U_{t-1}``, i.e. the reset lags the spike by one
   step (``mem_reset`` runs *before* ``state_function``). Note this makes
   ``r_t == S_{t-1}`` exactly, since both are H(U_{t-1} - theta).
2. The comparison is ``(U - theta) > 0``, not ``U > theta``. In floating point
   these are not the same predicate -- snntorch materialises
   ``mem_shift = mem - threshold`` and then tests ``> 0``.
3. The membrane accumulates in fp32 regardless of the input dtype, and
   ``beta`` is clamped to [0, 1] inside the step (snntorch clamps in forward,
   not in-place on the parameter).

Inference only -- there is no backward pass here by design. Training still runs
through the original ``spikeinfer.reference.lif.LIFNeuron`` / snntorch path.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on installs without triton
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _lif_multistep_fwd(
        I_ptr,
        S_ptr,
        U_ptr,
        U0_ptr,
        beta_ptr,
        th_ptr,
        M,
        N,
        stride_it,
        stride_im,
        stride_in,
        stride_st,
        stride_sm,
        stride_sn,
        stride_um,
        stride_un,
        T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_U0: tl.constexpr,
        STORE_U: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask = (offs_m[:, None] < M) & mask_n[None, :]

        # beta/threshold broadcast over the trailing (channel) axis only.
        beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        beta = tl.minimum(tl.maximum(beta, 0.0), 1.0)  # == Tensor.clamp(0, 1)
        th = tl.load(th_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        beta = beta[None, :]
        th = th[None, :]

        off_i = offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
        off_s = offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        off_u = offs_m[:, None] * stride_um + offs_n[None, :] * stride_un

        if HAS_U0:
            u = tl.load(U0_ptr + off_u, mask=mask, other=0.0).to(tl.float32)
        else:
            u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for t in tl.static_range(T):
            cur = tl.load(I_ptr + t * stride_it + off_i, mask=mask, other=0.0).to(tl.float32)
            # Reset uses the membrane from the *previous* step (delayed reset).
            r = tl.where(u - th > 0.0, 1.0, 0.0)
            u = beta * u + cur - r * th
            spk = tl.where(u - th > 0.0, 1.0, 0.0)
            tl.store(S_ptr + t * stride_st + off_s, spk.to(S_ptr.dtype.element_ty), mask=mask)

        if STORE_U:
            tl.store(U_ptr + off_u, u, mask=mask)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_blocks(m: int, n: int):
    """Heuristic tile choice.

    Deliberately *not* triton.autotune: autotune benchmarks candidate configs by
    launching kernels, which is illegal during CUDA graph capture and would make
    the first captured replay nondeterministic. A fixed heuristic keeps the
    kernel graph-safe.
    """
    block_n = min(_next_pow2(n), 256)
    block_m = max(1, min(_next_pow2(m), 4096 // block_n))
    num_warps = 8 if block_m * block_n >= 2048 else 4
    return block_m, block_n, num_warps


def lif_multistep_ref(x, beta, threshold, mem0=None, out_dtype=None):
    """Pure-PyTorch reference. Same recurrence, used on CPU and in tests."""
    T = x.shape[0]
    beta_c = beta.clamp(0, 1).to(torch.float32)
    th = threshold.to(torch.float32)
    u = torch.zeros(x.shape[1:], device=x.device, dtype=torch.float32) if mem0 is None else mem0.to(torch.float32).clone()
    out_dtype = out_dtype or x.dtype
    spikes = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    for t in range(T):
        r = (u - th > 0).to(torch.float32)
        u = beta_c * u + x[t].to(torch.float32) - r * th
        spikes[t] = (u - th > 0).to(out_dtype)
    return spikes, u


def lif_multistep(x, beta, threshold, mem0=None, out_dtype=None, return_membrane=False):
    """Run the LIF recurrence over all T timesteps in one kernel launch.

    Args:
        x: input current, shape ``[T, *mid, N]``. ``N`` is the channel axis that
            ``beta``/``threshold`` broadcast over (head_dim for q/k/v,
            intermediate_size for the MLP gate) and must be the trailing dim.
        beta: ``[N]`` decay, clamped to [0, 1] inside the step.
        threshold: ``[N]``, must be > 0 (see note below).
        mem0: optional initial membrane ``[*mid, N]``; defaults to zeros, which
            is what the model does on every forward.
        out_dtype: spike dtype. Defaults to ``x.dtype`` (fp32 in the bit-exact
            path); pass ``torch.uint8`` for the packed KV cache.
        return_membrane: also return the final membrane potential.

    Returns:
        spikes ``[T, *mid, N]``, and the final membrane ``[*mid, N]`` (fp32) if
        ``return_membrane``.

    Precondition: ``threshold > 0``. A non-positive threshold makes the t=0 reset
    fire (H(0 - theta) = 1 when theta <= 0), diverging from the zero-initialised
    membrane assumption. It is checked by ``validate_thresholds()`` at setup
    rather than per call -- see the note in the body. ``calibrate.py`` floors
    thresholds at 0.05, so this holds in practice.
    """
    assert x.ndim >= 2, "expected [T, *mid, N]"
    N = x.shape[-1]
    assert beta.numel() == N and threshold.numel() == N, (
        f"beta/threshold must match trailing dim: got {beta.numel()}/{threshold.numel()} vs N={N}"
    )

    out_dtype = out_dtype or x.dtype

    if not _HAS_TRITON or not x.is_cuda:
        spikes, u = lif_multistep_ref(x, beta, threshold, mem0=mem0, out_dtype=out_dtype)
        return (spikes, u) if return_membrane else spikes

    # NB: the threshold > 0 precondition is deliberately NOT checked here.
    # `bool(torch.all(...))` on a CUDA tensor forces a device-to-host sync, which
    # is illegal inside a CUDA graph capture ("operation not permitted when
    # stream is capturing") and would cost a stall on every call. Validate once
    # at setup with validate_thresholds() instead.
    orig_shape = x.shape
    T = orig_shape[0]
    x3 = x.reshape(T, -1, N)  # copies only if x is not flattenable as-is
    M = x3.shape[1]

    beta = beta.contiguous()
    threshold = threshold.contiguous()

    spikes = torch.empty((T, M, N), device=x.device, dtype=out_dtype)
    need_u = return_membrane
    u_out = torch.empty((M, N), device=x.device, dtype=torch.float32) if need_u else spikes  # dummy ptr
    u_in = mem0.reshape(M, N).contiguous().to(torch.float32) if mem0 is not None else spikes

    block_m, block_n, num_warps = _pick_blocks(M, N)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _lif_multistep_fwd[grid](
        x3,
        spikes,
        u_out,
        u_in,
        beta,
        threshold,
        M,
        N,
        x3.stride(0),
        x3.stride(1),
        x3.stride(2),
        spikes.stride(0),
        spikes.stride(1),
        spikes.stride(2),
        N if not need_u else u_out.stride(0),
        1 if not need_u else u_out.stride(1),
        T=T,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HAS_U0=mem0 is not None,
        STORE_U=need_u,
        num_warps=num_warps,
        # Required for bit-exactness. Triton/ptxas would otherwise contract
        # `beta * u + cur` into a single FMA (one rounding), while PyTorch eager
        # runs mul and add as separate kernels (two roundings). The 1-ulp
        # difference is enough to flip a spike when the membrane lands near
        # threshold -- observed at T=8, N=4864. This maps to ptxas --fmad=false.
        enable_fp_fusion=False,
    )

    spikes = spikes.view(orig_shape)
    if return_membrane:
        return spikes, u_out.view(orig_shape[1:])
    return spikes


def validate_thresholds(model):
    """Check the ``threshold > 0`` precondition across every LIF site.

    Call once after loading weights. Kept out of ``lif_multistep`` because the
    check requires a device-to-host sync, which is both a per-call stall and
    illegal during CUDA graph capture.
    """
    bad = []
    for name, module in model.named_modules():
        lif = getattr(module, "lif", None)
        th = getattr(lif, "threshold", None) if lif is not None else None
        if th is not None and not bool(torch.all(th > 0)):
            bad.append(f"{name} (min={th.min().item():.4g})")
    if bad:
        raise ValueError(
            "lif_multistep requires threshold > 0; violated at: " + ", ".join(bad)
        )
    return True
