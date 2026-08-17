"""Tests for SpikeKVCache / forward_with_cache / generate.

Why an incremental cache should be exactly (or near-exactly) equivalent to
recomputing the full prefix is explained in ``spikeinfer/kv_cache.py``:
the membrane potential is per-position and never mixes across the sequence
axis, so caching k/v spikes for past positions and starting a new position's
membrane at zero reproduces the full-prefix computation.

Two different equivalences are checked here, and they have different
tolerances:

* A *single* cached prefill call over the whole sequence uses the exact same
  GEMM shapes (M = T*B*S) as the uncached ``fast.model(ids, T=T)`` call, so
  it is bit-exact (``torch.equal``).
* Incremental decode spreads the same computation over several forward calls
  with *different* M (prefill chunk, then M = T*B*1 per decode step), which
  changes which cuBLAS/reduction kernel gets selected per call (same class of
  effect that makes the fast path diverge slightly from the reference in
  ``test_equivalence.py``). That is only ``allclose(atol=1e-5)`` (~2.4e-7
  measured), not bit-exact.

Greedy generation (temperature=0.0) is still exactly reproducible token-for-
token against a naive reference-model greedy loop, since the tiny numeric
drift never flips an argmax in these small test configs.
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_fast_from_ref, build_ref_model, requires_cuda, small_config
from spikeinfer.kv_cache import SpikeKVCache, forward_with_cache, generate


def _random_ids(B, S, vocab_size, device, seed):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (B, S), device=device)


def _cleanup(*modules):
    for m in modules:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# cached prefill (single call, full sequence) bit-exact vs uncached
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("B,S,T", [(1, 16, 4), (2, 24, 2), (1, 1, 4)])
def test_cached_prefill_bit_exact_vs_uncached(B, S, T, device):
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    ids = _random_ids(B, S, cfg.vocab_size, device, seed=3)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype = fast.model.embed_tokens.weight.dtype

    with torch.no_grad():
        hidden_uncached = fast.model(ids, T=T)

        cache = SpikeKVCache(
            cfg.num_hidden_layers, T, B, cfg.num_key_value_heads, head_dim,
            max_len=S, device=device, dtype=dtype,
        )
        hidden_cached = forward_with_cache(fast.model, ids, cache, T=T)

    n_diff = (hidden_cached != hidden_uncached).sum().item()
    max_diff = (hidden_cached - hidden_uncached).abs().max().item()
    assert torch.equal(hidden_cached, hidden_uncached), (
        f"cached prefill vs uncached mismatch B={B},S={S},T={T}: "
        f"{n_diff}/{hidden_uncached.numel()} elements differ, max abs diff {max_diff:.3e}"
    )

    _cleanup(ref, fast)


# ---------------------------------------------------------------------------
# incremental decode vs full-prefix recompute: allclose, NOT bit-exact
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("B,P,K,T", [(1, 8, 5, 4), (2, 5, 3, 2)])
def test_incremental_decode_matches_full_prefix_recompute(B, P, K, T, device):
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    S_total = P + K
    ids_full = _random_ids(B, S_total, cfg.vocab_size, device, seed=5)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype = fast.model.embed_tokens.weight.dtype

    with torch.no_grad():
        recompute_hidden = fast.model(ids_full, T=T)[:, -1, :]

        cache = SpikeKVCache(
            cfg.num_hidden_layers, T, B, cfg.num_key_value_heads, head_dim,
            max_len=S_total, device=device, dtype=dtype,
        )
        h = forward_with_cache(fast.model, ids_full[:, :P], cache, T=T)
        for i in range(P, S_total):
            h = forward_with_cache(fast.model, ids_full[:, i:i + 1], cache, T=T)
        decode_hidden = h[:, -1, :]

    max_diff = (decode_hidden - recompute_hidden).abs().max().item()
    assert torch.allclose(decode_hidden, recompute_hidden, atol=1e-5), (
        f"incremental decode vs full-prefix recompute diverged beyond atol=1e-5 for "
        f"B={B},P={P},K={K},T={T}: max abs diff {max_diff:.3e}"
    )
    # Different GEMM M per call (prefill chunk vs single-token decode steps)
    # selects different cuBLAS kernels than one large recompute call -- this
    # is expected NOT to be bit-exact (unlike the single-shot cached prefill
    # case above).
    assert not torch.equal(decode_hidden, recompute_hidden), (
        f"expected incremental decode to diverge slightly (cuBLAS kernel selection "
        f"differs across GEMM shapes) but got bit-exact equality for B={B},P={P},K={K},T={T}"
    )

    _cleanup(ref, fast)


# ---------------------------------------------------------------------------
# greedy generation token IDs vs a naive reference-model greedy loop
# ---------------------------------------------------------------------------

def _reference_greedy_generate(ref_model, ids, max_new_tokens, T):
    out_ids = ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = ref_model(out_ids, T=T)["logits"]
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out_ids = torch.cat([out_ids, nxt], dim=1)
    return out_ids


@requires_cuda
def test_greedy_generation_matches_reference_loop(device):
    T = 4
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    ids = _random_ids(1, 6, cfg.vocab_size, device, seed=9)
    max_new_tokens = 8

    with torch.no_grad():
        fast_ids = generate(
            fast, ids.clone(), max_new_tokens=max_new_tokens, T=T, temperature=0.0,
        )
    ref_ids = _reference_greedy_generate(ref, ids.clone(), max_new_tokens, T)

    assert torch.equal(fast_ids, ref_ids), (
        f"greedy generation token id mismatch:\nfast={fast_ids.tolist()}\nref ={ref_ids.tolist()}"
    )

    _cleanup(ref, fast)


# ---------------------------------------------------------------------------
# SpikeKVCache.append raises on overflow past max_len
# ---------------------------------------------------------------------------

def test_kv_cache_append_raises_on_overflow(device):
    T, B, KV_H, D, max_len = 2, 1, 2, 16, 4
    cache = SpikeKVCache(
        num_layers=1, T=T, batch_size=B, kv_heads=KV_H, head_dim=D,
        max_len=max_len, device=device, dtype=torch.float32,
    )
    k = torch.zeros(T, B, KV_H, max_len, D, device=device)
    v = torch.zeros_like(k)
    cache.append(0, k, v)  # exactly fills max_len -- OK
    cache.advance(max_len)

    overflow = torch.zeros(T, B, KV_H, 1, D, device=device)
    with pytest.raises(ValueError):
        cache.append(0, overflow, overflow)
