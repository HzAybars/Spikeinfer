"""Tests for CUDA graph capture (GraphedForward / GraphedDecoder).

CUDA graph capture is inherently GPU-only, so every test in this file
requires CUDA.

* ``GraphedForward`` must reproduce eager fast-model output bit-for-bit
  across SEVERAL different input batches replayed through the SAME captured
  graph -- this specifically catches stale-buffer bugs (e.g. forgetting to
  copy new input into the static input buffer, or aliasing issues in the
  captured output).
* ``GraphedDecoder`` must produce token-for-token identical greedy output to
  eager ``generate(..., temperature=0.0)``.
* ``GraphedForward`` must raise on an input shape that does not match what
  it was captured for (capture bakes in fixed (batch_size, seq_len)).
"""
from __future__ import annotations

import pytest
import torch

from conftest import build_fast_from_ref, build_ref_model, requires_cuda, small_config
from spikeinfer.graph_runner import GraphedDecoder, GraphedForward
from spikeinfer.kv_cache import generate


def _random_ids(B, S, vocab_size, device, seed):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (B, S), device=device)


def _cleanup(*modules):
    for m in modules:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@requires_cuda
def test_graphed_forward_bit_exact_vs_eager_multiple_batches(device):
    T = 4
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    B, S = 2, 16
    graphed = GraphedForward(fast, batch_size=B, seq_len=S, T=T, device=device)

    for i in range(5):
        ids = _random_ids(B, S, cfg.vocab_size, device, seed=100 + i)
        with torch.no_grad():
            eager_logits = fast(ids, T=T)["logits"]
            graph_logits = graphed(ids)["logits"]

        n_diff = (graph_logits != eager_logits).sum().item()
        max_diff = (graph_logits - eager_logits).abs().max().item()
        assert torch.equal(graph_logits, eager_logits), (
            f"GraphedForward diverged from eager on batch #{i}: "
            f"{n_diff}/{eager_logits.numel()} elements differ, max abs diff {max_diff:.3e}"
        )

    _cleanup(ref, fast, graphed)


@requires_cuda
def test_graphed_forward_raises_on_wrong_input_shape(device):
    T = 4
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    graphed = GraphedForward(fast, batch_size=2, seq_len=16, T=T, device=device)

    bad_batch = _random_ids(3, 16, cfg.vocab_size, device, seed=1)
    with pytest.raises(ValueError):
        graphed(bad_batch)

    bad_seq = _random_ids(2, 8, cfg.vocab_size, device, seed=2)
    with pytest.raises(ValueError):
        graphed(bad_seq)

    _cleanup(ref, fast, graphed)


@requires_cuda
def test_graphed_decoder_matches_eager_generate_greedy(device):
    T = 4
    cfg = small_config(T=T)
    ref = build_ref_model(cfg, device, seed=0)
    fast = build_fast_from_ref(cfg, ref, device)

    ids = _random_ids(1, 6, cfg.vocab_size, device, seed=3)
    max_new_tokens = 8
    max_len = ids.shape[1] + max_new_tokens

    with torch.no_grad():
        eager_ids = generate(
            fast, ids.clone(), max_new_tokens=max_new_tokens, T=T,
            temperature=0.0, max_len=max_len,
        )

        dec = GraphedDecoder(fast, batch_size=1, max_len=max_len, T=T, device=device)
        logits = dec.prefill(ids.clone())
        graph_ids = ids.clone()
        nxt = logits.argmax(dim=-1, keepdim=True)
        graph_ids = torch.cat([graph_ids, nxt], dim=1)
        for _ in range(max_new_tokens - 1):
            logits = dec.step(nxt)
            nxt = logits.argmax(dim=-1, keepdim=True)
            graph_ids = torch.cat([graph_ids, nxt], dim=1)

    assert torch.equal(graph_ids, eager_ids), (
        f"GraphedDecoder greedy token ids differ from eager generate:\n"
        f"graph={graph_ids.tolist()}\neager={eager_ids.tolist()}"
    )

    _cleanup(ref, fast, dec)
