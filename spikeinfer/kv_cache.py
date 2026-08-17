"""Spike KV cache and cached generation for the fast SpikingQwen path.

This is the simple single-stream path, kept because it is easy to reason about
and is the oracle the paged engine is validated against (see
``tests/test_engine.py``). For serving, use ``spikeinfer.LLMEngine``, whose
cache is paged and bit-packed.

Without a cache, generation re-runs the entire T-step x 24-layer forward over
the *whole prefix* for every new token: generating n tokens from a prompt of
length p costs O(sum over i of (p+i)^2) attention work plus n full-depth passes.
This module makes decode incremental.

Why an incremental cache is exactly equivalent here
---------------------------------------------------
The membrane potential in this architecture is **per position and independent
across positions** -- ``init_state`` allocates zeros of shape [B, H, S, D] on
every forward and the neuron never mixes information along the sequence axis
(the only recurrence is over T). So a token's spikes depend only on its own
input current, which depends causally on positions <= itself. Caching k/v spikes
for past positions and starting the new position's membrane at zero therefore
reproduces the full-prefix recompute exactly -- verified in the test suite by
comparing cached generation against uncached generation.

Storage dtype
-------------
Spikes are binary, so uint8 would cut cache memory 4x. It is *not* used, because
``torch.matmul`` cannot consume uint8: attention would have to upcast to fp32
first, which reads 0.5x the bytes but then writes and re-reads a full-size fp32
temporary -- a net bandwidth loss. The cache is kept in the compute dtype. A
uint8 cache only pays off alongside a custom attention kernel that consumes
packed spikes directly, which is out of scope here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SpikeKVCache:
    """Preallocated per-layer cache of k/v spikes, shape [T, B, KV_H, max_len, D]."""

    def __init__(self, num_layers, T, batch_size, kv_heads, head_dim, max_len, device, dtype):
        self.max_len = max_len
        self.length = 0
        shape = (T, batch_size, kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]

    def append(self, layer_idx, k_spk, v_spk):
        """Write [T,B,KV_H,S_new,D] at the current offset; returns the valid slices."""
        s_new = k_spk.shape[3]
        end = self.length + s_new
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > max_len={self.max_len}")
        self.k[layer_idx][:, :, :, self.length:end] = k_spk
        self.v[layer_idx][:, :, :, self.length:end] = v_spk
        return self.k[layer_idx][:, :, :, :end], self.v[layer_idx][:, :, :, :end]

    def advance(self, s_new):
        self.length += s_new

    def reset(self):
        self.length = 0


@torch.no_grad()
def forward_with_cache(model, input_ids, cache, T=None, use_sdpa=False):
    """One forward pass through FastSpikingQwenModel using (and filling) ``cache``.

    Handles both prefill (S > 1, causal mask) and decode (S == 1, no mask needed
    since a single query attends to every cached position). Returns the
    rate-coded hidden state [B, S, D].
    """
    T = T or model.config.T
    B, S = input_ids.shape
    device, dtype = input_ids.device, model.embed_tokens.weight.dtype
    offset = cache.length

    embeds = model.embed_tokens(input_ids)
    position_ids = torch.arange(offset, offset + S, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = model.rotary_emb(embeds, position_ids)

    if S > 1:
        mask = torch.full((S, offset + S), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=offset + 1)[None, None, :, :]
    else:
        mask = None  # a single query position sees all cached keys

    h = embeds.unsqueeze(0).expand(T, B, S, -1).contiguous()

    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        q_spk, k_spk, v_spk = attn.spike_qkv(layer.input_layernorm(h), cos, sin)
        k_all, v_all = cache.append(i, k_spk, v_spk)
        out = attn.attend(q_spk, k_all, v_all, mask, use_sdpa)
        h = h + attn.o_proj(out)
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    cache.advance(S)
    return model.norm(h).mean(dim=0)


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=60, T=None, temperature=0.8, top_k=40,
             use_sdpa=False, eos_token_id=None, max_len=None):
    """Incremental generation. ``model`` is a FastSpikingQwenForCausalLM."""
    cfg = model.config
    T = T or cfg.T
    B, S = input_ids.shape
    device = input_ids.device
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    max_len = max_len or (S + max_new_tokens)

    cache = SpikeKVCache(
        cfg.num_hidden_layers, T, B, cfg.num_key_value_heads, head_dim,
        max_len, device, model.model.embed_tokens.weight.dtype,
    )

    hidden = forward_with_cache(model.model, input_ids, cache, T, use_sdpa)
    out_ids = input_ids
    for _ in range(max_new_tokens):
        logits = model.lm_head(hidden[:, -1])
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        else:
            nxt = logits.argmax(dim=-1, keepdim=True)

        out_ids = torch.cat([out_ids, nxt], dim=1)
        if eos_token_id is not None and (nxt == eos_token_id).all():
            break
        hidden = forward_with_cache(model.model, nxt, cache, T, use_sdpa)

    return out_ids
