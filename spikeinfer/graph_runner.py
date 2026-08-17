"""CUDA graph capture for the fast SpikingQwen path.

This is the single highest-leverage optimization in the project. A decode step
issues ~28k kernel launches in the reference model, and on Windows WDDM launch
overhead runs ~10 us/kernel -- which is why decoding one token used to cost about
as much as prefilling 128 (135.7 ms vs 143.6 ms). Capturing the work as a graph
replays it as one driver submission.

Two runners:

* ``GraphedForward``  -- fixed (B, S) forward, for evaluation/prefill workloads
  where every batch has the same shape (``evaluate.py`` uses seq_len=128).
* ``GraphedDecoder``  -- graphed incremental decode on top of a preallocated
  KV cache, for generation.

Capture rules that shape the code below: no CPU-side branching on tensor values,
no ``.item()``, no allocation whose size depends on a value, and every input must
live in a buffer that is written in place. For the decoder that means the cache
is allocated at ``max_len`` up front and attention always runs over the full
``max_len``, with a validity mask masking the unwritten tail. Padding is nearly
free at decode -- a single query against max_len keys is trivial next to the
24 layers of length-independent projections that dominate the step.

Note the LIF kernel deliberately avoids ``triton.autotune``: autotuning launches
benchmark kernels, which is illegal during capture.
"""
from __future__ import annotations

import torch

from .kv_cache import forward_with_cache


def _warmup(fn, iters=3):
    """Run on a side stream before capture, as required by the CUDA graph API."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


class GraphedForward:
    """Captures ``model(input_ids, T=T)`` for one fixed (batch_size, seq_len)."""

    def __init__(self, model, batch_size, seq_len, T=None, use_sdpa=False, device="cuda"):
        self.model = model
        self.T = T or model.config.T
        self.static_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        def run():
            return model(self.static_ids, T=self.T, use_sdpa=use_sdpa)

        with torch.no_grad():
            _warmup(run)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = run()

    @torch.no_grad()
    def __call__(self, input_ids):
        if input_ids.shape != self.static_ids.shape:
            raise ValueError(
                f"GraphedForward captured for {tuple(self.static_ids.shape)}, got {tuple(input_ids.shape)}"
            )
        self.static_ids.copy_(input_ids)
        self.graph.replay()
        return {"loss": None, "logits": self.static_out["logits"].clone()}


class GraphedDecoder:
    """Graphed single-token decode against a preallocated spike KV cache.

    Usage::

        dec = GraphedDecoder(model, batch_size=1, max_len=512)
        logits = dec.prefill(prompt_ids)      # eager, fills the cache
        logits = dec.step(next_token_ids)     # graph replay, one token
    """

    def __init__(self, model, batch_size=1, max_len=512, T=None, use_sdpa=False, device="cuda"):
        cfg = model.config
        self.model = model
        self.cfg = cfg
        self.T = T or cfg.T
        self.max_len = max_len
        self.use_sdpa = use_sdpa
        self.length = 0

        dtype = model.model.embed_tokens.weight.dtype
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.head_dim = head_dim
        shape = (self.T, batch_size, cfg.num_key_value_heads, max_len, head_dim)
        n = cfg.num_hidden_layers
        self.cache_k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n)]
        self.cache_v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n)]

        # Buffers mutated in place between replays.
        self.static_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.static_pos = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.write_idx = torch.zeros((1,), dtype=torch.long, device=device)
        # [1,1,1,max_len]: 0 where a key is valid, -inf on the unwritten tail.
        self.valid_mask = torch.full((1, 1, 1, max_len), float("-inf"), device=device, dtype=dtype)

        self._captured = False
        self._device = device

    def _decode_body(self):
        model = self.model.model
        embeds = model.embed_tokens(self.static_ids)
        cos, sin = model.rotary_emb(embeds, self.static_pos)
        h = embeds.unsqueeze(0).expand(self.T, -1, -1, -1).contiguous()

        for i, layer in enumerate(model.layers):
            attn = layer.self_attn
            q_spk, k_spk, v_spk = attn.spike_qkv(layer.input_layernorm(h), cos, sin)
            # index_copy_ with a tensor index keeps the write graph-safe;
            # a Python slice `cache[..., length:end]` would bake in the offset.
            self.cache_k[i].index_copy_(3, self.write_idx, k_spk)
            self.cache_v[i].index_copy_(3, self.write_idx, v_spk)
            out = attn.attend(q_spk, self.cache_k[i], self.cache_v[i],
                              self.valid_mask, self.use_sdpa)
            h = h + attn.o_proj(out)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        # [:, -1] keeps the return shape [B, vocab], matching prefill().
        return self.model.lm_head(model.norm(h).mean(dim=0))[:, -1]

    def _capture(self):
        # Capture against a scratch state so the real cache is not polluted.
        saved = [t.clone() for t in self.cache_k] + [t.clone() for t in self.cache_v]
        # An all -inf mask would make softmax produce NaN during warmup; mark one
        # key valid so warmup runs on well-formed values.
        self._set_valid(max(self.length, 1))
        with torch.no_grad():
            _warmup(self._decode_body)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_logits = self._decode_body()
        for t, s in zip(self.cache_k + self.cache_v, saved):
            t.copy_(s)
        self._captured = True

    def _set_valid(self, length):
        self.valid_mask[..., :length] = 0.0
        if length < self.max_len:
            self.valid_mask[..., length:] = float("-inf")

    @torch.no_grad()
    def reset(self):
        self.length = 0
        for t in self.cache_k + self.cache_v:
            t.zero_()

    @torch.no_grad()
    def prefill(self, input_ids):
        """Eager prefill (shape varies with the prompt, so it is not captured)."""
        B, S = input_ids.shape
        if S > self.max_len:
            raise ValueError(f"prompt of {S} exceeds max_len={self.max_len}")

        class _View:
            """Adapts the preallocated buffers to the SpikeKVCache interface."""
            def __init__(self, outer):
                self.o = outer
                self.length = outer.length

            def append(self, i, k, v):
                end = self.length + k.shape[3]
                self.o.cache_k[i][:, :, :, self.length:end] = k
                self.o.cache_v[i][:, :, :, self.length:end] = v
                return self.o.cache_k[i][:, :, :, :end], self.o.cache_v[i][:, :, :, :end]

            def advance(self, s):
                self.length += s

        view = _View(self)
        hidden = forward_with_cache(self.model.model, input_ids, view, self.T, self.use_sdpa)
        self.length = view.length
        return self.model.lm_head(hidden[:, -1])

    @torch.no_grad()
    def step(self, input_ids):
        """Decode one token. ``input_ids`` is [B, 1]."""
        if not self._captured:
            self._capture()
        if self.length >= self.max_len:
            raise ValueError(f"KV cache full (max_len={self.max_len})")

        self.static_ids.copy_(input_ids)
        self.static_pos.fill_(self.length)
        self.write_idx.fill_(self.length)
        self._set_valid(self.length + 1)

        self.graph.replay()
        self.length += 1
        return self.static_logits.clone()
