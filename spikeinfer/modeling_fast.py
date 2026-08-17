"""Layer-major, T-batched SpikingQwen for inference.

The reference model in ``spikeinfer.reference.modeling_spiking_qwen`` runs

    for t in range(T):
        for layer in layers:  ...

which replays the entire 24-layer stack T times. Every ``nn.Linear`` therefore
runs T separate times on a [B, S, D] input -- at decode that is T sequential
GEMVs, each re-reading the full weight matrix from HBM.

This module swaps the loops:

    for layer in layers:
        for t in range(T):  ...

That reordering is *bit-identical* (verified with ``torch.equal`` on the real
model): ``state[i]`` is only ever touched by layer ``i``, the per-timestep order
of those state updates is preserved, and the T residual streams are mutually
independent. What it buys is that the residual stream can be carried as a single
``[T, B, S, D]`` tensor, so ``nn.Linear`` folds T into the GEMM's M dimension for
free -- one GEMM instead of T. Measured on the 896->4864 projection: 57.8 us of
T-looped GEMVs collapse to 13.6 us as one batched GEMM (4.25x).

The LIF sites then become a single fused kernel call each (see
``kernels/lif_triton.py``), since the whole T recurrence is now visible at once.

Parameter names mirror the reference model exactly, so::

    fast.load_state_dict(reference.state_dict())

works with no key translation. The snntorch ``LIFNeuron`` modules are kept purely
as parameter holders -- their ``forward`` is never called; only ``.lif.beta`` and
``.lif.threshold`` are read.

Numerical fidelity
------------------
This model is **not** bit-exact against the reference, and cannot be made so
while keeping the T-batching. Typical relative logit error is ~1.2e-7, argmax
agreement was 100% across every shape tested, and WikiText-2 perplexity moves
390.2473 -> 390.4453 (+0.05%).

Rarely a spike whose membrane lands exactly on its threshold flips, which is a
discrete crossing rather than float noise and shows up as a localized jump (one
observed case hit 6.3e-4 relative). This is intrinsic to a spiking model, not to
this implementation: injecting 5e-7 of relative noise into a LIF input flips
about 1 spike in 2.5 million. Tests should therefore assert argmax agreement and
a loose relative bound, never a tight per-element one.

The divergence is entirely in *how* the same arithmetic is scheduled, not in the
arithmetic itself. Carrying a leading T dimension changes tensor shapes, and both
cuBLAS and PyTorch's reduction kernels select different algorithms (hence
different summation orders) for different shapes. Three separate places do this:
``nn.Linear`` (M becomes T*B*S), the batched attention matmul (batch becomes
T*B), and ``Qwen2RMSNorm``'s ``mean(-1)``. The effect is shape-dependent and
easy to miss -- it vanishes at S=16 and S=32 but appears at S=4..12 and S=15.

What *is* verified bit-exact, and is where correctness actually rests:
  * the fused LIF kernel vs snntorch (``kernels/lif_triton.py``, incl. a
    boundary-stress case where the membrane sits on the threshold);
  * the loop reordering itself, checked in pure PyTorch on the reference model
    before any kernel was involved;
  * RoPE-by-broadcast vs ``apply_rotary_pos_emb``;
  * the spike-vs-spike QK^T matmul (both operands binary, contraction 64, so the
    products and their sum are exact integers in fp32 regardless of order).

Inference only: no backward. Fine-tuning stays on the reference path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    repeat_kv,
    rotate_half,
)

from .attention import AttentionMetadata, paged_attention
from .kernels import lif_multistep
from .reference.lif import LIFNeuron
from .reference.modeling_spiking_qwen import SpikingQwenConfig


def _apply_rope_t(q, k, cos, sin):
    """RoPE over a [T, B, H, S, D] tensor.

    Identical arithmetic to ``transformers.apply_rotary_pos_emb`` -- ``(x * cos)
    + (rotate_half(x) * sin)`` -- with cos/sin broadcast over the leading T and
    head axes instead of being materialised. Broadcasting keeps it bit-exact
    against the reference while avoiding a T-fold copy.
    """
    cos = cos[None, :, None, :, :]  # [1, B, 1, S, D]
    sin = sin[None, :, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class FastSpikingQwenAttention(nn.Module):
    def __init__(self, config: SpikingQwenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.lif_q = LIFNeuron(self.head_dim, config.beta_init, config.threshold_init)
        self.lif_k = LIFNeuron(self.head_dim, config.beta_init, config.threshold_init)
        self.lif_v = LIFNeuron(self.head_dim, config.beta_init, config.threshold_init)

    def spike_qkv(self, x, cos, sin):
        """[T,B,S,D] -> q/k/v spikes, each [T,B,H,S,Dh]. Shared by prefill and decode."""
        T, B, S, _ = x.shape
        # One GEMM per projection with M = T*B*S (this is the T-batching win).
        q = self.q_proj(x).view(T, B, S, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).view(T, B, S, self.num_kv_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x).view(T, B, S, self.num_kv_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # RoPE must precede the neuron: rotating a binary vector is not binary.
        q, k = _apply_rope_t(q, k, cos, sin)

        q_spk = lif_multistep(q, self.lif_q.lif.beta, self.lif_q.lif.threshold)
        k_spk = lif_multistep(k, self.lif_k.lif.beta, self.lif_k.lif.threshold)
        v_spk = lif_multistep(v, self.lif_v.lif.beta, self.lif_v.lif.threshold)
        return q_spk, k_spk, v_spk

    def attend(self, q_spk, k_spk, v_spk, causal_mask, use_sdpa):
        """q [T,B,Hq,Sq,Dh] against k/v [T,B,Hkv,Sk,Dh] -> [T,B,Sq,Hq*Dh]."""
        T, B, Hq, Sq, Dh = q_spk.shape

        q = q_spk.reshape(T * B, Hq, Sq, Dh)
        k = k_spk.reshape(T * B, self.num_kv_heads, -1, Dh)
        v = v_spk.reshape(T * B, self.num_kv_heads, -1, Dh)

        if use_sdpa:
            # enable_gqa avoids materialising repeat_kv's n_rep-fold copy.
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                is_causal=causal_mask is None and Sq > 1,
                enable_gqa=True,
            )
        else:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if causal_mask is not None:
                scores = scores + causal_mask
            out = torch.matmul(F.softmax(scores, dim=-1), v)

        return out.transpose(1, 2).reshape(T, B, Sq, Hq * Dh)

    def forward(self, x, cos, sin, causal_mask, use_sdpa=False):
        q_spk, k_spk, v_spk = self.spike_qkv(x, cos, sin)
        return self.o_proj(self.attend(q_spk, k_spk, v_spk, causal_mask, use_sdpa))

    def forward_paged(self, x, cos, sin, k_cache, v_cache, meta: AttentionMetadata):
        """Server path: attend against the paged spike cache instead of a
        dense [T,B,H,S,D] tensor. ``x`` is [T, 1, n_tok, D]."""
        q_spk, k_spk, v_spk = self.spike_qkv(x, cos, sin)
        ctx = paged_attention(
            q_spk, k_spk, v_spk, k_cache, v_cache, meta,
            self.num_heads, self.num_kv_heads, self.head_dim,
        )
        return self.o_proj(ctx)


class FastSpikingQwenMLP(nn.Module):
    def __init__(self, config: SpikingQwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.lif_gate = LIFNeuron(config.intermediate_size, config.beta_init, config.threshold_init)

    def forward(self, x):
        gate_spk = lif_multistep(
            self.gate_proj(x), self.lif_gate.lif.beta, self.lif_gate.lif.threshold
        )
        return self.down_proj(gate_spk * self.up_proj(x))


class FastSpikingQwenDecoderLayer(nn.Module):
    def __init__(self, config: SpikingQwenConfig, layer_idx: int):
        super().__init__()
        self.self_attn = FastSpikingQwenAttention(config, layer_idx)
        self.mlp = FastSpikingQwenMLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, cos, sin, causal_mask, use_sdpa=False):
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, causal_mask, use_sdpa)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h

    def forward_paged(self, h, cos, sin, k_cache, v_cache, meta: AttentionMetadata):
        h = h + self.self_attn.forward_paged(
            self.input_layernorm(h), cos, sin, k_cache, v_cache, meta
        )
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class FastSpikingQwenModel(nn.Module):
    def __init__(self, config: SpikingQwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [FastSpikingQwenDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        self._mask_cache = {}

    def _causal_mask(self, seq_len, attention_mask, device, dtype):
        key = (seq_len, device, dtype, None if attention_mask is None else id(attention_mask))
        if attention_mask is None:
            cached = self._mask_cache.get(key)
            if cached is not None:
                return cached
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * float("-inf")
            pad = torch.nan_to_num(pad, nan=0.0)
            return mask[None, None, :, :] + pad
        mask = mask[None, None, :, :]
        self._mask_cache[key] = mask
        return mask

    def forward(self, input_ids, attention_mask=None, T=None, return_all_timesteps=False,
                use_sdpa=False):
        T = T or self.config.T
        B, S = input_ids.shape
        device, dtype = input_ids.device, self.embed_tokens.weight.dtype

        embeds = self.embed_tokens(input_ids)
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(embeds, position_ids)

        causal_mask = None if use_sdpa and attention_mask is None else \
            self._causal_mask(S, attention_mask, device, dtype)
        if causal_mask is not None and causal_mask.shape[0] != 1:
            # Padding-dependent mask is per-batch [B,1,S,S], but attention runs on
            # the flattened [T*B, ...] view whose index is t*B + b -- tile to match.
            causal_mask = causal_mask.repeat(T, 1, 1, 1)

        # T independent residual streams, carried as one tensor so every Linear
        # below sees M = T*B*S.
        h = embeds.unsqueeze(0).expand(T, B, S, -1).contiguous()

        for layer in self.layers:
            h = layer(h, cos, sin, causal_mask, use_sdpa)

        h = self.norm(h)
        if return_all_timesteps:
            return h
        return h.mean(dim=0)  # rate-coded readout

    def forward_paged(self, input_ids, positions, kv_caches, meta: AttentionMetadata, T=None):
        """Flattened, paged forward -- the shape the server runs.

        All scheduled sequences are concatenated into one token axis, so every
        ``nn.Linear`` still sees a single GEMM with M = T * n_tok and the
        T-batching win survives continuous batching. Sequence boundaries exist
        only inside attention, carried by ``meta``.

        Args:
            input_ids: ``[n_tok]`` token ids, decode tokens first.
            positions: ``[n_tok]`` absolute position of each token in its own
                sequence (this is what makes RoPE correct for a ragged batch).
            kv_caches: per layer ``(k_cache, v_cache)``.

        Returns:
            ``[n_tok, D]`` rate-coded hidden states.
        """
        T = T or self.config.T
        n_tok = input_ids.shape[0]

        embeds = self.embed_tokens(input_ids.unsqueeze(0))  # [1, n_tok, D]
        cos, sin = self.rotary_emb(embeds, positions.unsqueeze(0))
        h = embeds.unsqueeze(0).expand(T, 1, n_tok, -1).contiguous()

        for i, layer in enumerate(self.layers):
            h = layer.forward_paged(h, cos, sin, kv_caches[i][0], kv_caches[i][1], meta)

        return self.norm(h).mean(dim=0).squeeze(0)


class FastSpikingQwenForCausalLM(nn.Module):
    def __init__(self, config: SpikingQwenConfig):
        super().__init__()
        self.config = config
        self.model = FastSpikingQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward_paged(self, input_ids, positions, kv_caches, meta, T=None):
        """Hidden states only -- see :meth:`compute_logits` for the reason."""
        return self.model.forward_paged(input_ids, positions, kv_caches, meta, T=T)

    def compute_logits(self, hidden, indices=None):
        """Project only the rows that will actually be sampled.

        A 4096-token prefill produces 4096 hidden states but samples one. At
        vocab 151936 the unprojected rows would cost 2.4 GB of fp32 logits, so
        the row selection happens *before* ``lm_head``, not after.
        """
        if indices is not None:
            hidden = hidden.index_select(0, indices)
        return self.lm_head(hidden)

    def forward(self, input_ids, attention_mask=None, labels=None, T=None, use_sdpa=False):
        hidden = self.model(input_ids, attention_mask=attention_mask, T=T, use_sdpa=use_sdpa)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}
