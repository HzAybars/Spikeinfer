"""SpikingQwen: Qwen2.5-0.5B decoder-only LM with LIF spiking neurons
inserted at attention Q/K/V and the SwiGLU MLP gate. See DESIGN.md.

Linear layers, RMSNorm and RoPE are reused/mirrored from the original
Qwen2 architecture (transformers==4.57.2) so pretrained weights transplant
1:1. Only the pointwise nonlinearities (SiLU gate, and the identity/softmax
input path for Q/K/V) are replaced by LIF neurons unrolled over T
timesteps with a constant (direct-coded) input at every timestep.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .lif import LIFNeuron


class SpikingQwenConfig(Qwen2Config):
    model_type = "spiking_qwen2"

    def __init__(self, T: int = 4, beta_init: float = 0.9, threshold_init: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.T = T
        self.beta_init = beta_init
        self.threshold_init = threshold_init


class SpikingQwenAttention(nn.Module):
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

    def init_state(self, batch_size: int, seq_len: int, device, dtype):
        q_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
        kv_shape = (batch_size, self.num_kv_heads, seq_len, self.head_dim)
        return {
            "mem_q": torch.zeros(q_shape, device=device, dtype=dtype),
            "mem_k": torch.zeros(kv_shape, device=device, dtype=dtype),
            "mem_v": torch.zeros(kv_shape, device=device, dtype=dtype),
        }

    def forward(self, x, state, cos, sin, causal_mask):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE applied to continuous pre-spike currents (a rotation of a
        # binary vector is not binary, so it must happen before LIF).
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q_spk, state["mem_q"] = self.lif_q(q, state["mem_q"])
        k_spk, state["mem_k"] = self.lif_k(k, state["mem_k"])
        v_spk, state["mem_v"] = self.lif_v(v, state["mem_v"])

        k_spk = repeat_kv(k_spk, self.n_rep)
        v_spk = repeat_kv(v_spk, self.n_rep)

        scores = torch.matmul(q_spk, k_spk.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + causal_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_spk)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), state


class SpikingQwenMLP(nn.Module):
    def __init__(self, config: SpikingQwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.lif_gate = LIFNeuron(config.intermediate_size, config.beta_init, config.threshold_init)

    def init_state(self, batch_size: int, seq_len: int, intermediate_size: int, device, dtype):
        return {"mem_gate": torch.zeros(batch_size, seq_len, intermediate_size, device=device, dtype=dtype)}

    def forward(self, x, state):
        gate_lin = self.gate_proj(x)
        gate_spk, state["mem_gate"] = self.lif_gate(gate_lin, state["mem_gate"])
        h = gate_spk * self.up_proj(x)
        return self.down_proj(h), state


class SpikingQwenDecoderLayer(nn.Module):
    def __init__(self, config: SpikingQwenConfig, layer_idx: int):
        super().__init__()
        self.self_attn = SpikingQwenAttention(config, layer_idx)
        self.mlp = SpikingQwenMLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def init_state(self, batch_size, seq_len, intermediate_size, device, dtype):
        return {
            "attn": self.self_attn.init_state(batch_size, seq_len, device, dtype),
            "mlp": self.mlp.init_state(batch_size, seq_len, intermediate_size, device, dtype),
        }

    def forward(self, x, state, cos, sin, causal_mask):
        residual = x
        attn_out, state["attn"] = self.self_attn(self.input_layernorm(x), state["attn"], cos, sin, causal_mask)
        x = residual + attn_out

        residual = x
        mlp_out, state["mlp"] = self.mlp(self.post_attention_layernorm(x), state["mlp"])
        x = residual + mlp_out
        return x, state


class SpikingQwenModel(nn.Module):
    def __init__(self, config: SpikingQwenConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [SpikingQwenDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config)

    def forward(self, input_ids, attention_mask=None, T: int | None = None, return_all_timesteps=False):
        T = T or self.config.T
        bsz, seq_len = input_ids.shape
        device, dtype = input_ids.device, self.embed_tokens.weight.dtype

        embeds = self.embed_tokens(input_ids)  # [B, L, D], constant input replayed every timestep

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary_emb(embeds, position_ids)

        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * float("-inf")
            pad_mask = torch.nan_to_num(pad_mask, nan=0.0)
            causal_mask = causal_mask[None, None, :, :] + pad_mask
        else:
            causal_mask = causal_mask[None, None, :, :]

        states = [
            layer.init_state(bsz, seq_len, self.config.intermediate_size, device, dtype)
            for layer in self.layers
        ]

        hidden_per_t = []
        for _ in range(T):
            h = embeds
            for i, layer in enumerate(self.layers):
                h, states[i] = layer(h, states[i], cos, sin, causal_mask)
            h = self.norm(h)
            hidden_per_t.append(h)

        if return_all_timesteps:
            return torch.stack(hidden_per_t, dim=0)  # [T, B, L, D]
        return torch.stack(hidden_per_t, dim=0).mean(dim=0)  # [B, L, D] rate-coded readout


class SpikingQwenForCausalLM(nn.Module):
    def __init__(self, config: SpikingQwenConfig):
        super().__init__()
        self.config = config
        self.model = SpikingQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, labels=None, T: int | None = None):
        hidden = self.model(input_ids, attention_mask=attention_mask, T=T)
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
