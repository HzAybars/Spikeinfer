"""spikeinfer -- an inference engine for spiking (LIF) transformers.

Serves Qwen2-architecture models converted to spiking form, with the ergonomics
of a dense engine: an OpenAI-compatible server, continuous batching, a paged KV
cache, and CUDA graphs.

What makes the internals different from a dense engine is that after the LIF
neurons, ``q``, ``k`` and ``v`` are *binary*. That is exploited twice:

* the KV cache stores one bit per spike -- 32x smaller than fp32, and 4x smaller
  than the fp16 KV cache of the dense Qwen2.5-0.5B this model is converted from,
  despite carrying T timesteps;
* attention scores are ``popcount(q & k)``: exact integers computed with bit
  operations instead of a matmul (``kernels/spike_attention.py``).

The original single-stream optimizations are still here and still the reason
decode is fast at all -- a fused multi-timestep LIF Triton kernel, a layer-major
T-batched model, and CUDA graph capture, which together take a T=4 decode step
from ~28k CUDA launches to a single graph launch.

Serving::

    spikeinfer serve ./qwen-spiking --port 8000

Offline::

    from spikeinfer import LLM, SamplingParams
    llm = LLM("./qwen-spiking")
    print(llm.generate("Hello", SamplingParams(max_tokens=32))[0].text)

``spikeinfer.reference`` holds the unoptimized snntorch implementation the
engine is validated against. Parameter names match exactly, so
``engine.load_state_dict(reference.state_dict())`` needs no translation.

Inference only -- no backward pass. Fine-tuning stays on the reference path.
"""
__version__ = "0.2.0"

from .attention import AttentionMetadata, paged_attention
from .config import CacheConfig, EngineConfig, GraphConfig, ModelConfig, SchedulerConfig
from .engine import AsyncLLMEngine, LLMEngine, PagedSpikeCache, RequestOutput
from .graph_runner import GraphedDecoder, GraphedForward
from .kernels import (
    lif_multistep,
    lif_multistep_ref,
    pack_spikes,
    paged_spike_attn_decode,
    unpack_spikes,
    validate_thresholds,
)
from .kv_cache import SpikeKVCache, forward_with_cache, generate
from .llm import LLM
from .loader import load_model, save_model
from .modeling_fast import (
    FastSpikingQwenForCausalLM,
    FastSpikingQwenModel,
)
from .sampling_params import SamplingParams

__all__ = [
    "__version__",
    # serving
    "LLM",
    "LLMEngine",
    "AsyncLLMEngine",
    "SamplingParams",
    "RequestOutput",
    "EngineConfig",
    "ModelConfig",
    "CacheConfig",
    "SchedulerConfig",
    "GraphConfig",
    # model
    "FastSpikingQwenForCausalLM",
    "FastSpikingQwenModel",
    "load_model",
    "save_model",
    # cache and kernels
    "PagedSpikeCache",
    "AttentionMetadata",
    "paged_attention",
    "paged_spike_attn_decode",
    "pack_spikes",
    "unpack_spikes",
    "lif_multistep",
    "lif_multistep_ref",
    "validate_thresholds",
    # single-stream path, unchanged
    "SpikeKVCache",
    "forward_with_cache",
    "generate",
    "GraphedDecoder",
    "GraphedForward",
]
