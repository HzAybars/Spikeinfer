"""Engine configuration.

One dataclass per concern, assembled into :class:`EngineConfig`. Every field a
user can set from the CLI or the Python API lands here, and the engine reads
nothing else -- so ``EngineConfig`` is the complete description of a running
server.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import torch

from .kernels.packing import packed_bytes_per_token

DType = Literal["auto", "float32", "float16", "bfloat16"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(dtype: DType, device: str = "cuda") -> torch.dtype:
    """``"auto"`` picks bf16 on hardware that supports it, else fp32.

    fp16 is deliberately not the auto choice: the T residual streams accumulate
    across timesteps and fp16's narrow exponent range costs more than its
    precision buys at these sizes.
    """
    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return torch.float32
    return _DTYPES[dtype]


@dataclass
class ModelConfig:
    """Which model to run and how to load it."""

    model: str
    """Path to a spikeinfer model directory (see ``spikeinfer convert``)."""

    tokenizer: str | None = None
    """Defaults to ``model`` -- the converter copies the tokenizer alongside."""

    dtype: DType = "auto"
    timesteps: int | None = None
    """Override the checkpoint's T. Fewer timesteps is faster and less accurate."""

    max_model_len: int | None = None
    """Defaults to the checkpoint's ``max_position_embeddings``."""

    trust_remote_code: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self.model


@dataclass
class CacheConfig:
    """Paged spike KV cache sizing."""

    block_size: int = 32
    """Tokens per block. Must be a power of two -- it is a Triton tile width."""

    gpu_memory_utilization: float = 0.90
    """Fraction of total GPU memory the engine may occupy, weights included."""

    num_gpu_blocks_override: int | None = None
    """Skip the memory-profiling step and use exactly this many blocks."""

    swap_space_gb: float = 0.0
    """Reserved. CPU swap is not implemented; preemption recomputes instead."""

    def __post_init__(self) -> None:
        if self.block_size & (self.block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {self.block_size}")


@dataclass
class SchedulerConfig:
    """Continuous-batching limits."""

    max_num_seqs: int = 64
    """Sequences that may be in the running batch at once."""

    max_num_batched_tokens: int = 4096
    """Token budget per engine step, shared by prefill chunks and decodes."""

    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    """Split long prompts across steps so decodes are not starved behind them."""

    preemption_mode: Literal["recompute"] = "recompute"

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len and not self.enable_chunked_prefill:
            raise ValueError(
                "max_num_batched_tokens must be >= max_model_len when chunked "
                "prefill is disabled, otherwise a full-length prompt can never "
                "be scheduled"
            )


@dataclass
class GraphConfig:
    """CUDA graph capture for the decode path."""

    enabled: bool = True
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    """Buckets to capture. A decode batch is padded up to the next bucket."""

    max_seq_len: int = 4096
    """Capture-time upper bound on cached length; block tables are sized to it."""

    def buckets_upto(self, max_num_seqs: int) -> tuple[int, ...]:
        picked = tuple(b for b in self.batch_sizes if b <= max_num_seqs)
        return picked or (1,)


@dataclass
class EngineConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    device: str = "cuda"

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.model.dtype, self.device)

    def bytes_per_token(self, num_layers: int, timesteps: int, kv_heads: int, head_dim: int) -> int:
        return packed_bytes_per_token(num_layers, timesteps, kv_heads, head_dim)


def default_device() -> str:
    if os.environ.get("SPIKEINFER_DEVICE"):
        return os.environ["SPIKEINFER_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"
