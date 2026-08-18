"""The engine: requests in, tokens out.

``LLMEngine.step()`` is the whole loop. Each call asks the scheduler for a
batch, runs it, samples one token per sequence that finished its chunk,
detokenizes, applies stop conditions, and returns whatever changed. Nothing
blocks: a request added mid-generation joins the next step, and a request that
finishes leaves without disturbing the others. That is continuous batching.

Everything above this file (the OpenAI server, the CLI, the offline ``LLM``
API) is a client of ``add_request`` / ``step``.
"""
from __future__ import annotations

import math
import time
import uuid
import warnings
from dataclasses import dataclass, field

import torch

from ..config import EngineConfig, ModelConfig, SchedulerConfig
from ..loader import load_model
from ..sampling_params import SamplingParams
from .block_manager import BlockManager
from .detokenizer import Detokenizer, check_stop
from .metrics import Stats
from .model_runner import ModelRunner
from .sampler import Sampler
from .scheduler import Scheduler
from .sequence import (
    CompletionOutput,
    RequestMetrics,
    RequestOutput,
    Sequence,
    SequenceStatus,
)
from .spike_cache import PagedSpikeCache

_SAFETY_MARGIN = 0.95
"""Fraction of the computed free budget actually handed to the cache. Covers
allocator fragmentation and the CUDA graph memory pool, which is reserved after
sizing."""


@dataclass
class _Request:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    params: SamplingParams
    seqs: list[Sequence] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return all(s.status.is_finished for s in self.seqs)


class LLMEngine:
    def __init__(self, config: EngineConfig, tokenizer=None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.torch_dtype
        if config.devices.cpu_threads is not None:
            torch.set_num_threads(config.devices.cpu_threads)

        self.tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(config.model)
        self.plan = _build_plan(config, self.device, self.dtype)
        self.model, self.model_config = load_model(
            config.model.model,
            dtype=self.dtype,
            device=self.device,
            timesteps=config.model.timesteps,
            plan=self.plan,
        )
        self.timesteps = self.model_config.T
        self.head_dim = getattr(
            self.model_config,
            "head_dim",
            self.model_config.hidden_size // self.model_config.num_attention_heads,
        )
        self.eos_token_id = _eos_token_id(self.tokenizer, self.model_config)

        max_len = config.model.max_model_len or self.model_config.max_position_embeddings
        config.scheduler.max_model_len = min(max_len, config.scheduler.max_model_len or max_len)
        self.max_model_len = config.scheduler.max_model_len

        num_blocks = self._size_cache()
        self.cache = PagedSpikeCache(
            num_layers=self.model_config.num_hidden_layers,
            timesteps=self.timesteps,
            num_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.head_dim,
            num_blocks=num_blocks,
            block_size=config.cache.block_size,
            device=self.device,
            layer_devices=_cache_devices(self.plan, self.device),
        )
        self.block_manager = BlockManager(
            self.cache.num_allocatable_blocks, config.cache.block_size
        )
        self.scheduler = Scheduler(config.scheduler, self.block_manager)
        self.runner = ModelRunner(
            self.model, self.cache, self.block_manager, config, self.device, self.dtype
        )
        self.sampler = Sampler(self.device, self.eos_token_id)
        self.detokenizer = Detokenizer(self.tokenizer) if self.tokenizer else None

        self.stats = Stats()
        self.stats.gpu_blocks = self.cache.num_allocatable_blocks
        self._requests: dict[str, _Request] = {}
        self._seq_counter = 0

        self._graph_note = _graph_note(self.plan, config)
        if self._graph_note:
            config.graph.enabled = False
        self.captured_graphs = self.runner.capture_graphs()

    # -- construction helpers --------------------------------------------

    @classmethod
    def from_pretrained(cls, model: str, **kwargs) -> LLMEngine:
        """Build an engine from a model path plus flat keyword overrides."""
        from ..config import CacheConfig, GraphConfig

        model_fields = {f for f in ModelConfig.__dataclass_fields__ if f != "model"}
        sched_fields = set(SchedulerConfig.__dataclass_fields__)
        cache_fields = set(CacheConfig.__dataclass_fields__)
        graph_fields = set(GraphConfig.__dataclass_fields__)

        model_kw, sched_kw, cache_kw, graph_kw = {}, {}, {}, {}
        device = kwargs.pop("device", None)
        for key, value in kwargs.items():
            if key in model_fields:
                model_kw[key] = value
            elif key in sched_fields:
                sched_kw[key] = value
            elif key in cache_fields:
                cache_kw[key] = value
            elif key in graph_fields:
                graph_kw[key] = value
            else:
                raise TypeError(f"unknown engine option {key!r}")

        from ..config import default_device

        config = EngineConfig(
            model=ModelConfig(model=model, **model_kw),
            cache=CacheConfig(**cache_kw),
            scheduler=SchedulerConfig(**sched_kw),
            graph=GraphConfig(**graph_kw),
            device=device or default_device(),
        )
        return cls(config)

    def _size_cache(self) -> int:
        """Decide how many blocks fit, by measuring rather than guessing.

        Runs one worst-case prefill against a throwaway cache, then gives the
        rest of the memory budget to the real one.
        """
        override = self.config.cache.num_gpu_blocks_override
        if override is not None:
            return override
        if self.device.type != "cuda":
            return self._size_host_cache()

        block_size = self.config.cache.block_size
        per_block = PagedSpikeCache.bytes_per_block(
            self.model_config.num_hidden_layers,
            self.timesteps,
            self.model_config.num_key_value_heads,
            self.head_dim,
            block_size,
        )
        probe_tokens = min(
            self.config.scheduler.max_num_batched_tokens, self.config.scheduler.max_model_len
        )
        probe_blocks = math.ceil(probe_tokens / block_size) + 1

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        probe = PagedSpikeCache(
            num_layers=self.model_config.num_hidden_layers,
            timesteps=self.timesteps,
            num_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.head_dim,
            num_blocks=probe_blocks,
            block_size=block_size,
            device=self.device,
            # The probe runs a real forward through the real placement, so its
            # cache has to be laid out like the real one -- a CPU layer handed
            # an all-CUDA probe cache fails on the first index_copy_.
            layer_devices=_cache_devices(self.plan, self.device),
        )
        self._dummy_prefill(probe, probe_tokens)
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated()
        del probe
        torch.cuda.empty_cache()

        free, total = torch.cuda.mem_get_info()
        non_torch = max(0, (total - free) - torch.cuda.memory_reserved())
        budget = total * self.config.cache.gpu_memory_utilization
        working_set = peak - probe_blocks * per_block
        available = (budget - non_torch - working_set) * _SAFETY_MARGIN

        num_blocks = int(available // per_block)

        num_blocks = min(num_blocks, self._block_ceiling(block_size))

        if num_blocks < 2:
            raise RuntimeError(
                "not enough GPU memory for a KV cache after loading weights "
                f"(need > {2 * per_block / 2**20:.1f} MiB, have "
                f"{available / 2**20:.1f} MiB). Lower --max-num-batched-tokens or "
                "raise --gpu-memory-utilization."
            )
        return num_blocks

    def _block_ceiling(self, block_size: int) -> int:
        """Blocks beyond which the scheduler can never fill the cache.

        A packed spike cache is cheap enough that a pure memory budget would
        reserve millions of tokens: the scheduler will not run more than
        ``max_num_seqs`` sequences, none longer than ``max_model_len``.
        Reserving past that is startup cost and nothing else.
        """
        return (
            math.ceil(
                self.config.scheduler.max_num_seqs
                * self.config.scheduler.max_model_len
                / block_size
            )
            + 1
        )

    def _size_host_cache(self) -> int:
        """Blocks for a CPU-resident cache, sized against host RAM.

        There is no equivalent of the CUDA probe here: host allocations are not
        reserved up front and ``max_memory_allocated`` has no host counterpart,
        so the working set cannot be measured the same way. Instead a fraction
        of *available* RAM is taken (available, not total -- the weights are
        already resident by this point and must not be counted twice).
        """
        from ..sysinfo import available_ram_bytes, format_bytes

        devices = self.config.devices
        block_size = self.config.cache.block_size
        per_block = PagedSpikeCache.bytes_per_block(
            self.model_config.num_hidden_layers,
            self.timesteps,
            self.model_config.num_key_value_heads,
            self.head_dim,
            block_size,
        )

        if devices.cpu_kv_gb is not None:
            budget = devices.cpu_kv_gb * 2**30
        else:
            budget = available_ram_bytes() * devices.cpu_memory_utilization

        num_blocks = min(int(budget // per_block), self._block_ceiling(block_size))
        if num_blocks < 2:
            raise RuntimeError(
                "not enough host memory for a KV cache "
                f"(need > {format_bytes(2 * per_block)}, budget is "
                f"{format_bytes(budget)}). Raise --cpu-kv-gb or "
                "--cpu-memory-utilization, or lower --max-model-len."
            )
        return num_blocks

    @torch.no_grad()
    def _dummy_prefill(self, cache: PagedSpikeCache, num_tokens: int) -> None:
        from ..attention import AttentionMetadata

        device = self.device
        ids = torch.zeros(num_tokens, dtype=torch.long, device=device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
        table = torch.arange(
            math.ceil(num_tokens / cache.block_size), dtype=torch.int32, device=device
        )
        meta = AttentionMetadata(
            slot_mapping=slots,
            block_size=cache.block_size,
            num_decode_tokens=0,
            prefill_query_starts=[0],
            prefill_query_lens=[num_tokens],
            prefill_context_lens=[0],
            prefill_block_tables=[table],
        )
        hidden = self.model.forward_paged(ids, positions, cache.layers, meta, T=self.timesteps)
        self.model.compute_logits(
            hidden, torch.tensor([num_tokens - 1], dtype=torch.long, device=device)
        )

    # -- request lifecycle ------------------------------------------------

    def add_request(
        self,
        prompt: str | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> str:
        params = sampling_params or SamplingParams()
        request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"

        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError("pass either prompt or prompt_token_ids")
            if self.tokenizer is None:
                raise RuntimeError("no tokenizer loaded; pass prompt_token_ids instead")
            prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("empty prompt")
        if len(prompt_token_ids) >= self.max_model_len:
            raise ValueError(
                f"prompt is {len(prompt_token_ids)} tokens, model limit is {self.max_model_len}"
            )

        request = _Request(request_id, prompt, list(prompt_token_ids), params)
        for i in range(params.n):
            self._seq_counter += 1
            seq = Sequence(
                seq_id=f"{request_id}-{i}",
                request_id=request_id,
                prompt_token_ids=list(prompt_token_ids),
                sampling_params=params.clone() if params.n > 1 else params,
            )
            if params.n > 1 and params.seed is not None:
                # Sibling completions must not be identical.
                seq.sampling_params.seed = params.seed + i
            if self.detokenizer:
                self.detokenizer.prime(seq)
            request.seqs.append(seq)
            self.scheduler.add(seq)

        self._requests[request_id] = request
        return request_id

    def abort_request(self, request_id: str) -> None:
        for seq in self.scheduler.abort(request_id):
            self.sampler.free(seq.seq_id)
        request = self._requests.pop(request_id, None)
        if request:
            for seq in request.seqs:
                if not seq.status.is_finished:
                    seq.finish(SequenceStatus.FINISHED_ABORTED)

    def has_unfinished_requests(self) -> bool:
        return bool(self._requests) and self.scheduler.has_work

    # -- the step ---------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        started = time.monotonic()
        sched = self.scheduler.schedule()
        if sched.is_empty:
            return self._collect({s.request_id for s in sched.ignored}) if sched.ignored else []

        logits, seqs = self.runner.execute(sched)

        for item in sched.scheduled:
            item.seq.num_computed_tokens += item.num_tokens

        touched: set[str] = {item.seq.request_id for item in sched.scheduled}
        touched.update(seq.request_id for seq in sched.ignored)
        generated = 0
        if seqs:
            sampled = self.sampler.sample(logits, seqs)
            for seq, token_id, logprobs in zip(seqs, sampled.token_ids, sampled.logprobs):
                self._accept_token(seq, token_id, logprobs)
                generated += 1

        prompted = sched.num_batched_tokens - generated
        self.stats.record_step(time.monotonic() - started, generated, max(0, prompted))
        self.stats.preemptions += len(sched.preempted)

        outputs = self._collect(touched)
        for seq in self.scheduler.free_finished():
            self.sampler.free(seq.seq_id)
        self._update_gauges()
        return outputs

    def _accept_token(self, seq: Sequence, token_id: int, logprobs) -> None:
        params = seq.sampling_params
        was_first = seq.first_token_time is None
        seq.append_token(token_id, logprobs)
        if was_first and seq.first_token_time is not None:
            self.stats.record_ttft(seq.first_token_time - seq.arrival_time)

        is_stop_token = token_id in params.stop_token_ids or (
            not params.ignore_eos and token_id == self.eos_token_id
        )
        if self.detokenizer and not is_stop_token:
            self.detokenizer.step(seq, token_id)
        check_stop(seq, self.eos_token_id, self.max_model_len)

    def _collect(self, request_ids: set[str]) -> list[RequestOutput]:
        outputs = []
        for request_id in request_ids:
            request = self._requests.get(request_id)
            if request is None:
                continue
            completions = []
            for index, seq in enumerate(request.seqs):
                delta = seq.output_text[seq.streamed_offset :]
                seq.streamed_offset = len(seq.output_text)
                completions.append(
                    CompletionOutput(
                        index=index,
                        text=seq.output_text,
                        token_ids=list(seq.output_token_ids),
                        cumulative_logprob=seq.cumulative_logprob,
                        logprobs=seq.logprobs if seq.sampling_params.logprobs else None,
                        finish_reason=seq.status.finish_reason,
                        stop_reason=seq.stop_reason,
                        delta=delta,
                        delta_token_ids=seq.output_token_ids[-1:] if delta else [],
                    )
                )
            finished = request.finished
            if finished:
                self._requests.pop(request_id, None)
                self.stats.finished_requests += 1
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    prompt=request.prompt,
                    prompt_token_ids=request.prompt_token_ids,
                    outputs=completions,
                    finished=finished,
                    metrics=RequestMetrics(
                        arrival_time=request.seqs[0].arrival_time,
                        first_token_time=request.seqs[0].first_token_time,
                        finish_time=request.seqs[0].finish_time,
                        prompt_tokens=len(request.prompt_token_ids),
                        generated_tokens=sum(s.output_len for s in request.seqs),
                    ),
                )
            )
        return outputs

    def _update_gauges(self) -> None:
        self.stats.num_running = len(self.scheduler.running)
        self.stats.num_waiting = len(self.scheduler.waiting)
        self.stats.cache_usage = self.block_manager.utilization

    # -- introspection ----------------------------------------------------

    def describe(self) -> dict:
        return {
            "model": self.config.model.model,
            "dtype": str(self.dtype).replace("torch.", ""),
            "timesteps": self.timesteps,
            "max_model_len": self.max_model_len,
            "block_size": self.cache.block_size,
            "gpu_blocks": self.cache.num_allocatable_blocks,
            "kv_cache_mib": round(self.cache.nbytes / 2**20, 1),
            "kv_capacity_tokens": self.cache.capacity_tokens,
            "captured_graph_batch_sizes": self.captured_graphs,
            "device": str(self.device),
            "kv_cache_device": self.cache.device_summary,
            "cuda_graphs": self._graph_status(),
            "placement": self.plan.summary() if self.plan is not None else None,
        }

    def _graph_status(self) -> str:
        """Why decode is or is not replaying a captured graph.

        Worth reporting rather than inferring: an empty
        ``captured_graph_batch_sizes`` looks the same whether the user disabled
        capture, is on a CPU, or capture failed and fell back.
        """
        if self.captured_graphs:
            return "enabled"
        if self._graph_note:
            return self._graph_note
        if not self.config.graph.enabled:
            return "disabled (--no-cuda-graph)"
        if self.device.type != "cuda":
            return f"unavailable (device={self.device.type})"
        return "not captured"  # pragma: no cover - capture either works or raises



def _build_plan(config: EngineConfig, device: torch.device, dtype: torch.dtype):
    """The plan for this engine, or ``None`` when nothing was asked for.

    Returning ``None`` rather than a trivial plan matters: it is what keeps the
    ordinary single-device path free of the executor, the ring and the
    per-device metadata mirrors entirely.
    """
    from ..loader import load_config
    from ..placement import plan_from_config

    devices = config.devices
    if (
        devices.gpu_layers is None
        and devices.offload_layers is None
        and not devices.offload_embeddings
        and not devices.adaptive_mlp
    ):
        return None

    model_config = load_config(config.model.model)
    if config.model.timesteps is not None:
        model_config.T = config.model.timesteps

    free_vram = 0
    if device.type == "cuda":
        free_vram = int(torch.cuda.mem_get_info()[0] * config.cache.gpu_memory_utilization)

    spike_stats = None
    if devices.adaptive_mlp:
        from pathlib import Path

        from .. import spike_stats as spike_stats_mod

        candidate = Path(config.model.model) / "spike_stats.json"
        if candidate.exists():
            spike_stats = spike_stats_mod.load(candidate)
        else:
            warnings.warn(
                f"--adaptive-mlp with no {candidate}: the hot/cold split will use a "
                "flat 20% of channels per layer instead of measured firing rates. "
                "Run `spikeinfer spike-stats <model> --write` for a fitted split.",
                RuntimeWarning,
                stacklevel=2,
            )

    plan = plan_from_config(
        model_config, devices, dtype, free_vram, spike_stats=spike_stats
    )
    if devices.adaptive_mlp and device.type == "cuda":
        warnings.warn(
            "--adaptive-mlp on CUDA trades CUDA graph capture for a per-layer "
            "host round trip. Measured on this project's reference setup it is "
            "slower than --offload-layers at more VRAM (28.1 vs 34.9 tok/s, "
            "833 vs 389 MiB), because the round trip costs far more than the "
            "transfer it avoids. It pays on CPU (+30%), on slow links, and on "
            "platforms with cheap synchronisation. See spikeinfer/offload.py.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None if plan.is_trivial else plan


def _cache_devices(plan, device: torch.device):
    """Where each layer's KV slabs go. ``None`` means all on ``device``."""
    if plan is None:
        return None
    from ..placement import Placement

    return [torch.device("cpu") if p is Placement.CPU else device for p in plan.layers]


def _graph_note(plan, config: EngineConfig) -> str | None:
    """Why capture is off under this plan, or ``None`` if it can stay on.

    Capture bakes in a fixed sequence of kernel launches. A hybrid split leaves
    that stream mid-way to run on the CPU, and the adaptive MLP path decides how
    many channels to fetch from a device-to-host read of the gate mask -- neither
    is capturable at all.

    Dense weight streaming *is* capturable: a copy from pinned host memory is a
    legal graph node, the host pointers are stable for the process's life, and
    the per-layer sequence is identical every step. It needs the copy stream
    forked and joined inside the capture region, which
    ``StreamingLayerRing.begin_pass`` arranges. Capture is attempted and falls
    back to eager if the driver disagrees -- see ``ModelRunner.capture_graphs``.
    """
    if plan is None or not config.graph.enabled:
        return None
    from ..placement import Placement

    if any(p is Placement.CPU for p in plan.layers):
        return "unavailable (hybrid cpu/gpu split cannot be captured as one graph)"
    if any(p is Placement.ADAPTIVE for p in plan.layers):
        return "unavailable (--adaptive-mlp reads the gate mask on the host each layer)"
    if Placement.CPU in (plan.embeddings, plan.head):
        # The embedding lookup would run on the CPU inside the captured region.
        # Capture does not merely skip it -- it poisons the stream, and the
        # failure surfaces later as an unrelated CUDA error, so refuse up front.
        return "unavailable (--offload-embeddings puts the embedding lookup on the host)"
    return None


def _load_tokenizer(model_config: ModelConfig):
    """Load the tokenizer, or warn and continue with token-id-only serving."""
    from pathlib import Path

    source = Path(str(model_config.tokenizer))
    if source.is_dir() and not any(
        (source / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    ):
        # A weights-only directory (tests, ad-hoc exports). Nothing to load and
        # nothing to warn about -- token-id serving is the intended mode.
        return None

    try:
        from transformers import AutoTokenizer

        from ..hf_compat import register_auto_classes

        register_auto_classes()
        return AutoTokenizer.from_pretrained(
            model_config.tokenizer, trust_remote_code=model_config.trust_remote_code
        )
    except Exception as exc:  # pragma: no cover - depends on what is on disk
        warnings.warn(
            f"could not load a tokenizer from {model_config.tokenizer!r}: "
            f"{type(exc).__name__}: {exc}. The engine will only accept "
            "prompt_token_ids, and text endpoints will fail.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _eos_token_id(tokenizer, config) -> int | None:
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    eos = getattr(config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return eos[0] if eos else None
    return eos
