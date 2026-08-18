"""Where each layer's weights live, and where each layer computes.

The engine used to assume one thing about placement: that the whole model fits
on one CUDA device. Three of the modes this package now supports break that
assumption in different directions, and they are all the same question asked
per layer --

``GPU``
    weights resident in VRAM, compute on the GPU. What the engine has always
    done, and still the default.
``STREAM``
    weights parked in host RAM, copied into a small ring of GPU buffers as the
    forward pass reaches them. Trades PCIe bandwidth for VRAM: the resident
    footprint stops scaling with depth.
``ADAPTIVE``
    the layer's dense half stays resident, but ``up_proj``/``down_proj`` are
    split: the channels that usually fire stay in VRAM and the tail lives in
    host RAM, fetched only when it fires. The MLP's input *is* sparse -- gate
    spikes are binary, so a silent channel's row and column do no work -- and on
    the reference checkpoint only 5.5% of channels fire per token. A different
    trade from ``STREAM``: less VRAM saved, but almost no per-step transfer.
    See :mod:`spikeinfer.spike_stats`.
``CPU``
    weights resident in host RAM, compute on the CPU. Layers split between
    devices, llama.cpp's ``-ngl``.

-- so they are one enum and one plan, not four code paths bolted onto the
model. :func:`apply_plan` is the only thing that touches the model, and what it
returns is an *executor*: an object owning the layer loop, which
``FastSpikingQwenModel.forward_paged`` defers to when one is installed. That
keeps ``modeling_fast`` about the arithmetic and keeps placement here.

Sizing (:func:`auto_plan`, :func:`describe_plan`) is deliberately separable from
execution: ``spikeinfer plan`` reports what a machine could run *without loading
any weights*, which is the only way to answer "will this fit" before spending a
minute on I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from .kernels.packing import packed_bytes_per_token


class Placement(str, Enum):
    GPU = "gpu"
    STREAM = "stream"
    ADAPTIVE = "adaptive"
    CPU = "cpu"

    def __str__(self) -> str:  # so f-strings and JSON read as "gpu", not "Placement.GPU"
        return self.value

    @property
    def computes_on_gpu(self) -> bool:
        return self is not Placement.CPU

    @property
    def weights_in_host_ram(self) -> bool:
        return self in (Placement.STREAM, Placement.ADAPTIVE, Placement.CPU)


@dataclass
class DevicePlan:
    """A complete placement decision. Everything the loader needs to act on."""

    layers: list[Placement]
    embeddings: Placement = Placement.GPU
    head: Placement = Placement.GPU
    stream_buffers: int = 2
    pin_memory: bool = True
    hot_channels: list[int] = field(default_factory=list)
    """Per layer, how many intermediate channels stay resident under ADAPTIVE.
    Empty when no layer is adaptive."""

    def count(self, placement: Placement) -> int:
        return sum(1 for p in self.layers if p is placement)

    @property
    def is_trivial(self) -> bool:
        """True when this plan is what the engine would do with no plan at all.

        Worth asking explicitly: the executor, the ring buffers and the
        host-side copies are all pure overhead in the ordinary case, so the
        model keeps its original loop unless a plan actually asks for something.
        """
        return (
            all(p is Placement.GPU for p in self.layers)
            and self.embeddings is Placement.GPU
            and self.head is Placement.GPU
        )

    @property
    def gpu_layer_count(self) -> int:
        return sum(1 for p in self.layers if p.computes_on_gpu)

    def summary(self) -> dict:
        out = {
            "layers": len(self.layers),
            "resident": self.count(Placement.GPU),
            "streamed": self.count(Placement.STREAM),
            "adaptive": self.count(Placement.ADAPTIVE),
            "cpu": self.count(Placement.CPU),
            "embeddings": str(self.embeddings),
            "head": str(self.head),
        }
        adaptive = [n for n in self.hot_channels if n]
        if adaptive:
            out["hot_channels_mean"] = round(sum(adaptive) / len(adaptive), 1)
        return out


# -- weight accounting, without loading weights ---------------------------


def _head_dim(config) -> int:
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


@dataclass
class WeightSizes:
    """Parameter counts derived from the config alone.

    Mirrors :class:`spikeinfer.modeling_fast.FastSpikingQwenDecoderLayer`'s
    constructor exactly. If that gains a parameter this must too, which is why
    ``tests/test_placement.py`` checks these numbers against a real model.
    """

    attention: int
    gate: int
    up: int
    down: int
    norms: int
    lif: int
    embedding: int
    tied_head: bool

    @property
    def mlp(self) -> int:
        return self.gate + self.up + self.down

    @property
    def per_layer(self) -> int:
        return self.attention + self.mlp + self.norms + self.lif

    @property
    def attention_share(self) -> float:
        return self.attention / self.per_layer

    def layer_bytes(self, dtype: torch.dtype) -> int:
        return self.per_layer * dtype.itemsize

    def embedding_bytes(self, dtype: torch.dtype) -> int:
        return self.embedding * (1 if self.tied_head else 2) * dtype.itemsize


def weight_sizes(config) -> WeightSizes:
    hidden = config.hidden_size
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = _head_dim(config)
    inter = config.intermediate_size

    q = hidden * heads * head_dim + heads * head_dim  # bias=True
    k = hidden * kv_heads * head_dim + kv_heads * head_dim
    v = hidden * kv_heads * head_dim + kv_heads * head_dim
    o = heads * head_dim * hidden  # bias=False

    return WeightSizes(
        attention=q + k + v + o,
        gate=hidden * inter,
        up=hidden * inter,
        down=inter * hidden,
        norms=2 * hidden,
        # beta + threshold at each of q/k/v (head_dim wide) and the gate.
        lif=2 * (3 * head_dim + inter),
        embedding=config.vocab_size * hidden,
        tied_head=bool(getattr(config, "tie_word_embeddings", False)),
    )


def transfer_fraction(sizes: WeightSizes, union_rate: float) -> float:
    """Share of a layer's weights still crossing PCIe under ADAPTIVE.

    Attention is dense, ``gate_proj`` is dense because it produces the mask, and
    only ``up_proj``/``down_proj`` scale with the firing rate. This is the
    ceiling on the whole adaptive idea and it is not close to zero: at
    ``union_rate = 0`` the layer still moves attention plus the gate.
    """
    per_layer = sizes.per_layer
    dense = sizes.attention + sizes.gate + sizes.norms + sizes.lif
    sparse = (sizes.up + sizes.down) * union_rate
    return (dense + sparse) / per_layer


# -- planning --------------------------------------------------------------


def auto_plan(
    config,
    devices,
    dtype: torch.dtype,
    free_vram_bytes: int,
    kv_cache_bytes: int = 0,
    spike_stats: dict | None = None,
) -> DevicePlan:
    """Turn a :class:`~spikeinfer.config.DeviceConfig` into a concrete plan.

    Explicit settings win; ``auto`` and ``None`` are resolved against the VRAM
    budget. The order matters: layers the user pinned to the CPU are removed
    first, then whatever is left is fitted into VRAM, streaming the tail if it
    does not fit.
    """
    num_layers = config.num_hidden_layers
    sizes = weight_sizes(config)

    gpu_layers = num_layers if devices.gpu_layers is None else min(devices.gpu_layers, num_layers)
    layers = [Placement.GPU] * gpu_layers + [Placement.CPU] * (num_layers - gpu_layers)

    embeddings = Placement.CPU if devices.offload_embeddings else Placement.GPU
    if gpu_layers == 0:
        # Nothing computes on the GPU; keeping the embedding there would mean a
        # PCIe round trip for a lookup whose result immediately goes to the CPU.
        embeddings = Placement.CPU

    budget = free_vram_bytes - kv_cache_bytes
    if embeddings is Placement.GPU:
        budget -= sizes.embedding_bytes(dtype)

    requested = devices.offload_layers
    if requested == "auto":
        fits = max(0, int(budget // max(1, sizes.layer_bytes(dtype))))
        to_stream = max(0, gpu_layers - fits)
    elif isinstance(requested, int):
        to_stream = min(requested, gpu_layers)
    else:
        to_stream = 0

    # Stream the *deepest* GPU layers. The shallow ones are reached first every
    # step, so leaving them resident means the first prefetch has a full layer's
    # compute to hide behind rather than starting cold.
    for index in range(gpu_layers - to_stream, gpu_layers):
        layers[index] = Placement.STREAM

    # Adaptive is what a layer does *instead of* being plainly resident: its
    # dense half stays in VRAM and only the MLP tail is fetched. It therefore
    # applies to the layers streaming did not claim, not to the streamed ones.
    if devices.adaptive_mlp:
        for index in range(gpu_layers - to_stream):
            layers[index] = Placement.ADAPTIVE

    hot = []
    if devices.adaptive_mlp:
        hot = _hot_channel_counts(config, devices, spike_stats, layers)

    result = DevicePlan(
        layers=layers,
        embeddings=embeddings,
        head=embeddings if getattr(config, "tie_word_embeddings", False) else Placement.GPU,
        stream_buffers=devices.stream_buffers,
        pin_memory=devices.pin_memory,
        hot_channels=hot,
    )
    # Carried rather than passed separately: apply_plan needs the same
    # measurement the hot sizes came from, to order the channels the same way.
    result._spike_stats = spike_stats
    return result


def _hot_channel_counts(config, devices, spike_stats, layers) -> list[int]:
    """How many channels each adaptive layer keeps resident.

    A flat fraction across layers would be wrong for this checkpoint: measured
    union rates run from 0.4% (layer 3) to 43.8% (layer 0). Sizing each layer
    from its own coverage curve puts the resident budget where it is actually
    used.
    """
    inter = config.intermediate_size
    if devices.hot_fraction is not None:
        count = int(round(devices.hot_fraction * inter))
        return [count if p is Placement.ADAPTIVE else 0 for p in layers]

    if spike_stats is None:
        # No measurement: 20% is where the coverage curve knees on the one
        # checkpoint we have numbers for. Better to say so than to pretend.
        return [int(0.2 * inter) if p is Placement.ADAPTIVE else 0 for p in layers]

    counts = []
    for index, placement in enumerate(layers):
        if placement is not Placement.ADAPTIVE or index >= len(spike_stats["layers"]):
            counts.append(0)
            continue
        freq = torch.tensor(spike_stats["layers"][index]["channel_union_freq"])
        counts.append(_coverage_cut(freq, HOT_COVERAGE, inter))
    return counts


HOT_COVERAGE = 0.9
"""Share of expected firings the resident hot slice should cover.

Sizing by a frequency threshold instead looks reasonable and is not: on the
reference checkpoint a 0.125 cut keeps 74% of layer 0's channels resident and
saves nothing, because that layer fires densely. Coverage adapts -- a layer
that fires everywhere gets a small slice (nothing would help it) and a layer
with a sharp head gets exactly the head."""


def _coverage_cut(freq: torch.Tensor, coverage: float, intermediate: int) -> int:
    """Smallest prefix of frequency-sorted channels covering ``coverage``.

    Capped at half the channels: past that the resident slice costs more VRAM
    than the fetching it avoids is worth, and the layer should just be dense.
    """
    ordered = torch.sort(freq, descending=True).values
    total = float(ordered.sum())
    if total <= 0:
        return 0
    cumulative = torch.cumsum(ordered, dim=0) / total
    cut = int(torch.searchsorted(cumulative, coverage).item()) + 1
    return max(1, min(cut, intermediate // 2))


def plan_from_config(config, devices, dtype, free_vram_bytes, **kwargs) -> DevicePlan:
    """``auto_plan``, but returns the trivial plan when nothing was asked for.

    Keeps the common path free of the executor entirely.
    """
    nothing_requested = (
        devices.gpu_layers is None
        and devices.offload_layers is None
        and not devices.offload_embeddings
        and not devices.adaptive_mlp
    )
    if nothing_requested:
        return DevicePlan(layers=[Placement.GPU] * config.num_hidden_layers)
    return auto_plan(config, devices, dtype, free_vram_bytes, **kwargs)


def describe_plan(
    config,
    plan: DevicePlan,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    union_rate: float | None = None,
    h2d_gb_s: float | None = None,
) -> dict:
    """Everything ``spikeinfer plan`` prints, computed from the config alone."""
    from .sysinfo import format_bytes

    sizes = weight_sizes(config)
    layer_bytes = sizes.layer_bytes(dtype)
    resident = plan.count(Placement.GPU) * layer_bytes
    if plan.embeddings is Placement.GPU:
        resident += sizes.embedding_bytes(dtype)
    ring = min(plan.stream_buffers, max(1, plan.count(Placement.STREAM) + plan.count(Placement.ADAPTIVE)))
    ring_bytes = ring * layer_bytes if plan.count(Placement.STREAM) + plan.count(Placement.ADAPTIVE) else 0

    per_token = packed_bytes_per_token(
        config.num_hidden_layers, config.T, config.num_key_value_heads, _head_dim(config)
    )
    kv_bytes = num_blocks * block_size * per_token

    streamed_layers = plan.count(Placement.STREAM)
    adaptive_layers = plan.count(Placement.ADAPTIVE)
    per_step = streamed_layers * layer_bytes
    if adaptive_layers:
        fraction = (
            transfer_fraction(sizes, union_rate) if union_rate is not None else 1.0
        )
        per_step += adaptive_layers * layer_bytes * fraction

    report = {
        "placement": plan.summary(),
        "dtype": str(dtype).replace("torch.", ""),
        "weights": {
            "per_layer_bytes": layer_bytes,
            "per_layer": format_bytes(layer_bytes),
            "attention_share": round(sizes.attention_share, 3),
            "mlp_share": round(sizes.mlp / sizes.per_layer, 3),
            "embedding": format_bytes(sizes.embedding_bytes(dtype)),
            "total": format_bytes(
                config.num_hidden_layers * layer_bytes + sizes.embedding_bytes(dtype)
            ),
        },
        "vram": {
            "resident_weights": format_bytes(resident),
            "stream_ring": format_bytes(ring_bytes),
            "kv_cache": format_bytes(kv_bytes),
            "total": format_bytes(resident + ring_bytes + kv_bytes),
        },
        "kv_cache": {
            "blocks": num_blocks,
            "block_size": block_size,
            "capacity_tokens": num_blocks * block_size,
            "bytes_per_token": per_token,
        },
    }

    if per_step:
        report["per_decode_step"] = {
            "host_to_device": format_bytes(per_step),
            "bytes": int(per_step),
        }
        if h2d_gb_s:
            seconds = per_step / (h2d_gb_s * 1e9)
            report["per_decode_step"]["transfer_ms"] = round(seconds * 1000, 2)
            report["per_decode_step"]["tok_s_ceiling_from_pcie"] = round(1 / seconds, 1)
        if union_rate is not None and adaptive_layers:
            report["per_decode_step"]["adaptive_transfer_fraction"] = round(
                transfer_fraction(sizes, union_rate), 3
            )
    return report


# -- execution -------------------------------------------------------------


class LayerExecutor:
    """Owns the layer loop when a plan is anything other than trivial.

    ``FastSpikingQwenModel.forward_paged`` delegates here rather than growing
    branches of its own. Everything placement-dependent -- which module holds a
    layer's weights right now, which device the hidden state has to be on, when
    the next prefetch goes out -- is decided here, so the model file stays about
    the arithmetic.

    One instance handles all four placements. A pure-CPU model does not build
    one at all (its plan is trivial once ``device`` is ``cpu``); a pure-GPU
    model does not either. Only mixtures pay.
    """

    def __init__(
        self,
        plan: DevicePlan,
        layer_devices: list[torch.device],
        ring=None,
        streamed: frozenset[int] = frozenset(),
    ) -> None:
        self.plan = plan
        self.layer_devices = layer_devices
        self.ring = ring
        self.streamed = streamed
        self.output_device = layer_devices[-1] if layer_devices else torch.device("cpu")
        self._rope_cache: dict[str, tuple] = {}

    def entry_device(self) -> torch.device:
        return self.layer_devices[0]

    def _rope_on(self, cos, sin, device):
        """cos/sin follow the hidden state across a device boundary.

        Recomputed nowhere: they are a function of position only, so one copy
        per device per step is both correct and cheap ([1, n_tok, head_dim]).
        """
        if cos.device == device:
            return cos, sin
        key = str(device)
        cached = self._rope_cache.get(key)
        if cached is None or cached[0].shape != cos.shape:
            cached = (cos.to(device), sin.to(device))
            self._rope_cache[key] = cached
        else:
            cached[0].copy_(cos, non_blocking=True)
            cached[1].copy_(sin, non_blocking=True)
        return cached

    def run(self, model, h, cos, sin, kv_caches, meta):
        self._rope_cache.clear()
        ring = self.ring
        lookahead = ring.num_slots - 1 if ring else 0

        if ring:
            ring.begin_pass()
            for index in sorted(self.streamed)[: ring.num_slots]:
                ring.prefetch(index)

        for index, layer in enumerate(model.layers):
            device = self.layer_devices[index]
            if h.device != device:
                h = h.to(device, non_blocking=True)
            layer_cos, layer_sin = self._rope_on(cos, sin, device)

            if index in self.streamed:
                module = ring.acquire(index)
            else:
                module = layer

            k_cache, v_cache = kv_caches[index]
            h = module.forward_paged(
                h, layer_cos, layer_sin, k_cache, v_cache, meta.on(device)
            )

            if index in self.streamed:
                ring.release(index)
                ring.prefetch(index + lookahead + 1)

        if h.device != self.output_device:
            h = h.to(self.output_device, non_blocking=True)
        return h


def apply_plan(model, plan: DevicePlan, device, dtype: torch.dtype) -> LayerExecutor | None:
    """Move a CPU-resident model into the shape ``plan`` describes.

    The model must already be built and loaded on the CPU -- which is what
    ``load_model`` does anyway before its ``.to(device)``, so nothing extra is
    read from disk. Returns the executor to install, or ``None`` when the plan
    is trivial and the model's own loop should be left alone.
    """
    device = torch.device(device)
    if plan.is_trivial:
        model.to(device=device)
        return None

    from .offload import FlatLayout, LayerWeightStore, StreamingLayerRing

    layers = model.model.layers
    cpu = torch.device("cpu")
    layer_devices = [cpu if p is Placement.CPU else device for p in plan.layers]

    # Everything that is not streamed simply moves to where it computes.
    streamed_indices = frozenset(i for i, p in enumerate(plan.layers) if p is Placement.STREAM)
    adaptive_indices = [i for i, p in enumerate(plan.layers) if p is Placement.ADAPTIVE]
    placed_later = streamed_indices | set(adaptive_indices)
    for index, layer in enumerate(layers):
        if index not in placed_later:
            layer.to(device=layer_devices[index])

    if adaptive_indices:
        _build_adaptive_layers(model, layers, adaptive_indices, plan, device, dtype)

    ring = None
    if streamed_indices:
        if device.type != "cuda":
            raise ValueError("weight streaming needs a CUDA device to stream to")
        template = layers[next(iter(sorted(streamed_indices)))]
        layout = FlatLayout.of(template)
        stores = {
            index: LayerWeightStore(layout, layers[index], dtype, plan.pin_memory)
            for index in sorted(streamed_indices)
        }
        # Slots are separate module instances so the ModuleList keeps holding
        # valid (host-side) layers -- useful for describe/debug, and it keeps
        # `model.layers[i]` meaning what it has always meant.
        slot_count = min(plan.stream_buffers, len(streamed_indices))
        slot_modules = [
            type(template)(model.config, 0).to(device=device, dtype=dtype)
            for _ in range(slot_count)
        ]
        for module in slot_modules:
            module.eval()
        ring = StreamingLayerRing(slot_modules, stores, layout, device, dtype)
        # The host stores now own the weights; drop the ModuleList's duplicates
        # so the model is not holding two copies of every streamed layer.
        for index in streamed_indices:
            _rebind_to_store(layers[index], layout, stores[index])

    embed_device = cpu if plan.embeddings is Placement.CPU else device
    model.model.embed_tokens.to(device=embed_device)
    model.lm_head.to(device=cpu if plan.head is Placement.CPU else device)
    # RoPE and the final norm are tiny and are needed wherever the last layer
    # ends up, so they follow the deepest compute device rather than the plan.
    model.model.rotary_emb.to(device=layer_devices[-1])
    model.model.norm.to(device=layer_devices[-1])

    model.eval()
    return LayerExecutor(plan, layer_devices, ring=ring, streamed=streamed_indices)



def _build_adaptive_layers(model, layers, indices, plan: DevicePlan, device, dtype) -> None:
    """Permute each adaptive layer's channels, then split its MLP hot/cold.

    The permutation has to happen first and on the whole layer: it reorders
    ``gate_proj``, ``lif_gate`` and both MLP matrices together, and only after
    that is "the hot set" a contiguous slice rather than a scatter.
    """
    from .offload import AdaptiveMLP

    stats = getattr(plan, "_spike_stats", None)
    intermediate = model.config.intermediate_size
    for index in indices:
        layer = layers[index]
        if stats is not None:
            apply_channel_permutation(
                layer, hot_channel_permutation(stats, index, intermediate)
            )
        n_hot = plan.hot_channels[index] if index < len(plan.hot_channels) else intermediate // 5
        AdaptiveMLP.replace_in(
            layer, n_hot, torch.device(device), dtype, pin_memory=plan.pin_memory
        )
        layer.self_attn.to(device=device, dtype=dtype)
        layer.input_layernorm.to(device=device, dtype=dtype)
        layer.post_attention_layernorm.to(device=device, dtype=dtype)


def _rebind_to_store(layer, layout, store) -> None:
    """Point a streamed layer's parameters at its host store's flat buffer."""
    from .offload import flatten_module

    flatten_module(layer, layout, store.flat)


# -- adaptive MLP: channel permutation -------------------------------------


def apply_channel_permutation(layer, permutation: torch.Tensor) -> None:
    """Reorder one layer's MLP intermediate channels, in place.

    The intermediate axis is permutation-equivariant. ``gate_proj`` and
    ``up_proj`` produce it (so their *rows* move), ``lif_gate``'s beta and
    threshold are per-channel (so they move with it), the ``gate_spk * up``
    product is elementwise, and ``down_proj`` consumes it (so its *columns*
    move). Apply the same permutation to all five and the layer computes the
    same function.

    The point is to make the hot set contiguous: with channels sorted by firing
    frequency, "the 1000 channels most likely to fire" is the slice ``[:1000]``,
    which can stay resident in VRAM and be used by a plain GEMM. Without this
    the hot set would be a scattered index list and every step would pay a
    gather over it.

    Not bit-exact: ``down_proj`` reduces over 4864 terms and reordering them
    changes the rounding. The magnitude is the same class as the T-batching
    noise ``modeling_fast`` documents (~1e-7 relative), and the tests assert
    argmax agreement rather than equality for exactly this reason.
    """
    mlp = layer.mlp
    index = permutation.to(mlp.gate_proj.weight.device)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight.index_select(0, index))
        mlp.up_proj.weight.copy_(mlp.up_proj.weight.index_select(0, index))
        mlp.down_proj.weight.copy_(mlp.down_proj.weight.index_select(1, index))
        lif = mlp.lif_gate.lif
        lif.beta.copy_(lif.beta.index_select(0, index))
        lif.threshold.copy_(lif.threshold.index_select(0, index))


def hot_channel_permutation(spike_stats: dict, layer_index: int, intermediate: int) -> torch.Tensor:
    """Channel order for one layer, most frequently firing first."""
    layers = spike_stats.get("layers", [])
    if layer_index >= len(layers):
        return torch.arange(intermediate)
    freq = torch.tensor(layers[layer_index]["channel_union_freq"], dtype=torch.float32)
    if freq.numel() != intermediate:
        raise ValueError(
            f"spike_stats layer {layer_index} has {freq.numel()} channels, "
            f"model has {intermediate} -- stats are from a different checkpoint"
        )
    return torch.sort(freq, descending=True, stable=True).indices
