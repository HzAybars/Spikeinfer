"""Streaming layer weights from host RAM, so depth stops costing VRAM.

The engine's resident footprint is ``num_layers x layer_bytes`` plus the KV
cache. The cache is already tiny here -- 3 KB per token, bit-packed -- so on a
model that does not fit, it is the weights that do not fit, and the only lever
left is to stop keeping all of them resident.

Three facts about this model make that unusually clean:

**Every decoder layer has identical parameter shapes**, and ``layer_idx`` is
stored in ``__init__`` and never read at runtime (``modeling_fast.py:105``). So
one physical layer module can host any logical layer's weights in turn. The ring
below is ``stream_buffers`` such modules; nothing is reallocated per step and no
parameter is ever rebound.

**One layer is 20 tensors.** Copying them individually is 20 ``cudaMemcpyAsync``
calls per layer, 480 per step at 24 layers, which on Windows/WDDM costs more in
launch overhead than the bytes cost in bandwidth. Instead every layer -- host
side and slot side -- is one flat contiguous buffer with the parameters as
views into it, so streaming a layer is exactly **one** copy.

**The copy engine is separate silicon.** Layer ``i``'s compute overlaps layer
``i+1``'s copy, so with enough buffers the transfer is free until PCIe
saturates. Two buffers is enough to overlap one layer ahead; a third helps when
copies are jittery.

What it costs
-------------
Decode becomes PCIe-bound. At the measured 26.6 GB/s on this machine, streaming
all 24 layers of the 0.5B model is 685 MiB per step, about 26 ms, which is a
ceiling of ~39 tok/s no matter how fast the GPU is. That is the price of running
a model that would otherwise not run at all -- and it is what
:mod:`spikeinfer.spike_stats` and the adaptive path exist to reduce.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FlatLayout:
    """Where each parameter sits inside a layer's single contiguous buffer.

    Built once from a template module and reused for every host store and every
    GPU slot, so the two are guaranteed to agree -- a mismatch would not raise,
    it would silently stream one tensor's bytes into another's slot.
    """

    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    offsets: tuple[int, ...]
    numel: int

    @classmethod
    def of(cls, module: nn.Module) -> FlatLayout:
        names, shapes, offsets, cursor = [], [], [], 0
        for name, param in module.named_parameters():
            names.append(name)
            shapes.append(param.shape)
            offsets.append(cursor)
            cursor += param.numel()
        return cls(tuple(names), tuple(shapes), tuple(offsets), cursor)

    def nbytes(self, dtype: torch.dtype) -> int:
        return self.numel * dtype.itemsize

    def view(self, flat: torch.Tensor, index: int) -> torch.Tensor:
        start = self.offsets[index]
        shape = self.shapes[index]
        return flat[start : start + shape.numel()].view(shape)


def _set_parameter(module: nn.Module, dotted: str, tensor: torch.Tensor) -> None:
    """Rebind ``module.<dotted>`` to a view, without copying."""
    *path, leaf = dotted.split(".")
    target = module
    for part in path:
        target = getattr(target, part)
    target._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)


def flatten_module(
    module: nn.Module, layout: FlatLayout, flat: torch.Tensor, copy_from: nn.Module | None = None
) -> torch.Tensor:
    """Point every parameter of ``module`` at a slice of ``flat``.

    When ``copy_from`` is given its values are written into ``flat`` first, so
    this both packs and rebinds. Otherwise ``flat`` is assumed already filled.
    """
    source = copy_from or module
    params = dict(source.named_parameters())
    for index, name in enumerate(layout.names):
        destination = layout.view(flat, index)
        if copy_from is not None or source is module:
            destination.copy_(params[name].detach())
        _set_parameter(module, name, destination)
    return flat


class LayerWeightStore:
    """One logical layer's weights, flat and (usually) pinned, in host RAM."""

    def __init__(self, layout: FlatLayout, source: nn.Module, dtype: torch.dtype, pin: bool):
        self.layout = layout
        try:
            self.flat = torch.empty(layout.numel, dtype=dtype, pin_memory=pin)
        except RuntimeError:
            # Pinning is a finite resource and a large model can exhaust it.
            # Falling back is much better than failing to load: the copies stop
            # overlapping compute, which is slower, not wrong.
            self.flat = torch.empty(layout.numel, dtype=dtype)
            self.pinned = False
        else:
            self.pinned = pin
        with torch.no_grad():
            params = dict(source.named_parameters())
            for index, name in enumerate(layout.names):
                self.layout.view(self.flat, index).copy_(params[name].detach().to(dtype).cpu())

    @property
    def nbytes(self) -> int:
        return self.flat.numel() * self.flat.element_size()


class StreamingLayerRing:
    """``stream_buffers`` GPU-resident layer slots, filled round-robin.

    Slot assignment is ``logical_index % num_slots``, which is what makes the
    two-way synchronisation below tractable: the slot layer ``i`` will use is
    the one layer ``i - num_slots`` used, and that layer's compute is long
    finished by the time the prefetch for ``i`` is issued.
    """

    def __init__(
        self,
        slot_modules: list[nn.Module],
        stores: dict[int, LayerWeightStore],
        layout: FlatLayout,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not slot_modules:
            raise ValueError("a streaming ring needs at least one slot")
        self.layout = layout
        self.stores = stores
        self.device = device
        self.modules = slot_modules
        self.num_slots = len(slot_modules)

        self.flats: list[torch.Tensor] = []
        for module in slot_modules:
            flat = torch.empty(layout.numel, dtype=dtype, device=device)
            flatten_module(module, layout, flat)
            self.flats.append(flat)

        self.copy_stream = torch.cuda.Stream(device=device)
        self.copy_done = [torch.cuda.Event() for _ in range(self.num_slots)]
        self.compute_done = [torch.cuda.Event() for _ in range(self.num_slots)]
        self.pass_start = torch.cuda.Event()
        self.resident: list[int | None] = [None] * self.num_slots

    def slot_of(self, layer_index: int) -> int:
        return layer_index % self.num_slots

    def begin_pass(self) -> None:
        """Start a forward pass: forget which slot holds what, and order this
        pass's copies after everything the compute stream already has queued.

        Both halves exist for CUDA graph capture. Forgetting the residency
        means every layer's copy is issued -- and therefore *captured* -- so a
        replay is self-contained rather than depending on which slots happened
        to be warm when capture ran. The single ``pass_start`` event replaces
        what would otherwise be a wait on events recorded before capture began,
        which CUDA rejects (``cudaErrorStreamCaptureIsolation``): recorded here,
        inside the pass, it is inside the capture region and legal, while still
        being strong enough to keep the previous pass's last layers from having
        their weights overwritten mid-compute.
        """
        self.resident = [None] * self.num_slots
        stream = torch.cuda.current_stream(self.device)
        self.pass_start.record(stream)
        self.copy_stream.wait_event(self.pass_start)

    def prefetch(self, layer_index: int) -> None:
        """Start layer ``layer_index``'s copy, if it is one we stream."""
        store = self.stores.get(layer_index)
        if store is None:
            return
        slot = self.slot_of(layer_index)
        if self.resident[slot] == layer_index:
            return
        if self.resident[slot] is not None:
            # The slot is being recycled within this pass, so its previous
            # occupant's compute must finish before the copy lands on it.
            self.copy_stream.wait_event(self.compute_done[slot])
        with torch.cuda.stream(self.copy_stream):
            self.flats[slot].copy_(store.flat, non_blocking=store.pinned)
        self.copy_done[slot].record(self.copy_stream)
        self.resident[slot] = layer_index

    def acquire(self, layer_index: int) -> nn.Module:
        """Block the compute stream until layer ``layer_index`` has landed."""
        slot = self.slot_of(layer_index)
        if self.resident[slot] != layer_index:
            # Nothing prefetched it -- issue the copy now and eat the stall.
            self.prefetch(layer_index)
        torch.cuda.current_stream(self.device).wait_event(self.copy_done[slot])
        return self.modules[slot]

    def release(self, layer_index: int) -> None:
        """Mark the slot reusable once this layer's compute is queued."""
        slot = self.slot_of(layer_index)
        self.compute_done[slot].record(torch.cuda.current_stream(self.device))

    @property
    def nbytes(self) -> int:
        return sum(f.numel() * f.element_size() for f in self.flats)

    def host_nbytes(self) -> int:
        return sum(store.nbytes for store in self.stores.values())


class AdaptiveMLP(nn.Module):
    """An MLP that only touches the channels its gate actually fires.

    ``down_proj(gate_spk * up_proj(x))`` is exactly zero wherever ``gate_spk``
    is zero, so row j of ``up_proj`` and column j of ``down_proj`` do no work
    for a token whose channel j stays silent. On the reference checkpoint the
    union over T of fired channels is 5.5% at batch 1
    (:mod:`spikeinfer.spike_stats`), so most of both matrices is idle.

    ``gate_proj`` is not optional -- it is what computes the mask -- which sets
    the floor: a layer still moves attention plus a third of its MLP however
    quiet the gate is, about 41% of the layer. The measured 5.5% lands at 45%.

    Layout
    ------
    Channels are permuted by firing frequency before this is built
    (:func:`spikeinfer.placement.apply_channel_permutation`), so:

    * ``[:n_hot]`` -- the channels that usually fire. Resident, plain GEMM, no
      gather, no transfer. This is what keeps the common case cheap.
    * ``[n_hot:]`` -- the tail. Kept in host memory, and only the rows that
      fired this step are gathered and copied.

    The cold half of ``down_proj`` is stored **transposed**. Its channels are
    columns of a ``[hidden, C]`` matrix, and gathering columns on the host means
    a strided read; transposed they are rows, and ``index_select`` runs at
    memory bandwidth.

    What it costs, measured
    -----------------------
    The mask is a device tensor and the gather is a host operation, so each
    layer needs one device-to-host read per step. That is a synchronisation
    point, which means **a layer using this cannot be inside a CUDA graph**, and
    the engine refuses capture when any layer is adaptive.

    On an RTX 4070 SUPER / Windows / WDDM, Qwen2.5-0.5B at T=4, bf16, batch 1,
    the sparsity does everything it promises and it still does not pay::

        per-step host-to-device   685 MiB dense streaming -> 2.5 MiB   (170x less)
        resident weights          989 MiB -> 833 MiB      (574 with --offload-embeddings)
        throughput                48.7 tok/s eager dense  -> 28.1 tok/s

    Isolating the cost, all eager, no graphs::

        dense resident                              20.9 ms/token
        adaptive, hot=100% (no cold path, no sync)  22.0 ms/token   <- the split is free
        adaptive, hot=90%  (rare cold fetches)      28.1 ms/token
        adaptive, hot=17%  (coverage-sized)         36.0 ms/token

    So restructuring the MLP costs ~5%, and the remaining 14 ms is 24 host round
    trips at ~0.25-0.6 ms each. The transfer those round trips exist to shrink
    is 0.1 ms. On this machine ``--offload-layers`` is simply better: it reaches
    389 MiB resident at 34.9 tok/s because it keeps CUDA graphs.

    Where it does pay:

    * **CPU** -- there is no sync and no transfer, and the same mask is a
      straight reduction in GEMM work: 9.5 -> 12.6 tok/s at batch 1 and
      28.6 -> 35.9 tok/s at four concurrent, on the same checkpoint.
    * links slow enough, or models large enough, that 685 MiB per step is the
      wall rather than launch overhead;
    * platforms where a synchronisation is not ~0.25 ms. WDDM batches
      submissions, which is exactly what makes a per-layer round trip expensive
      here and much cheaper on Linux.
    """

    def __init__(
        self,
        dense_mlp: nn.Module,
        n_hot: int,
        device: torch.device,
        dtype: torch.dtype,
        pin_memory: bool = True,
        scratch_channels: int | None = None,
    ) -> None:
        super().__init__()
        up = dense_mlp.up_proj.weight.detach()
        down = dense_mlp.down_proj.weight.detach()
        intermediate = up.shape[0]
        self.n_hot = max(0, min(n_hot, intermediate))
        self.n_cold = intermediate - self.n_hot
        self.hidden = up.shape[1]
        self.device = device
        self.streams_to_device = device.type == "cuda"

        self.gate_proj = dense_mlp.gate_proj.to(device=device, dtype=dtype)
        self.lif_gate = dense_mlp.lif_gate.to(device=device, dtype=dtype)

        # .contiguous() is load-bearing, not tidiness: `up[:n_hot]` is a *view*
        # of the full [C, hidden] weight, and if the source already sits on the
        # target device `.to()` is a no-op that hands the view straight back --
        # keeping the entire matrix alive and saving no memory at all. Forcing
        # a copy is what actually lets the cold rows be freed.
        self.register_buffer(
            "up_hot", up[: self.n_hot].contiguous().to(device=device, dtype=dtype)
        )
        self.register_buffer(
            "down_hot", down[:, : self.n_hot].contiguous().to(device=device, dtype=dtype)
        )

        cold_up = up[self.n_hot :].to(dtype=dtype).cpu().contiguous()  # same reason
        # [C_cold, hidden]: down's columns become rows so the gather is contiguous.
        cold_down = down[:, self.n_hot :].t().to(dtype=dtype).cpu().contiguous()
        if pin_memory and self.streams_to_device and self.n_cold:
            try:
                cold_up = cold_up.pin_memory()
                cold_down = cold_down.pin_memory()
            except RuntimeError:  # pragma: no cover - pinned memory exhausted
                pass
        self.cold_up_host = cold_up
        self.cold_down_host = cold_down

        # Scratch is a cap, not a limit: more channels than this fire and the
        # cold set is processed in chunks, so correctness never depends on it.
        capacity = scratch_channels or max(1, min(self.n_cold, max(64, self.n_cold // 4)))
        self.capacity = min(capacity, self.n_cold) if self.n_cold else 0
        if self.n_cold and self.streams_to_device:
            self.register_buffer(
                "scratch_up", torch.empty(self.capacity, self.hidden, device=device, dtype=dtype)
            )
            self.register_buffer(
                "scratch_down", torch.empty(self.capacity, self.hidden, device=device, dtype=dtype)
            )
            self.staging_up = torch.empty(
                self.capacity, self.hidden, dtype=dtype, pin_memory=cold_up.is_pinned()
            )
            self.staging_down = torch.empty(
                self.capacity, self.hidden, dtype=dtype, pin_memory=cold_down.is_pinned()
            )

        self.cold_fetches = 0
        self.cold_channels_fetched = 0
        self.calls = 0

    @classmethod
    def replace_in(cls, layer, n_hot, device, dtype, **kwargs) -> AdaptiveMLP:
        """Swap a layer's dense MLP for an adaptive one, in place."""
        adaptive = cls(layer.mlp, n_hot, device, dtype, **kwargs)
        layer.mlp = adaptive
        return adaptive

    def _cold_contribution(self, x, cold_spk, index):
        """``down_cold(cold_spk * up_cold(x))`` over the fired channels only."""
        k = index.numel()
        if self.streams_to_device:
            take = min(k, self.capacity)
            host_index = index[:take]
            torch.index_select(self.cold_up_host, 0, host_index, out=self.staging_up[:take])
            torch.index_select(self.cold_down_host, 0, host_index, out=self.staging_down[:take])
            self.scratch_up[:take].copy_(self.staging_up[:take], non_blocking=True)
            self.scratch_down[:take].copy_(self.staging_down[:take], non_blocking=True)
            up_w, down_w = self.scratch_up[:take], self.scratch_down[:take]
            device_index = host_index.to(cold_spk.device, non_blocking=True)
        else:
            take = k
            up_w = self.cold_up_host.index_select(0, index)
            down_w = self.cold_down_host.index_select(0, index)
            device_index = index

        selected = cold_spk.index_select(-1, device_index)
        activated = selected * torch.nn.functional.linear(x, up_w)
        out = activated @ down_w

        if take < k:
            # More fired than the scratch holds; the rest go round again.
            out = out + self._cold_contribution(x, cold_spk, index[take:])
        return out

    def forward(self, x):
        from .kernels import lif_multistep

        self.calls += 1
        gate_spk = lif_multistep(
            self.gate_proj(x), self.lif_gate.lif.beta, self.lif_gate.lif.threshold
        )

        out = None
        if self.n_hot:
            hot = gate_spk[..., : self.n_hot] * torch.nn.functional.linear(x, self.up_hot)
            out = torch.nn.functional.linear(hot, self.down_hot)

        if self.n_cold:
            cold_spk = gate_spk[..., self.n_hot :]
            # Union over T and over every token in the batch: one fetched set
            # serves the whole step, which is why the saving shrinks as
            # concurrency grows (see spike_stats' batch saturation table).
            fired = cold_spk.reshape(-1, self.n_cold).any(dim=0)
            index = fired.nonzero(as_tuple=False).squeeze(-1)
            if self.streams_to_device:
                index = index.cpu()  # the gather is a host operation
            count = index.numel()
            if count:
                self.cold_fetches += 1
                self.cold_channels_fetched += count
                contribution = self._cold_contribution(x, cold_spk, index)
                out = contribution if out is None else out + contribution

        if out is None:  # pragma: no cover - n_hot and n_cold cannot both be 0
            raise RuntimeError("adaptive MLP has no channels")
        return out

    @property
    def stats(self) -> dict:
        """What the profiler reports: how much of the tail actually moved."""
        return {
            "calls": self.calls,
            "cold_fetches": self.cold_fetches,
            "mean_cold_channels": (
                round(self.cold_channels_fetched / self.calls, 1) if self.calls else 0.0
            ),
            "n_hot": self.n_hot,
            "n_cold": self.n_cold,
        }
