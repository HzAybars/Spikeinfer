"""Where a decode step's time actually goes.

``spikeinfer bench`` reports what the server delivers. This reports why. The
two questions it exists to answer came out of building the placement layer and
are not answerable from throughput alone:

* **Is the step launch-bound or bandwidth-bound?** They call for opposite
  fixes. Launch-bound wants CUDA graphs; bandwidth-bound wants fewer bytes.
  Weight streaming turns the first into the second, and at that point capturing
  graphs stops mattering -- which is only visible if the two are measured
  separately.
* **Under ``--adaptive-mlp``, is the sparsity paying for its own overhead?**
  It fetches far fewer bytes and pays a host round trip per layer to decide
  which. Reporting the bytes without the round trip makes it look like a win it
  is not.

``bench/profile_kernels.py`` remains the tool for model-level kernel work --
per-op launch counts across the reference model, the fast model and a captured
graph. This one profiles the serving engine instead, which is where placement
lives.

The launch count here is CPU-side operator dispatches, not device-side kernels:
CUPTI's device tracing is unavailable on Windows/WDDM, and on this model
essentially every dispatch issues CUDA work, so it is a faithful proxy. See
``bench/profile_kernels.py`` for the verification of that claim.
"""
from __future__ import annotations

import time

import torch

from .sampling_params import SamplingParams


def _decode_only_step(engine, prompt_len: int, seqs: int):
    """Put ``seqs`` sequences into steady-state decode and return a stepper.

    Prefill and decode have completely different profiles -- one is
    compute-bound, the other is what this project is about -- so measuring a
    mixture measures neither.
    """
    params = SamplingParams(max_tokens=10_000, temperature=0.0, ignore_eos=True)
    for index in range(seqs):
        engine.add_request(
            prompt_token_ids=[(index * 37 + 5) % 500 + 1] * prompt_len,
            sampling_params=params,
        )
    # Step until every prefill has finished and only decodes remain.
    for _ in range(prompt_len + 4):
        engine.step()
    return lambda: engine.step()


def time_decode(engine, prompt_len: int = 32, seqs: int = 1, iterations: int = 30) -> dict:
    """Milliseconds per decode step, measured in steady state."""
    step = _decode_only_step(engine, prompt_len, seqs)
    for _ in range(5):
        step()
    if engine.device.type == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iterations):
        step()
    if engine.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    per_step = elapsed / iterations
    return {
        "decode_batch": seqs,
        "ms_per_step": round(per_step * 1000, 3),
        "tokens_per_second": round(seqs / per_step, 1),
    }


def count_dispatches(engine, prompt_len: int = 32, seqs: int = 1) -> dict:
    """CPU-side operator dispatches in one decode step.

    The number this project was built to shrink: ~27,859 for the reference
    model, ~5,727 for the fast model eager, and a handful once a decode step
    replays as one captured graph.
    """
    step = _decode_only_step(engine, prompt_len, seqs)
    for _ in range(3):
        step()
    if engine.device.type == "cuda":
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if engine.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        step()
        if engine.device.type == "cuda":
            torch.cuda.synchronize()

    events = prof.events()
    return {
        "cpu_dispatches_per_step": len(events),
        "graph_replay": bool(engine.captured_graphs),
    }


def transfer_report(engine) -> dict | None:
    """Bytes the placement moves per step, and what decides them.

    ``None`` when nothing is offloaded, which is the ordinary case.
    """
    from .offload import AdaptiveMLP, StreamingLayerRing
    from .placement import Placement

    plan = getattr(engine, "plan", None)
    if plan is None:
        return None

    executor = engine.model.model.executor
    ring: StreamingLayerRing | None = getattr(executor, "ring", None)
    report: dict = {"placement": plan.summary()}

    if ring is not None:
        streamed = plan.count(Placement.STREAM)
        layer_bytes = ring.layout.nbytes(ring.flats[0].dtype)
        report["streaming"] = {
            "layers": streamed,
            "bytes_per_step": streamed * layer_bytes,
            "mib_per_step": round(streamed * layer_bytes / 2**20, 2),
            "ring_slots": ring.num_slots,
            "ring_vram_mib": round(ring.nbytes / 2**20, 2),
            "host_weights_mib": round(ring.host_nbytes() / 2**20, 2),
            "pinned": all(store.pinned for store in ring.stores.values()),
        }

    adaptive = [m for m in engine.model.modules() if isinstance(m, AdaptiveMLP)]
    if adaptive:
        calls = sum(m.calls for m in adaptive)
        fetched = sum(m.cold_channels_fetched for m in adaptive)
        hidden = adaptive[0].hidden
        itemsize = adaptive[0].up_hot.dtype.itemsize
        per_layer = fetched / max(1, calls)
        # up row + down column, per fetched channel.
        bytes_per_step = len(adaptive) * per_layer * hidden * 2 * itemsize
        report["adaptive"] = {
            "layers": len(adaptive),
            "hot_channels_mean": round(sum(m.n_hot for m in adaptive) / len(adaptive), 1),
            "cold_channels_fetched_per_layer": round(per_layer, 1),
            "mib_per_step": round(bytes_per_step / 2**20, 3),
            "host_syncs_per_step": len(adaptive),
            "note": (
                "each layer costs one device-to-host read to learn which channels "
                "fired; that round trip, not the transfer, is the dominant cost on "
                "a batched-submission driver such as WDDM"
            ),
        }
    return report


def profile_engine(
    engine_config,
    prompt_len: int = 32,
    batches: tuple[int, ...] = (1, 8),
    iterations: int = 30,
) -> dict:
    """A full report for one engine configuration."""
    from .engine.llm_engine import LLMEngine

    engine = LLMEngine(engine_config)
    report = {
        "engine": engine.describe(),
        "decode": [],
    }
    for seqs in batches:
        if seqs > engine_config.scheduler.max_num_seqs:
            continue
        timing = time_decode(engine, prompt_len, seqs, iterations)
        timing.update(count_dispatches(engine, prompt_len, seqs))
        report["decode"].append(timing)
        for request_id in list(engine._requests):
            engine.abort_request(request_id)

    transfers = transfer_report(engine)
    if transfers:
        report["transfers"] = transfers
    report["bound_by"] = _diagnose(report)
    del engine
    return report


def _diagnose(report: dict) -> str:
    """A one-line reading of the numbers, so the report says what it means."""
    decode = report["decode"][0] if report["decode"] else None
    if decode is None:  # pragma: no cover - only with an empty batch list
        return "no decode measured"

    transfers = report.get("transfers", {})
    moved = 0.0
    for section in ("streaming", "adaptive"):
        moved += transfers.get(section, {}).get("mib_per_step", 0.0)

    if moved:
        # 26.6 GB/s is this project's reference machine; a wildly different link
        # shifts the crossover, which is why the raw bytes are reported too.
        transfer_ms = moved / 26_600 * 1000
        share = transfer_ms / max(decode["ms_per_step"], 1e-9)
        if share > 0.5:
            return f"bandwidth-bound (~{share:.0%} of the step is host-to-device transfer)"
        if transfers.get("adaptive"):
            return (
                "sync-bound (transfer is small, but each adaptive layer costs a "
                "host round trip and capture is unavailable)"
            )
    if not decode["graph_replay"] and decode["cpu_dispatches_per_step"] > 1000:
        return (
            f"launch-bound ({decode['cpu_dispatches_per_step']} dispatches per step, "
            "no graph replay)"
        )
    return "compute-bound"
