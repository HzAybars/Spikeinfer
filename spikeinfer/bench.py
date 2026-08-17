"""Serving benchmark: what the engine does under load, not what a kernel does.

Feeds the engine synthetic requests at a fixed concurrency and reports the
numbers that matter for serving -- output throughput, time to first token, and
inter-token latency -- alongside the KV cache footprint, which is where a
spiking model's packed cache shows up.

``bench/benchmark.py`` and ``bench/profile_kernels.py`` remain the right tools
for kernel-level work; this one measures the whole system.
"""
from __future__ import annotations

import statistics
import time

from .config import EngineConfig
from .engine.llm_engine import LLMEngine
from .sampling_params import SamplingParams


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def run_benchmark(
    config: EngineConfig,
    num_requests: int = 64,
    concurrency: int = 16,
    input_len: int = 128,
    output_len: int = 64,
    warmup: int = 2,
    engine: LLMEngine | None = None,
) -> dict:
    """Drive ``num_requests`` through the engine, ``concurrency`` in flight."""
    engine = engine or LLMEngine(config)
    vocab = engine.model_config.vocab_size
    params = SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)

    def prompt_ids(index: int) -> list[int]:
        # Distinct prompts so no accidental cache reuse flatters the numbers.
        return [(index * 7919 + i * 31) % (vocab - 1) + 1 for i in range(input_len)]

    for i in range(warmup):
        engine.add_request(prompt_token_ids=prompt_ids(10_000 + i), sampling_params=params)
    while engine.has_unfinished_requests():
        engine.step()

    submitted = 0
    in_flight = 0
    ttfts: list[float] = []
    latencies: list[float] = []
    per_request_tokens: list[int] = []
    arrivals: dict[str, float] = {}

    started = time.monotonic()
    while submitted < num_requests or in_flight > 0:
        while in_flight < concurrency and submitted < num_requests:
            request_id = engine.add_request(
                prompt_token_ids=prompt_ids(submitted), sampling_params=params
            )
            arrivals[request_id] = time.monotonic()
            submitted += 1
            in_flight += 1
        for output in engine.step():
            if output.metrics and output.metrics.time_to_first_token is not None:
                if output.request_id in arrivals:
                    ttfts.append(output.metrics.time_to_first_token)
                    arrivals.pop(output.request_id, None)
            if output.finished:
                in_flight -= 1
                latencies.append(time.monotonic() - started)
                per_request_tokens.append(sum(len(o.token_ids) for o in output.outputs))
    elapsed = time.monotonic() - started

    generated = sum(per_request_tokens)
    prompt_tokens = num_requests * input_len
    info = engine.describe()
    return {
        "config": {
            "num_requests": num_requests,
            "concurrency": concurrency,
            "input_len": input_len,
            "output_len": output_len,
            "dtype": info["dtype"],
            "timesteps": info["timesteps"],
            "block_size": info["block_size"],
            "cuda_graph_batch_sizes": info["captured_graph_batch_sizes"],
        },
        "throughput": {
            "requests_per_s": round(num_requests / elapsed, 3),
            "output_tokens_per_s": round(generated / elapsed, 2),
            "total_tokens_per_s": round((generated + prompt_tokens) / elapsed, 2),
            "elapsed_s": round(elapsed, 3),
        },
        "latency_ms": {
            "ttft_mean": round(1000 * statistics.fmean(ttfts), 2) if ttfts else None,
            "ttft_p50": round(1000 * _percentile(ttfts, 50), 2),
            "ttft_p99": round(1000 * _percentile(ttfts, 99), 2),
            "itl_mean": round(engine.stats.avg_itl_ms, 3),
        },
        "cache": {
            "gpu_blocks": info["gpu_blocks"],
            "kv_cache_mib": info["kv_cache_mib"],
            "capacity_tokens": info["kv_capacity_tokens"],
            "peak_usage_ratio": round(engine.stats.cache_usage, 4),
        },
        "engine": {
            "steps": engine.stats.steps,
            "preemptions": engine.stats.preemptions,
            "generated_tokens": generated,
        },
    }


def run_latency(
    config: EngineConfig,
    input_len: int = 128,
    output_len: int = 64,
    iterations: int = 5,
    engine: LLMEngine | None = None,
) -> dict:
    """Single-stream decode latency -- the number CUDA graphs move."""
    engine = engine or LLMEngine(config)
    vocab = engine.model_config.vocab_size
    params = SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)
    ids = [(i * 31) % (vocab - 1) + 1 for i in range(input_len)]

    samples = []
    for i in range(iterations + 1):
        started = time.monotonic()
        engine.add_request(prompt_token_ids=ids, sampling_params=params)
        while engine.has_unfinished_requests():
            engine.step()
        if i:  # discard the first pass
            samples.append(time.monotonic() - started)

    per_token = [s / output_len for s in samples]
    return {
        "input_len": input_len,
        "output_len": output_len,
        "ms_per_token_median": round(1000 * statistics.median(per_token), 3),
        "ms_per_token_min": round(1000 * min(per_token), 3),
        "tokens_per_s": round(output_len / statistics.median(samples), 2),
        "total_s_median": round(statistics.median(samples), 4),
    }
