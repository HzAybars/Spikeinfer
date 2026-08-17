"""Engine counters and their Prometheus rendering.

Deliberately tiny: a few counters, a few gauges, and two rolling windows for the
latency numbers people actually tune against (time to first token, inter-token
latency). No dependency on ``prometheus_client`` -- the exposition format is a
few lines of text and adding a dependency to a server for that is not worth it.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

_WINDOW = 256


@dataclass
class Stats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    finished_requests: int = 0
    preemptions: int = 0
    steps: int = 0

    num_running: int = 0
    num_waiting: int = 0
    cache_usage: float = 0.0
    gpu_blocks: int = 0

    start_time: float = field(default_factory=time.monotonic)
    _step_times: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))
    _ttfts: deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW))

    def record_step(self, duration: float, generated: int, prompted: int) -> None:
        self.steps += 1
        self.generation_tokens += generated
        self.prompt_tokens += prompted
        if generated:
            self._step_times.append(duration / generated)

    def record_ttft(self, seconds: float) -> None:
        self._ttfts.append(seconds)

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def avg_itl_ms(self) -> float:
        return 1000 * sum(self._step_times) / len(self._step_times) if self._step_times else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        return 1000 * sum(self._ttfts) / len(self._ttfts) if self._ttfts else 0.0

    @property
    def throughput(self) -> float:
        """Generated tokens per second, averaged over the process lifetime."""
        return self.generation_tokens / self.uptime if self.uptime > 0 else 0.0

    def to_prometheus(self, model: str) -> str:
        labels = f'{{model="{model}"}}'
        lines = [
            ("spikeinfer_prompt_tokens_total", "counter", self.prompt_tokens),
            ("spikeinfer_generation_tokens_total", "counter", self.generation_tokens),
            ("spikeinfer_request_success_total", "counter", self.finished_requests),
            ("spikeinfer_preemptions_total", "counter", self.preemptions),
            ("spikeinfer_engine_steps_total", "counter", self.steps),
            ("spikeinfer_num_requests_running", "gauge", self.num_running),
            ("spikeinfer_num_requests_waiting", "gauge", self.num_waiting),
            ("spikeinfer_gpu_cache_usage_ratio", "gauge", round(self.cache_usage, 4)),
            ("spikeinfer_gpu_cache_blocks", "gauge", self.gpu_blocks),
            ("spikeinfer_avg_ttft_ms", "gauge", round(self.avg_ttft_ms, 3)),
            ("spikeinfer_avg_itl_ms", "gauge", round(self.avg_itl_ms, 3)),
            ("spikeinfer_tokens_per_second", "gauge", round(self.throughput, 3)),
        ]
        out = []
        for name, kind, value in lines:
            out.append(f"# TYPE {name} {kind}")
            out.append(f"{name}{labels} {value}")
        return "\n".join(out) + "\n"
