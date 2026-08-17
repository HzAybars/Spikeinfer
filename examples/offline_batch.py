"""Offline batch generation.

Every prompt is submitted before the first step runs, so the engine batches them
continuously instead of walking the list one at a time. Throughput here is what
the engine does when it is kept fed; see ``streaming.py`` for the latency view.

    python examples/offline_batch.py ./qwen-spiking
"""
from __future__ import annotations

import json
import sys
import time

from spikeinfer import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "In 1969, humans first walked on",
    "def fibonacci(n):",
    "The three laws of robotics are",
]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "./qwen-spiking"
    llm = LLM(model, max_model_len=1024)
    print(json.dumps(llm.describe(), indent=2))

    params = SamplingParams(max_tokens=48, temperature=0.0)
    started = time.monotonic()
    outputs = llm.generate(PROMPTS, params)
    elapsed = time.monotonic() - started

    for output in outputs:
        completion = output.outputs[0]
        print(f"\n--- {output.prompt!r} ({completion.finish_reason}) ---")
        print(completion.text)

    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    ttfts = [o.metrics.time_to_first_token for o in outputs if o.metrics]
    print(f"\n{len(outputs)} prompts, {generated} tokens in {elapsed:.2f}s")
    print(f"{generated / elapsed:.1f} tok/s")
    if ttfts and all(t is not None for t in ttfts):
        print(f"time to first token: {1000 * min(ttfts):.0f}-{1000 * max(ttfts):.0f} ms")


if __name__ == "__main__":
    main()
