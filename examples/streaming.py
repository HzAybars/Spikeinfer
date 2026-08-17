"""Token-by-token streaming straight from the engine -- no server involved.

``LLM.stream`` yields text deltas as the engine produces them, which is the same
data the HTTP server puts on the wire as SSE. Time to first token is dominated
by prefill; everything after it is the per-token decode cost.

    python examples/streaming.py ./qwen-spiking "Once upon a time"
"""
from __future__ import annotations

import sys
import time

from spikeinfer import LLM, SamplingParams


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "./qwen-spiking"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"

    llm = LLM(model, max_model_len=1024)
    params = SamplingParams(max_tokens=128, temperature=0.7, top_p=0.95)

    print(prompt, end="", flush=True)
    started = time.monotonic()
    first_token_at = None
    chunks = 0

    for delta in llm.stream(prompt, params):
        if first_token_at is None:
            first_token_at = time.monotonic()
        chunks += 1
        print(delta, end="", flush=True)

    elapsed = time.monotonic() - started
    print("\n")
    if first_token_at is None:
        print("no output")
        return

    ttft = first_token_at - started
    decode = elapsed - ttft
    print(f"time to first token : {1000 * ttft:7.1f} ms")
    print(f"decode              : {1000 * decode:7.1f} ms for {chunks} chunks")
    if chunks > 1:
        print(f"                      {1000 * decode / (chunks - 1):7.2f} ms/token")


if __name__ == "__main__":
    main()
