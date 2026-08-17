"""Talking to a running spikeinfer server with the official OpenAI SDK.

Start the server first:

    spikeinfer serve ./qwen-spiking --port 8000

then:

    pip install openai
    python examples/openai_client.py

The point of this example is that there is nothing spikeinfer-specific in it.
"""
from __future__ import annotations

import sys

BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen-spiking"


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")

    client = OpenAI(base_url=BASE_URL, api_key="not-needed")

    print("models:", [m.id for m in client.models.list().data])

    completion = client.completions.create(
        model=MODEL, prompt="The capital of France is", max_tokens=24, temperature=0.0
    )
    print("\ncompletion:", completion.choices[0].text)
    print("usage:", completion.usage)

    print("\nstreaming chat:")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Write one sentence about spiking neurons."}],
        max_tokens=64,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
