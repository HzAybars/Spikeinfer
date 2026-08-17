# spikeinfer

[![CI](https://github.com/HzAybars/Spikeinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/HzAybars/Spikeinfer/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

An inference engine and OpenAI-compatible server for **spiking (LIF) transformers** —
what vLLM is for dense models, for models whose activations are spikes.

```bash
spikeinfer serve ./qwen-spiking --port 8000
```

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen-spiking",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Continuous batching, a paged KV cache, CUDA graphs, streaming, per-request sampling —
the usual serving stack. What is not usual is what it stores and how it computes
attention, because in a spiking model the tensors are **binary**.

## Contents

- [The idea: spikes are bits, so store bits](#the-idea-spikes-are-bits-so-store-bits)
- [Numbers](#numbers)
- [Install](#install)
- [Getting a model](#getting-a-model)
- [Use](#use)
- [Correctness](#correctness)
- [What it does not do](#what-it-does-not-do)
- [Layout](#layout)
- [License](#license)

## The idea: spikes are bits, so store bits

LIF neurons sit on the output of `q_proj`, `k_proj`, `v_proj` and the MLP gate. After the
neuron, `q`, `k` and `v` are 0 or 1 — nothing else. Two things follow, and the engine is
built on both.

**The KV cache is one bit per spike.** A head_dim-64 key is 8 bytes packed instead of 256
in fp32. That matters more than a constant factor suggests, because a spiking model runs
`T` timesteps and its KV cache starts out *T times larger* than the dense model's.
Packing more than pays that back:

| KV cache, per token, Qwen2.5-0.5B | bytes |
|---|---|
| spikes in fp32 | 98,304 |
| spikes in bf16 | 49,152 |
| **spikes bit-packed (this engine)** | **3,072** |
| *the dense fp16 model it was converted from* | *12,288* |

At T=4 the spiking model's cache ends up **4× smaller than the dense model's** — 349k
tokens of KV per GiB.

**Attention scores are a popcount, not a matmul.** With `q` and `k` both binary,

```
q · k  =  popcount(q_bits & k_bits)
```

an exact integer in `[0, head_dim]` — no accumulation order to worry about, no tensor
cores involved. The value side is the same trick from the other direction: with `v`
binary, `p · v` is the sum of the attention weights whose value bit is set.
`spikeinfer/kernels/spike_attention.py` is a paged, flash-style decode kernel doing
exactly that, reading 32× fewer bytes than an fp32 cache would. (Triton 3.7 has no
`popc` intrinsic, so it uses a SWAR popcount — which is bit-exact and, conveniently,
portable across Triton versions.)

Prefill deliberately does *not* use it. Prefill is compute-bound and SDPA already
saturates the tensor cores; trading those for CUDA-core popcounts would lose. Decode is
bandwidth-bound, which is exactly where reading 1-bit keys wins.

## Numbers

Qwen2.5-0.5B converted to spiking form, T=4, bf16, RTX 4070 SUPER, Windows/WDDM,
torch 2.6.0+cu124, 128-token prompts, 64-token outputs.

| concurrent requests | output tok/s | TTFT p50 | inter-token latency |
|---|---|---|---|
| 1 | 207 | 31 ms | 4.82 ms |
| 8 | 1,066 | 125 ms | 0.93 ms |
| 32 | 2,016 | 235 ms | 1.71 ms |
| 64 | **2,608** | 235 ms | 1.59 ms |

CUDA graphs remain the single largest win:

| | single stream | 32 concurrent |
|---|---|---|
| eager | 18.56 ms/token | 1,125 tok/s |
| CUDA graphs | **4.88 ms/token** | **2,016 tok/s** |

```bash
spikeinfer bench ./qwen-spiking --concurrency 32 --input-len 128 --output-len 64
```

### Where the speed came from originally

Spiking LLMs are usually *slower* than the dense models they replace, and the intuitive
diagnosis — "binary spikes are being simulated as dense FP32 matmuls" — is wrong for this
architecture. LIF neurons sit on the **output** of `q/k/v/gate_proj`, so **no `nn.Linear`
ever consumes a spike tensor**. A sparse spike-driven GEMV kernel has nothing to attach
to, and INT8 tensor cores measured *slower* than fp16 at these shapes (0.28x).

The real bottleneck was **kernel launch count**. One T=4 decode step issued **27,859 CUDA
launches**, and Windows WDDM costs roughly 10 us per launch. The tell: decoding a single
token (135.7 ms) cost about as much as prefilling 128 tokens (143.6 ms) — a launch-bound
signature, not a compute-bound one. A naive spiking implementation decodes at 146 ms/token;
the fused LIF kernel and the layer-major T-batching brought that to 18.65, and CUDA graphs
to 4.32.

CPU-side dispatches per decode step: **27,859 → 5,727 → 6**.

> These are **CPU-side dispatch counts**. CUPTI device-side kernel tracing is unavailable
> on this Windows/WDDM setup, so no tool here can see *inside* a captured graph — by
> construction the whole 24-layer × T=4 step replays as a single graph launch. The 6
> remaining dispatches are the bookkeeping around `replay()`.

Full detail in [docs/architecture.md](docs/architecture.md).

## Install

```bash
pip install -e ".[server]"
```

Triton is optional but is where the speed comes from — without it the LIF kernel, the
packing and the paged attention fall back to pure PyTorch and the package still runs.

- **Linux:** Triton ships inside the torch wheel.
- **Windows:** `pip install triton-windows`. Triton JIT-compiles through MSVC and needs
  `cl.exe` on `PATH` plus a `CUDA_HOME` matching your torch build; `tools\env.bat` sets
  both up. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Getting a model

There is no pretrained spiking LLM to download; you convert one. `spikeinfer convert`
writes a model directory (safetensors + config + tokenizer):

```bash
# from a dense Qwen2 checkpoint
spikeinfer convert --from-hf Qwen/Qwen2.5-0.5B --out ./qwen-spiking --timesteps 4

# or from an existing SpikingQwen .pt checkpoint
spikeinfer convert --from-checkpoint spiking_qwen.pt --out ./qwen-spiking \
                   --tokenizer Qwen/Qwen2.5-0.5B --save-dtype bfloat16
```

**A transplanted model is not yet a working model.** Every real-valued weight carries over
1:1 — the spiking architecture keeps Qwen2's linear layers, RMSNorms and RoPE — but the
LIF thresholds start uniform and have not been fitted to the activation scales they now
gate. Until they are calibrated the output is fluent-looking noise. Calibration is
training, and it lives outside an inference engine; see
[docs/converting.md](docs/converting.md).

Inspect a model without loading its weights:

```bash
spikeinfer info ./qwen-spiking
```

## Use

```bash
spikeinfer serve ./qwen-spiking --port 8000 --max-num-seqs 64
```

Endpoints: `/v1/completions`, `/v1/chat/completions` (both streaming), `/v1/models`,
`/tokenize`, `/detokenize`, `/health`, `/metrics` (Prometheus), `/stats`. Any OpenAI
client works unmodified:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
stream = client.chat.completions.create(
    model="qwen-spiking",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Offline batch:

```python
from spikeinfer import LLM, SamplingParams

llm = LLM("./qwen-spiking")
outputs = llm.generate(
    ["The capital of France is", "def fibonacci(n):"],
    SamplingParams(max_tokens=64, temperature=0.7, top_p=0.95),
)
for output in outputs:
    print(output.outputs[0].text)
```

Terminal:

```bash
spikeinfer chat ./qwen-spiking
spikeinfer generate ./qwen-spiking -p "Once upon a time" --max-tokens 128
```

The low-level pieces stay importable, and the fused LIF kernel is useful on its own
against any spiking model:

```python
from spikeinfer.kernels import lif_multistep, pack_spikes

spikes = lif_multistep(current, beta, threshold)   # current: [T, *dims, N]
packed = pack_spikes(spikes)                       # 1 bit per spike
```

## Correctness

**330 tests.** The one that matters most is
`tests/test_engine.py::test_greedy_matches_the_eager_path`: greedy generation through the
entire serving stack — paged cache, bit-packed spikes, popcount attention, continuous
batching, CUDA graph replay — must produce the **same token ids** as the simple eager
path, which is itself validated against the unoptimized snntorch reference in
`spikeinfer/reference/`. On the real 0.5B model that is currently 72/72 tokens across
three prompts.

Also asserted:

- packing is exactly invertible — including a 32-channel group that fills the int32 sign
  bit, and channel counts that are not a multiple of 32;
- the paged attention kernel matches an SDPA oracle to 1.8e-7 relative, with physical
  blocks assigned in scrambled order, ragged batch lengths, GQA ratios from 1:1 to 8:1,
  and an all-zero query (a token that fires no spikes at all);
- attention reads *only* the blocks a sequence owns — overwriting every other block must
  not change its output, and neither must the padding tail of its last block;
- a prompt split across steps by chunked prefill yields the same tokens as one that was
  not;
- a sequence preempted mid-generation and recomputed from scratch produces identical
  output;
- results do not depend on batch composition: run alone or 32-at-a-time, same tokens.

128 of the tests run without a GPU, and CI runs those. The rest need CUDA and skip
themselves — which means **CI passing does not mean the kernels are correct**; run the
full suite on a GPU before trusting a change.

### The one thing to know if you touch the LIF kernel

Spiking models are unusually sensitive to 1-ulp changes: a membrane potential sitting
exactly on its threshold flips a *discrete* spike. Triton/ptxas will contract
`beta*u + current` into a single FMA (one rounding) where PyTorch eager does two, and that
difference alone flips spikes near threshold — hence `enable_fp_fusion=False` on the
kernel. Also, snntorch's reset is delayed by one step and its comparison is `(U - θ) > 0`
rather than `U > θ`, which are not the same predicate in floating point. Getting any of
these wrong degrades the model silently rather than failing.

## What it does not do

- **Model quality is the conversion's problem, not the engine's.** The engine reproduces
  the reference implementation faithfully; a poorly calibrated checkpoint produces poor
  output, faithfully. The 0.5B checkpoint this was developed against sits at WikiText-2
  perplexity ~390.
- **One GPU.** No tensor or pipeline parallelism, no CPU offload.
- **Qwen2 architecture** (GQA supported). The kernels are architecture-agnostic; the model
  class is not.
- **No prefix caching, no beam search.** Preemption recomputes rather than swapping to
  host memory — with a packed cache a block is ~96 KB, so refilling one beats moving it
  over PCIe.
- **Inference only.** No backward pass in the kernels; training stays on the reference
  path.
- **At compute-bound batch sizes a T-step spiking model cannot beat the dense ANN.** T=4
  means 4× the FLOPs, and once launch overhead is gone that is the floor. The wins here
  are in memory footprint and in launch-bound decoding — which is where LLM inference
  actually lives.

## Layout

```
spikeinfer/
  kernels/
    lif_triton.py         fused multi-timestep LIF, one launch for all T
    spike_attention.py    paged popcount attention over packed spikes
    packing.py            bit-packing (Triton fast path + torch fallback)
  engine/
    llm_engine.py         requests in, tokens out
    scheduler.py          continuous batching, chunked prefill, preemption
    block_manager.py      paged block allocator
    spike_cache.py        the packed KV cache itself
    model_runner.py       batch construction + CUDA graph capture
    sampler.py            batched per-request sampling
    async_engine.py       asyncio wrapper for the server
  entrypoints/
    cli.py                serve / chat / generate / convert / bench / info
    openai/               OpenAI-compatible HTTP API
  attention.py            paged attention wiring
  modeling_fast.py        layer-major, T-batched model
  loader.py, convert.py   model directory format
  reference/              unoptimized snntorch implementation (golden reference)
tests/                    330 tests
bench/                    kernel-level timing and CUDA launch counting
docs/                     architecture and conversion notes
examples/                 runnable scripts
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
