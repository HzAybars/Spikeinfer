# Placement: running a model that does not fit

Until now this engine assumed the model fits on one CUDA device. Three ways out
of that, and they turn out to be one question asked per layer — *where do this
layer's weights live, and where does it compute?*

| placement | weights | computes | what it trades |
|---|---|---|---|
| `gpu` | VRAM | GPU | nothing — the default |
| `stream` | host RAM, copied in per step | GPU | VRAM for PCIe bandwidth |
| `adaptive` | dense half in VRAM, MLP tail in host RAM | GPU | VRAM for a host round trip per layer |
| `cpu` | host RAM | CPU | speed for not needing VRAM at all |

`spikeinfer plan` reports what a machine could run under each, reading the
config only — no weights are loaded, so it answers "will this fit" in
milliseconds rather than after a minute of I/O.

```bash
spikeinfer plan ./qwen-spiking --vram-gb 2 --offload-layers auto
```

## Numbers

Qwen2.5-0.5B converted to spiking form, T=4, bf16, RTX 4070 SUPER,
Windows/WDDM, batch 1, KV cache pinned at 64 blocks so only weight residency
moves. `resident` is `torch.cuda.memory_allocated()` after construction.

| configuration | resident | tok/s | CUDA graphs |
|---|---|---|---|
| everything resident | 989 MiB | **170.4** | enabled |
| everything resident, `--no-cuda-graph` | 998 MiB | 48.7 | off |
| `--offload-layers 12` | 872 MiB | 56.2 | enabled |
| `--offload-layers 24` | **389 MiB** | 34.9 | enabled |
| `--offload-layers 24 --offload-embeddings` | **325 MiB** | 27.1 | unavailable |
| `--adaptive-mlp` | 833 MiB | 28.1 | unavailable |
| `--adaptive-mlp --offload-embeddings` | 574 MiB | 13.6 | unavailable |
| `--gpu-layers 18` (fp32) | — | 20.8 | unavailable |
| `--gpu-layers 12` (fp32) | — | 15.2 | unavailable |
| `--device cpu` (fp32) | — | 9.5 | n/a |

**Use `--offload-layers`.** It is the only option here that is better on both
axes than the alternatives, and the reason is CUDA graphs: streamed weights
stay capturable, so the step keeps replaying as a single launch while the
copies overlap compute.

## Correctness

Streaming is **bit-identical** to running resident — same device, same
arithmetic, only the weights' address changed. `tests/test_placement.py`
asserts exactly that, and `spikeinfer validate --offload-layers 24` checks it on
a real checkpoint.

Splitting layers across devices is not, and cannot be. CPU and GPU reduce in
different orders, and in a spiking model a 1-ulp difference flips a spike whose
membrane sits exactly on its threshold — the same effect
`spikeinfer/modeling_fast.py` documents for T-batching. `--adaptive-mlp` is the
same case for the same reason: it permutes `down_proj`'s 4864-term reduction.

Greedy decoding cascades, so a total agreement count over a continuation
measures how early the first flip landed rather than how often flips happen.
What separates a working placement from a broken one is *where* it diverges,
so `validate` gates on the matching prefix. On the 0.5B checkpoint, three
prompts, 12 tokens each:

| | `spikeinfer validate` |
|---|---|
| `--offload-layers 24` | 36/36 tokens **identical** |
| `--gpu-layers 12` | no divergence; 36/36 agree |
| `--adaptive-mlp` | diverges after 69% of each continuation; 32/36 agree |

A broken placement fails this immediately — it produces garbage from token 0.

## Weight streaming

Three facts about this model make streaming unusually clean:

- **Every decoder layer has identical parameter shapes**, and `layer_idx` is
  stored in `__init__` and never read at runtime. So one physical layer module
  can host any logical layer's weights in turn. The ring is `--stream-buffers`
  such modules; nothing is reallocated per step and no parameter is ever
  rebound.
- **A layer is 20 tensors**, which would be 480 `cudaMemcpyAsync` calls per
  step. Instead every layer — host side and slot side — is one flat contiguous
  buffer with the parameters as views into it, so streaming a layer is *one*
  copy.
- **The copy engine is separate silicon**, so layer *i*'s compute overlaps layer
  *i+1*'s copy. Two buffers is enough to stay one layer ahead; a third measured
  no faster and cost 248 MiB, so the default is 2.

The step becomes bandwidth-bound: 683 MiB at 26.6 GB/s is ~26 ms, and `profile`
reports it as "~91% of the step is host-to-device transfer". That is the price
of running a model that would otherwise not run.

### Graph capture with streaming

A copy from pinned host memory is a legal CUDA graph node and the per-layer
sequence is identical every step, so capture works — with two things arranged
for it, both in `StreamingLayerRing.begin_pass`:

1. **Slot residency is forgotten at the start of every pass**, so every layer's
   copy is issued and therefore captured. Otherwise a replay would depend on
   which slots happened to be warm when capture ran, and the second replay would
   read the wrong layer's weights.
2. **One `pass_start` event forks the copy stream inside the pass.** Waiting on
   events recorded before capture began is rejected
   (`cudaErrorStreamCaptureIsolation`); recorded inside the pass it is legal,
   and still strong enough to stop the previous pass's last layers having their
   weights overwritten mid-compute.

Capture is attempted and falls back to eager with a warning if the driver
disagrees — losing a 3.5× speedup silently would be worse than saying so.

`--offload-embeddings` is incompatible with capture: the embedding lookup would
run on the host inside the captured region, which does not merely get skipped —
it poisons the stream and fails later somewhere unrelated. The engine refuses
capture up front instead.

## The sparse MLP, and why it does not pay here

This is the most interesting negative result in the project, so it is written
up rather than quietly dropped.

The README says LIF neurons sit on the *output* of `q/k/v/gate_proj`, so no
`nn.Linear` consumes a spike tensor. That is true for attention. It is **not**
true for the MLP:

```python
gate_spk = lif_multistep(self.gate_proj(x), beta, threshold)   # binary
return self.down_proj(gate_spk * self.up_proj(x))              # zero where gate is zero
```

Row *j* of `up_proj` and column *j* of `down_proj` do no work for a token whose
channel *j* stays silent. `gate_proj` produces the mask and can never be
skipped, which sets the floor.

`spikeinfer spike-stats` measures the rate that matters — the union over T, not
the per-timestep rate, because a channel firing at any one timestep needs its
weights and then serves all T. On the calibrated 0.5B checkpoint, 3258 tokens
of mixed prose and code:

```
per-timestep fire rate   2.1%
union over T  (r)        5.5%
union at 4 tokens       11.5%      <- one fetched set serves the whole batch,
union at 16 tokens      19.7%         so the saving shrinks with concurrency
union at 64 tokens      29.8%
per-layer union rate     min 0.4% (layer 3)   max 43.8% (layer 0)
top 20% of channels cover 90.8% of what a token needs
```

The MLP is 87.6% of a decoder layer's parameters (1.84M attention of 14.92M
total), so per-layer transfer scales as
`0.123 + 0.876·(1/3 + 2/3·r)` — **45% at r = 5.5%, and never below 41%**.

### What was built

Channels are permuted by firing frequency at load time so the hot set is a
contiguous `[:n_hot]` slice needing no gather. The intermediate axis is
permutation-equivariant across five tensors (`gate_proj` rows, `up_proj` rows,
`down_proj` columns, `lif_gate`'s beta and threshold); moving all five together
computes the same function to 2.9e-7 relative — the same class as the
T-batching noise already documented. The hot slice stays resident; the tail
lives in pinned host memory and only the rows that fired are gathered and
copied, with `down_proj`'s cold half stored transposed so gathering its columns
is a contiguous row read.

`n_hot` is sized per layer by coverage — the smallest prefix covering 90% of
expected firings, capped at half the channels. A flat frequency threshold looks
reasonable and is not: a 0.125 cut keeps 74% of layer 0 resident and saves
nothing, because that layer fires densely. Coverage gives 69 channels to the
quiet layer and 2432 to the dense one.

### What it measured

It does everything it promised, and still loses:

```
per-step host-to-device   683 MiB streaming -> 3.8 MiB    (180x less)
resident weights          989 MiB -> 833 MiB
throughput                48.7 tok/s eager dense -> 28.1 tok/s
```

Isolating the cost, all eager, no graphs:

| | ms/token |
|---|---|
| dense resident | 20.9 |
| adaptive, `hot=100%` (no cold path, no sync) | 22.0 |
| adaptive, `hot=90%` (rare cold fetches) | 28.1 |
| adaptive, `hot=17%` (coverage-sized) | 36.0 |

Restructuring the MLP costs ~5%. The other 14 ms is 24 host round trips at
~0.25–0.6 ms each — and the transfer they exist to shrink is 0.1 ms. The mask
is a device tensor, the gather is a host operation, so the round trip is not
removable without putting the cold weights back in VRAM.

**Where it does pay: CPU.** No sync, no transfer, and the same mask is a
straight reduction in GEMM work.

| | batch 1 | 4 concurrent |
|---|---|---|
| `--device cpu` | 9.5 tok/s | 28.6 tok/s |
| `--device cpu --adaptive-mlp` | **12.6 tok/s** | **35.9 tok/s** |

It should also pay on links slow enough, or models large enough, that hundreds
of MiB per step is the wall rather than launch overhead, and on platforms where
a synchronisation is not ~0.25 ms — WDDM batches submissions, which is exactly
what makes a per-layer round trip expensive here.

So `--adaptive-mlp` is opt-in, warns on CUDA, and is a reasonable default on
CPU.

### Not attempted

Predicting the mask so `gate_proj` could be skipped too — a learned predictor
(DejaVu-style) needs training, which lives in the conversion project and not in
an inference engine; a low-rank SVD approximation of `gate_proj` needs no
training but changes the model's output, and this engine's contract is that it
reproduces the reference faithfully.

## Hybrid CPU/GPU split

`--gpu-layers N` runs layers `0..N-1` on the GPU and the rest on the CPU. Each
layer's KV slabs live where that layer computes, which is affordable only
because a packed spike block is ~96 KiB — mirroring the block count into host
RAM costs almost nothing, and one `BlockManager` still owns the whole address
space because both sides use the same block count.

The hidden state crosses at the boundary: `[T, 1, n_tok, D]` is ~460 KiB at
T=4, n_tok=64, bf16, twice per step. Negligible next to the CPU layers
themselves.

CUDA graphs are unavailable: a capture cannot leave its stream mid-way to run on
the CPU. Capturing just the GPU prefix as a segment is possible and is not
implemented — on this checkpoint the CPU layers dominate so completely that it
would recover a small fraction of the step, but on a machine with more cores it
would be worth having.

Hybrid is slower than streaming for the same VRAM saving on this hardware. It is
the right tool when host RAM is plentiful and the link is slow — the opposite of
the regime streaming wants.

## Pure CPU

```bash
spikeinfer serve ./qwen-spiking --device cpu --cpu-threads 8
```

The KV cache is sized against available host RAM (`--cpu-kv-gb`, or
`--cpu-memory-utilization` of what is free). Decode attention runs
`kernels/spike_attention_cpu.py`, which keeps the cache bit-packed but unpacks
one bounded chunk at a time for compute: on a CPU, BLAS beats a vectorised
popcount by ~2× at every shape measured, which is the opposite of the GPU
trade-off and is why the Triton kernel's approach is not simply reused.

Expect single-digit tok/s for 0.5B. A T=4 spiking model does 4× the FLOPs of the
dense model it was converted from, and no amount of kernel work changes that.
The value of CPU mode is running without a GPU at all, and making the engine's
logic — scheduling, paging, chunked prefill, preemption — testable in CI, which
it was not before: 301 tests now run without CUDA, up from 128.
