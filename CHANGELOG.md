# Changelog

## 0.3.0

Running a model that does not fit, and running without a GPU at all.

Until this release the engine assumed one thing about placement: that the whole
model fits on one CUDA device. Everything below follows from removing that
assumption, and it turns out to be a single question asked per layer — *where
do this layer's weights live, and where does it compute?*

All numbers here were measured on Qwen2.5-0.5B converted to spiking form, T=4,
RTX 4070 SUPER, Windows/WDDM, torch 2.6.0+cu124.

### Weight offload

`--offload-layers N|auto` keeps a layer's weights in host RAM and streams them
into a small ring of GPU buffers as the forward pass reaches them.

| | resident VRAM | tok/s |
|---|---|---|
| everything resident | 989 MiB | 170.4 |
| `--offload-layers 12` | 872 MiB | 56.2 |
| `--offload-layers 24` | **389 MiB** | 34.9 |
| `--offload-layers 24 --offload-embeddings` | **325 MiB** | 27.1 |

Output is **bit-identical** to running resident — same device, same arithmetic,
only the weights' address changed.

Two things make it fast enough to be worth using:

- Every decoder layer has identical parameter shapes and `layer_idx` is never
  read at runtime, so one physical module can host any logical layer's weights
  in turn, and a layer is **one contiguous buffer** rather than 20 tensors —
  one `cudaMemcpyAsync` per layer instead of 480 per step.
- **It stays inside a captured CUDA graph.** A copy from pinned host memory is
  a legal graph node, so the step still replays as a single launch while the
  copies overlap compute. Capture is attempted and falls back to eager with a
  warning if the driver refuses.

### Pure CPU mode

`--device cpu` now works end to end, with the KV cache sized against available
host RAM (`--cpu-kv-gb`, `--cpu-memory-utilization`) and `--cpu-threads`.

New `kernels/spike_attention_cpu.py` replaces the per-sequence Python loop in
the decode fallback with a batched, memory-bounded one — 2.7× at a 8-way decode
batch. It keeps the cache bit-packed but unpacks a bounded chunk for compute:
on a CPU, BLAS beats a vectorised popcount by ~2× at every shape measured,
which is the opposite of the GPU trade-off.

Expect single-digit tok/s for 0.5B. A T=4 spiking model does 4× the FLOPs of
the dense model it came from, and no kernel work changes that; the point of CPU
mode is running without a GPU, and making the engine testable without one.

### Hybrid CPU/GPU split

`--gpu-layers N` runs layers `0..N-1` on the GPU and the rest on the CPU, in
the style of llama.cpp's `-ngl`. Each layer's KV slabs live where that layer
computes, which is affordable only because a packed spike block is ~96 KiB, so
one `BlockManager` still owns the whole address space.

Not bit-identical, and cannot be: CPU and GPU reduce in different orders, and a
1-ulp difference flips a spike sitting exactly on its threshold. Measured on
the 0.5B checkpoint, `--gpu-layers 18/12/0` reproduce the resident engine's
greedy tokens exactly anyway, but the claim is agreement, not equality.

CUDA graphs are unavailable in this mode — a capture cannot leave its stream
mid-way to run on the CPU. Capturing just the GPU prefix as a segment is
possible and is not implemented.

### Sparsity-aware offload (`--adaptive-mlp`)

The README used to say LIF neurons sit on the *output* of `q/k/v/gate_proj`, so
no `nn.Linear` ever consumes a spike tensor. That is true for attention. It is
**not** true for the MLP, which is 87.6% of a decoder layer's parameters:

```python
gate_spk = lif_multistep(self.gate_proj(x), beta, threshold)   # binary
return self.down_proj(gate_spk * self.up_proj(x))              # zero where the gate is zero
```

So `up_proj`'s rows and `down_proj`'s columns are dead weight for a token whose
channel stays silent. New `spikeinfer spike-stats` measures the rate that
actually matters — the union over T, not the per-timestep rate — and on the
calibrated checkpoint it is **5.5%** at batch 1.

`--adaptive-mlp` exploits it: channels are permuted by firing frequency at load
time so the hot set is a contiguous resident slice, and the tail is fetched
from pinned host memory only when it fires. Per-step transfer drops from
683 MiB to **3.8 MiB** (180×).

**And it still loses on this hardware**, which is documented rather than
hidden. Isolating the cost, all eager:

| | ms/token |
|---|---|
| dense resident | 20.9 |
| adaptive, `hot=100%` (no cold path, no sync) | 22.0 |
| adaptive, `hot=17%` (coverage-sized) | 36.0 |

Restructuring the MLP costs ~5%; the other 14 ms is 24 host round trips at
~0.25–0.6 ms each, and the transfer they exist to shrink is 0.1 ms. The mask is
a device tensor and the gather is a host operation, so the round trip also
rules out CUDA graph capture.

Where it does pay is **CPU**, where there is no sync and no transfer and the
same mask is a straight reduction in GEMM work: 9.5 → **12.6 tok/s** at batch 1,
28.6 → **35.9 tok/s** at four concurrent. It is opt-in, and warns on CUDA.

Full write-up, including what was deliberately not attempted, in
[docs/placement.md](docs/placement.md).

### Five new commands

| | |
|---|---|
| `spikeinfer doctor` | torch/CUDA/Triton, `cl.exe` on PATH, `CUDA_HOME` against the torch build, host RAM, measured PCIe bandwidth, and a live kernel compile |
| `spikeinfer plan <model>` | what fits on this machine and what it costs, from the config alone — no weights loaded |
| `spikeinfer validate <model>` | the correctness gates against a real checkpoint and a real placement |
| `spikeinfer profile <model>` | where a decode step's time goes, and whether it is launch-, bandwidth- or sync-bound |
| `spikeinfer spike-stats <model>` | gate firing rates — decides whether `--adaptive-mlp` pays |

### Testing

**301 of the tests run without a GPU, up from 128.** The engine tests —
scheduling, paging, chunked prefill, preemption, sampling — used to skip
themselves without CUDA even though none of that is device-specific, so CI was
not checking the engine at all. Same for the server tests, the loader tests and
84 fast-vs-reference equivalence tests. `SPIKEINFER_TEST_DEVICE=cpu` reproduces
the CI configuration on a GPU machine.

390 tests total, up from 330. New: `tests/test_placement.py` (the plan, flat
buffers, streaming, hybrid, the adaptive MLP) and `tests/test_cpu_backend.py`
(the CPU attention path against the reference oracle).

### Fixes

- `torch.cuda.is_available()` returns `True` while `device_count()` is 0 under
  `CUDA_VISIBLE_DEVICES=""` on torch 2.6.0+cu124/Windows, so anything trusting
  it walked into "Invalid device id". `sysinfo.cuda_is_usable()` checks both,
  and `default_device()` and `resolve_dtype()` now use it — masking devices is
  the standard way to force a CPU run and it has to work.
- `LLMEngine._size_cache()` on a non-CUDA device returned a placeholder block
  count instead of sizing against anything.
- `compute_logits` and the sampler assumed the hidden state and the indices
  were on the same device.

### Compatibility

No breaking changes. Every new option defaults to the previous behaviour, and
the engine builds no placement machinery at all unless one is requested.
`EngineConfig` gains a `devices: DeviceConfig` field; `EngineConfig.device` is
unchanged.

## 0.2.0

Initial release.
