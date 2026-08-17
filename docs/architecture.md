# Architecture

How a request becomes tokens, and why each piece is shaped the way it is.

## The shape of the problem

A spiking transformer replaces the pointwise nonlinearities of a dense one with LIF
neurons unrolled over `T` timesteps. In this architecture the neurons sit on the outputs
of `q_proj`, `k_proj`, `v_proj` and the MLP gate; the input embedding is replayed
identically at every timestep (direct coding) and the readout is the mean over `T`.

Two consequences drive everything else:

1. **`q`, `k`, `v` and the MLP gate are binary.** Nothing downstream of a neuron is a
   float until it hits the next `nn.Linear`.
2. **Everything runs `T` times.** Naively that means `T`× the kernels, `T`× the KV cache,
   and `T`× the launch overhead.

The second is what made spiking LLMs slow. The first is what makes them cheap to serve
once the second is fixed.

## Layer 0: making a single step fast

This layer predates the engine and is still the foundation.

| Piece | What it does | Measured |
|---|---|---|
| `kernels/lif_triton.py` | Runs **all T timesteps in one launch** with the membrane potential in registers, replacing ~10 kernels per neuron site per timestep | 15.3× on the LIF path |
| `modeling_fast.py` | Reorders `for t: for layer:` → `for layer: for t:`, carrying the T residual streams as one `[T, B, S, D]` tensor so every Linear folds T into the GEMM's M dimension — one GEMM instead of T | 4.25× on the 896→4864 projection |
| `graph_runner.py`, `engine/model_runner.py` | CUDA graph capture — the whole decode step replays as one driver submission | 27,859 → 6 CPU-side dispatches |

The loop reorder is the load-bearing trick. It is *mathematically identical*: each layer's
neuron state is touched only by that layer, in the same per-timestep order, and the `T`
residual streams are mutually independent. It is also what makes both the T-batched GEMMs
and the single-launch LIF kernel possible — you cannot fuse a recurrence you cannot see
all of at once.

### Why the intuitive optimization does not apply

"Binary spikes are being simulated as dense FP32 matmuls" is the obvious diagnosis and it
is wrong here. LIF neurons sit on the **output** of the projections, so no `nn.Linear`
ever consumes a spike tensor. A sparse spike-driven GEMV kernel has nothing to attach to.
INT8 tensor cores measured 0.28× — slower than fp16 — at these shapes.

The spikes *are* exploitable, but in attention, not in the projections. That is layer 1.

## Layer 1: the packed spike cache

`engine/spike_cache.py`, `kernels/packing.py`

Per layer, two `int32 [T, H_kv, num_slots, W]` slabs, where `W = ceil(head_dim / 32)`.
The bit-order convention, which every reader and writer must agree on:

```
bit i of word w  <->  channel  w * 32 + i
```

`num_slots = num_blocks * block_size`. A block is `block_size` consecutive slots, and a
sequence's logical position `i` lives at
`block_table[i // block_size] * block_size + i % block_size`. Putting the slot axis
second-to-last makes a block contiguous for a fixed `(timestep, head)`, which is the
decode kernel's inner-loop access pattern.

Writing is a plain `index_copy_` along the slot axis with the scheduler's slot mapping —
no custom kernel needed.

Packing itself is one Triton launch (`_pack_kernel`). This matters more than it looks: the
decode path packs `q`, `k` and `v` for all 24 layers on every step, and the tensor-op
version costs about seven elementwise kernels each — roughly 500 extra kernels per token,
worth ~0.66 ms/token even inside a CUDA graph.

The last block is **reserved and never allocated**. CUDA-graph decode batches are padded up
to a bucket size, and the padding rows still write a key and a value somewhere; they write
there.

## Layer 2: popcount attention

`kernels/spike_attention.py`

With `q` and `k` binary, the score is exact integer arithmetic:

```
q · k = popcount(q_bits & k_bits)      in [0, head_dim]
```

and with `v` binary, `p · v` is the sum of attention weights whose value bit is set. The
kernel is flash-style: one program per `(sequence, query head, timestep)`, walking the
sequence's blocks with an online softmax so nothing of size `seq_len` is materialised.

Triton 3.7 exposes no `popc` intrinsic, so it uses the SWAR divide-and-conquer popcount.
That is exact on negative inputs too: every intermediate mask (`0x55..`, `0x33..`,
`0x0F..`) has bit 31 clear, so the sign extension an arithmetic `>>` introduces is masked
away before it can pollute a count. Being pure bit ops, it is also portable across Triton
versions.

**Decode only.** Prefill goes through `gather_kv_unpacked` + SDPA:

- decode is bandwidth-bound, so reading 1-bit keys is close to a 32× cut on the dominant
  cost;
- prefill is compute-bound and SDPA already reaches the tensor cores, which CUDA-core
  popcounts cannot match.

Prefill unpacking is uniform: it always reads the sequence's whole cached prefix back out,
which is what makes chunked prefill and a partially-cached sequence the same code path.

There is no `triton.autotune` anywhere, for the same reason in both kernels: autotune
benchmarks candidates by launching kernels, which is illegal during CUDA graph capture.
Block sizes come from fixed heuristics.

## Layer 3: the serving loop

### Batch layout

Every scheduled sequence is flattened into one token axis, so the model sees
`[T, 1, n_tok, D]` and every `nn.Linear` still gets a single GEMM with `M = T * n_tok`.
Sequence boundaries exist only inside attention, carried by `AttentionMetadata`.

Token order is **decodes first, then prefills**, so the decode region is a contiguous
prefix a CUDA graph can replay.

### Scheduling

`engine/scheduler.py` — FCFS with chunked prefill:

- running sequences are considered first, so an admitted request keeps making progress;
- a sequence contributes `min(uncomputed, remaining_budget)` tokens, which collapses
  prefill and decode into one code path — a decode is a one-token chunk that happens to
  reach the end of the sequence;
- when the cache is full, the newest running sequences are preempted (blocks freed, tokens
  requeued for recompute) until the batch fits.

Preemption **recomputes** rather than swapping to host memory. With a packed cache a
32-token block costs ~96 KB across the whole model, so refilling one is cheaper than
moving it over PCIe. A preempted sequence keeps its generated tokens and re-prefills
prompt-plus-output, so preemption is invisible in the output — asserted in
`test_preemption_and_recompute_preserve_output`.

Two subtleties that were bugs first:

- a sequence preempted during a step must not be re-admitted **in that same step**, or the
  scheduler thrashes;
- a sequence that could not fit even in an *empty* cache must be dropped with an error
  rather than preempting everything, failing, preempting itself, and looping forever. Only
  reachable via a hand-set block count, but a server must not livelock.

### CUDA graphs

`engine/model_runner.py` captures one decode graph per batch-size bucket (1, 2, 4, …, 64);
a decode batch is padded up to the next bucket. Capture runs entirely against the reserved
padding block — every static slot points at it and every block table entry is it — so
warmup and capture cannot disturb real sequence data and no save/restore of the cache is
needed. On replay only the live rows are written, so the padding rows keep the values set
at capture time.

Anything with a prefill in it runs eager: prefill shapes vary per step and it is
compute-bound anyway.

`lm_head` sits **outside** the graph. A 4096-token prefill produces 4096 hidden states but
samples one; at vocab 151936 the unprojected rows would be 2.4 GB of fp32 logits, so row
selection happens before the projection.

### Sampling

`engine/sampler.py`. Per-row temperature, top-k, top-p, min-p, penalties, logit bias and
`min_tokens`, applied in the order clients expect. Two things worth noting: an all-greedy
batch skips the probability path entirely (a softmax and multinomial draw over 151936
tokens is not free — 0.24 ms → 0.06 ms), and per-request seeds fall back to a Python loop
because reproducibility must not depend on what else was in the batch.

## Numerical fidelity

The engine is **not bit-exact** against the reference model, and cannot be while keeping
T-batching. Typical relative logit error is ~1.2e-7 and argmax agreement was 100% across
every shape tested; WikiText-2 perplexity moves 390.2473 → 390.4453 (+0.05%).

The divergence is *scheduling, not arithmetic*. A leading `T` dimension changes tensor
shapes, so cuBLAS and PyTorch's reduction kernels pick different algorithms — and
therefore different summation orders — in three places: `nn.Linear` with `M = T·B·S`, the
batched attention matmul with batch `T·B`, and `Qwen2RMSNorm`'s `mean(-1)`. It is
shape-dependent and easy to miss: it vanishes at S=16/32 but appears at S=4–12 and S=15.

Rarely a spike whose membrane lands exactly on its threshold flips — a discrete crossing,
not float noise — producing a localized jump (one observed case hit 6.3e-4 relative). This
is intrinsic to spiking models: injecting 5e-7 of relative noise into a LIF input flips
about 1 spike in 2.5 million. Tests therefore assert argmax agreement plus a loose relative
bound, never a tight per-element one.

What **is** verified bit-exact:

- the fused LIF kernel against snntorch, including a boundary-stress case where the
  membrane sits exactly on the threshold;
- the loop reordering itself, checked in pure PyTorch before any kernel was involved;
- RoPE-by-broadcast against `apply_rotary_pos_emb`;
- bit-packing round-trips;
- attention isolation — a sequence's output does not change when every block it does not
  own is overwritten.

### Two things to know before touching the LIF kernel

1. **FMA contraction breaks bit-exactness.** Triton/ptxas will contract `beta*u + current`
   into a single fused multiply-add (one rounding) while PyTorch eager runs mul and add as
   separate kernels (two roundings). The 1-ulp difference is enough to flip a spike near
   threshold, so the kernel launches with `enable_fp_fusion=False` (ptxas `--fmad=false`).
2. **The reset is delayed by one step.** snntorch computes `r_t = H(U_{t-1} − θ)` *before*
   the membrane update, so `r_t == S_{t-1}`. The comparison is also `(U − θ) > 0`, not
   `U > θ` — in floating point those are not the same predicate. Getting either wrong
   degrades the model silently rather than failing.

## Sizing the cache

The engine measures rather than guesses: it runs one worst-case prefill against a
throwaway cache, records the peak allocation, and gives the rest of the memory budget to
the real cache. The result is then **capped** at what the scheduler could ever use
(`max_num_seqs × max_model_len`), because a packed spike cache is cheap enough that the
memory budget alone would otherwise reserve millions of tokens no workload can fill — 7.8
GB where 192 MB does the job.

## What is deliberately absent

- **Prefix caching / block sharing.** Would need reference counts and copy-on-write in the
  block manager.
- **A packed prefill kernel.** See above — it would lose to SDPA.
- **Swapping to host memory.** Recompute is cheaper for a cache this small.
- **Multi-GPU.** No tensor or pipeline parallelism.
- **Flash-decoding style sequence splitting** in the attention kernel. At batch 1 the
  decode kernel launches `B × H_q × T` programs (56 at the 0.5B config), which
  under-occupies a large GPU at short context. Splitting the sequence across programs with
  a reduction pass would fix it.
