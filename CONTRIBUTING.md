# Contributing

## Getting set up

```sh
git clone https://github.com/HzAybars/Spikeinfer.git
cd spikeinfer
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,reference]"
```

Triton is not a dependency — the package falls back to pure PyTorch without it,
just slowly. To get the fast paths:

- **Linux:** already inside the torch wheel.
- **Windows:** `pip install triton-windows`. Triton JIT-compiles through MSVC, so
  it needs `cl.exe` on `PATH` and a `CUDA_HOME` matching the CUDA your torch was
  built against — a machine can easily have a newer `nvcc` on `PATH` than torch
  expects, which breaks the build. `tools/env.bat` sets both:

  ```
  cmd /c "tools\env.bat && python -m pytest tests -q"
  ```

## Running the tests

```sh
pytest tests -q
```

330 tests. 128 run anywhere; the other 202 need CUDA and skip themselves when no
device is present, which is why CI is green without proving much. **Run the full
suite on a GPU before opening a PR** if you touched anything under
`spikeinfer/kernels/`, `spikeinfer/attention.py`, `spikeinfer/engine/`, or the
model.

## Before you open a PR

- `ruff check .`
- Keep `tests/test_engine.py::test_greedy_matches_the_eager_path` green. It
  compares the entire serving stack — paged cache, bit-packed spikes, popcount
  attention, continuous batching, CUDA graphs — against the simple eager path,
  and it is the test that catches "fast but subtly wrong".
- **Never optimize `spikeinfer/reference/`.** It is the unoptimized snntorch
  implementation everything else is validated against; its only job is to be
  obviously correct. Optimize elsewhere and prove equivalence against it.
- New kernels need a pure-PyTorch oracle next to them, the way
  `paged_spike_attn_decode_ref` sits beside the Triton kernel. Tests compare the
  two; without an oracle a kernel is untestable.
- Spiking models are unusually sensitive to 1-ulp changes: a membrane potential
  sitting exactly on its threshold flips a discrete spike, which is why the LIF
  kernel launches with `enable_fp_fusion=False`. If a "harmless" numerical
  refactor changes output tokens, that is the reason.

## Project layout

- `spikeinfer/` — the package. `engine/` is the serving loop (scheduler, block
  manager, model runner, sampler); `kernels/` holds the Triton kernels and
  bit-packing; `entrypoints/` is the CLI and HTTP server; `reference/` is the
  golden reference.
- `tests/` — equivalence tests first, unit tests second.
- `bench/` — kernel-level timing and CUDA launch counting.
- `docs/` — architecture and conversion notes.
- `examples/` — runnable scripts.
- `tools/` — `env.bat`, the Windows Triton toolchain setup.

## Reporting a bug

Include:

- GPU model and driver, `torch.__version__`, `triton.__version__`, OS
- the output of `spikeinfer info <model-dir>`
- the engine settings you ran with (`--max-num-seqs`, `--block-size`, `--dtype`)
- whether it also happens with `--no-cuda-graph`, and with `--dtype float32`

Those last two split the search space in half immediately: a bug that vanishes
without CUDA graphs is a capture/replay problem, and one that vanishes in fp32
is a precision problem.
