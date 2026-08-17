"""GPU timing harness for SpikingQwen (SNN) vs. the reference Qwen2.5-0.5B
(ANN) forward pass, across a few fixed workload shapes.

Naive timing code uses bare
time.time() around a single un-warmed-up call with no
torch.cuda.synchronize() -- on an async CUDA queue under Windows WDDM this
mostly measures kernel-launch queuing time, not actual GPU execution, and a
single sample says nothing about the (large, WDDM-driver-induced) tail
latency. This harness instead: runs warmup iterations, synchronizes around
each individually-timed iteration, and reports median + p10/p90 rather than
a single number or a plain mean.

Run from the repo root, after `call tools\\env.bat` (needed if you plan to
let Triton/inductor JIT anything -- this script itself does not use
torch.compile, which is broken in this environment: triton 3.7 here is
missing AttrsDescriptor):

    .venv\\Scripts\\python.exe bench\\benchmark.py --configs decode --compare-ann

This harness can measure either implementation via ``--model``:

    --model reference   spikeinfer.reference.SpikingQwenForCausalLM (the
                         unoptimized, python-loop, snntorch-based model)
    --model engine       spikeinfer.FastSpikingQwenForCausalLM (the fused,
                         T-batched, Triton-kernel engine) -- the default,
                         since this is an engine repo.

``--sdpa`` switches the engine's attention to F.scaled_dot_product_attention.
``--graph`` captures the engine's fixed-shape forward in a CUDA graph and
times replays only (capture cost is reported separately, once per config, as
"capture: N ms"). ``--compare-reference`` runs both implementations in one
invocation and reports the speedup.

Known-good reference numbers (sanity check for this harness itself): random
weights, fp32, T=4, RTX 4070 SUPER --
    reference decode (B=1,S=1)   ~140-148 ms   engine decode (eager) ~17-24 ms   engine+graph ~6.6 ms
    reference prefill (B=1,S=128) ~143 ms      engine prefill        ~30 ms
    reference batch  (B=4,S=128) ~159 ms       engine batch          ~109 ms
SNN decode is not much cheaper than SNN prefill despite S=1 vs S=128 --
that's expected here: the reference model has no KV cache, and at T=4
timesteps x 24 layers x several LIF neurons per layer, per-step overhead
(kernel launch count) dominates over the FLOPs difference from sequence
length. That's the whole reason bench/profile_kernels.py exists: kernel-launch
count, not raw FLOPs, is the bottleneck this project is trying to drive down
-- and the engine's whole reason for existing is to collapse that count.

Note on --dtype bfloat16: the reference LIF neuron (snntorch Leaky)
always emits float32 spike/membrane tensors regardless of its input
dtype, so a plain model.to(bfloat16) crashes at the first bf16-weight
Linear layer with a dtype mismatch. Since spikeinfer/reference/ is out of scope
here, this harness works around it by running the timed forward pass
under torch.autocast(dtype=bfloat16) instead of relying on strict tensor
dtypes to already agree -- see autocast_ctx() below. The engine's fused LIF
kernel does not have this quirk (it respects the input dtype), so autocast
is a no-op safety net for it rather than a required workaround.
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

from spikeinfer import FastSpikingQwenForCausalLM, GraphedForward, validate_thresholds
from spikeinfer.reference import SpikingQwenConfig, SpikingQwenForCausalLM

# Qwen2.5-0.5B base hyperparameters, used to build a random-weight model
# when --ckpt is not given (SpikingQwenConfig adds T / beta_init /
# threshold_init on top of these).
BASE_CFG_KWARGS = dict(
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    max_position_embeddings=32768,
    rope_theta=1e6,
    tie_word_embeddings=True,
)

CONFIGS = OrderedDict(
    [
        ("prefill", dict(B=1, S=128)),
        ("decode", dict(B=1, S=1)),
        ("batch", dict(B=4, S=128)),
    ]
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

MODEL_CLASSES = {"reference": SpikingQwenForCausalLM, "engine": FastSpikingQwenForCausalLM}


def autocast_ctx(device, dtype):
    """Context manager for the timed forward call.

    For bfloat16 this is not merely an optimization: SpikingQwen's LIF
    neuron (snntorch Leaky) always emits float32 spike/membrane tensors
    regardless of its input dtype (verified empirically), so a plain
    model.to(dtype=bfloat16) crashes downstream at the first bf16-weight
    Linear layer with a dtype-mismatch RuntimeError. Wrapping the forward
    pass in torch.autocast makes every Linear/matmul op autocast its
    operands consistently, independent of that float32 leakage, which
    avoids the crash. This is a bench/-side workaround for a modeling
    quirk in spikeinfer/reference/lif.py -- that file is out of scope here."""
    if device == "cuda" and dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device if device == "cuda" else "cpu", enabled=False)


def benchmark(fn, iters, warmup):
    """Time `fn()` (a zero-arg callable that runs one forward pass).

    Does `warmup` untimed calls first, then times `iters` calls
    individually, synchronizing the CUDA stream immediately before and
    after each timed call so async kernel-launch queuing doesn't leak into
    (or out of) the measured window. Returns median/p10/p90/mean in
    milliseconds plus the raw per-iteration samples.
    """
    is_cuda = torch.cuda.is_available()

    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    samples_ms = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    arr = np.array(samples_ms)
    p10, median, p90 = np.percentile(arr, [10, 50, 90])
    return {
        "median_ms": float(median),
        "p10_ms": float(p10),
        "p90_ms": float(p90),
        "mean_ms": float(arr.mean()),
        "raw_ms": samples_ms,
    }


def ann_config_from_snn(snn_cfg):
    """Build a plain Qwen2Config with matching architecture hyperparameters
    to the given SpikingQwenConfig, so the ANN comparison model is exactly
    the dense equivalent of whatever SNN config was actually used (random
    default, or one loaded from a checkpoint)."""
    return Qwen2Config(
        vocab_size=snn_cfg.vocab_size,
        hidden_size=snn_cfg.hidden_size,
        intermediate_size=snn_cfg.intermediate_size,
        num_hidden_layers=snn_cfg.num_hidden_layers,
        num_attention_heads=snn_cfg.num_attention_heads,
        num_key_value_heads=snn_cfg.num_key_value_heads,
        max_position_embeddings=snn_cfg.max_position_embeddings,
        rope_theta=snn_cfg.rope_theta,
        tie_word_embeddings=snn_cfg.tie_word_embeddings,
    )


def load_model(model_kind, ckpt_path, T, device, dtype):
    """model_kind: 'reference' or 'engine'. Same constructor / state_dict keys
    either way, so this differs from the original single-model loader only in
    which class gets built, plus a validate_thresholds() call for the engine
    (cheap, and required precondition documented in spikeinfer/__init__.py)."""
    cls = MODEL_CLASSES[model_kind]
    if ckpt_path:
        # A spikeinfer model directory, not a pickled .pt: unpickling executes
        # code from the file. Convert legacy checkpoints once with
        # `spikeinfer convert --from-checkpoint`.
        from safetensors.torch import load_file

        from spikeinfer.loader import load_config

        cfg = load_config(ckpt_path)
        model = cls(cfg)
        state = load_file(str(Path(ckpt_path) / "model.safetensors"), device="cpu")
        if cfg.tie_word_embeddings and "lm_head.weight" not in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        model.load_state_dict(state)
        mode = f"checkpoint weights ({ckpt_path})"
    else:
        cfg = SpikingQwenConfig(T=T, beta_init=0.9, threshold_init=1.0, **BASE_CFG_KWARGS)
        model = cls(cfg)
        mode = "RANDOM weights (no --ckpt given)"
    model = model.to(device=device, dtype=dtype).eval()
    if model_kind == "engine":
        validate_thresholds(model)
    return model, cfg, mode


def reset_peak_mem(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_mem_mb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def build_forward_fn(kind, model, cfg, device, dtype, T, use_sdpa):
    """B, S -> zero-arg callable that runs one eager forward pass.

    kind is 'reference', 'engine', or 'ann'; each has a slightly different
    call signature (reference has no use_sdpa kwarg, ann takes use_cache
    instead of T)."""

    def model_fn(B, S):
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)

        def _run():
            with torch.no_grad(), autocast_ctx(device, dtype):
                if kind == "reference":
                    model(input_ids, T=T)
                elif kind == "ann":
                    model(input_ids, use_cache=False)
                else:  # engine
                    model(input_ids, T=T, use_sdpa=use_sdpa)

        return _run

    return model_fn


def run_configs(model_fn, config_names, device, iters, warmup):
    """model_fn(B, S) -> zero-arg callable that runs one forward pass.
    Returns {config_name: {timing..., peak_mem_mb}}."""
    out = {}
    for name in config_names:
        shape = CONFIGS[name]
        B, S = shape["B"], shape["S"]
        reset_peak_mem(device)
        fn = model_fn(B, S)
        stats = benchmark(fn, iters, warmup)
        stats["peak_mem_mb"] = peak_mem_mb(device)
        stats["B"], stats["S"] = B, S
        out[name] = stats
        print(
            f"    {name:8s} B={B} S={S}  median={stats['median_ms']:8.2f}ms "
            f"p10={stats['p10_ms']:8.2f}ms p90={stats['p90_ms']:8.2f}ms  "
            f"peak_vram={stats['peak_mem_mb']:7.1f}MB"
        )
    return out


def run_configs_graph(model, cfg, config_names, device, dtype, T, use_sdpa, iters, warmup):
    """Like run_configs, but the forward pass is a CUDA-graph replay.

    Per config: first `warmup` EAGER calls (untimed) so Triton JIT-compiles
    its kernels -- compiling during capture is illegal (it launches its own
    benchmark kernels), so this must happen before GraphedForward() is built.
    Then GraphedForward(...) itself is built/captured -- also untimed except
    for a single wall-clock measurement of the whole call, reported
    separately as "capture: N ms" and excluded from the replay stats. The
    timed loop that follows measures replay-only cost (benchmark(..., warmup=0),
    since the graph is already warm by construction).
    """
    out = {}
    for name in config_names:
        shape = CONFIGS[name]
        B, S = shape["B"], shape["S"]
        reset_peak_mem(device)
        input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)

        with torch.no_grad(), autocast_ctx(device, dtype):
            for _ in range(warmup):
                model(input_ids, T=T, use_sdpa=use_sdpa)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        graphed = GraphedForward(model, B, S, T=T, use_sdpa=use_sdpa, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - t0) * 1000.0

        def _run(graphed=graphed, input_ids=input_ids):
            graphed(input_ids)

        stats = benchmark(_run, iters, warmup=0)
        stats["peak_mem_mb"] = peak_mem_mb(device)
        stats["B"], stats["S"] = B, S
        stats["capture_ms"] = capture_ms
        out[name] = stats
        print(
            f"    {name:8s} B={B} S={S}  median={stats['median_ms']:8.2f}ms "
            f"p10={stats['p10_ms']:8.2f}ms p90={stats['p90_ms']:8.2f}ms  "
            f"peak_vram={stats['peak_mem_mb']:7.1f}MB  capture: {capture_ms:.2f} ms"
        )
    return out


def row_key_for(kind, graph, sdpa):
    if kind != "engine":
        return kind
    suffix = "+".join(s for s, flag in (("graph", graph), ("sdpa", sdpa)) if flag)
    return f"{kind}+{suffix}" if suffix else kind


def print_summary(results, config_names, row_keys, speedup_pairs):
    print("\n=== Summary ===")
    header = f"{'config':10s}"
    for rk in row_keys:
        header += f" {rk + ' median':>14s} {rk + ' p10-p90':>18s}"
    for label, _, _ in speedup_pairs:
        header += f" {label:>10s}"
    header += f" {row_keys[0] + ' VRAM':>12s}"
    print(header)
    print("-" * len(header))
    for name in config_names:
        c = results["configs"][name]
        row = f"{name:10s}"
        for rk in row_keys:
            st = c[rk]
            rng = f"[{st['p10_ms']:.1f}-{st['p90_ms']:.1f}]"
            row += f" {st['median_ms']:12.2f}ms {rng:>18s}"
        for label, num_key, den_key in speedup_pairs:
            num, den = c[num_key]["median_ms"], c[den_key]["median_ms"]
            ratio = num / den if den else float("inf")
            c[num_key].setdefault("ratios", {})[label] = ratio
            row += f" {ratio:9.2f}x"
        row += f" {c[row_keys[0]]['peak_mem_mb']:11.1f}MB"
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default="prefill,decode,batch", help="comma-separated subset of: " + ",".join(CONFIGS))
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default="engine",
                         help="which SpikingQwen implementation to benchmark (default: engine)")
    parser.add_argument("--sdpa", action="store_true",
                         help="engine only: use F.scaled_dot_product_attention instead of the manual matmul/softmax path")
    parser.add_argument("--graph", action="store_true",
                         help="engine only: capture the fixed-shape forward pass in a CUDA graph, built once "
                              "before the timing loop, and time replays only (capture cost reported separately)")
    parser.add_argument("--compare-ann", action="store_true")
    parser.add_argument("--compare-reference", action="store_true",
                         help="benchmark BOTH --model reference and --model engine in this invocation "
                              "(engine still respects --sdpa/--graph) and report the speedup per config")
    parser.add_argument("--ckpt", default=None, help="path to a spikeinfer model directory; omit to use random weights")
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    parser.add_argument("--json", default=None, help="optional path to dump results as JSON")
    parser.add_argument("--label", default="unlabeled", help="tag stored in the JSON output, for diffing runs across optimization stages")
    args = parser.parse_args()

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in config_names:
        if c not in CONFIGS:
            parser.error(f"unknown config {c!r}, must be one of {list(CONFIGS)}")

    if not args.compare_reference and args.model == "reference":
        if args.sdpa:
            parser.error("--sdpa only applies to --model engine (the reference implementation has no SDPA path)")
        if args.graph:
            parser.error("--graph only applies to --model engine (CUDA graph capture is not supported for the reference implementation)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available -- timings will not be meaningful for this project's target hardware.")
    dtype = DTYPES[args.dtype]

    print(
        f"device={device}  dtype={args.dtype}  T={args.T}  iters={args.iters}  warmup={args.warmup}  "
        f"model={args.model}  sdpa={args.sdpa}  graph={args.graph}  "
        f"compare_reference={args.compare_reference}  compare_ann={args.compare_ann}  label={args.label!r}"
    )

    results = {
        "label": args.label,
        "T": args.T,
        "dtype": args.dtype,
        "iters": args.iters,
        "warmup": args.warmup,
        "model": args.model,
        "sdpa": args.sdpa,
        "graph": args.graph,
        "compare_reference": args.compare_reference,
        "compare_ann": args.compare_ann,
        "ckpt": args.ckpt,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "configs": {},
    }

    row_keys = []
    reference_row_key = None
    engine_row_key = None
    primary_cfg = None

    def record(name_stats, rk, kind, graph, sdpa):
        for name in config_names:
            s = name_stats[name]
            s["model"], s["graph"], s["sdpa"] = kind, graph, sdpa
            results["configs"].setdefault(name, {})[rk] = s

    if args.compare_reference:
        print("\n=== SpikingQwen reference ===")
        ref_model, ref_cfg, ref_mode = load_model("reference", args.ckpt, args.T, device, dtype)
        print(f"  weight mode: {ref_mode}")
        # NOTE: build_forward_fn(...) is passed inline (not bound to a local
        # name) so its closure -- which captures the model by reference --
        # doesn't outlive this call. If it were kept in a variable here, that
        # variable would keep the model alive after `del ref_model` below,
        # `torch.cuda.empty_cache()` would then free nothing, and the next
        # model built would sit on GPU *alongside* this one instead of after it.
        ref_stats = run_configs(
            build_forward_fn("reference", ref_model, ref_cfg, device, dtype, args.T, False),
            config_names, device, args.iters, args.warmup,
        )
        reference_row_key = "reference"
        row_keys.append(reference_row_key)
        record(ref_stats, reference_row_key, "reference", False, False)
        primary_cfg = ref_cfg
        del ref_model
        if device == "cuda":
            torch.cuda.empty_cache()

        print("\n=== spikeinfer engine ===")
        eng_model, eng_cfg, eng_mode = load_model("engine", args.ckpt, args.T, device, dtype)
        print(f"  weight mode: {eng_mode}")
        engine_row_key = row_key_for("engine", args.graph, args.sdpa)
        if args.graph:
            eng_stats = run_configs_graph(eng_model, eng_cfg, config_names, device, dtype, args.T, args.sdpa, args.iters, args.warmup)
        else:
            eng_stats = run_configs(
                build_forward_fn("engine", eng_model, eng_cfg, device, dtype, args.T, args.sdpa),
                config_names, device, args.iters, args.warmup,
            )
        row_keys.append(engine_row_key)
        record(eng_stats, engine_row_key, "engine", args.graph, args.sdpa)
        primary_cfg = eng_cfg
        del eng_model
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        kind = args.model
        label = "SpikingQwen reference" if kind == "reference" else "spikeinfer engine"
        print(f"\n=== {label} ===")
        model, cfg, mode = load_model(kind, args.ckpt, args.T, device, dtype)
        print(f"  weight mode: {mode}")
        rk = row_key_for(kind, args.graph, args.sdpa)
        if args.graph:
            stats = run_configs_graph(model, cfg, config_names, device, dtype, args.T, args.sdpa, args.iters, args.warmup)
        else:
            stats = run_configs(
                build_forward_fn(kind, model, cfg, device, dtype, args.T, args.sdpa),
                config_names, device, args.iters, args.warmup,
            )
        row_keys.append(rk)
        record(stats, rk, kind, args.graph, args.sdpa)
        primary_cfg = cfg
        if kind == "reference":
            reference_row_key = rk
        else:
            engine_row_key = rk
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- ANN (optional) ---
    if args.compare_ann:
        print("\n=== Qwen2ForCausalLM (ANN, dense reference) ===")
        ann_cfg = ann_config_from_snn(primary_cfg)
        ann_model = Qwen2ForCausalLM(ann_cfg).to(device=device, dtype=dtype).eval()
        print("  weight mode: RANDOM weights (architecture-matched dense reference)")
        ann_stats = run_configs(
            build_forward_fn("ann", ann_model, ann_cfg, device, dtype, args.T, False),
            config_names, device, args.iters, args.warmup,
        )
        row_keys.append("ann")
        record(ann_stats, "ann", "ann", False, False)
        del ann_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- summary table ---
    speedup_pairs = []
    if reference_row_key and engine_row_key:
        speedup_pairs.append(("speedup", reference_row_key, engine_row_key))
    if args.compare_ann:
        baseline_row = engine_row_key or reference_row_key
        speedup_pairs.append(("vs ANN", baseline_row, "ann"))

    print_summary(results, config_names, row_keys, speedup_pairs)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
