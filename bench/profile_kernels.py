"""Kernel-launch-count profiler for SpikingQwen forward passes.

The whole point of this project is to cut launch-overhead: on Windows/WDDM,
async CUDA kernel launches queue through the driver one at a time and each
one carries real, fixed CPU-side dispatch overhead, so a model built out of
many small ops (LIF neurons unrolled over T timesteps, in this case) pays
for that overhead T x num_layers x num_LIF_sites times per forward pass.
The headline metric this script reports -- CUDA kernel launch count -- is
what that overhead scales with; the project's target is to drive it from a
baseline of ~27,859 (reference, decode, T=4) down under 1,000.

This script can profile either implementation via ``--model``:

    --model reference   spikeinfer.reference.SpikingQwenForCausalLM
    --model engine       spikeinfer.FastSpikingQwenForCausalLM -- the default.

``--sdpa`` switches the engine's attention to F.scaled_dot_product_attention.
``--graph`` builds+captures a CUDA graph for the fixed-shape forward BEFORE
profiling and profiles a single replay, not the capture (capture issues its
own warmup/recording kernels that would otherwise pollute the launch count).
``--compare`` profiles reference, engine (eager), and engine+graph in one run
and prints their launch counts side by side -- the single clearest
illustration of what this project buys, since each stage should show a large
drop: ~28k -> ~5.7k -> ~21.

Measurement note: true per-kernel CUPTI device-side tracing is not
available in this environment -- verified by exporting a chrome trace and
finding zero "kernel"-category events, only "cpu_op" ones (this is a known
Windows/WDDM CUPTI limitation, not a bug in this script). What IS reliably
available is CUDA-event-based GPU time attribution per CPU-side op
(torch.profiler's self_device_time_total), because that only needs
cudaEventRecord, not privileged GPU performance counters. So "kernel
launch count" here is the count of profiled CPU-side operator dispatches
in the profiled region (torch.profiler.profile(...).events()) -- on this
model, almost every one of those directly issues CUDA work (self GPU time
> 0 for ~99.5% of them in spot checks), so it's a faithful proxy for the
actual number of cudaLaunchKernel calls, and it lines up with the
project's existing ~27,859 baseline figure to within ~1%.

Usage (from repo root, after `call tools\\env.bat` if you need Triton/cl):
    .venv\\Scripts\\python.exe bench\\profile_kernels.py --config decode
    .venv\\Scripts\\python.exe bench\\profile_kernels.py --config decode --compare
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spikeinfer import FastSpikingQwenForCausalLM, GraphedForward, validate_thresholds
from spikeinfer.reference import SpikingQwenConfig, SpikingQwenForCausalLM

# Same Qwen2.5-0.5B base hyperparameters as bench/benchmark.py.
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

TARGET_KERNEL_LAUNCHES = 1000

MODEL_CLASSES = {"reference": SpikingQwenForCausalLM, "engine": FastSpikingQwenForCausalLM}


def build_model(model_kind, T, device):
    cfg = SpikingQwenConfig(T=T, beta_init=0.9, threshold_init=1.0, **BASE_CFG_KWARGS)
    model = MODEL_CLASSES[model_kind](cfg).to(device=device).eval()
    if model_kind == "engine":
        validate_thresholds(model)
    return model, cfg


def build_forward_callable(model_kind, model, input_ids, T, use_sdpa, device, use_graph, pre_capture_warmup=3):
    """Returns a zero-arg callable to profile.

    For use_graph=True, the CUDA graph is built+captured HERE, outside the
    profiled region -- capture itself launches its own warmup and recording
    kernels, which would otherwise be counted as if they were part of a
    steady-state forward pass. `pre_capture_warmup` eager calls run first so
    Triton JIT-compiles before capture (compiling mid-capture is illegal:
    autotune/JIT launches benchmark kernels).
    """
    if use_graph:
        B, S = input_ids.shape
        with torch.no_grad():
            for _ in range(pre_capture_warmup):
                model(input_ids, T=T, use_sdpa=use_sdpa)
        if device == "cuda":
            torch.cuda.synchronize()
        graphed = GraphedForward(model, B, S, T=T, use_sdpa=use_sdpa, device=device)

        def fn():
            with torch.no_grad():
                graphed(input_ids)

        return fn

    def fn():
        with torch.no_grad():
            if model_kind == "reference":
                model(input_ids, T=T)
            else:
                model(input_ids, T=T, use_sdpa=use_sdpa)

    return fn


def profile_one_forward(fn, warmup, device):
    """Profile a single call to `fn()`. Runs `warmup` untimed/unprofiled
    calls first (so Triton JIT-compiles before the profiled call -- moot for
    a graph replay, which is already warm, but harmless), then wraps exactly
    one more call in torch.profiler."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        if device == "cuda":
            torch.cuda.synchronize()

    return prof


def summarize(prof, top_n):
    events = prof.events()
    kernel_launch_count = len(events)

    key_avgs = prof.key_averages()
    total_self_gpu_us = sum(e.self_device_time_total for e in key_avgs)
    total_self_cpu_us = sum(e.self_cpu_time_total for e in key_avgs)

    ranked = sorted(key_avgs, key=lambda e: e.self_device_time_total, reverse=True)
    top_ops = []
    for e in ranked[:top_n]:
        pct = (e.self_device_time_total / total_self_gpu_us * 100.0) if total_self_gpu_us else 0.0
        top_ops.append(
            {
                "name": e.key,
                "count": e.count,
                "self_gpu_ms": e.self_device_time_total / 1000.0,
                "self_cpu_ms": e.self_cpu_time_total / 1000.0,
                "pct_self_gpu": pct,
            }
        )

    return {
        "kernel_launch_count": kernel_launch_count,
        "total_self_gpu_ms": total_self_gpu_us / 1000.0,
        "total_self_cpu_ms": total_self_cpu_us / 1000.0,
        "distinct_ops": len(key_avgs),
        "top_ops": top_ops,
    }


def mode_str(model_kind, graph, sdpa):
    mode = model_kind
    if model_kind == "engine":
        if graph:
            mode += "+graph"
        if sdpa:
            mode += "+sdpa"
    return mode


def print_report(config_name, args, summary):
    mode = mode_str(args.model, args.graph, args.sdpa)
    print(f"\n=== profile_kernels: config={config_name}  T={args.T}  model={mode} ===")
    print(f"Total CUDA kernel launch count: {summary['kernel_launch_count']:,}   (target: < {TARGET_KERNEL_LAUNCHES:,})")
    if summary["kernel_launch_count"] <= TARGET_KERNEL_LAUNCHES:
        print("  -> at or under target.")
    else:
        factor = summary["kernel_launch_count"] / TARGET_KERNEL_LAUNCHES
        print(f"  -> {factor:.1f}x over target.")
    print(f"Total self GPU time: {summary['total_self_gpu_ms']:.3f} ms   (self CPU time: {summary['total_self_cpu_ms']:.3f} ms, {summary['distinct_ops']} distinct ops)")

    print(f"\nTop {len(summary['top_ops'])} ops by self GPU time:")
    header = f"{'op':30s} {'count':>8s} {'self GPU ms':>12s} {'% self GPU':>11s} {'self CPU ms':>12s}"
    print(header)
    print("-" * len(header))
    for op in summary["top_ops"]:
        print(f"{op['name'][:30]:30s} {op['count']:8d} {op['self_gpu_ms']:12.3f} {op['pct_self_gpu']:10.2f}% {op['self_cpu_ms']:12.3f}")


def run_compare(args, device):
    """Profile reference eager, engine eager, and engine+graph replay in one
    invocation and print launch counts side by side -- the single clearest
    illustration of what this project buys. Never holds two models on GPU at
    once (each fp32 0.5B model is ~2GB; only ~6GB VRAM is free)."""
    shape = CONFIGS[args.config]
    variants = [
        ("reference", "reference", False, False),
        ("engine (eager)", "engine", False, args.sdpa),
        ("engine + CUDA graph", "engine", True, args.sdpa),
    ]

    rows = []
    for label, kind, use_graph, use_sdpa in variants:
        print(f"\n--- profiling: {label} ---")
        model, cfg = build_model(kind, args.T, device)
        input_ids = torch.randint(0, cfg.vocab_size, (shape["B"], shape["S"]), device=device)
        fn = build_forward_callable(kind, model, input_ids, args.T, use_sdpa, device, use_graph)
        prof = profile_one_forward(fn, warmup=(1 if use_graph else 3), device=device)
        summary = summarize(prof, args.top)
        print(f"    launches={summary['kernel_launch_count']:,}   self_gpu={summary['total_self_gpu_ms']:.3f}ms")
        rows.append({"label": label, "model": kind, "graph": use_graph, "sdpa": use_sdpa, **summary})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n=== profile_kernels --compare: config={args.config}  T={args.T} ===")
    header = f"{'variant':22s} {'launches':>10s} {'self GPU ms':>12s} {'vs target':>12s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        factor = r["kernel_launch_count"] / TARGET_KERNEL_LAUNCHES
        tag = "OK" if r["kernel_launch_count"] <= TARGET_KERNEL_LAUNCHES else f"{factor:.1f}x over"
        print(f"{r['label']:22s} {r['kernel_launch_count']:10,d} {r['total_self_gpu_ms']:12.3f} {tag:>12s}")

    ref_count = rows[0]["kernel_launch_count"]
    print()
    for r in rows[1:]:
        reduction = ref_count / r["kernel_launch_count"] if r["kernel_launch_count"] else float("inf")
        print(f"  {r['label']} vs reference: {reduction:.1f}x fewer launches ({ref_count:,} -> {r['kernel_launch_count']:,})")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=list(CONFIGS), default="decode")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--model", choices=list(MODEL_CLASSES), default="engine",
                         help="which SpikingQwen implementation to profile (default: engine)")
    parser.add_argument("--sdpa", action="store_true",
                         help="engine only: use F.scaled_dot_product_attention instead of the manual matmul/softmax path")
    parser.add_argument("--graph", action="store_true",
                         help="engine only: build+capture a CUDA graph before profiling, and profile a replay (not the capture)")
    parser.add_argument("--compare", action="store_true",
                         help="profile reference, engine (eager), and engine+graph side by side; ignores --model/--graph")
    parser.add_argument("--json", default=None, help="optional path to dump the summary as JSON")
    parser.add_argument("--trace", default=None, help="optional path to export a chrome trace (not supported with --compare)")
    args = parser.parse_args()

    if args.compare and args.trace:
        parser.error("--trace is not supported together with --compare (there are multiple profiled regions)")
    if not args.compare and args.model == "reference":
        if args.sdpa:
            parser.error("--sdpa only applies to --model engine")
        if args.graph:
            parser.error("--graph only applies to --model engine")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available -- kernel-launch profiling is only meaningful on GPU.")

    print(f"device={device}  config={args.config}  T={args.T}")

    if args.compare:
        rows = run_compare(args, device)
        if args.json:
            out_path = Path(args.json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shape = CONFIGS[args.config]
            payload = {
                "config": args.config,
                "T": args.T,
                "B": shape["B"],
                "S": shape["S"],
                "target_kernel_launches": TARGET_KERNEL_LAUNCHES,
                "variants": rows,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nWrote results to {out_path}")
        return

    mode = mode_str(args.model, args.graph, args.sdpa)
    print(f"  model={mode}")
    model, cfg = build_model(args.model, args.T, device)

    shape = CONFIGS[args.config]
    input_ids = torch.randint(0, cfg.vocab_size, (shape["B"], shape["S"]), device=device)

    fn = build_forward_callable(args.model, model, input_ids, args.T, args.sdpa, device, args.graph)
    prof = profile_one_forward(fn, warmup=(2 if args.graph else 3), device=device)
    summary = summarize(prof, args.top)
    print_report(args.config, args, summary)

    if args.trace:
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        print(f"\nWrote chrome trace to {trace_path}")

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": args.config,
            "T": args.T,
            "B": shape["B"],
            "S": shape["S"],
            "model": args.model,
            "sdpa": args.sdpa,
            "graph": args.graph,
            "target_kernel_launches": TARGET_KERNEL_LAUNCHES,
            **summary,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
