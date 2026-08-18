"""The ``spikeinfer`` command line.

    spikeinfer convert  --from-hf Qwen/Qwen2.5-0.5B --out ./qwen-spiking
    spikeinfer serve    ./qwen-spiking --port 8000
    spikeinfer chat     ./qwen-spiking
    spikeinfer generate ./qwen-spiking -p "The capital of France is"
    spikeinfer bench    ./qwen-spiking --concurrency 16
    spikeinfer info     ./qwen-spiking
    spikeinfer spike-stats ./qwen-spiking
    spikeinfer doctor
    spikeinfer plan     ./qwen-spiking --offload-layers auto
    spikeinfer validate ./qwen-spiking
    spikeinfer profile  ./qwen-spiking --batch 1 --batch 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from ..config import (
    CacheConfig,
    DeviceConfig,
    EngineConfig,
    GraphConfig,
    ModelConfig,
    SchedulerConfig,
    default_device,
)
from ..sampling_params import SamplingParams


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("engine")
    group.add_argument("model", help="path to a spikeinfer model directory")
    group.add_argument("--tokenizer", default=None, help="defaults to the model directory")
    group.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    group.add_argument("--timesteps", "-T", type=int, default=None, help="override the model's T")
    group.add_argument("--max-model-len", type=int, default=None)
    group.add_argument("--max-num-seqs", type=int, default=64)
    group.add_argument("--max-num-batched-tokens", type=int, default=4096)
    group.add_argument("--block-size", type=int, default=32)
    group.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    group.add_argument("--num-gpu-blocks-override", type=int, default=None)
    group.add_argument("--no-chunked-prefill", action="store_true")
    group.add_argument("--no-cuda-graph", action="store_true")
    group.add_argument("--device", default=None, help="cuda, cuda:N or cpu")
    group.add_argument("--seed", type=int, default=0)

    placement = parser.add_argument_group(
        "placement",
        "Where weights live and where layers compute. The defaults keep every "
        "weight resident on --device, which is what fits on one GPU.",
    )
    placement.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        metavar="N",
        help="run the first N layers on the GPU and the rest on the CPU "
        "(default: all on the GPU)",
    )
    placement.add_argument(
        "--offload-layers",
        default=None,
        metavar="N|auto",
        help="keep this many layers' weights in host RAM and stream them per step",
    )
    placement.add_argument(
        "--offload-embeddings",
        action="store_true",
        help="keep the embedding/lm_head matrix in host RAM",
    )
    placement.add_argument(
        "--adaptive-mlp",
        action="store_true",
        help="stream only the MLP channels the gate spikes activate "
        "(needs spike_stats.json; see `spikeinfer spike-stats`)",
    )
    placement.add_argument("--hot-fraction", type=float, default=None, metavar="F")
    placement.add_argument("--stream-buffers", type=int, default=2, metavar="K")
    placement.add_argument("--no-pin-memory", action="store_true")
    placement.add_argument(
        "--cpu-threads", type=int, default=None, metavar="N", help="torch.set_num_threads"
    )
    placement.add_argument(
        "--cpu-kv-gb",
        type=float,
        default=None,
        metavar="GB",
        help="host RAM budget for the KV cache (default: a fraction of what is free)",
    )
    placement.add_argument("--cpu-memory-utilization", type=float, default=0.5, metavar="F")


def _engine_config(args: argparse.Namespace) -> EngineConfig:
    max_len = args.max_model_len or 4096
    return EngineConfig(
        model=ModelConfig(
            model=args.model,
            tokenizer=args.tokenizer,
            dtype=args.dtype,
            timesteps=args.timesteps,
            max_model_len=args.max_model_len,
            seed=args.seed,
        ),
        cache=CacheConfig(
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            num_gpu_blocks_override=args.num_gpu_blocks_override,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=max_len,
            enable_chunked_prefill=not args.no_chunked_prefill,
        ),
        graph=GraphConfig(enabled=not args.no_cuda_graph, max_seq_len=max_len),
        devices=DeviceConfig(
            gpu_layers=args.gpu_layers,
            offload_layers=_offload_layers(args.offload_layers),
            adaptive_mlp=args.adaptive_mlp,
            hot_fraction=args.hot_fraction,
            offload_embeddings=args.offload_embeddings,
            stream_buffers=args.stream_buffers,
            pin_memory=not args.no_pin_memory,
            cpu_threads=args.cpu_threads,
            cpu_kv_gb=args.cpu_kv_gb,
            cpu_memory_utilization=args.cpu_memory_utilization,
        ),
        device=args.device or default_device(),
    )


def _offload_layers(value: str | None) -> int | str | None:
    """``--offload-layers`` takes a count or the word ``auto``."""
    if value is None or value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"--offload-layers wants a number or 'auto', got {value!r}") from None


def _sampling_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sampling")
    group.add_argument("--max-tokens", "-n", type=int, default=128)
    group.add_argument("--temperature", type=float, default=0.7)
    group.add_argument("--top-p", type=float, default=0.95)
    group.add_argument("--top-k", type=int, default=-1)
    group.add_argument("--min-p", type=float, default=0.0)
    group.add_argument("--repetition-penalty", type=float, default=1.0)
    group.add_argument("--stop", action="append", default=None)
    group.add_argument("--sampling-seed", type=int, default=None)


def _sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        stop=args.stop or [],
        seed=args.sampling_seed,
    )


# -- commands -------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from .openai.server import run_server

    run_server(
        _engine_config(args),
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name or Path(args.model).name,
        api_key=args.api_key,
        log_level=args.log_level,
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    llm = _build_llm(args)
    params = _sampling_params(args)

    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.extend(
            line.strip() for line in Path(args.prompt_file).read_text("utf-8").splitlines() if line.strip()
        )
    if not prompts:
        prompts = [sys.stdin.read()]

    if len(prompts) == 1 and not args.no_stream:
        print(prompts[0], end="", flush=True)
        started = time.monotonic()
        count = 0
        for delta in llm.stream(prompts[0], params):
            print(delta, end="", flush=True)
            count += 1
        elapsed = time.monotonic() - started
        print(f"\n\n[{count} tokens in {elapsed:.2f}s = {count / max(elapsed, 1e-9):.1f} tok/s]")
        return 0

    started = time.monotonic()
    outputs = llm.generate(prompts, params)
    elapsed = time.monotonic() - started
    total = 0
    for output in outputs:
        for completion in output.outputs:
            total += len(completion.token_ids)
            print(f"--- {output.request_id} ---")
            print(output.prompt or "", end="")
            print(completion.text)
    print(f"\n[{total} tokens in {elapsed:.2f}s = {total / max(elapsed, 1e-9):.1f} tok/s]")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    llm = _build_llm(args)
    params = _sampling_params(args)
    tokenizer = llm.tokenizer
    has_template = tokenizer is not None and getattr(tokenizer, "chat_template", None)

    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    print("spikeinfer chat -- /reset clears history, /exit quits, Ctrl-C interrupts")
    if not has_template:
        print("(this model has no chat template; using a plain transcript)")
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return 0
        if user == "/reset":
            history = history[:1] if args.system else []
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": user})
        prompt = (
            llm.apply_chat_template(history)
            if has_template
            else "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\nassistant:"
        )

        print()
        pieces = []
        try:
            for delta in llm.stream(prompt, params):
                pieces.append(delta)
                print(delta, end="", flush=True)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        print()
        history.append({"role": "assistant", "content": "".join(pieces)})


def cmd_convert(args: argparse.Namespace) -> int:
    import torch

    from ..convert import convert_legacy_checkpoint, transplant_from_hf

    dtype = {None: None, "float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.save_dtype]

    if args.from_hf:
        transplant_from_hf(
            args.from_hf,
            args.out,
            timesteps=args.timesteps,
            beta_init=args.beta_init,
            threshold_init=args.threshold_init,
            dtype=dtype or torch.float32,
        )
    else:
        convert_legacy_checkpoint(
            args.from_checkpoint, args.out, tokenizer_src=args.tokenizer, dtype=dtype
        )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from ..engine.spike_cache import PagedSpikeCache
    from ..loader import load_config

    config = load_config(args.model)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    per_block = PagedSpikeCache.bytes_per_block(
        config.num_hidden_layers, config.T, config.num_key_value_heads, head_dim, args.block_size
    )
    per_token = per_block / args.block_size
    dense_fp16 = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * 2

    info = {
        "path": str(args.model),
        "architecture": "spiking_qwen2",
        "timesteps_T": config.T,
        "hidden_size": config.hidden_size,
        "layers": config.num_hidden_layers,
        "attention_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "kv_cache": {
            "block_size": args.block_size,
            "bytes_per_block": per_block,
            "bytes_per_token": round(per_token, 1),
            "bytes_per_token_dense_fp16_equivalent": dense_fp16,
            "ratio_vs_dense_fp16": round(per_token / dense_fp16, 3),
            "tokens_per_gib": int(2**30 / per_token),
        },
    }
    print(json.dumps(info, indent=2))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from ..profiling import profile_engine

    report = profile_engine(
        _engine_config(args),
        prompt_len=args.prompt_len,
        batches=tuple(args.batch) if args.batch else (1, 8),
        iterations=args.iterations,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from ..validation import run

    config = _engine_config(args)
    result = run(
        config,
        max_tokens=args.max_tokens,
        skip_reference=args.skip_reference,
    )
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.render())
    return 1 if result.failed else 0


def cmd_plan(args: argparse.Namespace) -> int:
    import torch

    from ..config import DeviceConfig, resolve_dtype
    from ..engine.spike_cache import PagedSpikeCache
    from ..loader import load_config
    from ..placement import plan_from_config
    from ..sysinfo import cuda_is_usable

    config = load_config(args.model)
    if args.timesteps:
        config.T = args.timesteps
    device = "cuda" if (args.device or "").startswith("cuda") or (
        args.device is None and cuda_is_usable()
    ) else "cpu"
    dtype = resolve_dtype(args.dtype, device)

    if args.vram_gb is not None:
        free_vram = int(args.vram_gb * 2**30)
    elif cuda_is_usable():
        free_vram = torch.cuda.mem_get_info()[0]
    else:
        free_vram = 0

    stats = None
    stats_path = args.spike_stats or (Path(args.model) / "spike_stats.json")
    if Path(stats_path).exists():
        from .. import spike_stats as spike_stats_mod

        stats = spike_stats_mod.load(stats_path)

    devices = DeviceConfig(
        gpu_layers=args.gpu_layers,
        offload_layers=_offload_layers(args.offload_layers),
        adaptive_mlp=args.adaptive_mlp,
        hot_fraction=args.hot_fraction,
        offload_embeddings=args.offload_embeddings,
    )
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    per_block = PagedSpikeCache.bytes_per_block(
        config.num_hidden_layers, config.T, config.num_key_value_heads, head_dim, args.block_size
    )
    ceiling = math.ceil(args.max_num_seqs * args.max_model_len / args.block_size) + 1

    plan = plan_from_config(config, devices, dtype, free_vram)
    # Size the cache against what the weights leave behind, then re-plan so the
    # two agree -- the placement decides the leftover and the leftover decides
    # the cache, so one pass each way is the cheapest fixed point that is honest.
    from ..placement import Placement, weight_sizes

    sizes = weight_sizes(config)
    resident = plan.count(Placement.GPU) * sizes.layer_bytes(dtype)
    if plan.embeddings is Placement.GPU:
        resident += sizes.embedding_bytes(dtype)
    kv_budget = max(0, free_vram - resident) * args.gpu_memory_utilization
    num_blocks = max(2, min(int(kv_budget // per_block), ceiling)) if free_vram else ceiling

    from ..diagnostics import measured_h2d_bandwidth
    from ..diagnostics import run as run_diagnostics
    from ..placement import describe_plan

    h2d = args.h2d_gb_s
    if h2d is None and args.measure_pcie:
        h2d = measured_h2d_bandwidth(run_diagnostics())

    union = stats["summary"]["mean_union_rate"] if stats else None
    report = describe_plan(
        config, plan, dtype, args.block_size, num_blocks, union_rate=union, h2d_gb_s=h2d
    )
    report["model"] = str(args.model)
    report["free_vram"] = None if not free_vram else _format_bytes(free_vram)
    report["spike_stats"] = str(stats_path) if stats else None

    print(json.dumps(report, indent=2))
    return 0


def _format_bytes(n):
    from ..sysinfo import format_bytes

    return format_bytes(n)


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..diagnostics import run

    report = run()
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return 1 if report.failed else 0


def cmd_spike_stats(args: argparse.Namespace) -> int:
    import torch

    from .. import spike_stats
    from ..config import default_device, resolve_dtype
    from ..loader import load_model

    device = torch.device(args.device or default_device())
    dtype = resolve_dtype(args.dtype, str(device))
    model, config = load_model(args.model, dtype=dtype, device=device, timesteps=args.timesteps)

    texts = spike_stats.SAMPLE_TEXT
    if args.prompt_file:
        texts = [
            line
            for line in Path(args.prompt_file).read_text("utf-8").splitlines()
            if line.strip()
        ]

    tokenizer = _tokenizer_for(args.tokenizer or args.model)
    if tokenizer is None:
        print(
            f"no tokenizer in {args.tokenizer or args.model}; firing rates are "
            "input-dependent and random ids would not represent real traffic",
            file=sys.stderr,
        )
        return 2
    batches = spike_stats.build_batches(tokenizer, texts, args.max_len)
    if not batches:
        print("no usable text to measure", file=sys.stderr)
        return 2

    stats = spike_stats.collect(model, config, batches, device)
    stats["model"] = str(args.model)
    print(spike_stats.format_summary(stats))

    out = args.out
    if out is None and args.write:
        out = str(Path(args.model) / "spike_stats.json")
    if out:
        spike_stats.write(stats, out)
        print()
        print(f"wrote {out}")
    return 0


def _tokenizer_for(source: str):
    try:
        from transformers import AutoTokenizer

        from ..hf_compat import register_auto_classes

        register_auto_classes()
        return AutoTokenizer.from_pretrained(source)
    except Exception:
        return None


def cmd_bench(args: argparse.Namespace) -> int:
    from ..bench import run_benchmark

    result = run_benchmark(
        _engine_config(args),
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        input_len=args.input_len,
        output_len=args.output_len,
        warmup=args.warmup,
    )
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def _build_llm(args: argparse.Namespace):
    from ..llm import LLM

    llm = LLM.__new__(LLM)
    from ..engine.llm_engine import LLMEngine

    llm.engine = LLMEngine(_engine_config(args))
    return llm


# -- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spikeinfer",
        description="Inference engine for spiking (LIF) transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the OpenAI-compatible HTTP server")
    _add_engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None)
    serve.add_argument("--api-key", default=None)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    generate = sub.add_parser("generate", help="complete one or more prompts")
    _add_engine_args(generate)
    _sampling_args(generate)
    generate.add_argument("--prompt", "-p", action="append", default=None)
    generate.add_argument("--prompt-file", default=None, help="one prompt per line")
    generate.add_argument("--no-stream", action="store_true")
    generate.set_defaults(func=cmd_generate)

    chat = sub.add_parser("chat", help="interactive chat REPL")
    _add_engine_args(chat)
    _sampling_args(chat)
    chat.add_argument("--system", default=None, help="system prompt")
    chat.set_defaults(func=cmd_chat)

    convert = sub.add_parser("convert", help="build a spikeinfer model directory")
    source = convert.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-hf", help="dense Qwen2 model path or hub id to transplant")
    source.add_argument("--from-checkpoint", help="legacy .pt SpikingQwen checkpoint")
    convert.add_argument("--out", required=True, help="output model directory")
    convert.add_argument("--tokenizer", default=None, help="where to copy tokenizer files from")
    convert.add_argument("--timesteps", "-T", type=int, default=4)
    convert.add_argument("--beta-init", type=float, default=0.9)
    convert.add_argument("--threshold-init", type=float, default=1.0)
    convert.add_argument(
        "--save-dtype", default=None, choices=["float32", "float16", "bfloat16"]
    )
    convert.set_defaults(func=cmd_convert)

    info = sub.add_parser("info", help="describe a model directory without loading weights")
    info.add_argument("model")
    info.add_argument("--block-size", type=int, default=32)
    info.set_defaults(func=cmd_info)

    profile = sub.add_parser(
        "profile",
        help="where a decode step's time goes: launches, transfers, and what "
        "the step is bound by",
        description="Profiles the serving engine. For model-level kernel launch "
        "counts across the reference model, the fast model and a captured graph, "
        "use bench/profile_kernels.py instead.",
    )
    _add_engine_args(profile)
    profile.add_argument("--prompt-len", type=int, default=32)
    profile.add_argument(
        "--batch", type=int, action="append", default=None, help="decode batch size, repeatable"
    )
    profile.add_argument("--iterations", type=int, default=30)
    profile.set_defaults(func=cmd_profile)

    validate = sub.add_parser(
        "validate",
        help="run the correctness gates against a real checkpoint and placement",
    )
    _add_engine_args(validate)
    validate.add_argument("--max-tokens", "-n", type=int, default=16)
    validate.add_argument(
        "--skip-reference",
        action="store_true",
        help="skip the snntorch comparison (it builds a second full model)",
    )
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)

    plan_cmd = sub.add_parser(
        "plan",
        help="what would fit on this machine, and what it would cost -- reads "
        "the config only, never the weights",
    )
    plan_cmd.add_argument("model")
    plan_cmd.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    plan_cmd.add_argument("--timesteps", "-T", type=int, default=None)
    plan_cmd.add_argument("--device", default=None)
    plan_cmd.add_argument("--vram-gb", type=float, default=None, help="default: what is free now")
    plan_cmd.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    plan_cmd.add_argument("--block-size", type=int, default=32)
    plan_cmd.add_argument("--max-model-len", type=int, default=4096)
    plan_cmd.add_argument("--max-num-seqs", type=int, default=64)
    plan_cmd.add_argument("--gpu-layers", type=int, default=None)
    plan_cmd.add_argument("--offload-layers", default=None, metavar="N|auto")
    plan_cmd.add_argument("--offload-embeddings", action="store_true")
    plan_cmd.add_argument("--adaptive-mlp", action="store_true")
    plan_cmd.add_argument("--hot-fraction", type=float, default=None)
    plan_cmd.add_argument("--spike-stats", default=None, help="default: <model>/spike_stats.json")
    plan_cmd.add_argument("--h2d-gb-s", type=float, default=None)
    plan_cmd.add_argument(
        "--measure-pcie", action="store_true", help="time a real host-to-device copy"
    )
    plan_cmd.set_defaults(func=cmd_plan)

    doctor = sub.add_parser(
        "doctor", help="check this machine can run spikeinfer, and how fast"
    )
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    spike_stats_cmd = sub.add_parser(
        "spike-stats",
        help="measure how often the MLP gate fires (decides if --adaptive-mlp pays)",
    )
    spike_stats_cmd.add_argument("model")
    spike_stats_cmd.add_argument("--tokenizer", default=None)
    spike_stats_cmd.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    spike_stats_cmd.add_argument("--timesteps", "-T", type=int, default=None)
    spike_stats_cmd.add_argument("--device", default=None)
    spike_stats_cmd.add_argument("--prompt-file", default=None, help="one text per line")
    spike_stats_cmd.add_argument("--max-len", type=int, default=256)
    spike_stats_cmd.add_argument("--out", default=None, help="write the JSON here")
    spike_stats_cmd.add_argument(
        "--write", action="store_true", help="write spike_stats.json into the model directory"
    )
    spike_stats_cmd.set_defaults(func=cmd_spike_stats)

    bench = sub.add_parser("bench", help="measure throughput and latency")
    _add_engine_args(bench)
    bench.add_argument("--num-requests", type=int, default=64)
    bench.add_argument("--concurrency", type=int, default=16)
    bench.add_argument("--input-len", type=int, default=128)
    bench.add_argument("--output-len", type=int, default=64)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--json", default=None, help="also write the result here")
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from .. import __version__

        print(f"spikeinfer {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
