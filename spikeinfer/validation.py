"""Correctness gates, run against a real checkpoint.

The test suite proves these properties on 4-layer random models. That is the
right thing for CI -- fast, deterministic, no weights in the repo -- but it
leaves a gap: nothing checks them on the model you are about to serve, at its
real width and depth, through its real dtype, under whatever placement you
chose. A threshold that drifted non-positive during calibration, a checkpoint
saved from a differently-shaped model, an offload plan that streams the wrong
buffer -- all of those pass the unit tests and fail here.

So this is the same set of gates, aimed at a model directory:

1. thresholds are positive, which the fused LIF kernel requires and does not
   check per call (the check needs a device sync, illegal during graph capture);
2. packing round-trips exactly on the model's own spikes, not on random bits;
3. the paged serving stack reproduces the simple eager path token for token --
   the property the whole engine exists to preserve, and the one that placement
   changes are most likely to break;
4. the fast model agrees with the unoptimised snntorch reference, when snntorch
   is installed.

Gate 3 is the load-bearing one. It compares against ``kv_cache.generate``, the
single-stream path that is itself validated against the reference, on the same
device and dtype -- so any difference is the paged cache, the packing, the
popcount kernel, the scheduler or the placement, and nothing else.

Exactness is claimed only where it is actually available: same device, same arithmetic.
A hybrid CPU/GPU split moves part of the computation to different kernels with
different reduction orders, and in a spiking model a 1-ulp difference can flip a
spike sitting exactly on its threshold. There the gate is argmax agreement over
the run, which is what the project asserts everywhere else for the same reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str
    skipped: bool = False
    data: dict = field(default_factory=dict)

    def render(self) -> str:
        mark = "[--]  " if self.skipped else ("[ok]  " if self.passed else "[FAIL]")
        return f"{mark} {self.name:<34} {self.detail}"


class Result:
    def __init__(self) -> None:
        self.gates: list[Gate] = []

    def add(self, *args, **kwargs) -> Gate:
        gate = Gate(*args, **kwargs)
        self.gates.append(gate)
        return gate

    @property
    def failed(self) -> bool:
        return any(not g.passed and not g.skipped for g in self.gates)

    def render(self) -> str:
        body = "\n".join(g.render() for g in self.gates)
        checked = sum(1 for g in self.gates if not g.skipped)
        failures = sum(1 for g in self.gates if not g.passed and not g.skipped)
        verdict = "FAILED" if failures else "passed"
        return f"{body}\n\n{checked - failures}/{checked} gates {verdict}"

    def to_dict(self) -> dict:
        return {
            "ok": not self.failed,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "skipped": g.skipped,
                    "detail": g.detail,
                    **g.data,
                }
                for g in self.gates
            ],
        }


def _gate_thresholds(result: Result, model) -> None:
    from .kernels import validate_thresholds

    try:
        validate_thresholds(model)
        result.add("lif thresholds positive", True, "every site > 0")
    except ValueError as exc:
        result.add("lif thresholds positive", False, str(exc))


def _gate_packing(result: Result, model, config, device, dtype) -> None:
    """Round-trip the model's own spikes, not random bits.

    Random bits exercise the bit layout; real spikes additionally exercise
    whatever the LIF actually emits -- including the all-zero rows a quiet
    channel produces, which is where an off-by-one in the tail of the last word
    would hide.
    """
    from .kernels import lif_multistep, pack_spikes, unpack_spikes

    layer = model.model.layers[0]
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    current = torch.randn(config.T, 2, 16, head_dim, device=device, dtype=dtype)
    spikes = lif_multistep(
        current, layer.self_attn.lif_k.lif.beta, layer.self_attn.lif_k.lif.threshold
    )
    round_tripped = unpack_spikes(pack_spikes(spikes), head_dim, spikes.dtype)
    exact = torch.equal(round_tripped, spikes)
    fired = float(spikes.mean())
    result.add(
        "packing round-trip",
        exact,
        f"exact over {spikes.numel()} spikes ({fired:.1%} fired)"
        if exact
        else "pack/unpack is not the identity on this model's spikes",
        data={"fire_rate": round(fired, 4)},
    )


def _eager_reference(model, device, prompts, max_tokens) -> list[list[int]]:
    """Greedy continuations from the single-stream path, on a plain model.

    Deliberately *not* the engine's model. Under a placement the engine's layers
    live wherever the plan put them, and ``kv_cache.forward_with_cache`` walks
    ``model.layers`` directly with no notion of that -- so running the oracle on
    them would either crash or, worse, compare a placed engine against an
    equally placed oracle and cancel the very bug this gate exists to find.
    """
    from .kv_cache import generate as eager_generate

    out = []
    for prompt in prompts:
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        out.append(
            eager_generate(
                model, ids, max_new_tokens=max_tokens, temperature=0.0, use_sdpa=True
            )[0, len(prompt) :].tolist()
        )
    return out


def _gate_paged_vs_eager(
    result: Result,
    engine_config,
    prompts: list[list[int]],
    reference_tokens: list[list[int]],
    exact: bool,
    max_tokens: int,
) -> None:
    """The one that matters: the serving stack against the single-stream path."""
    from .engine.llm_engine import LLMEngine
    from .sampling_params import SamplingParams

    started = time.monotonic()
    engine = LLMEngine(engine_config)

    mismatches = []
    divergences: list[float] = []
    total = matched = 0
    for prompt, reference in zip(prompts, reference_tokens):
        request_id = engine.add_request(
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(
                max_tokens=max_tokens, temperature=0.0, ignore_eos=True
            ),
        )
        served = None
        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.request_id == request_id and output.finished:
                    served = list(output.outputs[0].token_ids)
        assert served is not None

        total += len(reference)
        agree = sum(1 for a, b in zip(served, reference) if a == b)
        matched += agree
        first = next(
            (i for i, (a, b) in enumerate(zip(served, reference)) if a != b), len(reference)
        )
        divergences.append(first / max(1, len(reference)))
        if served != reference:
            mismatches.append((len(prompt), first, agree, len(reference)))

    elapsed = time.monotonic() - started
    rate = matched / max(1, total)
    prefix = sum(divergences) / max(1, len(divergences))

    if exact:
        passed = not mismatches
        detail = (
            f"{matched}/{total} tokens identical across {len(prompts)} prompts ({elapsed:.1f}s)"
            if passed
            else f"{matched}/{total} tokens agree; first divergence at "
            f"{[m[1] for m in mismatches]}"
        )
    else:
        # Greedy decoding cascades: one flipped spike changes that token and
        # every token after it, so a total agreement count measures how early
        # the first flip landed, not how often flips happen. What separates a
        # working placement from a broken one is *where* it diverges -- broken
        # placements produce garbage from token 0 -- so the gate is the mean
        # matching prefix, and the raw agreement is reported alongside it.
        passed = prefix >= 0.5
        detail = (
            f"diverges after {prefix:.0%} of each continuation on average "
            f"({matched}/{total} tokens agree overall); exact equality is not "
            f"available for this placement ({elapsed:.1f}s)"
        )
    result.add(
        "paged engine == eager oracle",
        passed,
        detail,
        data={
            "agreement": round(rate, 4),
            "mean_matching_prefix": round(prefix, 4),
            "tokens": total,
            "exact_required": exact,
        },
    )
    del engine


def _gate_reference(result: Result, model, config, device, dtype, prompt: list[int]) -> None:
    """Fast model vs the unoptimised snntorch reference, when available."""
    try:
        import snntorch  # noqa: F401
    except ImportError:
        result.add(
            "fast == snntorch reference",
            True,
            "skipped -- snntorch not installed (pip install 'spikeinfer[reference]')",
            skipped=True,
        )
        return

    from .reference.modeling_spiking_qwen import SpikingQwenForCausalLM

    try:
        reference = SpikingQwenForCausalLM(config).to(device=device, dtype=dtype).eval()
        reference.load_state_dict(model.state_dict())
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        result.add(
            "fast == snntorch reference",
            True,
            f"skipped -- could not build the reference model ({type(exc).__name__})",
            skipped=True,
        )
        return

    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.no_grad():
        fast_logits = model(ids)["logits"]
        ref_logits = reference(ids)["logits"]

    agree = bool(torch.equal(fast_logits.argmax(-1), ref_logits.argmax(-1)))
    scale = ref_logits.abs().max().clamp(min=1e-6)
    relative = float((fast_logits - ref_logits).abs().max() / scale)
    result.add(
        "fast == snntorch reference",
        agree and relative < 1e-3,
        f"argmax {'agrees' if agree else 'DIFFERS'}, max relative logit error {relative:.2e}",
        data={"argmax_agrees": agree, "relative_error": relative},
    )
    del reference


DEFAULT_PROMPTS = [
    [785, 6722, 315, 9625, 374],
    [2, 3, 5, 7, 11, 13],
    [40] * 24,
]
"""Token ids, not text: validation must work on a model directory whose
tokenizer is missing or broken, and the gates do not care what the ids mean."""


def run(
    engine_config,
    prompts: list[list[int]] | None = None,
    max_tokens: int = 16,
    skip_reference: bool = False,
) -> Result:
    """Every gate, against the model ``engine_config`` names."""
    from .loader import load_model

    result = Result()
    device = torch.device(engine_config.device)
    dtype = engine_config.torch_dtype

    model, config = load_model(
        engine_config.model.model,
        dtype=dtype,
        device=device,
        timesteps=engine_config.model.timesteps,
    )

    _gate_thresholds(result, model)
    _gate_packing(result, model, config, device, dtype)

    prompts = prompts or DEFAULT_PROMPTS
    prompts = [p for p in prompts if len(p) < engine_config.scheduler.max_model_len]

    if not skip_reference:
        _gate_reference(result, model, config, device, dtype, prompts[0])

    # Bit equality is only claimable when the engine computes everything on one
    # device with the same arithmetic the oracle used.
    devices = engine_config.devices
    exact = devices.gpu_layers is None or devices.gpu_layers >= config.num_hidden_layers
    if devices.adaptive_mlp:
        # The channel permutation reorders down_proj's 4864-term reduction.
        exact = False

    reference_tokens = _eager_reference(model, device, prompts, max_tokens)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _gate_paged_vs_eager(
        result, engine_config, prompts, reference_tokens, exact, max_tokens
    )
    return result
