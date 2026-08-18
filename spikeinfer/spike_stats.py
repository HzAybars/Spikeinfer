"""How often the MLP gate actually fires, and whether that is exploitable.

The adaptive offload path in :mod:`spikeinfer.offload` rests on one empirical
claim: that ``lif_gate`` fires on only a small fraction of channels, so most of
``up_proj``'s rows and ``down_proj``'s columns need never cross PCIe for a given
token. Whether that claim holds is a property of the *checkpoint* -- of how its
thresholds were calibrated -- not of the architecture, so it has to be measured
before anything is built on it.

What matters is not the per-timestep firing rate but the **union over T**: a
channel that fires at any one of the T timesteps needs its weights fetched, and
the same weights then serve all T. Nor is it the union over one token: a decode
batch needs the union over every token in flight, which is why this module
reports how the union saturates as the batch grows. Both numbers are usually
much larger than the naive per-timestep rate, and quoting that one instead is
the easy way to talk yourself into a feature that does not pay.

Where the sparsity is
---------------------
Only at the gate. ``q/k/v_proj`` also feed LIF neurons, but on the *output*
side, and nothing downstream consumes those spikes as a matmul input --
``o_proj`` sees a softmax-weighted context, which is dense. The MLP is the
exception::

    gate_spk = lif(gate_proj(x))          # binary
    down_proj(gate_spk * up_proj(x))      # zero wherever gate_spk is zero

so row j of ``up_proj`` and column j of ``down_proj`` are dead weight unless
channel j fires. ``gate_proj`` itself produces the mask and can never be
skipped, which is what bounds the whole idea: a third of the MLP always moves.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch

from .kernels import lif_multistep
from .modeling_fast import FastSpikingQwenMLP

SAMPLE_TEXT = [
    "The capital of France is Paris, a city on the river Seine.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "In 1969 Apollo 11 landed the first humans on the Moon.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "Machine learning models are trained by minimising a loss with gradient descent.",
    "She opened the letter carefully, unsure whether the news inside would be good.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
]
"""Fallback corpus when no ``--prompt-file`` is given. Small and deliberately
mixed (prose, code, facts): firing rates are input-dependent, and a corpus of
one register would flatter or damn the checkpoint for the wrong reason."""

BATCH_PROBES = (1, 4, 16, 64)
"""Token counts to report the union over, standing in for decode batch sizes."""

MIN_TOKENS_FOR_CONFIDENCE = 2000
"""Below this the rates are indicative only. Firing is input-dependent, and a
hundred tokens of sample text is a hundred tokens of one writer habits."""

FALLBACK_ATTENTION_SHARE = 0.123
"""Share of a decoder layer's parameters in q/k/v/o_proj, which are dense.

Used only when no config is to hand. The real value is computed from the config
by :func:`spikeinfer.placement.weight_sizes` -- for Qwen2.5-0.5B it is 1.84M of
14.92M, so the MLP is 87.6% of a layer and the ceiling on adaptive streaming is
better than a first guess from the README's head counts suggests."""


class GateRecorder:
    """Collects gate spikes per layer by intercepting the MLP's forward.

    ``lif_gate`` is only a parameter holder -- its ``forward`` is never called
    (see ``modeling_fast``) -- so a module hook has nothing to attach to and the
    spikes have to be caught where they are actually produced.
    """

    def __init__(self, num_layers: int, intermediate_size: int) -> None:
        self.fire_counts = torch.zeros(num_layers, intermediate_size, dtype=torch.float64)
        self.union_counts = torch.zeros(num_layers, intermediate_size, dtype=torch.float64)
        self.batch_unions = {
            n: torch.zeros(num_layers, dtype=torch.float64) for n in BATCH_PROBES
        }
        self.batch_groups = dict.fromkeys(BATCH_PROBES, 0)
        self.tokens = 0
        self._index: dict[int, int] = {}

    def record(self, layer_index: int, gate_spk: torch.Tensor) -> None:
        """``gate_spk`` is ``[T, B, S, C]`` of 0/1."""
        spikes = gate_spk.detach().to(torch.bool).cpu()
        timesteps = spikes.shape[0]
        flat = spikes.reshape(timesteps, -1, spikes.shape[-1])  # [T, tokens, C]
        per_token = flat.any(dim=0)  # [tokens, C] -- the set one token needs

        self.fire_counts[layer_index] += flat.sum(dim=(0, 1)).to(torch.float64) / timesteps
        self.union_counts[layer_index] += per_token.sum(dim=0).to(torch.float64)
        if layer_index == 0:
            self.tokens += per_token.shape[0]

        # How the union grows when several tokens are decoded together: one
        # streamed weight set serves the whole batch, so this is what decides
        # whether the saving survives concurrency.
        n_tokens = per_token.shape[0]
        for probe in BATCH_PROBES:
            groups = n_tokens // probe
            if groups == 0:
                continue
            grouped = per_token[: groups * probe].reshape(groups, probe, -1).any(dim=1)
            self.batch_unions[probe][layer_index] += float(
                grouped.to(torch.float64).mean(dim=1).sum()
            )
            if layer_index == 0:
                self.batch_groups[probe] += groups


@contextlib.contextmanager
def recording(model, recorder: GateRecorder):
    """Swap ``FastSpikingQwenMLP.forward`` for one that reports its gate.

    Patching the class rather than the instances keeps this entirely inside
    this module -- the model gains no recording hook it would carry into
    serving, where it would cost a device-to-host copy per layer per step.
    """
    for index, mlp in enumerate(m for m in model.modules() if isinstance(m, FastSpikingQwenMLP)):
        recorder._index[id(mlp)] = index

    original = FastSpikingQwenMLP.forward

    def instrumented(self, x):
        gate_spk = lif_multistep(
            self.gate_proj(x), self.lif_gate.lif.beta, self.lif_gate.lif.threshold
        )
        recorder.record(recorder._index[id(self)], gate_spk)
        return self.down_proj(gate_spk * self.up_proj(x))

    FastSpikingQwenMLP.forward = instrumented
    try:
        yield
    finally:
        FastSpikingQwenMLP.forward = original


def coverage_curve(union_freq: torch.Tensor) -> dict[str, float]:
    """The hot/cold split's payoff curve.

    Fraction of a token's needed channels that the hottest x% of channels
    account for. If the hottest 20% cover 80% of what a token needs, a resident
    hot slice removes most of the traffic; if the curve is a diagonal, firing is
    uniform across channels and splitting buys nothing over streaming the lot.
    """
    ordered = torch.sort(union_freq, descending=True).values
    total = float(ordered.sum())
    fractions = (0.05, 0.1, 0.2, 0.5)
    if total <= 0:
        return {f"top_{int(f * 100)}pct": 0.0 for f in fractions}
    cumulative = torch.cumsum(ordered, dim=0) / total
    out = {}
    for fraction in fractions:
        cut = max(1, int(round(fraction * ordered.numel())))
        out[f"top_{int(fraction * 100)}pct"] = round(float(cumulative[cut - 1]), 4)
    return out


def transfer_fraction(union_rate: float, config=None) -> float:
    """Share of a decoder layer's weights still crossing PCIe at this rate.

    Attention is dense, ``gate_proj`` is dense because it is what produces the
    mask, and only ``up_proj``/``down_proj`` scale with the firing rate. So even
    a gate that never fires still moves attention plus a third of the MLP --
    about 41% of the layer for Qwen2.5-0.5B. That is the ceiling, and it is why
    this is a ~2.4x saving rather than the 20x the raw firing rate suggests.
    """
    if config is not None:
        from .placement import transfer_fraction as exact
        from .placement import weight_sizes

        return exact(weight_sizes(config), union_rate)
    share = FALLBACK_ATTENTION_SHARE
    return share + (1 - share) * (1 / 3 + 2 / 3 * union_rate)


def verdict(mean_union_rate: float, config=None) -> str:
    """The go/no-go this module exists to produce."""
    if mean_union_rate <= 0.25:
        return (
            "worth it -- adaptive MLP streaming should cut per-layer transfer to "
            f"~{transfer_fraction(mean_union_rate, config):.0%} of dense"
        )
    if mean_union_rate <= 0.5:
        return (
            "marginal -- a resident hot slice may pay, but dynamic cold fetching "
            "is unlikely to clear its own overhead"
        )
    return (
        "not worth it -- the gate fires too densely for adaptive streaming; "
        "use dense --offload-layers instead"
    )


def collect(model, config, token_batches, device) -> dict:
    """Run the batches through ``model`` and summarise gate firing."""
    recorder = GateRecorder(config.num_hidden_layers, config.intermediate_size)
    with torch.no_grad(), recording(model, recorder):
        for ids in token_batches:
            model(ids.to(device), use_sdpa=True)

    tokens = max(1, recorder.tokens)
    union_freq = recorder.union_counts / tokens  # [layers, C] -- per-channel P(needed)
    per_timestep = recorder.fire_counts / tokens

    layers = []
    for index in range(config.num_hidden_layers):
        layers.append(
            {
                "layer": index,
                "union_rate": round(float(union_freq[index].mean()), 4),
                "per_timestep_rate": round(float(per_timestep[index].mean()), 4),
                "dead_channels": int((union_freq[index] == 0).sum()),
                "always_on_channels": int((union_freq[index] >= 0.999).sum()),
                "coverage": coverage_curve(union_freq[index]),
                "channel_union_freq": [round(v, 4) for v in union_freq[index].tolist()],
            }
        )

    # A probe larger than any single sequence forms no groups at all. Reporting
    # that as 0.0 would read as "the union vanishes at batch 64", which is the
    # opposite of the truth -- it is unmeasured, so say so.
    saturation: dict[str, float | None] = {}
    for probe in BATCH_PROBES:
        groups = recorder.batch_groups[probe]
        saturation[str(probe)] = (
            round(float((recorder.batch_unions[probe] / groups).mean()), 4) if groups else None
        )

    mean_union = round(float(union_freq.mean()), 4)
    return {
        "timesteps": config.T,
        "intermediate_size": config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "tokens_measured": recorder.tokens,
        "summary": {
            "mean_union_rate": mean_union,
            "mean_per_timestep_rate": round(float(per_timestep.mean()), 4),
            "union_rate_by_batch": saturation,
            "attention_share_of_layer": round(
                _attention_share(config), 4
            ),
            "transfer_fraction_at_batch_1": round(transfer_fraction(mean_union, config), 4),
            "verdict": verdict(mean_union, config),
        },
        "layers": layers,
    }


def _attention_share(config) -> float:
    from .placement import weight_sizes

    return weight_sizes(config).attention_share


def build_batches(tokenizer, texts, max_len: int) -> list[torch.Tensor]:
    """One batch per text, truncated.

    Texts are kept in separate batches rather than padded together: a pad token
    fires its own spikes and would pollute the rates with positions no real
    request contains.
    """
    batches = []
    for text in texts:
        ids = tokenizer.encode(text)[:max_len]
        if ids:
            batches.append(torch.tensor([ids], dtype=torch.long))
    return batches


def write(stats: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hot_order(stats: dict, layer_index: int) -> list[int]:
    """Channel indices for one layer, hottest first.

    This is the permutation the adaptive path applies so that the resident hot
    slice is a contiguous ``[:n_hot]`` range and needs no gather.
    """
    freq = torch.tensor(stats["layers"][layer_index]["channel_union_freq"])
    return torch.sort(freq, descending=True).indices.tolist()


def format_summary(stats: dict) -> str:
    """The human-readable report ``spikeinfer spike-stats`` prints."""
    summary = stats["summary"]
    lines = [
        f"tokens measured          {stats['tokens_measured']}",
        f"timesteps T              {stats['timesteps']}",
        f"intermediate size        {stats['intermediate_size']}",
        f"layers                   {stats['num_layers']}",
        "",
        f"per-timestep fire rate   {summary['mean_per_timestep_rate']:.1%}",
        f"union over T  (r)        {summary['mean_union_rate']:.1%}"
        "   <- what adaptive streaming pays for",
        "",
        "union as the decode batch grows:",
    ]
    for batch, rate in summary["union_rate_by_batch"].items():
        shown = f"{rate:.1%}" if rate is not None else f"n/a (no sequence had {batch} tokens)"
        lines.append(f"  {batch:>3} tokens              {shown}")
    lines += [
        "",
        f"layer transfer vs dense  {summary['transfer_fraction_at_batch_1']:.0%} at batch 1",
        "",
        "hot/cold coverage (share of a token's needs met by the hottest channels):",
    ]
    for key in stats["layers"][0]["coverage"]:
        values = [layer["coverage"][key] for layer in stats["layers"]]
        lines.append(
            f"  {key:<12} min {min(values):.1%}   mean {sum(values) / len(values):.1%}"
            f"   max {max(values):.1%}"
        )

    rates = [layer["union_rate"] for layer in stats["layers"]]
    hottest = max(range(len(rates)), key=rates.__getitem__)
    coolest = min(range(len(rates)), key=rates.__getitem__)
    lines += [
        "",
        f"per-layer union rate     min {rates[coolest]:.1%} (layer {coolest})   "
        f"max {rates[hottest]:.1%} (layer {hottest})",
        "",
        f"VERDICT: {summary['verdict']}",
    ]
    if stats["tokens_measured"] < MIN_TOKENS_FOR_CONFIDENCE:
        lines.append(
            f"  (only {stats['tokens_measured']} tokens measured -- pass --prompt-file "
            f"with at least {MIN_TOKENS_FOR_CONFIDENCE} tokens of representative text "
            "before acting on this)"
        )
    return "\n".join(lines)
