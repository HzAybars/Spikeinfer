"""Placement: the plan, the flat buffers, and the executor that runs them.

Two kinds of test here. The arithmetic ones -- weight accounting, plan
resolution, flat-buffer layout -- run anywhere, and they are what stops the
planner drifting away from the model it claims to describe: ``weight_sizes``
duplicates the decoder layer's constructor by hand, so a test that compares it
against a real module is the only thing keeping the two in step.

The rest need a GPU because streaming weights into VRAM is what they check. All
of them assert the same property in different words: **placement must not
change the answer.** Where every layer computes on one device that means bit
equality, because it is the same arithmetic on the same hardware and only the
weights' address has changed. Where layers are split across devices it means
agreement, not equality -- CPU and GPU reduce in different orders, and a
1-ulp difference can flip a spike sitting on its threshold.
"""
from __future__ import annotations

import pytest
import torch

from conftest import requires_cuda, small_config
from spikeinfer.config import DeviceConfig
from spikeinfer.offload import FlatLayout, LayerWeightStore, flatten_module
from spikeinfer.placement import (
    DevicePlan,
    Placement,
    auto_plan,
    plan_from_config,
    transfer_fraction,
    weight_sizes,
)

GiB = 2**30


# -- weight accounting -----------------------------------------------------


def test_weight_sizes_match_a_real_layer():
    """The planner counts parameters from the config; the model builds them.

    If those two ever disagree, every VRAM estimate this package prints is
    wrong and nothing else would notice.
    """
    from spikeinfer.modeling_fast import FastSpikingQwenDecoderLayer

    config = small_config()
    layer = FastSpikingQwenDecoderLayer(config, 0)
    actual = sum(p.numel() for p in layer.parameters())
    assert weight_sizes(config).per_layer == actual


def test_weight_sizes_match_the_real_model_shape():
    """Same check at the shape the numbers in the docs were computed for."""
    config = small_config(
        hidden_size=896,
        intermediate_size=4864,
        num_attention_heads=14,
        num_key_value_heads=2,
        vocab_size=151936,
    )
    sizes = weight_sizes(config)
    assert sizes.per_layer == 14_922_496
    # The MLP dominates, which is the whole reason gate sparsity is worth
    # exploiting and attention sparsity would not be.
    assert sizes.attention_share == pytest.approx(0.123, abs=0.001)
    assert sizes.mlp / sizes.per_layer == pytest.approx(0.876, abs=0.001)


def test_transfer_fraction_is_bounded_below_by_the_dense_part():
    """gate_proj produces the mask, so it can never be skipped.

    A firing rate of zero does not mean zero transfer, and any claim that
    adaptive streaming approaches free is wrong.
    """
    sizes = weight_sizes(small_config(hidden_size=896, intermediate_size=4864))
    floor = transfer_fraction(sizes, 0.0)
    assert 0.35 < floor < 0.5
    assert transfer_fraction(sizes, 1.0) == pytest.approx(1.0)
    assert transfer_fraction(sizes, 0.055) > floor


# -- plan resolution -------------------------------------------------------


def test_no_options_means_no_plan():
    """The ordinary path must not pay for machinery it does not use."""
    config = small_config()
    plan = plan_from_config(config, DeviceConfig(), torch.float32, 8 * GiB)
    assert plan.is_trivial


def test_gpu_layers_splits_the_stack():
    config = small_config(num_hidden_layers=8)
    plan = auto_plan(config, DeviceConfig(gpu_layers=3), torch.float32, 8 * GiB)
    assert plan.layers == [Placement.GPU] * 3 + [Placement.CPU] * 5
    assert plan.gpu_layer_count == 3


def test_gpu_layers_zero_keeps_the_embedding_off_the_gpu():
    """Nothing computes there, so a GPU embedding would be a round trip for a
    lookup whose result immediately goes back to the host."""
    plan = auto_plan(small_config(num_hidden_layers=4), DeviceConfig(gpu_layers=0), torch.float32, 8 * GiB)
    assert plan.embeddings is Placement.CPU
    assert all(p is Placement.CPU for p in plan.layers)


def test_offload_auto_streams_only_the_overflow():
    config = small_config(num_hidden_layers=8)
    sizes = weight_sizes(config)
    # Room for four layers plus the embedding, and no more.
    budget = 4 * sizes.layer_bytes(torch.float32) + sizes.embedding_bytes(torch.float32)
    plan = auto_plan(config, DeviceConfig(offload_layers="auto"), torch.float32, budget)
    assert plan.count(Placement.GPU) == 4
    assert plan.count(Placement.STREAM) == 4


def test_offload_auto_streams_nothing_when_it_all_fits():
    config = small_config(num_hidden_layers=8)
    plan = auto_plan(config, DeviceConfig(offload_layers="auto"), torch.float32, 64 * GiB)
    assert plan.count(Placement.STREAM) == 0


def test_streaming_takes_the_deepest_layers():
    """The shallow layers run first every step, so leaving them resident gives
    the first prefetch a full layer of compute to hide behind."""
    plan = auto_plan(
        small_config(num_hidden_layers=6), DeviceConfig(offload_layers=2), torch.float32, 8 * GiB
    )
    assert plan.layers[:4] == [Placement.GPU] * 4
    assert plan.layers[4:] == [Placement.STREAM] * 2


def test_adaptive_applies_to_resident_layers_not_streamed_ones():
    """Adaptive is what a layer does instead of being plainly resident: its
    dense half stays in VRAM and only the MLP tail is fetched. A streamed layer
    is already moving all of its weights, so the two do not stack."""
    plan = auto_plan(
        small_config(num_hidden_layers=4),
        DeviceConfig(offload_layers=2, adaptive_mlp=True, hot_fraction=0.25),
        torch.float32,
        8 * GiB,
    )
    assert plan.layers == [Placement.ADAPTIVE] * 2 + [Placement.STREAM] * 2
    hot = [n for n in plan.hot_channels if n]
    assert hot == [64, 64]  # 25% of intermediate_size=256


def test_adaptive_alone_covers_every_gpu_layer():
    plan = auto_plan(
        small_config(num_hidden_layers=4),
        DeviceConfig(adaptive_mlp=True, hot_fraction=0.25),
        torch.float32,
        8 * GiB,
    )
    assert plan.count(Placement.ADAPTIVE) == 4


def test_offload_layers_cannot_exceed_the_gpu_layers():
    plan = auto_plan(
        small_config(num_hidden_layers=6),
        DeviceConfig(gpu_layers=2, offload_layers=99),
        torch.float32,
        8 * GiB,
    )
    assert plan.count(Placement.STREAM) == 2
    assert plan.count(Placement.CPU) == 4


# -- flat buffers ----------------------------------------------------------


def test_flat_layout_covers_every_parameter_exactly_once():
    from spikeinfer.modeling_fast import FastSpikingQwenDecoderLayer

    layer = FastSpikingQwenDecoderLayer(small_config(), 0)
    layout = FlatLayout.of(layer)
    assert len(layout.names) == len(list(layer.named_parameters()))
    assert layout.numel == sum(p.numel() for p in layer.parameters())
    # Offsets tile the buffer with no gaps and no overlap.
    expected = 0
    for offset, shape in zip(layout.offsets, layout.shapes):
        assert offset == expected
        expected += shape.numel()


def test_flatten_module_preserves_every_value():
    """Packing into one buffer and rebinding must be value-identical -- this is
    the step where a layout mismatch would quietly stream one tensor's bytes
    into another tensor's slot."""
    from spikeinfer.modeling_fast import FastSpikingQwenDecoderLayer

    config = small_config()
    torch.manual_seed(0)
    layer = FastSpikingQwenDecoderLayer(config, 0)
    before = {name: p.detach().clone() for name, p in layer.named_parameters()}

    layout = FlatLayout.of(layer)
    flat = torch.empty(layout.numel, dtype=torch.float32)
    flatten_module(layer, layout, flat)

    for name, param in layer.named_parameters():
        assert torch.equal(param, before[name]), name
        assert param.data_ptr() >= flat.data_ptr(), f"{name} is not a view into the buffer"


def test_weight_store_round_trips_through_the_flat_buffer():
    from spikeinfer.modeling_fast import FastSpikingQwenDecoderLayer

    torch.manual_seed(1)
    layer = FastSpikingQwenDecoderLayer(small_config(), 0)
    layout = FlatLayout.of(layer)
    store = LayerWeightStore(layout, layer, torch.float32, pin=False)

    for index, name in enumerate(layout.names):
        stored = layout.view(store.flat, index)
        assert torch.equal(stored, dict(layer.named_parameters())[name])


# -- end to end ------------------------------------------------------------


def _tokens(engine, prompt, max_tokens=8):
    from spikeinfer.sampling_params import SamplingParams

    request_id = engine.add_request(
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
    )
    out = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id == request_id and output.finished:
                out = list(output.outputs[0].token_ids)
    return out


def _engine(model_dir, device, graphs=False, **devices):
    from spikeinfer.config import (
        CacheConfig,
        EngineConfig,
        GraphConfig,
        ModelConfig,
        SchedulerConfig,
    )
    from spikeinfer.engine.llm_engine import LLMEngine

    return LLMEngine(
        EngineConfig(
            model=ModelConfig(model=str(model_dir), dtype="float32"),
            cache=CacheConfig(block_size=16, num_gpu_blocks_override=64),
            scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128, max_model_len=64),
            graph=GraphConfig(enabled=graphs, batch_sizes=(1, 2, 4), max_seq_len=64),
            devices=DeviceConfig(**devices),
            device=device,
        )
    )


PROMPT = [5, 91, 44, 7, 20, 13]


@requires_cuda
@pytest.mark.parametrize(
    "devices",
    [
        {"offload_layers": 4},
        {"offload_layers": 2},
        {"offload_layers": 4, "stream_buffers": 3},
        {"offload_layers": 4, "pin_memory": False},
        {"offload_layers": 4, "offload_embeddings": True},
    ],
    ids=["all", "half", "buffers-3", "unpinned", "with-embeddings"],
)
def test_streaming_is_bit_identical(tiny_model_dir, device, devices):
    """Same device, same arithmetic, only the weights' address changed."""
    baseline = _engine(tiny_model_dir, device)
    expected = _tokens(baseline, PROMPT)
    del baseline

    streamed = _engine(tiny_model_dir, device, **devices)
    assert _tokens(streamed, PROMPT) == expected
    del streamed
    torch.cuda.empty_cache()


@requires_cuda
def test_streaming_survives_cuda_graph_capture(tiny_model_dir, device):
    """A captured replay copies the weights too, so it must not depend on which
    slot happened to be warm when capture ran."""
    baseline = _engine(tiny_model_dir, device)
    expected = _tokens(baseline, PROMPT, max_tokens=12)
    del baseline

    graphed = _engine(tiny_model_dir, device, graphs=True, offload_layers=4)
    assert graphed.captured_graphs, "capture silently fell back to eager"
    # Several requests in a row: the second replay is where a ring that leaked
    # state across passes would return the wrong layer's weights.
    for _ in range(3):
        assert _tokens(graphed, PROMPT, max_tokens=12) == expected
    del graphed
    torch.cuda.empty_cache()


@requires_cuda
def test_offloaded_embeddings_disable_capture_rather_than_corrupt_it(tiny_model_dir, device):
    """A host-side embedding lookup inside a captured region poisons the stream
    and fails later somewhere unrelated, so it has to be refused up front."""
    engine = _engine(tiny_model_dir, device, graphs=True, offload_layers=4, offload_embeddings=True)
    assert not engine.captured_graphs
    assert "offload-embeddings" in engine.describe()["cuda_graphs"]
    del engine
    torch.cuda.empty_cache()


@requires_cuda
@pytest.mark.parametrize("gpu_layers", [3, 1, 0])
def test_hybrid_split_agrees_with_a_single_device(tiny_model_dir, device, gpu_layers):
    """Not bit equality: CPU and GPU reduce in different orders, and this model
    turns a 1-ulp difference into a flipped spike. Agreement is the claim."""
    baseline = _engine(tiny_model_dir, device)
    expected = _tokens(baseline, PROMPT, max_tokens=12)
    del baseline

    hybrid = _engine(tiny_model_dir, device, gpu_layers=gpu_layers)
    got = _tokens(hybrid, PROMPT, max_tokens=12)
    agreement = sum(a == b for a, b in zip(expected, got)) / len(expected)
    assert agreement >= 0.75, f"{agreement:.0%} agreement, got {got} vs {expected}"
    del hybrid
    torch.cuda.empty_cache()


@requires_cuda
def test_hybrid_puts_each_layers_cache_where_that_layer_computes(tiny_model_dir, device):
    engine = _engine(tiny_model_dir, device, gpu_layers=2)
    devices = [d.type for d in engine.cache.layer_devices]
    assert devices == ["cuda", "cuda", "cpu", "cpu"]
    assert engine.describe()["kv_cache_device"] == "cpu+cuda"
    # One block manager owns both, so the block counts have to match.
    assert all(k.shape == engine.cache.k[0].shape for k in engine.cache.k)
    del engine
    torch.cuda.empty_cache()


def test_metadata_mirrors_are_reused(tiny_model_dir):
    """`meta.on(device)` is called once per layer per step; it must not rebuild
    the tensors every time."""
    from spikeinfer.attention import AttentionMetadata

    meta = AttentionMetadata(
        slot_mapping=torch.zeros(4, dtype=torch.int64),
        block_size=16,
        num_decode_tokens=4,
        block_tables=torch.zeros((4, 2), dtype=torch.int32),
        seq_lens=torch.ones(4, dtype=torch.int32),
    )
    assert meta.on(torch.device("cpu")) is meta
    if torch.cuda.is_available() and torch.cuda.device_count():
        first = meta.on(torch.device("cuda"))
        assert meta.on(torch.device("cuda")) is first
        assert first.slot_mapping.device.type == "cuda"


def test_plan_summary_omits_hot_channels_when_nothing_is_adaptive():
    plan = DevicePlan(layers=[Placement.GPU, Placement.STREAM], hot_channels=[0, 0])
    assert "hot_channels_mean" not in plan.summary()


# -- adaptive MLP ----------------------------------------------------------


def _mlp_layer(seed=0, intermediate=256):
    """A decoder layer whose gate fires on some channels and not others."""
    from spikeinfer.modeling_fast import FastSpikingQwenDecoderLayer

    torch.manual_seed(seed)
    config = small_config(intermediate_size=intermediate)
    layer = FastSpikingQwenDecoderLayer(config, 0).eval()
    lif = layer.mlp.lif_gate.lif
    lif.beta.data.uniform_(0.1, 0.95)
    # Spread thresholds so channels differ in how readily they fire -- a layer
    # where every channel behaves the same would make the hot/cold split
    # meaningless and hide an ordering bug.
    lif.threshold.data.uniform_(0.2, 1.4)
    return layer, config


def test_channel_permutation_preserves_the_function():
    """The intermediate axis is permutation-equivariant across five tensors.

    Not bit-exact -- down_proj's reduction order changes -- but the difference
    must stay in the same class as the T-batching noise the model already has.
    """
    from spikeinfer.placement import apply_channel_permutation

    layer, config = _mlp_layer()
    x = torch.randn(config.T, 1, 12, config.hidden_size)
    with torch.no_grad():
        before = layer.mlp(x)
        apply_channel_permutation(layer, torch.randperm(config.intermediate_size))
        after = layer.mlp(x)

    relative = float((after - before).abs().max() / before.abs().max())
    assert relative < 1e-5, f"permutation changed the output by {relative:.2e}"


def test_channel_permutation_moves_all_five_tensors():
    """Forgetting any one of them still "works" and silently corrupts the model,
    so check the wiring directly rather than only through the output."""
    from spikeinfer.placement import apply_channel_permutation

    layer, config = _mlp_layer()
    inter = config.intermediate_size
    gate = layer.mlp.gate_proj.weight.detach().clone()
    up = layer.mlp.up_proj.weight.detach().clone()
    down = layer.mlp.down_proj.weight.detach().clone()
    beta = layer.mlp.lif_gate.lif.beta.detach().clone()
    threshold = layer.mlp.lif_gate.lif.threshold.detach().clone()

    perm = torch.randperm(inter)
    apply_channel_permutation(layer, perm)

    assert torch.equal(layer.mlp.gate_proj.weight, gate[perm])
    assert torch.equal(layer.mlp.up_proj.weight, up[perm])
    assert torch.equal(layer.mlp.down_proj.weight, down[:, perm])
    assert torch.equal(layer.mlp.lif_gate.lif.beta, beta[perm])
    assert torch.equal(layer.mlp.lif_gate.lif.threshold, threshold[perm])


@pytest.mark.parametrize("n_hot", [0, 1, 32, 128, 255, 256])
def test_adaptive_mlp_matches_the_dense_mlp(n_hot):
    """Splitting the channel sum into a resident half and a fetched half is a
    reassociation of the same sum, nothing more."""
    import copy

    from spikeinfer.offload import AdaptiveMLP

    layer, config = _mlp_layer()
    x = torch.randn(config.T, 1, 10, config.hidden_size)
    with torch.no_grad():
        dense = layer.mlp(x)
        probe = copy.deepcopy(layer)
        AdaptiveMLP.replace_in(probe, n_hot, torch.device("cpu"), torch.float32, pin_memory=False)
        got = probe.mlp(x)

    relative = float((got - dense).abs().max() / dense.abs().max())
    assert relative < 1e-5, f"n_hot={n_hot} changed the output by {relative:.2e}"


def test_adaptive_mlp_chunks_when_more_fires_than_scratch_holds():
    """The scratch buffer is a cap on how much is fetched at once, never on how
    much *can* be fetched -- correctness must not depend on its size."""
    import copy

    from spikeinfer.offload import AdaptiveMLP

    layer, config = _mlp_layer()
    x = torch.randn(config.T, 1, 8, config.hidden_size)
    with torch.no_grad():
        dense = layer.mlp(x)
        probe = copy.deepcopy(layer)
        AdaptiveMLP.replace_in(
            probe, 8, torch.device("cpu"), torch.float32, pin_memory=False, scratch_channels=4
        )
        got = probe.mlp(x)
    assert float((got - dense).abs().max() / dense.abs().max()) < 1e-5


def test_adaptive_mlp_skips_the_cold_path_when_nothing_fires():
    from spikeinfer.offload import AdaptiveMLP

    layer, config = _mlp_layer()
    # A threshold nothing can reach: the gate stays silent, so no channel of the
    # cold tail is ever needed and no fetch should happen.
    layer.mlp.lif_gate.lif.threshold.data.fill_(1e9)
    AdaptiveMLP.replace_in(layer, 16, torch.device("cpu"), torch.float32, pin_memory=False)
    with torch.no_grad():
        out = layer.mlp(torch.randn(config.T, 1, 6, config.hidden_size))
    assert torch.equal(out, torch.zeros_like(out))
    assert layer.mlp.stats["cold_fetches"] == 0


def test_hot_channel_permutation_sorts_by_firing_frequency():
    from spikeinfer.placement import hot_channel_permutation

    stats = {"layers": [{"channel_union_freq": [0.1, 0.9, 0.0, 0.5]}]}
    assert hot_channel_permutation(stats, 0, 4).tolist() == [1, 3, 0, 2]


def test_hot_channel_permutation_rejects_stats_from_another_model():
    from spikeinfer.placement import hot_channel_permutation

    stats = {"layers": [{"channel_union_freq": [0.1, 0.9]}]}
    with pytest.raises(ValueError, match="different checkpoint"):
        hot_channel_permutation(stats, 0, 4)


def test_coverage_sizing_beats_a_flat_threshold_on_a_dense_layer():
    """A layer that fires everywhere should get a *small* resident slice --
    nothing would help it -- while a layer with a sharp head gets the head.
    A fixed frequency cut gets this backwards and keeps most of the dense
    layer resident for no benefit.
    """
    from spikeinfer.placement import HOT_COVERAGE, _coverage_cut

    peaked = torch.zeros(1000)
    peaked[:50] = 1.0
    flat = torch.full((1000,), 0.5)

    assert _coverage_cut(peaked, HOT_COVERAGE, 1000) <= 50
    assert _coverage_cut(flat, HOT_COVERAGE, 1000) == 500  # capped at half


@requires_cuda
def test_adaptive_engine_agrees_and_shrinks_the_resident_set(tiny_model_dir, device):
    baseline = _engine(tiny_model_dir, device)
    expected = _tokens(baseline, PROMPT, max_tokens=12)
    del baseline
    torch.cuda.empty_cache()

    adaptive = _engine(
        tiny_model_dir, device, graphs=True, adaptive_mlp=True, hot_fraction=0.25
    )
    got = _tokens(adaptive, PROMPT, max_tokens=12)
    agreement = sum(a == b for a, b in zip(expected, got)) / len(expected)
    assert agreement >= 0.75, f"{agreement:.0%} agreement: {got} vs {expected}"
    # Capture is refused rather than attempted: the gate mask is read on the host.
    assert not adaptive.captured_graphs
    assert "adaptive-mlp" in adaptive.describe()["cuda_graphs"]
    del adaptive
    torch.cuda.empty_cache()
