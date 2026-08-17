"""End-to-end engine behaviour.

The anchor test is equivalence: greedy generation through the full serving stack
-- paged cache, packed spikes, popcount attention, continuous batching, CUDA
graphs -- must produce the same token ids as the simple eager path in
``spikeinfer.kv_cache``, which is itself validated against the snntorch
reference. Everything the server adds is only allowed to change *when* work
happens, never *what* comes out.

The rest are the properties a server is expected to hold under load: results
must not depend on batch composition, on whether a prompt was chunked, or on
whether a sequence was preempted and recomputed halfway through.
"""
from __future__ import annotations

import pytest
import torch

from conftest import (
    CUDA_AVAILABLE,
    requires_cuda,
    run_to_completion,
    small_config,
)
from spikeinfer.config import (
    CacheConfig,
    EngineConfig,
    GraphConfig,
    ModelConfig,
    SchedulerConfig,
)
from spikeinfer.engine.llm_engine import LLMEngine
from spikeinfer.kv_cache import generate as eager_generate
from spikeinfer.sampling_params import SamplingParams

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="engine tests need CUDA")

GREEDY = dict(temperature=0.0, ignore_eos=True)
PROMPTS = [[5, 91, 44, 7, 200, 13], [1, 2, 3], [77] * 20]


def build_engine(model_dir, device, **overrides):
    cache = dict(block_size=16, num_gpu_blocks_override=128)
    scheduler = dict(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=128)
    graph = dict(enabled=False, batch_sizes=(1, 2, 4, 8), max_seq_len=128)
    cache.update(overrides.pop("cache", {}))
    scheduler.update(overrides.pop("scheduler", {}))
    graph.update(overrides.pop("graph", {}))
    return LLMEngine(
        EngineConfig(
            model=ModelConfig(model=str(model_dir), dtype="float32", **overrides),
            cache=CacheConfig(**cache),
            scheduler=SchedulerConfig(**scheduler),
            graph=GraphConfig(**graph),
            device=device,
        )
    )


def generate_ids(engine, prompt, max_tokens=12, **params):
    request_id = engine.add_request(
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, **{**GREEDY, **params}),
    )
    return run_to_completion(engine, [request_id])[0].outputs[0].token_ids


# -- equivalence ----------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_the_eager_path(tiny_engine, prompt):
    """The whole point: paging and batching must not change the output."""
    got = generate_ids(tiny_engine, prompt, max_tokens=12)

    ids = torch.tensor([prompt], dtype=torch.long, device=tiny_engine.device)
    reference = eager_generate(
        tiny_engine.model, ids, max_new_tokens=12, temperature=0.0, use_sdpa=True
    )
    expected = reference[0, len(prompt) :].tolist()
    assert got == expected


@requires_cuda
def test_cuda_graph_decode_matches_eager_decode(tiny_model_dir, device):
    """A captured graph replays the same computation it captured."""
    eager = build_engine(tiny_model_dir, device, graph={"enabled": False})
    graphed = build_engine(tiny_model_dir, device, graph={"enabled": True})
    assert graphed.captured_graphs, "no graphs were captured"

    for prompt in PROMPTS:
        assert generate_ids(eager, prompt) == generate_ids(graphed, prompt), (
            "graph replay diverged from eager decode"
        )
    del eager, graphed
    torch.cuda.empty_cache()


@requires_cuda
def test_batching_does_not_change_results(tiny_engine):
    """Run alone, then all together: identical tokens either way."""
    alone = [generate_ids(tiny_engine, p, max_tokens=10) for p in PROMPTS]

    ids = [
        tiny_engine.add_request(
            prompt_token_ids=list(p),
            sampling_params=SamplingParams(max_tokens=10, **GREEDY),
        )
        for p in PROMPTS
    ]
    together = [o.outputs[0].token_ids for o in run_to_completion(tiny_engine, ids)]
    assert together == alone


@requires_cuda
def test_chunked_prefill_matches_whole_prefill(tiny_model_dir, device):
    """A prompt split across steps must land in the same cache state."""
    prompt = list(range(1, 61))
    whole = build_engine(
        tiny_model_dir, device, scheduler={"max_num_batched_tokens": 256}
    )
    chunked = build_engine(
        tiny_model_dir,
        device,
        scheduler={"max_num_batched_tokens": 16, "enable_chunked_prefill": True},
    )
    assert generate_ids(whole, prompt) == generate_ids(chunked, prompt)
    del whole, chunked
    torch.cuda.empty_cache()


@requires_cuda
def test_preemption_and_recompute_preserve_output(tiny_model_dir, device):
    """Squeeze the cache until a running sequence is evicted mid-generation.

    Two 16-token prompts fill one 16-token block each, leaving nothing free.
    Both then need a second block for their 17th token, so one must be
    preempted, dropped from the cache, and recomputed from scratch later. Its
    output must come back identical.
    """
    prompts = [list(range(1, 17)), list(range(20, 36))]
    roomy = build_engine(tiny_model_dir, device, cache={"num_gpu_blocks_override": 128})
    expected = [generate_ids(roomy, p, max_tokens=8) for p in prompts]

    tight = build_engine(
        tiny_model_dir,
        device,
        cache={"num_gpu_blocks_override": 3},  # 2 usable blocks of 16 tokens
        scheduler={"max_num_seqs": 8, "max_num_batched_tokens": 64, "max_model_len": 128},
    )
    ids = [
        tight.add_request(
            prompt_token_ids=list(p), sampling_params=SamplingParams(max_tokens=8, **GREEDY)
        )
        for p in prompts
    ]
    outputs = run_to_completion(tight, ids)
    got = [o.outputs[0].token_ids for o in outputs]

    assert tight.stats.preemptions > 0, "the cache was not tight enough to force preemption"
    assert got == expected, "a recomputed sequence diverged from its uninterrupted run"
    del roomy, tight
    torch.cuda.empty_cache()


# -- request semantics ----------------------------------------------------


@requires_cuda
def test_max_tokens_is_respected(tiny_engine):
    for limit in (1, 5, 17):
        tokens = generate_ids(tiny_engine, PROMPTS[0], max_tokens=limit)
        assert len(tokens) == limit


@requires_cuda
def test_finish_reason_length(tiny_engine):
    request_id = tiny_engine.add_request(
        prompt_token_ids=PROMPTS[0], sampling_params=SamplingParams(max_tokens=4, **GREEDY)
    )
    output = run_to_completion(tiny_engine, [request_id])[0]
    assert output.outputs[0].finish_reason == "length"


@requires_cuda
def test_stop_token_id_ends_the_sequence(tiny_engine):
    """Stop on whatever the model would produce second, and check it stops."""
    baseline = generate_ids(tiny_engine, PROMPTS[0], max_tokens=6)
    target = baseline[2]

    request_id = tiny_engine.add_request(
        prompt_token_ids=PROMPTS[0],
        sampling_params=SamplingParams(
            max_tokens=6, temperature=0.0, ignore_eos=True, stop_token_ids=[target]
        ),
    )
    output = run_to_completion(tiny_engine, [request_id])[0]
    assert output.outputs[0].finish_reason == "stop"
    assert output.outputs[0].stop_reason == target
    assert output.outputs[0].token_ids[-1] == target
    assert len(output.outputs[0].token_ids) <= 3


@requires_cuda
def test_n_greater_than_one_returns_n_completions(tiny_engine):
    request_id = tiny_engine.add_request(
        prompt_token_ids=PROMPTS[1],
        sampling_params=SamplingParams(n=3, max_tokens=5, temperature=1.0, ignore_eos=True),
    )
    output = run_to_completion(tiny_engine, [request_id])[0]
    assert len(output.outputs) == 3
    assert [o.index for o in output.outputs] == [0, 1, 2]
    assert all(len(o.token_ids) == 5 for o in output.outputs)


@requires_cuda
def test_abort_stops_a_running_request(tiny_engine):
    request_id = tiny_engine.add_request(
        prompt_token_ids=PROMPTS[2], sampling_params=SamplingParams(max_tokens=64, **GREEDY)
    )
    tiny_engine.step()
    tiny_engine.abort_request(request_id)

    assert not tiny_engine.has_unfinished_requests()
    free_before = tiny_engine.block_manager.num_free_blocks
    assert free_before == tiny_engine.block_manager.num_blocks, "aborting leaked blocks"


@requires_cuda
def test_a_request_that_can_never_fit_is_dropped_not_looped(tiny_model_dir, device):
    """A cache too small for one sequence must error out, not livelock."""
    engine = build_engine(
        tiny_model_dir,
        device,
        cache={"num_gpu_blocks_override": 2},  # 1 usable block = 16 tokens
        scheduler={"max_num_batched_tokens": 128, "max_model_len": 128},
    )
    request_id = engine.add_request(
        prompt_token_ids=list(range(1, 40)),
        sampling_params=SamplingParams(max_tokens=4, **GREEDY),
    )
    outputs = run_to_completion(engine, [request_id])
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "abort"
    assert outputs[0].outputs[0].stop_reason == "kv_cache_too_small"
    del engine
    torch.cuda.empty_cache()


@requires_cuda
def test_prompt_longer_than_the_model_is_rejected(tiny_engine):
    with pytest.raises(ValueError, match="model limit"):
        tiny_engine.add_request(prompt_token_ids=list(range(1, 500)))


@requires_cuda
def test_empty_prompt_is_rejected(tiny_engine):
    with pytest.raises(ValueError, match="empty prompt"):
        tiny_engine.add_request(prompt_token_ids=[])


# -- resource accounting --------------------------------------------------


@requires_cuda
def test_blocks_are_returned_after_every_request(tiny_engine):
    total = tiny_engine.block_manager.num_blocks
    for prompt in PROMPTS:
        generate_ids(tiny_engine, prompt, max_tokens=6)
    assert tiny_engine.block_manager.num_free_blocks == total, "blocks leaked across requests"


@requires_cuda
def test_stats_track_generation(tiny_engine):
    before = tiny_engine.stats.generation_tokens
    generate_ids(tiny_engine, PROMPTS[0], max_tokens=7)
    assert tiny_engine.stats.generation_tokens == before + 7
    assert tiny_engine.stats.steps > 0


@requires_cuda
def test_describe_reports_the_cache_layout(tiny_engine):
    info = tiny_engine.describe()
    assert info["timesteps"] == small_config().T
    assert info["block_size"] == 16
    assert info["kv_capacity_tokens"] == info["gpu_blocks"] * 16
    assert info["dtype"] == "float32"


@requires_cuda
def test_cache_capacity_reflects_bit_packing(tiny_engine):
    """A packed block must cost 1/32 of an fp32 one, not more."""
    cache = tiny_engine.cache
    dense_fp32 = (
        2 * cache.num_layers * cache.timesteps * cache.num_kv_heads
        * cache.head_dim * cache.block_size * 4
    )
    packed = cache.bytes_per_block(
        cache.num_layers, cache.timesteps, cache.num_kv_heads,
        cache.head_dim, cache.block_size,
    )
    assert packed * 32 == dense_fp32
