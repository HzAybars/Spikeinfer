"""Shared fixtures and helpers for the SpikingQwen test suite.

Tests build SMALL models only -- never a full 0.5B model, never a real
checkpoint. See `small_config()`.
"""
from __future__ import annotations

import os

import pytest
import torch

from spikeinfer.modeling_fast import FastSpikingQwenForCausalLM
from spikeinfer.reference import SpikingQwenConfig, SpikingQwenForCausalLM
from spikeinfer.reference.lif import LIFNeuron
from spikeinfer.sysinfo import cuda_is_usable

# `torch.cuda.is_available()` alone is not enough -- see sysinfo.cuda_is_usable.
# SPIKEINFER_TEST_DEVICE=cpu forces the CPU suite on a machine that has a GPU,
# which is how the CI configuration is reproduced locally.
CUDA_AVAILABLE = (
    cuda_is_usable() and os.environ.get("SPIKEINFER_TEST_DEVICE", "").lower() != "cpu"
)

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA GPU required")


def small_config(**overrides):
    """Small SpikingQwenConfig for fast tests.

    Never build a full 0.5B model or load a real checkpoint in tests --
    this keeps the suite fast and comfortably inside a few hundred MB of VRAM.
    """
    kwargs = dict(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        T=4,
    )
    kwargs.update(overrides)
    return SpikingQwenConfig(**kwargs)


def randomize_lif(model):
    """Randomize beta/threshold on every LIFNeuron in place.

    Equivalence checks would be trivially satisfied if every LIF site kept
    its identical default init (beta=0.9, threshold=1.0 everywhere) -- this
    breaks that symmetry so tests actually exercise per-channel
    beta/threshold broadcasting, matching the top-level instructions.
    """
    for m in model.modules():
        if isinstance(m, LIFNeuron):
            m.lif.beta.data.copy_(torch.rand_like(m.lif.beta))
            m.lif.threshold.data.copy_(torch.rand_like(m.lif.threshold) * 0.5 + 0.1)


def build_ref_model(cfg, device, seed=0):
    """Construct the golden-reference SpikingQwenForCausalLM with randomized
    LIF parameters (deterministic given `seed`)."""
    torch.manual_seed(seed)
    model = SpikingQwenForCausalLM(cfg).to(device).eval()
    randomize_lif(model)
    return model


def build_fast_from_ref(cfg, ref_model, device):
    """Build a FastSpikingQwenForCausalLM sharing ref_model's exact weights
    (including its randomized LIF params) via a raw state_dict transfer.

    Asserts the transfer has no missing/unexpected keys -- per the fast
    module's docstring, parameter names are designed to mirror the reference
    model exactly, so no key translation should ever be needed.
    """
    fast = FastSpikingQwenForCausalLM(cfg).to(device).eval()
    result = fast.load_state_dict(ref_model.state_dict())
    assert not result.missing_keys, f"missing keys on state_dict transfer: {result.missing_keys}"
    assert not result.unexpected_keys, f"unexpected keys on state_dict transfer: {result.unexpected_keys}"
    return fast


def build_fast_model(cfg, device, seed=0):
    """A standalone fast model with randomized LIF params, no reference needed."""
    torch.manual_seed(seed)
    model = FastSpikingQwenForCausalLM(cfg).to(device).eval()
    randomize_lif(model)
    return model


def make_sequence(prompt_len=4, seq_id="s0", request_id=None, **params):
    """A Sequence with a synthetic prompt, for scheduler/block-manager tests."""
    from spikeinfer.engine.sequence import Sequence
    from spikeinfer.sampling_params import SamplingParams

    return Sequence(
        seq_id=seq_id,
        request_id=request_id or seq_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=SamplingParams(**params),
    )


@pytest.fixture(scope="session")
def device():
    return "cuda" if CUDA_AVAILABLE else "cpu"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A saved small model directory -- what the engine and loader consume.

    Session-scoped: writing it costs a few MB and the engine tests all want the
    same weights, so equivalence comparisons across tests stay meaningful.
    """
    from spikeinfer.loader import save_model

    cfg = small_config()
    model = build_fast_model(cfg, "cpu", seed=0)
    path = tmp_path_factory.mktemp("tiny_model")
    save_model(model, cfg, path)
    return path


@pytest.fixture(scope="session")
def tiny_engine(tiny_model_dir, device):
    """An engine over ``tiny_model_dir``, with CUDA graphs enabled on GPU.

    Session-scoped because engine construction captures graphs; tests must
    therefore leave it with no pending requests (they all run to completion).

    Runs on CPU when there is no GPU. That is the point: everything the engine
    does above the kernels -- scheduling, paging, chunked prefill, preemption,
    sampling, detokenization -- is device-independent, and until this fixture
    worked without CUDA none of it was covered by CI. Graph capture is the one
    part that genuinely needs a GPU, and it is disabled here rather than
    skipping the whole fixture.
    """
    from spikeinfer.config import (
        CacheConfig,
        EngineConfig,
        GraphConfig,
        ModelConfig,
        SchedulerConfig,
    )
    from spikeinfer.engine.llm_engine import LLMEngine

    config = EngineConfig(
        model=ModelConfig(model=str(tiny_model_dir), dtype="float32"),
        cache=CacheConfig(block_size=16, num_gpu_blocks_override=128),
        scheduler=SchedulerConfig(
            max_num_seqs=8, max_num_batched_tokens=256, max_model_len=128
        ),
        graph=GraphConfig(
            enabled=CUDA_AVAILABLE, batch_sizes=(1, 2, 4, 8), max_seq_len=128
        ),
        device=device,
    )
    engine = LLMEngine(config)
    yield engine
    del engine
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()


def run_to_completion(engine, request_ids=None):
    """Step the engine until idle, returning finished outputs by request id."""
    done = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                done[output.request_id] = output
    if request_ids is not None:
        return [done[r] for r in request_ids]
    return done
