"""Sampling, one row at a time and all at once.

The property that actually matters for a server is *isolation*: every sequence
in a batch samples under its own parameters, and a greedy request must stay
greedy no matter what the request next to it asked for. Several of these tests
therefore put contradictory parameters in the same batch on purpose.

Support-set assertions (top_k, top_p, min_p) draw repeatedly and check which
tokens can ever appear, which is exact -- a filtered token has probability
zero, not a small probability.
"""
from __future__ import annotations

import pytest
import torch

from conftest import make_sequence
from spikeinfer.engine.sampler import Sampler
from spikeinfer.sampling_params import SamplingParams

VOCAB = 8
DEVICE = torch.device("cpu")


def sampler(eos_token_id=None):
    return Sampler(DEVICE, eos_token_id=eos_token_id)


def logits_from(values):
    return torch.tensor([values], dtype=torch.float32, device=DEVICE)


def seq_with(**params):
    return make_sequence(prompt_len=3, **params)


def support(values, trials=400, **params):
    """Every token id that the given parameters can ever produce."""
    seen = set()
    for _ in range(trials):
        out = sampler().sample(logits_from(values), [seq_with(**params)])
        seen.add(out.token_ids[0])
    return seen


# -- greedy ---------------------------------------------------------------


def test_greedy_takes_the_argmax():
    values = [0.1, 5.0, 0.3, 2.0, 0.0, 0.0, 0.0, 0.0]
    out = sampler().sample(logits_from(values), [seq_with(temperature=0.0)])
    assert out.token_ids == [1]


def test_greedy_ignores_top_k_and_top_p():
    values = [9.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = sampler().sample(
        logits_from(values), [seq_with(temperature=0.0, top_k=1, top_p=0.01)]
    )
    assert out.token_ids == [0]


def test_greedy_is_deterministic():
    values = [1.0, 1.0001, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = {
        sampler().sample(logits_from(values), [seq_with(temperature=0.0)]).token_ids[0]
        for _ in range(20)
    }
    assert results == {1}


# -- filters --------------------------------------------------------------


def test_top_k_limits_the_support():
    values = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]
    assert support(values, top_k=2) == {0, 1}
    assert support(values, top_k=1) == {0}


def test_top_p_limits_the_support():
    # softmax over these is dominated by token 0; nucleus 0.5 keeps only it.
    values = [10.0, 5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert support(values, top_p=0.5) == {0}


def test_top_p_keeps_at_least_one_token():
    """A tiny top_p must not empty the distribution."""
    values = [1.0] * VOCAB
    assert len(support(values, top_p=0.01)) >= 1


def test_min_p_drops_tokens_far_below_the_peak():
    values = [10.0, 9.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert support(values, min_p=0.5) == {0, 1}


def test_no_filter_reaches_every_token():
    assert support([1.0] * VOCAB, temperature=2.0, trials=800) == set(range(VOCAB))


# -- penalties ------------------------------------------------------------


def test_repetition_penalty_suppresses_seen_tokens():
    seq = seq_with(temperature=0.0, repetition_penalty=2.0)
    seq.append_token(1)
    values = [1.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = sampler().sample(logits_from(values), [seq])
    assert out.token_ids == [0], "the repeated token should have been penalised below token 0"


def test_frequency_penalty_scales_with_count():
    seq = seq_with(temperature=0.0, frequency_penalty=1.0)
    for _ in range(3):
        seq.append_token(2)
    values = [0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = sampler().sample(logits_from(values), [seq])
    assert out.token_ids != [2], "3 repeats at penalty 1.0 should outweigh a 2.5 lead"


def test_presence_penalty_is_flat():
    seq = seq_with(temperature=0.0, presence_penalty=1.0)
    seq.append_token(2)
    values = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    out = sampler().sample(logits_from(values), [seq])
    assert out.token_ids != [2]


def test_no_penalty_leaves_logits_alone():
    seq = seq_with(temperature=0.0)
    seq.append_token(2)
    values = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert sampler().sample(logits_from(values), [seq]).token_ids == [2]


# -- bias and blocking ----------------------------------------------------


def test_logit_bias_shifts_a_token():
    values = [0.0] * VOCAB
    seq = seq_with(temperature=0.0, logit_bias={5: 10.0})
    assert sampler().sample(logits_from(values), [seq]).token_ids == [5]


def test_min_tokens_blocks_eos():
    values = [0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]
    seq = seq_with(temperature=0.0, min_tokens=5)
    out = sampler(eos_token_id=3).sample(logits_from(values), [seq])
    assert out.token_ids != [3], "EOS must be suppressed below min_tokens"


def test_min_tokens_stops_blocking_once_satisfied():
    values = [0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]
    seq = seq_with(temperature=0.0, min_tokens=2)
    seq.append_token(1)
    seq.append_token(1)
    out = sampler(eos_token_id=3).sample(logits_from(values), [seq])
    assert out.token_ids == [3]


# -- seeding --------------------------------------------------------------


def test_same_seed_gives_the_same_draw():
    values = [1.0] * VOCAB
    runs = []
    for _ in range(2):
        engine_sampler = sampler()
        seq = seq_with(temperature=1.0, seed=1234)
        runs.append(
            [engine_sampler.sample(logits_from(values), [seq]).token_ids[0] for _ in range(10)]
        )
    assert runs[0] == runs[1]


def test_different_seeds_diverge():
    values = [1.0] * VOCAB
    draws = []
    for seed in (1, 2):
        engine_sampler = sampler()
        seq = seq_with(temperature=1.0, seed=seed, seq_id=f"s{seed}")
        draws.append(
            [engine_sampler.sample(logits_from(values), [seq]).token_ids[0] for _ in range(20)]
        )
    assert draws[0] != draws[1]


# -- batching -------------------------------------------------------------


def test_each_row_uses_its_own_parameters():
    """The isolation property: a greedy row is unaffected by a sampled one."""
    logits = torch.tensor(
        [
            [5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # greedy -> 0
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0],  # greedy -> 7
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # top_k=1 -> ties, any
        ],
        dtype=torch.float32,
    )
    seqs = [
        seq_with(temperature=0.0, seq_id="a"),
        seq_with(temperature=0.0, seq_id="b"),
        seq_with(temperature=1.0, top_k=1, seq_id="c"),
    ]
    out = sampler().sample(logits.clone(), seqs)
    assert out.token_ids[0] == 0
    assert out.token_ids[1] == 7


def test_penalties_apply_per_row():
    logits = torch.tensor([[1.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float32)
    penalised = seq_with(temperature=0.0, repetition_penalty=2.0, seq_id="p")
    penalised.append_token(1)
    plain = seq_with(temperature=0.0, seq_id="q")
    plain.append_token(1)

    out = sampler().sample(logits.clone(), [penalised, plain])
    assert out.token_ids == [0, 1], "the penalty leaked across rows"


def test_batch_size_must_match():
    with pytest.raises(AssertionError):
        sampler().sample(torch.zeros(2, VOCAB), [seq_with()])


# -- logprobs -------------------------------------------------------------


def test_logprobs_are_returned_when_asked():
    values = [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    seq = seq_with(temperature=0.0, logprobs=3)
    out = sampler().sample(logits_from(values), [seq])

    entry = out.logprobs[0]
    assert entry is not None
    assert out.token_ids[0] in entry
    assert len(entry) >= 3
    assert all(value <= 0 for value in entry.values()), "log-probabilities must be <= 0"
    expected = torch.log_softmax(torch.tensor(values), dim=-1)[0].item()
    assert abs(entry[0] - expected) < 1e-5


def test_logprobs_are_none_by_default():
    out = sampler().sample(logits_from([1.0] * VOCAB), [seq_with(temperature=0.0)])
    assert out.logprobs == [None]


def test_logprobs_come_from_the_unfiltered_distribution():
    """top_k must not distort the reported probabilities."""
    values = [5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    seq = seq_with(temperature=0.0, top_k=1, logprobs=0)
    entry = sampler().sample(logits_from(values), [seq]).logprobs[0]
    expected = torch.log_softmax(torch.tensor(values), dim=-1)[0].item()
    assert abs(entry[0] - expected) < 1e-5


# -- parameter validation -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"max_tokens": 0},
        {"temperature": -1},
        {"top_p": 0},
        {"top_p": 1.5},
        {"top_k": 0},
        {"top_k": -2},
        {"min_p": 2},
        {"repetition_penalty": 0},
        {"logprobs": -1},
        {"min_tokens": 100, "max_tokens": 10},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_stop_string_is_normalised_to_a_list():
    assert SamplingParams(stop="END").stop == ["END"]
