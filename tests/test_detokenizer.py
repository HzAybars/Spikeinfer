"""Incremental detokenization and stop conditions.

Streaming means decoding one token at a time, and that is where two classic bugs
live: a multi-byte character split across tokens surfacing as U+FFFD, and BPE's
leading-space convention dropping a space when a token is decoded alone. Both
are checked against the ground truth of decoding the whole sequence at once.

A stub tokenizer stands in for a real one so these run without any model files;
one test then repeats the invariant on the real Qwen vocabulary when it is
available.
"""
from __future__ import annotations

import pytest

from conftest import make_sequence
from spikeinfer.engine.detokenizer import Detokenizer, check_stop
from spikeinfer.engine.sequence import SequenceStatus


class StubTokenizer:
    """A byte-level tokenizer with BPE's leading-space marker.

    ``convert_tokens_to_string`` joins and rewrites the marker, so decoding a
    token in isolation genuinely differs from decoding it in context -- which is
    the behaviour the sliding window exists to handle.
    """

    def __init__(self, vocab: dict[int, str]) -> None:
        self.vocab = vocab

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).replace("\u0120", " ")


ASCII_VOCAB = {
    1: "Hello",
    2: "\u0120world",
    3: "!",
    4: "\u0120how",
    5: "\u0120are",
    6: "\u0120you",
}


def detokenizer(vocab=None):
    return Detokenizer(StubTokenizer(vocab or ASCII_VOCAB))


def test_deltas_concatenate_to_the_full_decode():
    tok = detokenizer()
    seq = make_sequence(prompt_len=1)
    seq.prompt_token_ids = [1]
    tok.prime(seq)

    deltas = [tok.step(seq, token) for token in (2, 3, 4, 5, 6)]
    assert "".join(deltas) == " world! how are you"
    assert seq.output_text == " world! how are you"


def test_leading_space_survives_the_first_token():
    """Decoded alone, token 2 is 'Ġworld'; only left context makes it ' world'."""
    tok = detokenizer()
    seq = make_sequence(prompt_len=1)
    seq.prompt_token_ids = [1]
    tok.prime(seq)
    assert tok.step(seq, 2) == " world"


def test_multibyte_character_is_held_until_complete():
    """A character split across two tokens must never emit U+FFFD."""
    vocab = {1: "a", 2: "\ufffd", 3: "\u00e9"}  # token 2 decodes to a replacement char

    class SplitTokenizer(StubTokenizer):
        def convert_tokens_to_string(self, tokens):
            joined = "".join(tokens)
            # Emulate a two-token sequence that only resolves together.
            return joined.replace("\ufffd\u00e9", "\u00e9\u00e9")

    tok = Detokenizer(SplitTokenizer(vocab))
    seq = make_sequence(prompt_len=1)
    seq.prompt_token_ids = [1]
    tok.prime(seq)

    first = tok.step(seq, 2)
    assert "\ufffd" not in first, "a replacement character reached the client"
    second = tok.step(seq, 3)
    assert "\ufffd" not in second
    assert "\ufffd" not in seq.output_text


def test_window_does_not_grow_without_bound():
    tok = detokenizer()
    seq = make_sequence(prompt_len=1)
    seq.prompt_token_ids = [1]
    tok.prime(seq)
    for _ in range(200):
        tok.step(seq, 3)
    assert len(seq.tokens) < 32, "the detokenization window is leaking"
    assert seq.output_text == "!" * 200


# -- stop conditions ------------------------------------------------------


def test_stop_string_truncates_the_output():
    seq = make_sequence(stop=["STOP"], max_tokens=100)
    seq.output_text = "hello STOP world"
    seq.append_token(1)

    assert check_stop(seq, eos_token_id=None, max_model_len=1024)
    assert seq.output_text == "hello "
    assert seq.status is SequenceStatus.FINISHED_STOPPED
    assert seq.stop_reason == "STOP"


def test_stop_string_can_be_kept():
    seq = make_sequence(stop=["STOP"], include_stop_str_in_output=True, max_tokens=100)
    seq.output_text = "hello STOP world"
    seq.append_token(1)

    assert check_stop(seq, eos_token_id=None, max_model_len=1024)
    assert seq.output_text == "hello STOP"


def test_eos_finishes_the_sequence():
    seq = make_sequence(max_tokens=100)
    seq.append_token(50256)
    assert check_stop(seq, eos_token_id=50256, max_model_len=1024)
    assert seq.status is SequenceStatus.FINISHED_STOPPED
    assert seq.stop_reason == 50256


def test_ignore_eos_keeps_going():
    seq = make_sequence(max_tokens=100, ignore_eos=True)
    seq.append_token(50256)
    assert not check_stop(seq, eos_token_id=50256, max_model_len=1024)
    assert not seq.status.is_finished


def test_stop_token_id_finishes_even_with_ignore_eos():
    seq = make_sequence(max_tokens=100, ignore_eos=True, stop_token_ids=[7])
    seq.append_token(7)
    assert check_stop(seq, eos_token_id=50256, max_model_len=1024)
    assert seq.stop_reason == 7


def test_max_tokens_finishes_with_length():
    seq = make_sequence(max_tokens=2)
    seq.append_token(1)
    assert not check_stop(seq, None, 1024)
    seq.append_token(2)
    assert check_stop(seq, None, 1024)
    assert seq.status is SequenceStatus.FINISHED_LENGTH
    assert seq.status.finish_reason == "length"


def test_model_length_limit_finishes():
    seq = make_sequence(prompt_len=10, max_tokens=100)
    for i in range(5):
        seq.append_token(i)
    assert check_stop(seq, None, max_model_len=15)
    assert seq.status is SequenceStatus.FINISHED_LENGTH


def test_streamed_offset_never_exceeds_truncated_text():
    """A stop string can shorten text a stream already partly sent."""
    seq = make_sequence(stop=["END"], max_tokens=100)
    seq.output_text = "abcEND"
    seq.streamed_offset = 6
    seq.append_token(1)

    check_stop(seq, None, 1024)
    assert seq.output_text == "abc"
    assert seq.streamed_offset <= len(seq.output_text)


# -- against a real tokenizer, when one is around -------------------------


@pytest.mark.parametrize("text", ["Hello world!", "  spaced  out  ", "café", "日本語のテキスト", "🎉🎉"])
def test_matches_a_real_tokenizer_full_decode(text):
    """The invariant that matters, on a real vocabulary.

    Opt-in: set ``SPIKEINFER_TEST_TOKENIZER`` to a model path or hub id. The
    stub-based tests above cover the logic; this one covers the assumption that
    real BPE vocabularies behave as the sliding window expects, including on
    accents, CJK and emoji (all multi-token, multi-byte).
    """
    import os

    transformers = pytest.importorskip("transformers")
    source = os.environ.get("SPIKEINFER_TEST_TOKENIZER")
    if not source:
        pytest.skip("set SPIKEINFER_TEST_TOKENIZER to run against a real tokenizer")

    tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        pytest.skip("need at least two tokens to exercise incremental decoding")

    seq = make_sequence(prompt_len=1)
    seq.prompt_token_ids = ids[:1]
    tok = Detokenizer(tokenizer)
    tok.prime(seq)
    for token_id in ids[1:]:
        tok.step(seq, token_id)

    expected = tokenizer.decode(ids[1:])
    assert seq.output_text == expected
