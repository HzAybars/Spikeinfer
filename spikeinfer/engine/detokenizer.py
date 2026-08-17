"""Incremental detokenization and stop-string handling.

Decoding token-by-token is not as simple as ``decode([id])``: BPE tokens can be
fragments of a multi-byte character, and many tokenizers only insert the leading
space of a word when they see its neighbour. Both mean a token's text depends on
the tokens around it.

The fix is the standard sliding window: keep a small suffix of already-decoded
tokens, decode ``[prefix_offset:]`` and ``[prefix_offset:read_offset]``, and emit
the difference. A partially decoded multi-byte sequence surfaces as U+FFFD, and
holding the offsets back until it resolves is what stops a replacement character
from ever reaching the client.
"""
from __future__ import annotations

from .sequence import Sequence, SequenceStatus

REPLACEMENT_CHAR = "�"
_WINDOW = 6
"""Tokens of left context kept for re-decoding. Enough for any UTF-8 sequence
plus the word-boundary lookbehind BPE vocabularies need."""


class Detokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def prime(self, seq: Sequence) -> None:
        """Seed the sliding window from the tail of the prompt.

        Without this, the first generated token is decoded with no left context
        and loses its leading space on BPE vocabularies.
        """
        tail = seq.prompt_token_ids[-_WINDOW:]
        seq.tokens = self.tokenizer.convert_ids_to_tokens(tail)
        seq.prefix_offset = max(0, len(seq.tokens) - 1)
        seq.read_offset = len(seq.tokens)

    def step(self, seq: Sequence, token_id: int) -> str:
        """Append one token and return the newly readable text."""
        new = self.tokenizer.convert_ids_to_tokens([token_id])
        seq.tokens.extend(new)

        prefix = self._to_string(seq.tokens[seq.prefix_offset : seq.read_offset])
        whole = self._to_string(seq.tokens[seq.prefix_offset :])

        if len(whole) <= len(prefix) or whole.endswith(REPLACEMENT_CHAR):
            # Still mid-character: keep the offsets, emit nothing this step.
            return ""

        delta = whole[len(prefix) :]
        seq.prefix_offset = seq.read_offset
        seq.read_offset = len(seq.tokens)
        # Trim the window so it cannot grow without bound over a long generation.
        if seq.prefix_offset > _WINDOW * 2:
            drop = seq.prefix_offset - _WINDOW
            seq.tokens = seq.tokens[drop:]
            seq.prefix_offset -= drop
            seq.read_offset -= drop
        seq.output_text += delta
        return delta

    def _to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)


def check_stop(seq: Sequence, eos_token_id: int | None, max_model_len: int) -> bool:
    """Apply every stop condition to ``seq``; returns True if it finished.

    Stop strings truncate ``output_text`` at the match, so a client never sees
    the sentinel unless it asked for it.
    """
    params = seq.sampling_params
    last = seq.output_token_ids[-1] if seq.output_token_ids else None

    if last is not None and last in params.stop_token_ids:
        seq.finish(SequenceStatus.FINISHED_STOPPED, stop_reason=last)
        return True

    if (
        last is not None
        and not params.ignore_eos
        and eos_token_id is not None
        and last == eos_token_id
    ):
        # The EOS token itself is not part of the completion.
        seq.finish(SequenceStatus.FINISHED_STOPPED, stop_reason=eos_token_id)
        return True

    for stop in params.stop:
        if not stop:
            continue
        index = seq.output_text.find(stop)
        if index != -1:
            end = index + len(stop) if params.include_stop_str_in_output else index
            seq.output_text = seq.output_text[:end]
            seq.streamed_offset = min(seq.streamed_offset, len(seq.output_text))
            seq.finish(SequenceStatus.FINISHED_STOPPED, stop_reason=stop)
            return True

    if seq.output_len >= params.max_tokens:
        seq.finish(SequenceStatus.FINISHED_LENGTH)
        return True
    if seq.total_len >= max_model_len:
        seq.finish(SequenceStatus.FINISHED_LENGTH)
        return True
    return False
