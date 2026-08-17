"""Batched sampling.

Every scheduled sequence samples from its own distribution with its own
parameters in the same step, so the operators here are per-row: a temperature
vector, a top-k vector, a top-p vector. The only place that falls back to a
Python loop is per-request seeding, where reproducibility requires a generator
that does not depend on what else happened to be in the batch.

Order matters and follows the convention clients expect:
logit bias -> penalties -> min_tokens EOS block -> temperature -> min_p ->
top_k -> top_p -> sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .sequence import Sequence

_NEG_INF = float("-inf")


@dataclass
class SamplerOutput:
    token_ids: list[int]
    logprobs: list[dict[int, float] | None]


class Sampler:
    def __init__(self, device: torch.device, eos_token_id: int | None = None) -> None:
        self.device = device
        self.eos_token_id = eos_token_id
        self._generators: dict[str, torch.Generator] = {}

    def free(self, seq_id: str) -> None:
        self._generators.pop(seq_id, None)

    def _generator(self, seq: Sequence) -> torch.Generator:
        gen = self._generators.get(seq.seq_id)
        if gen is None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seq.sampling_params.seed))
            self._generators[seq.seq_id] = gen
        return gen

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> SamplerOutput:
        """``logits`` is ``[n_seqs, vocab]`` in fp32, one row per sequence."""
        assert logits.shape[0] == len(seqs), (logits.shape, len(seqs))
        params = [s.sampling_params for s in seqs]
        vocab = logits.shape[-1]

        raw_logprobs = None
        if any(p.logprobs is not None for p in params):
            raw_logprobs = torch.log_softmax(logits, dim=-1)

        logits = self._apply_logit_bias(logits, params)
        logits = self._apply_penalties(logits, seqs)
        logits = self._block_early_eos(logits, seqs, vocab)

        if all(p.is_greedy for p in params):
            # Skip the whole probability path. A softmax and a multinomial draw
            # over a 151936-token vocabulary is not free, and top-1 survives
            # every filter anyway, so argmax on the penalised logits is exact.
            tokens = logits.argmax(dim=-1)
        else:
            temps = torch.tensor(
                [1.0 if p.is_greedy else p.temperature for p in params],
                device=logits.device,
                dtype=logits.dtype,
            )
            logits = logits / temps.unsqueeze(-1)
            logits = self._filter(logits, params)

            greedy = torch.tensor([p.is_greedy for p in params], device=logits.device)
            probs = torch.softmax(logits, dim=-1)
            sampled = self._draw(probs, seqs)
            tokens = torch.where(greedy, logits.argmax(dim=-1), sampled)

        token_ids = tokens.tolist()
        return SamplerOutput(
            token_ids=token_ids,
            logprobs=self._collect_logprobs(raw_logprobs, params, token_ids),
        )

    # -- steps ------------------------------------------------------------

    def _apply_logit_bias(self, logits: torch.Tensor, params) -> torch.Tensor:
        for i, p in enumerate(params):
            if p.logit_bias:
                idx = torch.tensor(list(p.logit_bias), device=logits.device, dtype=torch.long)
                val = torch.tensor(
                    list(p.logit_bias.values()), device=logits.device, dtype=logits.dtype
                )
                logits[i].index_add_(0, idx, val)
        return logits

    def _apply_penalties(self, logits: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        for i, seq in enumerate(seqs):
            p = seq.sampling_params
            if not p.needs_penalties or not seq.token_counts:
                continue
            idx = torch.tensor(list(seq.token_counts), device=logits.device, dtype=torch.long)
            counts = torch.tensor(
                list(seq.token_counts.values()), device=logits.device, dtype=logits.dtype
            )
            row = logits[i]
            if p.repetition_penalty != 1.0:
                seen = row.index_select(0, idx)
                scaled = torch.where(
                    seen > 0, seen / p.repetition_penalty, seen * p.repetition_penalty
                )
                row.index_copy_(0, idx, scaled)
            if p.presence_penalty or p.frequency_penalty:
                delta = p.presence_penalty + p.frequency_penalty * counts
                row.index_add_(0, idx, -delta)
        return logits

    def _block_early_eos(self, logits: torch.Tensor, seqs: list[Sequence], vocab: int):
        for i, seq in enumerate(seqs):
            p = seq.sampling_params
            if p.min_tokens and seq.output_len < p.min_tokens:
                for tid in self._eos_ids(p):
                    if 0 <= tid < vocab:
                        logits[i, tid] = _NEG_INF
        return logits

    def _eos_ids(self, params) -> list[int]:
        ids = list(params.stop_token_ids)
        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def _filter(self, logits: torch.Tensor, params) -> torch.Tensor:
        min_p = [p.min_p for p in params]
        top_k = [p.top_k for p in params]
        top_p = [p.top_p for p in params]
        vocab = logits.shape[-1]

        if any(m > 0 for m in min_p):
            probs = torch.softmax(logits, dim=-1)
            thresh = torch.tensor(min_p, device=logits.device, dtype=probs.dtype).unsqueeze(-1)
            thresh = thresh * probs.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(probs < thresh, _NEG_INF)

        ks = [vocab if k == -1 else min(k, vocab) for k in top_k]
        needs_k = any(k < vocab for k in ks)
        needs_p = any(p < 1.0 for p in top_p)
        if not needs_k and not needs_p:
            return logits

        width = max(ks) if needs_k else vocab
        values, indices = torch.topk(logits, width, dim=-1)  # descending

        if needs_k:
            keep = torch.tensor(ks, device=logits.device).unsqueeze(-1)
            pos = torch.arange(width, device=logits.device).unsqueeze(0)
            values = values.masked_fill(pos >= keep, _NEG_INF)

        if needs_p:
            probs = torch.softmax(values, dim=-1)
            cumulative = probs.cumsum(dim=-1) - probs  # mass strictly before this token
            limit = torch.tensor(top_p, device=logits.device, dtype=probs.dtype).unsqueeze(-1)
            values = values.masked_fill(cumulative >= limit, _NEG_INF)

        out = torch.full_like(logits, _NEG_INF)
        return out.scatter_(-1, indices, values)

    def _draw(self, probs: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        seeded = [i for i, s in enumerate(seqs) if s.sampling_params.seed is not None]
        out = torch.multinomial(probs, num_samples=1).squeeze(-1)
        for i in seeded:
            out[i] = torch.multinomial(
                probs[i], num_samples=1, generator=self._generator(seqs[i])
            )[0]
        return out

    def _collect_logprobs(self, raw, params, token_ids):
        if raw is None:
            return [None] * len(token_ids)
        result: list[dict[int, float] | None] = []
        for i, p in enumerate(params):
            if p.logprobs is None:
                result.append(None)
                continue
            entry = {token_ids[i]: raw[i, token_ids[i]].item()}
            if p.logprobs > 0:
                top = torch.topk(raw[i], min(p.logprobs, raw.shape[-1]))
                for tid, lp in zip(top.indices.tolist(), top.values.tolist()):
                    entry[tid] = lp
            result.append(entry)
        return result
