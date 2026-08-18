"""Turns a scheduler decision into tensors, runs the model, returns logits.

Two responsibilities:

* **Input preparation.** Flatten every scheduled sequence into one token axis,
  compute each token's absolute position and its physical cache slot, and build
  the block tables the decode kernel walks. Sequences are ragged; the model
  never sees that.
* **Execution.** Decode-only batches replay a captured CUDA graph, which is
  what removes the ~28k launches per step that motivated this project in the
  first place. Anything with a prefill in it runs eager, because prefill shapes
  vary per step and are compute-bound anyway.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from ..attention import AttentionMetadata
from ..config import EngineConfig
from .scheduler import SchedulerOutput
from .sequence import Sequence
from .spike_cache import PagedSpikeCache


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    meta: AttentionMetadata
    sample_indices: torch.Tensor
    """Rows of the hidden state that produce a token this step."""
    sample_seqs: list[Sequence]
    num_decode_tokens: int
    is_decode_only: bool


class ModelRunner:
    def __init__(
        self,
        model,
        cache: PagedSpikeCache,
        block_manager,
        config: EngineConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.cache = cache
        self.block_manager = block_manager
        self.config = config
        self.device = device
        self.dtype = dtype
        self.timesteps = config.model.timesteps or model.config.T
        self.block_size = cache.block_size
        self.max_blocks_per_seq = math.ceil(
            config.scheduler.max_model_len / self.block_size
        )
        self.graph_runners: dict[int, CUDAGraphRunner] = {}

    # -- input preparation ------------------------------------------------

    def prepare(self, sched: SchedulerOutput) -> ModelInput:
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        sample_indices: list[int] = []
        sample_seqs: list[Sequence] = []

        decode_tables: list[list[int]] = []
        decode_lens: list[int] = []

        prefill_starts: list[int] = []
        prefill_lens: list[int] = []
        prefill_ctx: list[int] = []
        prefill_tables: list[torch.Tensor] = []

        num_decode_tokens = 0
        for item in sched.scheduled:
            seq = item.seq
            start = seq.num_computed_tokens
            count = item.num_tokens

            input_ids.extend(seq.slice_to_compute(count))
            positions.extend(range(start, start + count))
            slots.extend(self.block_manager.slot_mapping(seq, start, count))

            if item.is_decode:
                num_decode_tokens += 1
                table = seq.block_table[: self.max_blocks_per_seq]
                decode_tables.append(list(table))
                decode_lens.append(start + 1)
            else:
                prefill_starts.append(len(input_ids) - count)
                prefill_lens.append(count)
                prefill_ctx.append(start)
                prefill_tables.append(
                    torch.tensor(seq.block_table, dtype=torch.int32, device=self.device)
                )

            if item.will_sample:
                sample_indices.append(len(input_ids) - 1)
                sample_seqs.append(seq)

        pad_block = self.cache.pad_block
        block_tables = None
        seq_lens = None
        if decode_tables:
            padded = [
                row + [pad_block] * (self.max_blocks_per_seq - len(row)) for row in decode_tables
            ]
            block_tables = torch.tensor(padded, dtype=torch.int32, device=self.device)
            seq_lens = torch.tensor(decode_lens, dtype=torch.int32, device=self.device)

        meta = AttentionMetadata(
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=self.device),
            block_size=self.block_size,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_tables,
            seq_lens=seq_lens,
            prefill_query_starts=prefill_starts,
            prefill_query_lens=prefill_lens,
            prefill_context_lens=prefill_ctx,
            prefill_block_tables=prefill_tables,
        )

        return ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=self.device),
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            meta=meta,
            sample_indices=torch.tensor(sample_indices, dtype=torch.long, device=self.device),
            sample_seqs=sample_seqs,
            num_decode_tokens=num_decode_tokens,
            is_decode_only=num_decode_tokens == len(sched.scheduled) and num_decode_tokens > 0,
        )

    # -- execution --------------------------------------------------------

    @torch.no_grad()
    def execute(self, sched: SchedulerOutput) -> tuple[torch.Tensor, list[Sequence]]:
        model_input = self.prepare(sched)
        if not model_input.sample_seqs and model_input.input_ids.numel() == 0:
            return torch.empty(0, device=self.device), []

        runner = None
        if model_input.is_decode_only:
            runner = self._graph_for(model_input.num_decode_tokens)

        if runner is not None:
            hidden = runner.replay(model_input)
        else:
            hidden = self.model.forward_paged(
                model_input.input_ids,
                model_input.positions,
                self.cache.layers,
                model_input.meta,
                T=self.timesteps,
            )

        if not model_input.sample_seqs:
            return torch.empty(0, device=self.device), []

        indices = model_input.sample_indices
        if runner is not None:
            # Graph batches are decode-only, so row i of the hidden state is
            # sequence i and the indices are already 0..n-1.
            indices = None
            hidden = hidden[: len(model_input.sample_seqs)]
        logits = self.model.compute_logits(hidden, indices)
        # The sampler runs on the engine's device. Under a hybrid split or an
        # offloaded lm_head the logits may arrive from elsewhere; this is
        # [n_sampled, vocab], a couple of MiB at most, and it happens once per
        # step rather than once per layer.
        if logits.device != self.device:
            logits = logits.to(self.device)
        return logits.float(), model_input.sample_seqs

    def _graph_for(self, batch_size: int) -> CUDAGraphRunner | None:
        for size in sorted(self.graph_runners):
            if size >= batch_size:
                return self.graph_runners[size]
        return None

    # -- CUDA graph capture ----------------------------------------------

    def capture_graphs(self) -> list[int]:
        """Capture one decode graph per bucket. Call once, after loading."""
        if not self.config.graph.enabled or self.device.type != "cuda":
            return []
        captured = []
        for size in self.config.graph.buckets_upto(self.config.scheduler.max_num_seqs):
            runner = CUDAGraphRunner(self, size)
            try:
                runner.capture()
            except RuntimeError as exc:
                # Capture is an optimisation, never a correctness requirement.
                # A driver that refuses (most likely over a stream-capture rule
                # the streamed-weight path trips) must degrade to eager rather
                # than take the server down -- but silently losing a 3.6x
                # speedup would be worse, so say so.
                self.graph_runners.clear()
                warnings.warn(
                    f"CUDA graph capture failed at batch size {size} "
                    f"({type(exc).__name__}: {exc}); falling back to eager decode. "
                    "Throughput will drop substantially on Windows/WDDM.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return []
            self.graph_runners[size] = runner
            captured.append(size)
        return captured


class CUDAGraphRunner:
    """A decode step of one fixed batch size, replayable as a single launch.

    Capture runs entirely against the cache's reserved padding block: every
    static slot points at it and every block table entry is it, so warmup and
    capture cannot disturb real sequence data and no save/restore is needed.
    """

    def __init__(self, runner: ModelRunner, batch_size: int) -> None:
        self.runner = runner
        self.batch_size = batch_size
        self.graph: torch.cuda.CUDAGraph | None = None
        dev = runner.device
        pad_slot = runner.cache.pad_slot
        pad_block = runner.cache.pad_block

        self.input_ids = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self.positions = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self.slot_mapping = torch.full((batch_size,), pad_slot, dtype=torch.int64, device=dev)
        self.block_tables = torch.full(
            (batch_size, runner.max_blocks_per_seq), pad_block, dtype=torch.int32, device=dev
        )
        self.seq_lens = torch.ones(batch_size, dtype=torch.int32, device=dev)

        self.meta = AttentionMetadata(
            slot_mapping=self.slot_mapping,
            block_size=runner.block_size,
            num_decode_tokens=batch_size,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
        )
        self.static_hidden: torch.Tensor | None = None

    def _body(self):
        return self.runner.model.forward_paged(
            self.input_ids,
            self.positions,
            self.runner.cache.layers,
            self.meta,
            T=self.runner.timesteps,
        )

    @torch.no_grad()
    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_hidden = self._body()

    @torch.no_grad()
    def replay(self, model_input: ModelInput) -> torch.Tensor:
        n = model_input.num_decode_tokens
        if n > self.batch_size:
            raise ValueError(f"batch of {n} exceeds captured size {self.batch_size}")
        # Only the live rows are written; the padding rows keep the values set
        # at capture time and therefore stay pointed at the pad block.
        self.input_ids[:n].copy_(model_input.input_ids[:n])
        self.positions[:n].copy_(model_input.positions[:n])
        self.slot_mapping[:n].copy_(model_input.meta.slot_mapping[:n])
        self.block_tables[:n].copy_(model_input.meta.block_tables[:n])
        self.seq_lens[:n].copy_(model_input.meta.seq_lens[:n])
        self.graph.replay()
        return self.static_hidden
