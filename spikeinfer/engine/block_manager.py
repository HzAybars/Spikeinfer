"""Physical block allocator for the paged spike cache.

The cache is one flat slab of ``num_blocks * block_size`` slots per layer. A
sequence owns a list of physical block ids; its logical position ``i`` lives in
slot ``block_table[i // block_size] * block_size + i % block_size``. Blocks are
handed out one at a time as sequences grow, so memory is never reserved for
tokens that may not be generated -- the whole point of paging.

Bit-packing makes the blocks unusually cheap here. At the 0.5B config
(24 layers, T=4, 2 KV heads, head_dim 64) one 32-token block costs 96 KB across
the whole model, versus 3.1 MB for an fp32 spike cache.
"""
from __future__ import annotations

from .sequence import Sequence


class BlockAllocationError(RuntimeError):
    """Raised when a request can never fit, as opposed to not fitting *now*."""


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"need at least one block, got {num_blocks}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Handed out from the tail so ids stay warm in the common case.
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        """Additional blocks to hold ``num_new_tokens`` beyond what ``seq`` owns."""
        total = seq.num_computed_tokens + num_new_tokens
        return max(0, self.blocks_for(total) - len(seq.block_table))

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self.blocks_needed(seq, num_new_tokens) <= len(self._free)

    def allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        """Grow ``seq.block_table`` to cover ``num_new_tokens`` more tokens.

        Returns False and changes nothing if there is not enough room, so the
        caller can preempt and retry.
        """
        need = self.blocks_needed(seq, num_new_tokens)
        if need > len(self._free):
            return False
        for _ in range(need):
            seq.block_table.append(self._free.pop())
        return True

    def free(self, seq: Sequence) -> None:
        for block in seq.block_table:
            self._free.append(block)
        seq.block_table = []

    def reset(self) -> None:
        self._free = list(range(self.num_blocks - 1, -1, -1))

    def slot_mapping(self, seq: Sequence, start: int, count: int) -> list[int]:
        """Physical slots for logical positions ``[start, start + count)``."""
        slots = []
        for pos in range(start, start + count):
            block = seq.block_table[pos // self.block_size]
            slots.append(block * self.block_size + pos % self.block_size)
        return slots

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.num_blocks
