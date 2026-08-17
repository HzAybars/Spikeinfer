"""Block allocator invariants.

The allocator is the only thing standing between the engine and cache
corruption: two sequences handed the same physical block would silently read
each other's keys, and attention would not complain. So the properties worth
asserting are conservation (no block is lost or duplicated) and isolation (no
block is ever handed out twice).
"""
from __future__ import annotations

import pytest

from conftest import make_sequence
from spikeinfer.engine.block_manager import BlockManager


def test_starts_fully_free():
    manager = BlockManager(num_blocks=10, block_size=16)
    assert manager.num_free_blocks == 10
    assert manager.num_used_blocks == 0
    assert manager.utilization == 0.0


def test_rejects_an_empty_pool():
    with pytest.raises(ValueError, match="at least one block"):
        BlockManager(num_blocks=0, block_size=16)


@pytest.mark.parametrize(
    "tokens,block_size,expected",
    [(1, 16, 1), (16, 16, 1), (17, 16, 2), (32, 16, 2), (33, 16, 3), (0, 16, 0)],
)
def test_blocks_for_rounds_up(tokens, block_size, expected):
    assert BlockManager(8, block_size).blocks_for(tokens) == expected


def test_allocation_grows_only_as_needed():
    manager = BlockManager(num_blocks=10, block_size=16)
    seq = make_sequence(prompt_len=20)

    assert manager.allocate(seq, 20)  # 20 tokens -> 2 blocks
    assert len(seq.block_table) == 2
    assert manager.num_free_blocks == 8

    seq.num_computed_tokens = 20
    assert manager.allocate(seq, 1)  # 21st token still fits block 2
    assert len(seq.block_table) == 2, "allocated a block that was not needed"

    seq.num_computed_tokens = 32
    assert manager.allocate(seq, 1)  # 33rd token needs a third block
    assert len(seq.block_table) == 3
    assert manager.num_free_blocks == 7


def test_failed_allocation_changes_nothing():
    manager = BlockManager(num_blocks=2, block_size=16)
    seq = make_sequence(prompt_len=100)

    assert not manager.can_allocate(seq, 100)
    assert not manager.allocate(seq, 100)
    assert seq.block_table == [], "a failed allocation must not partially allocate"
    assert manager.num_free_blocks == 2, "a failed allocation leaked blocks"


def test_blocks_are_never_handed_out_twice():
    manager = BlockManager(num_blocks=16, block_size=16)
    seqs = [make_sequence(prompt_len=32, seq_id=f"s{i}") for i in range(8)]
    for seq in seqs:
        assert manager.allocate(seq, 32)

    handed_out = [block for seq in seqs for block in seq.block_table]
    assert len(handed_out) == len(set(handed_out)) == 16
    assert manager.num_free_blocks == 0


def test_free_returns_every_block():
    manager = BlockManager(num_blocks=8, block_size=16)
    seq = make_sequence(prompt_len=64)
    manager.allocate(seq, 64)
    assert manager.num_free_blocks == 4

    manager.free(seq)
    assert manager.num_free_blocks == 8
    assert seq.block_table == []


def test_slot_mapping_follows_the_block_table():
    manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_sequence(prompt_len=10)
    seq.block_table = [5, 2, 7]  # deliberately not contiguous

    assert manager.slot_mapping(seq, 0, 4) == [20, 21, 22, 23]  # block 5
    assert manager.slot_mapping(seq, 4, 4) == [8, 9, 10, 11]  # block 2
    assert manager.slot_mapping(seq, 8, 2) == [28, 29]  # block 7
    assert manager.slot_mapping(seq, 3, 3) == [23, 8, 9], "block boundary crossed wrongly"


def test_utilization_tracks_allocation():
    manager = BlockManager(num_blocks=4, block_size=16)
    seq = make_sequence(prompt_len=32)
    manager.allocate(seq, 32)
    assert manager.utilization == 0.5


def test_reset_reclaims_everything():
    manager = BlockManager(num_blocks=4, block_size=16)
    manager.allocate(make_sequence(prompt_len=64), 64)
    assert manager.num_free_blocks == 0
    manager.reset()
    assert manager.num_free_blocks == 4
