"""The paged spike KV cache itself.

Per layer, two ``int32 [T, H_kv, num_slots, W]`` slabs -- keys and values. Slots
are the flat address space the block manager hands out; a block is just
``block_size`` consecutive slots. Laying the slot axis second-to-last means a
block is contiguous for a fixed ``(t, head)``, which is exactly the access
pattern of the decode kernel's inner loop.

The last block is reserved and never given to the allocator: CUDA-graph decode
batches are padded up to a bucket size, and the padding rows still write a key
and a value somewhere. They write there.
"""
from __future__ import annotations

import torch

from ..kernels.packing import packed_width


class PagedSpikeCache:
    def __init__(
        self,
        num_layers: int,
        timesteps: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        device: str | torch.device = "cuda",
        layer_devices: list[str | torch.device] | None = None,
    ) -> None:
        """``layer_devices`` places each layer's slabs individually.

        A hybrid placement computes some layers on the CPU, and a layer's cache
        has to live where it computes. Both sides get the *same* ``num_blocks``
        so slot ids mean the same thing everywhere and one ``BlockManager``
        still owns the whole address space -- which is affordable only because
        a packed spike block is ~96 KiB, so mirroring the block count into host
        RAM costs almost nothing.
        """
        if num_blocks < 2:
            raise ValueError("need at least 2 blocks (one is reserved for padding)")
        self.num_layers = num_layers
        self.timesteps = timesteps
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.words = packed_width(head_dim)

        shape = (timesteps, num_kv_heads, num_blocks * block_size, self.words)
        self.layer_devices = [
            torch.device(d) for d in (layer_devices or [device] * num_layers)
        ]
        if len(self.layer_devices) != num_layers:
            raise ValueError(
                f"layer_devices has {len(self.layer_devices)} entries for {num_layers} layers"
            )
        self.k = [torch.zeros(shape, dtype=torch.int32, device=d) for d in self.layer_devices]
        self.v = [torch.zeros(shape, dtype=torch.int32, device=d) for d in self.layer_devices]
        self.layers = list(zip(self.k, self.v))

    # -- sizing -----------------------------------------------------------

    @staticmethod
    def bytes_per_block(
        num_layers: int, timesteps: int, num_kv_heads: int, head_dim: int, block_size: int
    ) -> int:
        return 2 * num_layers * timesteps * num_kv_heads * block_size * packed_width(head_dim) * 4

    @property
    def nbytes(self) -> int:
        return self.bytes_per_block(
            self.num_layers, self.timesteps, self.num_kv_heads, self.head_dim, self.block_size
        ) * self.num_blocks

    @property
    def num_allocatable_blocks(self) -> int:
        """Blocks the allocator may hand out -- everything but the pad block."""
        return self.num_blocks - 1

    @property
    def pad_block(self) -> int:
        return self.num_blocks - 1

    @property
    def pad_slot(self) -> int:
        return self.pad_block * self.block_size

    @property
    def capacity_tokens(self) -> int:
        return self.num_allocatable_blocks * self.block_size

    @property
    def device_summary(self) -> str:
        kinds = sorted({d.type for d in self.layer_devices})
        return "+".join(kinds)

    def reset(self) -> None:
        for t in self.k + self.v:
            t.zero_()
