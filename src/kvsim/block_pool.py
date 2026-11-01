from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from kvsim.block_hash import BlockHash


@dataclass(eq=False)
class KVCacheBlock:
    block_id: int
    ref_count: int = 0
    block_hash: BlockHash | None = None


class FreeBlockQueue:
    """Free blocks, iterated in the order they will be allocated."""

    def __init__(self, blocks: Iterable[KVCacheBlock]) -> None:
        self._blocks: OrderedDict[int, KVCacheBlock] = OrderedDict(
            (block.block_id, block) for block in blocks
        )

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._blocks

    def __iter__(self) -> Iterator[KVCacheBlock]:
        return iter(self._blocks.values())

    def popleft(self) -> KVCacheBlock:
        _, block = self._blocks.popitem(last=False)
        return block

    def append(self, block: KVCacheBlock) -> None:
        self._blocks[block.block_id] = block

    def remove(self, block: KVCacheBlock) -> None:
        del self._blocks[block.block_id]


class BlockPool:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
        self._free_queue = FreeBlockQueue(self._blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
        if not 0 <= num_blocks <= self.num_free_blocks:
            raise ValueError(f"cannot allocate {num_blocks} blocks, {self.num_free_blocks} free")
        blocks: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self._free_queue.popleft()
            block.ref_count = 1
            blocks.append(block)
        return blocks

    def release(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_count == 0:
                raise ValueError(f"block {block.block_id} is already free")
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_queue.append(block)

    def retain(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_count == 0:
                self._free_queue.remove(block)
            block.ref_count += 1
