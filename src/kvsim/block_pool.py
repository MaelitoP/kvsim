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

    def appendleft(self, block: KVCacheBlock) -> None:
        self._blocks[block.block_id] = block
        self._blocks.move_to_end(block.block_id, last=False)

    def remove(self, block: KVCacheBlock) -> None:
        del self._blocks[block.block_id]


class BlockPool:
    """Allocation uses every unhashed free block before evicting cached ones, oldest first."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
        self._free_queue = FreeBlockQueue(self._blocks)
        self._block_by_hash: dict[BlockHash, KVCacheBlock] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    def get_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        return self._block_by_hash.get(block_hash)

    def cache_full_blocks(
        self, blocks: Iterable[KVCacheBlock], block_hashes: Iterable[BlockHash]
    ) -> None:
        for block, block_hash in zip(blocks, block_hashes, strict=True):
            if block.ref_count == 0:
                raise ValueError(f"block {block.block_id} is free")
            if block.block_hash is not None:
                raise ValueError(f"block {block.block_id} is already cached")
            if block_hash not in self._block_by_hash:
                block.block_hash = block_hash
                self._block_by_hash[block_hash] = block

    def reset_prefix_cache(self) -> bool:
        if self.num_free_blocks < self.num_blocks:
            return False
        self._block_by_hash.clear()
        for block in self._blocks:
            block.block_hash = None
        return True

    def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
        if not 0 <= num_blocks <= self.num_free_blocks:
            raise ValueError(f"cannot allocate {num_blocks} blocks, {self.num_free_blocks} free")
        blocks: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self._free_queue.popleft()
            if block.block_hash is not None:
                del self._block_by_hash[block.block_hash]
                block.block_hash = None
            block.ref_count = 1
            blocks.append(block)
        return blocks

    def release(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_count == 0:
                raise ValueError(f"block {block.block_id} is already free")
            block.ref_count -= 1
            if block.ref_count > 0:
                continue
            if block.block_hash is None:
                self._free_queue.appendleft(block)
            else:
                self._free_queue.append(block)

    def retain(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_count == 0:
                self._free_queue.remove(block)
            block.ref_count += 1
