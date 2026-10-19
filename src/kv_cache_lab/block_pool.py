from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass(eq=False)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: bytes | None = None


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


@dataclass(eq=False)
class BlockPool:
    """Fixed set of blocks, each either free (`ref_cnt == 0`, in `free_queue`) or allocated.

    `get_new_blocks` pops from the head of the free queue; `free_blocks` re-queues a block at the
    tail when its last reference goes, in the order the caller passes them.
    """

    num_blocks: int
    blocks: list[KVCacheBlock] = field(init=False, repr=False)
    free_queue: FreeBlockQueue = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.blocks = [KVCacheBlock(block_id) for block_id in range(self.num_blocks)]
        self.free_queue = FreeBlockQueue(self.blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if not 0 <= num_blocks <= self.num_free_blocks:
            raise ValueError(f"cannot allocate {num_blocks} blocks, {self.num_free_blocks} free")
        blocks: list[KVCacheBlock] = []
        for _ in range(num_blocks):
            block = self.free_queue.popleft()
            block.ref_cnt = 1
            blocks.append(block)
        return blocks

    def free_blocks(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_cnt == 0:
                raise ValueError(f"block {block.block_id} is already free")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self.free_queue.append(block)

    def touch(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_queue.remove(block)
            block.ref_cnt += 1
