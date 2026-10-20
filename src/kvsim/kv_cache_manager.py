from kvsim.block_pool import BlockPool, KVCacheBlock
from kvsim.request import Request


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError(
                f"num_blocks and block_size must be >= 1, got {num_blocks}, {block_size}"
            )
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_pool = BlockPool(num_blocks)
        self._blocks_by_request: dict[str, list[KVCacheBlock]] = {}

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free_blocks

    @property
    def usage(self) -> float:
        return (self.num_blocks - self.num_free_blocks) / self.num_blocks

    def num_blocks_for(self, request: Request) -> int:
        return len(self._blocks_by_request.get(request.request_id, []))

    def allocate_slots(self, request: Request, num_new_tokens: int) -> list[KVCacheBlock] | None:
        """Return None without changing state when too few blocks are free."""
        num_tokens = request.num_computed_tokens + num_new_tokens
        num_required_blocks = -(-num_tokens // self.block_size)
        num_new_blocks = num_required_blocks - self.num_blocks_for(request)
        if num_new_blocks > self.num_free_blocks:
            return None
        new_blocks = self.block_pool.allocate(num_new_blocks)
        self._blocks_by_request.setdefault(request.request_id, []).extend(new_blocks)
        return new_blocks

    def free(self, request: Request) -> None:
        blocks = self._blocks_by_request.pop(request.request_id, [])
        self.block_pool.release(reversed(blocks))
