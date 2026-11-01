from collections.abc import Sequence

from kvsim.block_pool import BlockPool, KVCacheBlock
from kvsim.request import Request


class KVCacheManager:
    """`allocate_slots` changes nothing when short of blocks; hits may only start an empty table."""

    def __init__(self, num_blocks: int, block_size: int, *, enable_caching: bool) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError(
                f"num_blocks and block_size must be >= 1, got {num_blocks}, {block_size}"
            )
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_caching = enable_caching
        self.block_pool = BlockPool(num_blocks)
        self._blocks_by_request: dict[str, list[KVCacheBlock]] = {}
        self._num_cached_blocks: dict[str, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free_blocks

    @property
    def usage(self) -> float:
        return (self.num_blocks - self.num_free_blocks) / self.num_blocks

    def num_blocks_for(self, request: Request) -> int:
        return len(self._blocks_by_request.get(request.request_id, []))

    def get_computed_blocks(self, request: Request) -> list[KVCacheBlock]:
        if not self.enable_caching:
            return []
        request.update_block_hashes(self.block_size)
        computed_blocks: list[KVCacheBlock] = []
        max_num_hit_blocks = (request.num_tokens - 1) // self.block_size
        for block_hash in request.block_hashes[:max_num_hit_blocks]:
            block = self.block_pool.get_cached_block(block_hash)
            if block is None:
                break
            computed_blocks.append(block)
        return computed_blocks

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock] = (),
    ) -> list[KVCacheBlock] | None:
        num_reserved_tokens = (
            request.num_computed_tokens
            + len(new_computed_blocks) * self.block_size
            + num_new_tokens
        )
        num_required_blocks = -(-num_reserved_tokens // self.block_size)
        num_new_blocks = (
            num_required_blocks - self.num_blocks_for(request) - len(new_computed_blocks)
        )
        num_free_hits = sum(block.ref_count == 0 for block in new_computed_blocks)
        if num_new_blocks + num_free_hits > self.num_free_blocks:
            return None
        self.block_pool.retain(new_computed_blocks)
        new_blocks = self.block_pool.allocate(num_new_blocks)
        blocks = self._blocks_by_request.setdefault(request.request_id, [])
        blocks.extend(new_computed_blocks)
        blocks.extend(new_blocks)
        if self.enable_caching:
            request.update_block_hashes(self.block_size)
            num_cached_blocks = self._num_cached_blocks.get(request.request_id, 0) + len(
                new_computed_blocks
            )
            num_full_blocks = num_reserved_tokens // self.block_size
            self.block_pool.cache_full_blocks(
                blocks[num_cached_blocks:num_full_blocks],
                request.block_hashes[num_cached_blocks:num_full_blocks],
            )
            self._num_cached_blocks[request.request_id] = num_full_blocks
        return new_blocks

    def free(self, request: Request) -> None:
        blocks = self._blocks_by_request.pop(request.request_id, [])
        self._num_cached_blocks.pop(request.request_id, None)
        self.block_pool.release(reversed(blocks))
