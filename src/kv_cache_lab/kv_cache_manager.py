from dataclasses import dataclass, field

from kv_cache_lab.block_pool import BlockPool, KVCacheBlock
from kv_cache_lab.request import Request


@dataclass(eq=False)
class KVCacheManager:
    """Owns the block pool and one append-only block table per request.

    A request whose first `n` tokens are computed or scheduled holds `ceil(n / block_size)`
    blocks. `allocate_slots` returns `None` and changes nothing when the pool cannot supply the
    missing blocks. `request_id` must be unique among live requests; `num_new_tokens` is never
    negative.
    """

    num_blocks: int
    block_size: int
    block_pool: BlockPool = field(init=False, repr=False)
    req_to_blocks: dict[str, list[KVCacheBlock]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_blocks < 1 or self.block_size < 1:
            raise ValueError(
                f"num_blocks and block_size must be >= 1, got {self.num_blocks}, {self.block_size}"
            )
        self.block_pool = BlockPool(self.num_blocks)
        self.req_to_blocks = {}

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free_blocks

    @property
    def usage(self) -> float:
        return (self.num_blocks - self.num_free_blocks) / self.num_blocks

    def num_blocks_for(self, request: Request) -> int:
        return len(self.req_to_blocks.get(request.request_id, []))

    def allocate_slots(self, request: Request, num_new_tokens: int) -> list[KVCacheBlock] | None:
        num_tokens = request.num_computed_tokens + num_new_tokens
        num_required_blocks = -(-num_tokens // self.block_size)
        num_new_blocks = num_required_blocks - self.num_blocks_for(request)
        if num_new_blocks > self.num_free_blocks:
            return None
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        self.req_to_blocks.setdefault(request.request_id, []).extend(new_blocks)
        return new_blocks

    def free(self, request: Request) -> None:
        blocks = self.req_to_blocks.pop(request.request_id, [])
        self.block_pool.free_blocks(reversed(blocks))
