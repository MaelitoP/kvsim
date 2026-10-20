from collections import Counter
from collections.abc import Iterable

from kv_cache_lab.block_pool import BlockPool, KVCacheBlock
from kv_cache_lab.kv_cache_manager import KVCacheManager


def block_ids(blocks: Iterable[KVCacheBlock]) -> list[int]:
    return [block.block_id for block in blocks]


def check_pool(pool: BlockPool) -> None:
    assert [block.block_id for block in pool.blocks] == list(range(pool.num_blocks))
    for block in pool.free_queue:
        assert block is pool.blocks[block.block_id]
    for block in pool.blocks:
        assert block.ref_cnt >= 0
        assert (block.ref_cnt == 0) == (block in pool.free_queue)


def check_manager(manager: KVCacheManager) -> None:
    pool = manager.block_pool
    check_pool(pool)
    assert pool.num_blocks == manager.num_blocks
    references = Counter(
        block.block_id for blocks in manager.req_to_blocks.values() for block in blocks
    )
    for blocks in manager.req_to_blocks.values():
        for block in blocks:
            assert block is pool.blocks[block.block_id]
    for block in pool.blocks:
        assert block.ref_cnt == references[block.block_id]
        assert block.ref_cnt <= 1


def check_scheduler(scheduler: object) -> None:
    pass
