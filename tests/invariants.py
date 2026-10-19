from kv_cache_lab.block_pool import BlockPool


def check_pool(pool: BlockPool) -> None:
    assert [block.block_id for block in pool.blocks] == list(range(pool.num_blocks))
    for block in pool.free_queue:
        assert block is pool.blocks[block.block_id]
    for block in pool.blocks:
        assert block.ref_cnt >= 0
        assert (block.ref_cnt == 0) == (block in pool.free_queue)


def check_manager(manager: object) -> None:
    pass


def check_scheduler(scheduler: object) -> None:
    pass
