import math
from collections import Counter
from collections.abc import Iterable, Mapping

from kvsim.block_pool import BlockPool, KVCacheBlock
from kvsim.kv_cache_manager import KVCacheManager
from kvsim.request import Request, RequestStatus
from kvsim.scheduler import Scheduler


def block_ids(blocks: Iterable[KVCacheBlock]) -> list[int]:
    return [block.block_id for block in blocks]


def request_ids(requests: Iterable[Request]) -> list[str]:
    return [request.request_id for request in requests]


def check_pool(pool: BlockPool) -> None:
    assert [block.block_id for block in pool._blocks] == list(range(pool.num_blocks))
    for block in pool._free_queue:
        assert block is pool._blocks[block.block_id]
    for block in pool._blocks:
        assert block.ref_count >= 0
        assert (block.ref_count == 0) == (block in pool._free_queue)


def check_manager(manager: KVCacheManager) -> None:
    pool = manager.block_pool
    check_pool(pool)
    assert pool.num_blocks == manager.num_blocks
    references = Counter(
        block.block_id for blocks in manager._blocks_by_request.values() for block in blocks
    )
    for blocks in manager._blocks_by_request.values():
        for block in blocks:
            assert block is pool._blocks[block.block_id]
    for block in pool._blocks:
        assert block.ref_count == references[block.block_id]
        assert block.ref_count <= 1


def check_scheduler(scheduler: Scheduler) -> None:
    manager = scheduler.kv_cache_manager
    check_manager(manager)
    block_size = scheduler.config.block_size
    pending = scheduler._pending_batch
    scheduled: Mapping[str, int] = pending.num_scheduled_tokens if pending is not None else {}
    waiting = list(scheduler._waiting)
    running = scheduler._running
    if pending is not None:
        assert set(scheduled) == set(request_ids(running))
        assert all(num_tokens >= 1 for num_tokens in scheduled.values())
        assert sum(scheduled.values()) <= scheduler.config.max_tokens_per_step
    assert len(running) <= scheduler.config.max_running_requests
    assert len(waiting) + len(running) == len(scheduler._requests)
    assert set(request_ids(waiting + running)) == set(scheduler._requests)
    assert set(manager._blocks_by_request) <= set(request_ids(running))
    for request in waiting:
        assert scheduler._requests[request.request_id] is request
        assert request.status is RequestStatus.WAITING
        assert request.num_computed_tokens == 0
        assert manager.num_blocks_for(request) == 0
    for request in running:
        assert scheduler._requests[request.request_id] is request
        assert request.status is RequestStatus.RUNNING
        num_reserved_tokens = request.num_computed_tokens + scheduled.get(request.request_id, 0)
        assert num_reserved_tokens <= request.num_tokens
        assert manager.num_blocks_for(request) == math.ceil(num_reserved_tokens / block_size)
        if pending is None:
            if request.output_token_ids:
                assert request.num_computed_tokens == request.num_tokens - 1
            else:
                assert 1 <= request.num_computed_tokens < request.num_prompt_tokens
