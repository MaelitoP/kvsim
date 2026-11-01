import math
from collections import Counter
from collections.abc import Iterable, Mapping

from kvsim.block_hash import hash_full_blocks
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
    for block_hash, block in pool._block_by_hash.items():
        assert block is pool._blocks[block.block_id]
        assert block.block_hash == block_hash
    assert sum(block.block_hash is not None for block in pool._blocks) == len(pool._block_by_hash)
    cached_flags = [block.block_hash is not None for block in pool._free_queue]
    assert cached_flags == sorted(cached_flags)


def check_manager(manager: KVCacheManager) -> None:
    pool = manager.block_pool
    check_pool(pool)
    assert pool.num_blocks == manager.num_blocks
    references = Counter(
        block.block_id for blocks in manager._blocks_by_request.values() for block in blocks
    )
    for request_id, blocks in manager._blocks_by_request.items():
        assert len(set(block_ids(blocks))) == len(blocks)
        for block in blocks:
            assert block is pool._blocks[block.block_id]
        num_cached_blocks = manager._num_cached_blocks.get(request_id, 0)
        assert num_cached_blocks <= len(blocks)
        assert all(block.block_hash is None for block in blocks[num_cached_blocks:])
    assert set(manager._num_cached_blocks) <= set(manager._blocks_by_request)
    for block in pool._blocks:
        assert block.ref_count == references[block.block_id]
    if not manager.enable_caching:
        assert not manager._num_cached_blocks
        assert all(block.ref_count <= 1 for block in pool._blocks)
        assert not pool._block_by_hash


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
        if pending.preempted:
            assert pending.scheduled_resumed == pending.scheduled_new == ()
            assert set(pending.preempted) <= set(request_ids(waiting))
    assert len(running) <= scheduler.config.max_running_requests
    assert len(waiting) + len(running) == len(scheduler._requests)
    assert set(request_ids(waiting + running)) == set(scheduler._requests)
    assert set(manager._blocks_by_request) <= set(request_ids(running))
    assert scheduler.prefix_cache_hits <= scheduler.prefix_cache_queries
    assert scheduler.prefix_cache_hits % block_size == 0
    if not scheduler.config.enable_prefix_caching:
        assert scheduler.prefix_cache_queries == 0
        assert all(not request.block_hashes for request in waiting + running)
    preempted_flags = [request.status is RequestStatus.PREEMPTED for request in waiting]
    assert preempted_flags == sorted(preempted_flags, reverse=True)
    for request in waiting:
        assert scheduler._requests[request.request_id] is request
        assert request.num_computed_tokens == 0
        assert manager.num_blocks_for(request) == 0
        if request.status is RequestStatus.PREEMPTED:
            assert request.num_preemptions >= 1
            assert request.first_scheduled_step is not None
        else:
            assert request.status is RequestStatus.WAITING
            assert request.num_preemptions == 0
            assert request.first_scheduled_step is None
            assert not request.output_token_ids
    for request in running:
        assert scheduler._requests[request.request_id] is request
        assert request.status is RequestStatus.RUNNING
        num_reserved_tokens = request.num_computed_tokens + scheduled.get(request.request_id, 0)
        assert num_reserved_tokens <= request.num_tokens
        assert manager.num_blocks_for(request) == math.ceil(num_reserved_tokens / block_size)
        if scheduler.config.enable_prefix_caching:
            num_cached_blocks = manager._num_cached_blocks[request.request_id]
            assert num_cached_blocks == num_reserved_tokens // block_size
            block_hashes = request.block_hashes
            assert len(block_hashes) >= num_cached_blocks
            assert (
                block_hashes
                == hash_full_blocks(block_size, request.all_token_ids)[: len(block_hashes)]
            )
            blocks = manager._blocks_by_request[request.request_id]
            for block, block_hash in zip(
                blocks[:num_cached_blocks], block_hashes[:num_cached_blocks], strict=True
            ):
                assert block.block_hash in (None, block_hash)
        if pending is None:
            assert 1 <= request.num_computed_tokens < request.num_tokens
            if request.output_token_ids and request.num_preemptions == 0:
                assert request.num_computed_tokens == request.num_tokens - 1
