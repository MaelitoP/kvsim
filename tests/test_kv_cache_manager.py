import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariants import block_ids, check_manager
from kvsim.block_hash import BlockHash
from kvsim.block_pool import KVCacheBlock
from kvsim.kv_cache_manager import KVCacheManager
from kvsim.request import Request


def make_request(request_id: str = "r0", prompt_token_ids: tuple[int, ...] = (1, 2, 3)) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_output_tokens=8,
        arrival_step=0,
    )


Snapshot = tuple[
    list[tuple[int, BlockHash | None]], list[int], dict[str, list[int]], dict[str, int]
]


def snapshot(manager: KVCacheManager) -> Snapshot:
    return (
        [(block.ref_count, block.block_hash) for block in manager.block_pool._blocks],
        block_ids(manager.block_pool._free_queue),
        {
            request_id: block_ids(blocks)
            for request_id, blocks in manager._blocks_by_request.items()
        },
        dict(manager._num_cached_blocks),
    )


def admit(manager: KVCacheManager, request: Request) -> list[KVCacheBlock]:
    computed_blocks = manager.get_computed_blocks(request)
    num_computed_tokens = len(computed_blocks) * manager.block_size
    new_blocks = manager.allocate_slots(
        request, request.num_tokens - num_computed_tokens, computed_blocks
    )
    assert new_blocks is not None
    request.num_computed_tokens = request.num_tokens
    check_manager(manager)
    return computed_blocks


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 4), (4, 0)])
def test_rejects_an_empty_pool_or_block(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        KVCacheManager(num_blocks=num_blocks, block_size=block_size, enable_caching=False)


def test_a_new_block_is_allocated_every_block_size_tokens() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=4, enable_caching=False)
    request = make_request()
    for token in range(1, 13):
        new_blocks = manager.allocate_slots(request, num_new_tokens=1)
        assert new_blocks is not None
        assert len(new_blocks) == (1 if token % 4 == 1 else 0)
        request.num_computed_tokens += 1
        assert manager.num_blocks_for(request) == math.ceil(token / 4)
        check_manager(manager)
    assert block_ids(manager._blocks_by_request["r0"]) == [0, 1, 2]


def test_failed_allocation_and_unknown_free_leave_pool_and_tables_unchanged() -> None:
    manager = KVCacheManager(num_blocks=2, block_size=4, enable_caching=False)
    running = make_request("running")
    assert manager.allocate_slots(running, num_new_tokens=4) is not None
    running.num_computed_tokens = 4
    before = snapshot(manager)
    assert manager.allocate_slots(running, num_new_tokens=8) is None
    assert manager.allocate_slots(make_request("waiting"), num_new_tokens=9) is None
    manager.free(make_request("waiting"))
    assert snapshot(manager) == before
    assert "waiting" not in manager._blocks_by_request
    check_manager(manager)


def test_free_returns_blocks_to_the_head_in_table_order() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=1, enable_caching=False)
    request = make_request()
    new_blocks = manager.allocate_slots(request, num_new_tokens=3)
    assert new_blocks is not None
    assert block_ids(new_blocks) == [0, 1, 2]
    assert manager.usage == 0.75
    manager.free(request)
    assert block_ids(manager.block_pool._free_queue) == [0, 1, 2, 3]
    assert manager.num_blocks_for(request) == 0
    assert manager.usage == 0.0
    check_manager(manager)


def test_requests_sharing_a_prefix_share_its_cached_blocks() -> None:
    manager = KVCacheManager(num_blocks=16, block_size=16, enable_caching=True)
    prefix = tuple(range(64))
    first, second, third = (
        make_request(request_id, prefix + (1000 + index,) * 10)
        for index, request_id in enumerate(("first", "second", "third"))
    )
    assert admit(manager, first) == []
    shared = manager._blocks_by_request["first"][:4]
    assert admit(manager, second) == shared
    assert [block.ref_count for block in shared] == [2] * 4
    manager.free(first)
    assert [block.ref_count for block in shared] == [1] * 4
    manager.free(second)
    assert [block.ref_count for block in shared] == [0] * 4
    assert [
        manager.block_pool.get_cached_block(block_hash) for block_hash in first.block_hashes[:4]
    ] == shared
    assert admit(manager, third) == shared
    assert [block.ref_count for block in shared] == [1] * 4


def test_free_hit_blocks_count_against_capacity() -> None:
    manager = KVCacheManager(num_blocks=5, block_size=4, enable_caching=True)
    cached = make_request("cached", tuple(range(16)))
    admit(manager, cached)
    admit(manager, make_request("holder", (99,)))
    manager.free(cached)
    longer = make_request("longer", tuple(range(20)))
    computed_blocks = manager.get_computed_blocks(longer)
    assert len(computed_blocks) == 4
    before = snapshot(manager)
    assert manager.allocate_slots(longer, 4, computed_blocks) is None
    assert snapshot(manager) == before
    check_manager(manager)


def test_a_hit_at_the_head_of_the_free_queue_is_retained_before_allocating() -> None:
    manager = KVCacheManager(num_blocks=2, block_size=4, enable_caching=True)
    oldest = make_request("oldest", (1, 2, 3, 4))
    newest = make_request("newest", (5, 6, 7, 8))
    for request in (oldest, newest):
        admit(manager, request)
    oldest_blocks = manager._blocks_by_request["oldest"]
    for request in (oldest, newest):
        manager.free(request)
    assert list(manager.block_pool._free_queue)[:1] == oldest_blocks
    extending = make_request("extending", (1, 2, 3, 4, 9))
    assert admit(manager, extending) == oldest_blocks


def test_the_lookup_stops_at_the_first_evicted_block() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=4, enable_caching=True)
    cached = make_request("cached", tuple(range(12)))
    admit(manager, cached)
    first, *later = manager._blocks_by_request["cached"]
    manager.free(cached)
    manager.block_pool.retain(later)
    manager.block_pool.release(later)
    manager.block_pool.release(manager.block_pool.allocate(2))
    assert first.block_hash is None
    assert all(block.block_hash is not None for block in later)
    extending = make_request("extending", tuple(range(13)))
    assert manager.get_computed_blocks(extending) == []


@given(
    block_size=st.integers(1, 8),
    steps=st.lists(st.tuples(st.integers(0, 2), st.none() | st.integers(0, 12))),
)
def test_requests_grow_by_ceil_and_never_share_blocks(
    block_size: int, steps: list[tuple[int, int | None]]
) -> None:
    manager = KVCacheManager(num_blocks=12, block_size=block_size, enable_caching=False)
    requests = [make_request(f"r{index}") for index in range(3)]
    for index, num_new_tokens in steps:
        request = requests[index]
        if num_new_tokens is None:
            manager.free(request)
            request.num_computed_tokens = 0
        else:
            before = snapshot(manager)
            held = before[2].get(request.request_id, [])
            required = math.ceil((request.num_computed_tokens + num_new_tokens) / block_size)
            new_blocks = manager.allocate_slots(request, num_new_tokens)
            if new_blocks is None:
                assert required - len(held) > manager.num_free_blocks
                assert snapshot(manager) == before
            else:
                request.num_computed_tokens += num_new_tokens
                assert len(held) + len(new_blocks) == required
                assert block_ids(
                    manager._blocks_by_request[request.request_id]
                ) == held + block_ids(new_blocks)
        check_manager(manager)
