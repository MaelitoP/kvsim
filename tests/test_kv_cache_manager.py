import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariants import block_ids, check_manager
from kvsim.kv_cache_manager import KVCacheManager
from kvsim.request import Request


def make_request(request_id: str = "r0") -> Request:
    return Request(
        request_id=request_id, prompt_token_ids=(1, 2, 3), max_output_tokens=8, arrival_step=0
    )


def snapshot(manager: KVCacheManager) -> tuple[list[int], list[int], dict[str, list[int]]]:
    return (
        [block.ref_count for block in manager.block_pool._blocks],
        block_ids(manager.block_pool._free_queue),
        {
            request_id: block_ids(blocks)
            for request_id, blocks in manager._blocks_by_request.items()
        },
    )


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 4), (4, 0)])
def test_rejects_an_empty_pool_or_block(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        KVCacheManager(num_blocks=num_blocks, block_size=block_size)


def test_a_new_block_is_allocated_every_block_size_tokens() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=4)
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
    manager = KVCacheManager(num_blocks=2, block_size=4)
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


def test_free_returns_every_block_with_the_tail_block_first() -> None:
    manager = KVCacheManager(num_blocks=4, block_size=1)
    request = make_request()
    new_blocks = manager.allocate_slots(request, num_new_tokens=3)
    assert new_blocks is not None
    assert block_ids(new_blocks) == [0, 1, 2]
    assert manager.usage == 0.75
    manager.free(request)
    assert block_ids(manager.block_pool._free_queue) == [3, 2, 1, 0]
    assert manager.num_blocks_for(request) == 0
    assert manager.usage == 0.0
    check_manager(manager)


@given(
    block_size=st.integers(1, 8),
    steps=st.lists(st.tuples(st.integers(0, 2), st.none() | st.integers(0, 12))),
)
def test_requests_grow_by_ceil_and_never_share_blocks(
    block_size: int, steps: list[tuple[int, int | None]]
) -> None:
    manager = KVCacheManager(num_blocks=12, block_size=block_size)
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
