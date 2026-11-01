import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from invariants import block_ids, check_pool
from kvsim.block_hash import NONE_HASH, BlockHash, hash_block_tokens
from kvsim.block_pool import BlockPool


def make_hash(token: int) -> BlockHash:
    return hash_block_tokens(NONE_HASH, [token])


def test_allocation_follows_free_queue_order() -> None:
    pool = BlockPool(num_blocks=4)
    first = pool.allocate(2)
    assert block_ids(first) == [0, 1]
    pool.release([first[0]])
    assert block_ids(pool._free_queue) == [0, 2, 3]
    assert block_ids(pool.allocate(3)) == [0, 2, 3]
    assert pool.num_free_blocks == 0
    check_pool(pool)


def test_unhashed_blocks_are_reused_last_in_first_out() -> None:
    pool = BlockPool(num_blocks=3)
    blocks = pool.allocate(3)
    pool.release([blocks[2], blocks[0]])
    pool.release([blocks[1]])
    assert block_ids(pool.allocate(3)) == [1, 0, 2]
    check_pool(pool)


def test_cached_blocks_are_evicted_in_free_order_after_every_unhashed_block() -> None:
    pool = BlockPool(num_blocks=5)
    blocks = pool.allocate(5)
    pool.cache_full_blocks(blocks[:3], [make_hash(token) for token in range(3)])
    pool.release(reversed(blocks[:4]))
    pool.release([blocks[4]])
    assert block_ids(pool._free_queue) == [4, 3, 2, 1, 0]
    assert block_ids(pool.allocate(3)) == [4, 3, 2]
    assert blocks[2].block_hash is None
    assert pool.get_cached_block(make_hash(2)) is None
    assert pool.get_cached_block(make_hash(1)) is blocks[1]
    assert len(pool._block_by_hash) == 2
    check_pool(pool)


def test_a_freed_cached_block_stays_a_hit_and_can_be_retained() -> None:
    pool = BlockPool(num_blocks=2)
    (block,) = pool.allocate(1)
    pool.cache_full_blocks([block], [make_hash(7)])
    pool.release([block])
    assert pool.get_cached_block(make_hash(7)) is block
    pool.retain([block])
    assert block.ref_count == 1
    assert block.block_hash == make_hash(7)
    check_pool(pool)


def test_a_hash_already_cached_keeps_its_block_and_the_duplicate_stays_unhashed() -> None:
    pool = BlockPool(num_blocks=2)
    first, duplicate = pool.allocate(2)
    pool.cache_full_blocks([first], [make_hash(7)])
    pool.cache_full_blocks([duplicate], [make_hash(7)])
    assert pool.get_cached_block(make_hash(7)) is first
    assert duplicate.block_hash is None
    check_pool(pool)


def test_caching_stops_at_a_free_or_cached_block_and_keeps_the_earlier_ones() -> None:
    pool = BlockPool(num_blocks=3)
    cached, valid, free = pool.allocate(3)
    pool.cache_full_blocks([cached], [make_hash(0)])
    pool.release([free])
    with pytest.raises(ValueError, match="block 2 is free"):
        pool.cache_full_blocks([valid, free], [make_hash(1), make_hash(2)])
    with pytest.raises(ValueError, match="block 0 is already cached"):
        pool.cache_full_blocks([cached], [make_hash(3)])
    assert pool.get_cached_block(make_hash(1)) is valid
    assert pool.get_cached_block(make_hash(2)) is None
    assert pool.get_cached_block(make_hash(3)) is None
    check_pool(pool)


def test_reset_prefix_cache_needs_every_block_free() -> None:
    pool = BlockPool(num_blocks=2)
    blocks = pool.allocate(2)
    pool.cache_full_blocks(blocks, [make_hash(0), make_hash(1)])
    pool.release(blocks[1:])
    assert not pool.reset_prefix_cache()
    assert len(pool._block_by_hash) == 2
    pool.release(blocks[:1])
    assert pool.reset_prefix_cache()
    assert pool._block_by_hash == {}
    check_pool(pool)


def test_retain_removes_a_free_block_from_the_queue() -> None:
    pool = BlockPool(num_blocks=3)
    block = pool._blocks[1]
    pool.retain([block])
    assert block.ref_count == 1
    assert block_ids(pool.allocate(2)) == [0, 2]
    check_pool(pool)


def test_shared_block_is_freed_when_the_last_reference_goes() -> None:
    pool = BlockPool(num_blocks=2)
    (block,) = pool.allocate(1)
    pool.retain([block])
    assert block.ref_count == 2
    pool.release([block])
    assert block.ref_count == 1
    assert block not in pool._free_queue
    pool.release([block])
    assert block_ids(pool._free_queue) == [0, 1]
    check_pool(pool)


def test_failed_allocation_raises_without_side_effects() -> None:
    pool = BlockPool(num_blocks=3)
    pool.allocate(1)
    before = ([block.ref_count for block in pool._blocks], block_ids(pool._free_queue))
    with pytest.raises(ValueError, match="cannot allocate 3 blocks, 2 free"):
        pool.allocate(3)
    with pytest.raises(ValueError, match="cannot allocate -1 blocks"):
        pool.allocate(-1)
    assert ([block.ref_count for block in pool._blocks], block_ids(pool._free_queue)) == before
    check_pool(pool)


def test_double_free_raises() -> None:
    pool = BlockPool(num_blocks=1)
    blocks = pool.allocate(1)
    pool.release(blocks)
    with pytest.raises(ValueError, match="block 0 is already free"):
        pool.release(blocks)
    check_pool(pool)


HASHES = [make_hash(token) for token in range(4)]


class BlockPoolMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.pool = BlockPool(num_blocks=8)
        self.initial_blocks = list(self.pool._blocks)

    @rule(data=st.data())
    def allocate(self, data: st.DataObject) -> None:
        count = data.draw(st.integers(0, self.pool.num_free_blocks))
        expected = list(self.pool._free_queue)[:count]
        blocks = self.pool.allocate(count)
        assert blocks == expected
        assert all(block.ref_count == 1 and block.block_hash is None for block in blocks)

    @precondition(lambda self: any(block.ref_count > 0 for block in self.pool._blocks))
    @rule(data=st.data())
    def free(self, data: st.DataObject) -> None:
        allocated = [block for block in self.pool._blocks if block.ref_count > 0]
        self.pool.release(data.draw(st.lists(st.sampled_from(allocated), unique=True)))

    @rule(data=st.data())
    def retain(self, data: st.DataObject) -> None:
        self.pool.retain(data.draw(st.lists(st.sampled_from(self.pool._blocks), unique=True)))

    @precondition(
        lambda self: any(
            block.ref_count > 0 and block.block_hash is None for block in self.pool._blocks
        )
    )
    @rule(data=st.data())
    def cache(self, data: st.DataObject) -> None:
        cacheable = [
            block for block in self.pool._blocks if block.ref_count > 0 and block.block_hash is None
        ]
        blocks = data.draw(st.lists(st.sampled_from(cacheable), unique=True))
        block_hashes = [data.draw(st.sampled_from(HASHES)) for _ in blocks]
        expected = dict(self.pool._block_by_hash)
        for block, block_hash in zip(blocks, block_hashes, strict=True):
            expected.setdefault(block_hash, block)
        self.pool.cache_full_blocks(blocks, block_hashes)
        assert self.pool._block_by_hash == expected

    @rule()
    def reset_prefix_cache(self) -> None:
        all_free = self.pool.num_free_blocks == self.pool.num_blocks
        expected = {} if all_free else dict(self.pool._block_by_hash)
        assert self.pool.reset_prefix_cache() == all_free
        assert self.pool._block_by_hash == expected

    @invariant()
    def pool_is_consistent(self) -> None:
        check_pool(self.pool)
        assert self.pool._blocks == self.initial_blocks


TestBlockPool = BlockPoolMachine.TestCase
