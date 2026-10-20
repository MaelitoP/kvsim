import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from invariants import block_ids, check_pool
from kv_cache_lab.block_pool import BlockPool


def test_allocation_follows_free_queue_order() -> None:
    pool = BlockPool(num_blocks=4)
    first = pool.get_new_blocks(2)
    assert block_ids(first) == [0, 1]
    pool.free_blocks([first[0]])
    assert block_ids(pool.free_queue) == [2, 3, 0]
    assert block_ids(pool.get_new_blocks(3)) == [2, 3, 0]
    assert pool.num_free_blocks == 0
    check_pool(pool)


def test_free_appends_to_the_tail_in_the_given_order() -> None:
    pool = BlockPool(num_blocks=3)
    blocks = pool.get_new_blocks(3)
    pool.free_blocks([blocks[2], blocks[0]])
    pool.free_blocks([blocks[1]])
    assert block_ids(pool.free_queue) == [2, 0, 1]
    check_pool(pool)


def test_touch_removes_a_free_block_from_the_queue() -> None:
    pool = BlockPool(num_blocks=3)
    block = pool.blocks[1]
    pool.touch([block])
    assert block.ref_cnt == 1
    assert block_ids(pool.get_new_blocks(2)) == [0, 2]
    check_pool(pool)


def test_shared_block_is_freed_when_the_last_reference_goes() -> None:
    pool = BlockPool(num_blocks=2)
    (block,) = pool.get_new_blocks(1)
    pool.touch([block])
    assert block.ref_cnt == 2
    pool.free_blocks([block])
    assert block.ref_cnt == 1
    assert block not in pool.free_queue
    pool.free_blocks([block])
    assert block_ids(pool.free_queue) == [1, 0]
    check_pool(pool)


def test_failed_allocation_raises_without_side_effects() -> None:
    pool = BlockPool(num_blocks=3)
    pool.get_new_blocks(1)
    before = ([block.ref_cnt for block in pool.blocks], block_ids(pool.free_queue))
    with pytest.raises(ValueError, match="cannot allocate 3 blocks, 2 free"):
        pool.get_new_blocks(3)
    with pytest.raises(ValueError, match="cannot allocate -1 blocks"):
        pool.get_new_blocks(-1)
    assert ([block.ref_cnt for block in pool.blocks], block_ids(pool.free_queue)) == before
    check_pool(pool)


def test_double_free_raises() -> None:
    pool = BlockPool(num_blocks=1)
    blocks = pool.get_new_blocks(1)
    pool.free_blocks(blocks)
    with pytest.raises(ValueError, match="block 0 is already free"):
        pool.free_blocks(blocks)
    check_pool(pool)


class BlockPoolMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.pool = BlockPool(num_blocks=8)
        self.initial_blocks = list(self.pool.blocks)

    @rule(data=st.data())
    def allocate(self, data: st.DataObject) -> None:
        count = data.draw(st.integers(0, self.pool.num_free_blocks))
        blocks = self.pool.get_new_blocks(count)
        assert all(block.ref_cnt == 1 for block in blocks)

    @precondition(lambda self: any(block.ref_cnt > 0 for block in self.pool.blocks))
    @rule(data=st.data())
    def free(self, data: st.DataObject) -> None:
        allocated = [block for block in self.pool.blocks if block.ref_cnt > 0]
        self.pool.free_blocks(data.draw(st.lists(st.sampled_from(allocated), unique=True)))

    @rule(data=st.data())
    def touch(self, data: st.DataObject) -> None:
        self.pool.touch(data.draw(st.lists(st.sampled_from(self.pool.blocks), unique=True)))

    @invariant()
    def pool_is_consistent(self) -> None:
        check_pool(self.pool)
        assert self.pool.blocks == self.initial_blocks


TestBlockPool = BlockPoolMachine.TestCase
