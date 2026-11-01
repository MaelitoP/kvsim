from hypothesis import given
from hypothesis import strategies as st

from kvsim.block_hash import hash_full_blocks

TOKEN_IDS = st.integers(-(2**63), 2**63 - 1)


def test_hashes_match_the_pinned_encoding() -> None:
    assert [block_hash.hex() for block_hash in hash_full_blocks(2, [1, 2, 3, 4, 5])] == [
        "bc46af79e96bc03a3b5a1a88aeab5079b0dd4c6dfc5f9c14e318fec7de310f6b",
        "456085d4fd464c1c3933d729ca9c0ca386011432460b3a08c6ccf903fbed4d9c",
    ]


@given(block_size=st.integers(1, 8), token_ids=st.lists(TOKEN_IDS, max_size=40))
def test_only_full_blocks_are_hashed(block_size: int, token_ids: list[int]) -> None:
    assert len(hash_full_blocks(block_size, token_ids)) == len(token_ids) // block_size


@given(block_size=st.integers(1, 8), num_blocks=st.integers(1, 5), data=st.data())
def test_a_changed_token_changes_its_block_hash_and_every_later_one(
    block_size: int, num_blocks: int, data: st.DataObject
) -> None:
    num_full_tokens = num_blocks * block_size
    token_ids = data.draw(
        st.lists(TOKEN_IDS, min_size=num_full_tokens, max_size=num_full_tokens + block_size - 1)
    )
    position = data.draw(st.integers(0, num_full_tokens - 1))
    changed = list(token_ids)
    changed[position] = data.draw(TOKEN_IDS.filter(lambda token: token != token_ids[position]))
    original_hashes = hash_full_blocks(block_size, token_ids)
    changed_hashes = hash_full_blocks(block_size, changed)
    changed_block = position // block_size
    assert changed_hashes[:changed_block] == original_hashes[:changed_block]
    assert all(
        before != after
        for before, after in zip(
            original_hashes[changed_block:], changed_hashes[changed_block:], strict=True
        )
    )


@given(
    block_size=st.integers(1, 8),
    num_head_blocks=st.integers(1, 4),
    tail=st.lists(TOKEN_IDS, max_size=20),
    data=st.data(),
)
def test_a_chain_continues_from_its_last_hash(
    block_size: int, num_head_blocks: int, tail: list[int], data: st.DataObject
) -> None:
    head_size = num_head_blocks * block_size
    head = data.draw(st.lists(TOKEN_IDS, min_size=head_size, max_size=head_size))
    head_hashes = hash_full_blocks(block_size, head)
    assert hash_full_blocks(block_size, head + tail) == head_hashes + hash_full_blocks(
        block_size, tail, parent=head_hashes[-1]
    )
