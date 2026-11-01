import hashlib
import struct
from collections.abc import Sequence
from typing import NewType

BlockHash = NewType("BlockHash", bytes)

NONE_HASH = BlockHash(bytes(32))


def hash_block_tokens(parent: BlockHash, token_ids: Sequence[int]) -> BlockHash:
    hasher = hashlib.sha256(parent)
    hasher.update(struct.pack(f"<{len(token_ids)}q", *token_ids))
    return BlockHash(hasher.digest())


def hash_full_blocks(
    block_size: int, token_ids: Sequence[int], *, parent: BlockHash = NONE_HASH
) -> list[BlockHash]:
    block_hashes: list[BlockHash] = []
    block_hash = parent
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        block_hash = hash_block_tokens(block_hash, token_ids[start : start + block_size])
        block_hashes.append(block_hash)
    return block_hashes
