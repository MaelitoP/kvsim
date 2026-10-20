# Architecture

kv-cache-lab simulates the scheduler and paged KV-cache manager of an LLM inference engine, based on vLLM V1. It models block allocation, prefix caching, token budgets and preemption.

The simulator is deterministic. It does not execute a model or require a GPU, tokenizer or runtime dependencies.

## Modules

Code lives in `src/kv_cache_lab/`. The block pool, the request model and the KV cache manager are implemented.

| Module | Responsibility | Status |
| --- | --- | --- |
| `block_pool.py` | Fixed block pool, reference counts, free queue and cache eviction | Implemented |
| `request.py` | Request tokens, computed-token count, status and timeline | Implemented |
| `kv_cache_manager.py` | Per-request block tables and prefix-cache lookup on admission | Implemented, no prefix cache yet |
| `prefix_cache.py` | Hashing full blocks with chained parent hashes | Planned |
| `scheduler.py` | Token budget, chunked prefill, preemption and abort | Planned |
| `workload.py` | Deterministic workloads from a JSON spec | Planned |
| `simulator.py` | Workload replay and simulated time | Planned |
| `metrics.py` | Per-request metrics and aggregate results | Planned |
| `__main__.py` | `simulate` and `summarize` commands | Planned |

Tests live in `tests/`, with example tests and Hypothesis properties for each module. Shared helpers in `tests/invariants.py` (`check_pool`, `check_manager`, `check_scheduler`) assert the memory invariants below after each operation; `check_manager` verifies that every reference count equals the number of block tables holding the block. Scheduler state-machine tests will live in `tests/test_stateful.py`.

## Block pool

`BlockPool(num_blocks)` creates all blocks at initialization. Blocks are never added or removed.

Each `KVCacheBlock` has a `block_id`, a `ref_cnt` and an optional `block_hash`. All blocks start with no hash and a reference count of zero, queued in `block_id` order.

The free queue determines eviction order:

- `get_new_blocks(n)` takes blocks from the head and sets their reference counts to 1. A negative count or insufficient free blocks raises before changing any state.
- `free_blocks(blocks)` decrements reference counts. Blocks that reach zero go to the tail in the supplied order, so callers control their relative eviction order. Freeing a block already at zero raises.
- `touch(blocks)` increments reference counts, removing blocks from the free queue when they go from zero to one. This lets a request reuse a cached prefix that has not been evicted.

`FreeBlockQueue` uses an `OrderedDict` keyed by block ID for O(1) removal from the head, insertion at the tail and removal by ID.

## Requests

`Request` is a dataclass holding one generation request: `request_id`, the immutable `prompt_token_ids`, `max_output_tokens`, `arrival_step`, the growing `output_token_ids`, `num_computed_tokens` and a `status`. Derived views: `all_token_ids` (prompt followed by output), `num_prompt_tokens`, `num_output_tokens`, `num_tokens` and `is_finished`. There is no prefill-done flag: after a preemption the prompt and the generated tokens are recomputed together, so the scheduler reasons only about `num_tokens - num_computed_tokens`. The scheduler grows output through `append_output_token`.

`RequestStatus` has three live states, `WAITING`, `RUNNING` and `PREEMPTED`, and three finished states, `FINISHED_STOPPED` (EOS sampled), `FINISHED_LENGTH_CAPPED` (`max_output_tokens` reached) and `FINISHED_ABORTED`. `status.is_finished` distinguishes the two groups.

The scheduler finishes a request only through `finish(status, step)`. It sets `status` and `finished_step`, and raises if the status is a live one or the request is already finished, so a request finishes at most once. Transitions between live states are plain assignments made by the scheduler.

The timeline fields the metrics read are `first_scheduled_step`, `first_token_step` and `finished_step` (`None` until set) and `num_preemptions` (0 until the scheduler preempts the request). The scheduler fills them in; the request only stores them.

## KV cache manager

`KVCacheManager(num_blocks, block_size)` owns a `BlockPool` and `req_to_blocks`, one append-only block table per request keyed by `request_id`. Both sizes must be at least 1. Request ids must be unique among live requests; the scheduler and the workload generator own that uniqueness.

`allocate_slots(request, num_new_tokens)` computes the blocks the request needs to hold `num_computed_tokens + num_new_tokens` tokens, `ceil(n / block_size)`, subtracts the blocks already in its table and asks the pool for the difference. When the pool cannot supply them it returns `None` and nothing changes: no block, no reference count, no table. On success it appends the new blocks to the table and returns them, possibly an empty list when the current blocks still have room. A request has a table from its first successful allocation until `free`, even when that allocation added no block. `num_new_tokens` is never negative: the table never runs ahead of `num_computed_tokens` plus the tokens being scheduled, so the block delta is never negative either.

`free(request)` pops the table and frees its blocks in reverse order, so the free queue tail reads from the request's last block back to its first. Freeing a request without a table is a no-op, which lets the scheduler free aborted requests without checking whether they were ever scheduled.

`num_blocks_for(request)`, `num_free_blocks` and `usage` (fraction of allocated blocks) are the counters the scheduler and the metrics read.

Without caching, every block belongs to exactly one table, so two requests never share a block.

## Contracts

These rules define the simulator's behavior. State invariants must hold after every public operation and are checked by the shared test helpers.

### Determinism

The simulator never reads the wall clock or uses unseeded randomness. The same workload spec, configuration and seed produce a byte-identical trace.

Simulated time consists of a step counter and an estimated duration from an explicit cost model. Both are reported, with durations labelled as estimates.

### Memory

The cache contains `num_blocks` blocks, each holding `block_size` tokens. A request with `n` computed or scheduled tokens needs exactly `ceil(n / block_size)` blocks.

Each request has an append-only block table from its first allocation until it is freed. Finishing, aborting or preempting a request releases its entire table.

The simulator supports full attention only. It does not model hybrid allocators, sliding windows, multimodal inputs, LoRA, speculative decoding or async scheduling.

The following invariants apply:

- A block is either free (`ref_cnt == 0`, in the free queue) or allocated (`ref_cnt > 0`, outside the free queue).
- The total block count never changes.
- A block's reference count equals the number of request block tables containing it.
- Every hashed block has a matching hash-map entry pointing to it. Blocks outside the map have no hash.
- Cached blocks can be free and available for reuse until eviction, or allocated and shared by requests.
- Failed allocations leave blocks, reference counts and cache entries unchanged.
- Finished, aborted and preempted requests hold no blocks.
- The sum of block-table lengths equals the sum of reference counts.

### Scheduling

Each step calls `schedule()` once. Running requests are served first, then waiting requests in arrival order.

Prefill and decode share the `max_num_batched_tokens` budget. At most `max_num_seqs` requests can run at once. Chunked prefill limits each request to the remaining token budget.

If a running request cannot allocate blocks, running requests are preempted in LIFO order. A preempted request releases its blocks and returns to the head of the waiting queue. It keeps its generated tokens but must recompute them along with the prompt. A step that preempts a request admits no waiting requests.

Partial prefill produces no output token. Between decode steps, `num_computed_tokens == num_tokens - 1`: the newly sampled token is appended to the request and computed in the next step.

On admission, prefix-cache lookup covers at most `num_tokens - 1` tokens and reuses only full blocks. Even a fully cached prompt recomputes its last block, ensuring at least one token is computed.

Output token IDs depend only on the request, not on scheduling. Preempting and resuming a request produces the same output as running it uninterrupted.

### Measurements

Simulator results are estimates. Engine and load-client results are measurements. Reports, tables and plots must distinguish the two.

## Differences from vLLM

This section records simplifications as they are implemented.

- **Free queue:** the simulator uses an `OrderedDict`. vLLM stores doubly linked list pointers on the blocks themselves (`prev_free_block`, `next_free_block`) to avoid extra Python object allocations. Both support O(1) queue operations and removal by block.
- **Null block:** vLLM reserves block 0 as a placeholder, excludes it from usage and never caches it. The simulator has no reserved block. Every block is allocatable and counts toward usage.
