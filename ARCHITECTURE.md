# Architecture

kvsim models request scheduling and paged KV-cache allocation for a full-attention inference engine. It does not run a model or store KV tensors. Given the same request arrivals and sampled tokens, it produces the same batches and block assignments. It has no runtime dependencies and does not read the wall clock.

## Components and ownership

| Module | Responsibility | Owned state |
| --- | --- | --- |
| `block_pool.py` | Allocate, retain and release fixed-size blocks; register and evict cached blocks | Block objects, reference counts, free queue and hash-to-block map |
| `block_hash.py` | Hash full blocks of tokens into a chain of prefix fingerprints | None |
| `request.py` | Store request inputs, progress and outcome | Prompt, generated tokens, status, timeline and block hashes |
| `kv_cache_manager.py` | Allocate blocks for each request; find cached prefixes and cache full blocks | Block pool, request block tables and per-request cached-block counts |
| `scheduler.py` | Build batches, preempt under memory pressure, apply sampled tokens and abort requests | Waiting queue, running list, live request registry, pending batch, preemption total and prefix-cache counters |

`BlockPool`, `KVCacheManager` and `Scheduler` use explicit constructors. Queues, block storage and request tables are private. Callers use the existing counts and `usage` to inspect capacity. Tests inspect private state through the invariant helpers; there are no test-only public accessors.

`Request` and `KVCacheBlock` are mutable dataclasses with identity-based equality. `SchedulerConfig` and the returned `ScheduledBatch` are frozen dataclasses.

### Block allocation

`BlockPool(num_blocks)` creates blocks in ID order. `allocate(n)` takes blocks from the head of the free queue with a reference count of one. A negative count or insufficient capacity raises before changing state. `retain(blocks)` adds references and removes any newly referenced blocks from the free queue. `release(blocks)` removes references. A block that reaches zero without a hash goes to the head of the free queue, so unhashed blocks are reused last in, first out. A block with a hash goes to the tail, in the supplied order. Releasing an already free block raises.

The pool also holds the prefix cache: a map from `BlockHash` to the one block registered under it. `cache_full_blocks(blocks, block_hashes)` gives each allocated, unhashed block its hash and registers it. When the hash already maps to another block, the map keeps that block and the new one stays unhashed. A request's table can then hold a block whose content duplicates a cached one. Caching a free or already hashed block raises, and so do block and hash sequences of different lengths; the pairs before the invalid one stay cached. `get_cached_block(hash)` returns the registered block, free or allocated, or `None`. A freed cached block keeps its hash and stays a hit until `allocate` pops it from the free queue or `reset_prefix_cache` clears it. Popping evicts it: the hash leaves the block and the map. `retain` on a cached free block takes it out of the free queue without touching its hash. `reset_prefix_cache()` clears every hash and returns `True` only when every block is free; otherwise it changes nothing and returns `False`. The cleared blocks keep their places in the free queue.

The two ends of the free queue keep eviction cheap. Unhashed blocks hold nothing reusable, so they are consumed first. Cached blocks sit behind them, in the order they were freed, and are evicted least recently freed first. The caller releases a request's table in reverse, so a request's last block, the one covering the most tokens and the least likely to be shared, is evicted before its earlier blocks. A free queue is therefore always a run of unhashed blocks followed by a run of cached ones. Without prefix caching no block is ever hashed, every released block goes to the head, and eviction never happens.

The free queue uses an `OrderedDict` for constant-time removal from the head, insertion at both ends and removal by block ID.

`KVCacheManager(num_blocks, block_size, *, enable_caching)` requires both sizes to be positive. `allocate_slots(request, num_new_tokens, new_computed_blocks=())` reserves enough blocks for `request.num_computed_tokens`, plus the tokens covered by `new_computed_blocks`, plus `num_new_tokens`, rounded up by block size. The caller supplies a nonnegative count, never reserves past `request.num_tokens`, and advances computed tokens before allocating the next step. Cache hits may only be passed for a request with no table and no computed tokens, and the caller then sets its computed tokens to the hit tokens. Existing block tables only grow until freed.

With caching on, `get_computed_blocks(request)` returns the cached blocks of the request's longest cached prefix: it walks `request.block_hashes` in order and stops at the first hash the pool does not hold. It looks at most `(num_tokens - 1) // block_size` blocks deep, so at least one token is always left to compute and a fully cached prompt recomputes its whole last block: the model must run on the last token to sample the next one. The lookup retains nothing. With caching off it returns no blocks. The caller passes the hits to `allocate_slots`, which counts every hit block that is currently free as one more block taken from the free queue. It retains the hits before allocating new blocks; in the other order the pool could pop a hit block from the head of the free queue and evict it. The hits start the table, the new blocks follow.

After a successful allocation with caching on, the manager caches every block that will be full once the step's tokens are computed. Blocks are therefore cached at schedule time, so a second identical prompt admitted in the same step hits the first one's blocks. A per-request count of cached blocks makes caching incremental: hit blocks count as cached, and decoded tokens are cached as they complete blocks. The manager owns hashing: it calls `request.update_block_hashes(block_size)` before each lookup and each caching pass, always with its own block size.

An allocation returns the new blocks, possibly an empty list. It returns `None` without changing the pool or tables when capacity is insufficient. `free(request)` releases the entire table in reverse order: unhashed blocks end at the head in table order, and cached blocks end at the tail with the request's first block evicted last. Freeing a request without a table does nothing. `num_blocks_for`, `num_free_blocks` and `usage` report block counts and the fraction of allocated blocks.

### Block hashes

`hash_block_tokens(parent, token_ids)` returns the SHA-256 digest of the parent's 32-byte hash followed by the block's token IDs as little-endian 64-bit integers. The first block of a sequence uses `NONE_HASH`, 32 zero bytes, as its parent. A hash therefore fingerprints the whole prefix that ends with its block: for one block size, and barring SHA-256 collisions, two sequences share the hash of block `i` exactly when their first `i + 1` blocks hold the same tokens. `BlockHash` is a `NewType` over the digest bytes; a parent must be a real digest or `NONE_HASH`. The encoding is fixed, so the same tokens give the same hashes in every process and on every platform. Token IDs must fit in a signed 64-bit integer.

`hash_full_blocks(block_size, token_ids, parent=NONE_HASH)` hashes every full block in order and skips a partial tail; `parent` continues an existing chain. `Request.block_hashes` holds the chain for the request's full blocks. `Request.update_block_hashes(block_size)` hashes only the blocks completed since the last call, chaining from the last known hash. Tokens are only ever appended, so a hash is never recomputed and the list only grows. Every call must pass the same block size, the cache's; nothing checks it, and a request that already has hashes is not fresh, so `add_request` rejects it. With prefix caching on, the KV cache manager is the only caller, always with its own block size; with it off, nothing hashes.

### Request lifecycle

Callers construct a `Request` with its ID, prompt token tuple, output limit and arrival step. Progress fields cannot be passed to the constructor. The scheduler updates them after admission. `arrival_step` is metadata: the caller decides when to submit a request, and submission order determines queue order.

Requests move from `WAITING` to `RUNNING`, may alternate between `PREEMPTED` and `RUNNING`, then finish on EOS or the output limit, or are aborted from any live state. `finish(status, step)` rejects live statuses and a second finish. EOS takes precedence when both stop conditions apply. Completion and abort release the block table but keep tokens, computed-token counts and timestamps on the returned request.

## One scheduling step

```python
from kvsim.request import Request
from kvsim.scheduler import Scheduler, SchedulerConfig

scheduler = Scheduler(SchedulerConfig(num_blocks=8, block_size=4, max_tokens_per_step=16))
request = Request("r0", prompt_token_ids=(1, 2, 3, 4), max_output_tokens=2, arrival_step=0)
scheduler.add_request(request)

batch = scheduler.schedule()
assert batch.num_scheduled_tokens == {"r0": 4}
assert scheduler.complete_step(batch, {"r0": 7}) == []

batch = scheduler.schedule()
assert batch.num_scheduled_tokens == {"r0": 1}
assert scheduler.complete_step(batch, {"r0": 8}) == [request]
```

`add_request` rejects a live ID, a request that is not waiting with zero progress and no output, an empty prompt, or an output limit below one. It also rejects a request whose longest possible sequence, `num_prompt_tokens + max_output_tokens - 1` tokens, would not fit the whole cache or, with chunked prefill off, the whole token budget. That is the most a request ever holds in the cache, and the most a resumed request recomputes in one step. The check guarantees that a request alone in an empty cache always runs to completion, which is what keeps preemption from deadlocking. Rejection leaves the queues unchanged.

`schedule()` serves running requests in admission order, then waiting requests in submission order. Prefill and decode share `max_tokens_per_step`. At most `max_running_requests` requests run at once. If the head of the waiting queue cannot get its tokens or blocks, admission stops. Later requests do not skip ahead.

With `enable_chunked_prefill` (the default), each request gets its remaining tokens capped by the remaining budget, so a long prompt is computed over several steps. A request in partial prefill stays running, keeps its blocks and continues in the running loop of the next step. When `long_prefill_token_threshold` is positive and more than one request is live (running or waiting, admissible or not) at the start of the step, it also caps each request's tokens, so one long prompt cannot take the whole budget. The threshold requires chunked prefill. With chunking off, a waiting prompt must fit in the remaining budget, otherwise admission stops.

Running requests always fit the remaining budget. Only the last request scheduled in a step can have been cut by the budget, and nothing is admitted after it. Every earlier request needs no more tokens than it got: a decode needs one, and a chunk capped by the threshold stays capped while more than one request is live. A lone request has the whole budget.

With `enable_prefix_caching` (the default, as in vLLM), admission first asks the KV cache manager for the cached blocks of the request's prefix. Their tokens count as already computed: only the remaining tokens are scheduled, charged to the budget and checked against it with chunking off. On successful admission the scheduler adds the request's token count to `prefix_cache_queries` and the hit tokens to `prefix_cache_hits`, the token-based counters vLLM reports. A lookup that does not end in an admission is not counted. Preempted requests go through the same lookup when readmitted.

When a running request cannot get its blocks, the scheduler preempts the last request in the running list, the most recently admitted one, and retries. Preemption frees the victim's blocks, sets `PREEMPTED`, resets `num_computed_tokens` to zero, increments the request's `num_preemptions` and the scheduler's `num_preemptions`, and pushes the victim to the front of the waiting queue. Victims always come after the request being scheduled, so no request already scheduled in the step is preempted. If the request being scheduled becomes the victim itself, the running loop stops: every request behind it is already preempted. Successive victims are prepended one by one, so they return to the waiting queue in their running order, ahead of every fresh request. A step that preempts admits no waiting request, even when blocks and budget remain.

A preempted request keeps its output tokens and first-token step. On readmission it recomputes its prompt and generated tokens as one prefill, chunked like any other, and samples its next token only when that recomputation completes. With prefix caching, the readmission lookup can hit the request's own blocks that were cached and not yet evicted, and only the rest is recomputed. The head of the running list is never preempted, because every other request can be preempted for it and the `add_request` check guarantees it then fits. It therefore makes progress every step, so every request finishes. Chunked admission reserves blocks for the first chunk only, so partial prefills can exhaust the cache and cause preemptions that full-prompt admission would have avoided.

The returned batch records a step number, token counts, `scheduled_running`, `scheduled_resumed`, `scheduled_new` and `preempted`. Resumed requests are preempted requests readmitted in this step; new requests are admitted for the first time. The token-count mapping is a read-only copy in running, resumed, then new order; the ID sequences are tuples. `preempted` lists victims in the order they were preempted, youngest first. Step numbers start at zero, and `num_steps` counts returned batches, including empty ones. Scheduling reserves blocks without advancing computed tokens.

`complete_step(batch, sampled_token_ids)` accepts only the exact pending batch returned by this scheduler. A second `schedule()` before completion raises `RuntimeError`. A foreign, copied or replayed batch raises `ValueError`. Missing sampled IDs also raise `ValueError` before any state changes; extra IDs are ignored. After rejection, callers can correct the input and retry the pending batch.

The caller supplies one sampled token per scheduled request, as a model samples every sequence of its batch. Completion advances computed-token counts. It discards the token of a request whose prompt is still incomplete; every other request appends its token and records the first-token timestamp. It finishes requests, releases their blocks, removes them from the live registry and returns them in batch order. Only then does it clear the pending batch. Empty batches also need completion. Requests added while a batch is pending wait for the next step.

`finish_requests(*request_ids, status=FINISHED_ABORTED)` ends live requests in any state: waiting, preempted, running in partial prefill or decoding. Each one leaves the live registry and its queue, finishes with `status`, which must be a finished status, and releases its blocks, cached ones keeping their hashes; a live status raises before anything changes. Its `finished_step` is the most recently scheduled step, `num_steps - 1`, so its finish time is the end of that step like any other finished request; an abort before the first step records `-1`. Unknown and already finished IDs are ignored, a repeated ID included. It returns the requests it finished, in argument order. An abort may happen while a batch is pending. The batch keeps the aborted request's ID and token count; `complete_step` still requires its sampled token and then skips the ID: the request is gone, or a request re-added under that ID is waiting. Blocks the batch cached for that request's tokens keep their hashes, since completing the batch means it ran.

## Current invariants and tests

- After `update_block_hashes(block_size)` with the same block size on every call, a request's `block_hashes` equals `hash_full_blocks(block_size, all_token_ids)`: one hash per full block, none for a partial tail. Earlier hashes are the same objects as before the call.
- The pool contains the same block objects throughout its lifetime. A block is free exactly when its reference count is zero and it appears in the free queue.
- A block has a hash exactly when the map holds it under that hash. A cached block may be free or allocated.
- The free queue holds every unhashed free block before every cached one.
- A block's reference count matches the request tables holding it. A table never holds a block twice. Without prefix caching no block is shared and none has a hash.
- With prefix caching, a running request's cached-block count equals its reserved tokens divided by the block size, rounded down. Each of those blocks carries the request's hash for its position, or no hash when another block already held that hash. Every later block has no hash. The request's hashes are a prefix of the hash chain of its tokens.
- Prefix-cache hits never exceed queries and are a whole number of blocks; both stay zero without prefix caching.
- A request table holds exactly enough blocks for its computed tokens plus the tokens reserved by the pending batch. Waiting, preempted and finished requests hold no blocks.
- Each live request belongs to exactly one queue with the matching status and object identity. Finished requests leave the registry.
- The waiting queue holds preempted requests first, then fresh ones. A preempted request has been preempted at least once; a fresh one has never been scheduled and has no output.
- A pending batch schedules every running request, at least one token each, within the token budget. Its other IDs belong to requests aborted since it was scheduled, either gone or re-added and waiting. A batch that preempts admits nothing, and its live victims are in the waiting queue.
- Between steps, a running request has at least one computed token and at least one uncomputed token. A running request that was never preempted and has output has `num_computed_tokens == num_tokens - 1`; the sampled token gets computed in the next step. Without output, it is in partial prefill. A resumed request may have output and still be recomputing.

`tests/invariants.py` checks pool, table and queue state. A pinned digest guards the hash encoding against accidental changes. Example tests cover allocation order, failure behavior, admission, chunking, preemption order, resumption, prefix sharing, the lookup cap and its first-miss stop, hits on blocks at the head of the free queue, resumption from surviving cached blocks, aborts in every state and during a pending batch, stopping and batch ownership. The scheduler test helper also checks every step against the expected transitions. The head of the running list is scheduled. Victims leave from the tail and return to the waiting head. A request appends exactly the sampled token in the step its computation completes, and no other request's output changes. Hypothesis exercises block reference operations, request-table growth, and scheduling with varied arrivals, EOS positions, budgets, chunking settings and prefix caching on and off; its prompts share prefixes, so caching examples hit. Its pools are sized from the drawn prompts so that preemption is frequent. The scheduler property samples each token from the request's output position, checks that every request ends with the output that sampler implies, and has a step bound so stalled progress fails the test.

## vLLM references and differences

These comparisons use vLLM revision `8122776fba2dcfae2973a0659db2da6ac688306b`.

| Lab name | vLLM counterpart |
| --- | --- |
| `BlockPool.allocate`, `retain`, `release` | [`get_new_blocks`, `touch`, `free_blocks`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/block_pool.py) |
| `KVCacheManager.allocate_slots` | [`KVCacheManager.allocate_slots`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py) |
| `max_running_requests`, `max_tokens_per_step` | [`max_num_seqs`, `max_num_batched_tokens`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `enable_chunked_prefill`, `long_prefill_token_threshold` | [Same names in `SchedulerConfig`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/config/scheduler.py) |
| `finish_requests` | [`Scheduler.finish_requests`, `_free_request`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `ScheduledBatch`, `complete_step` | [`SchedulerOutput`, `update_from_output`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `Scheduler._preempt`, `ScheduledBatch.preempted`, `scheduled_resumed` | [`_preempt_request`, `preempted_req_ids`, `scheduled_resumed_reqs`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `add_request` longest-sequence check | [`verify_max_model_len`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/config/scheduler.py), [`_check_enough_kv_cache_memory`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_planning.py) |
| `KVCacheManager.get_computed_blocks` | [`KVCacheManager.get_computed_blocks`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py), [`FullAttentionManager.find_longest_cache_hit`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/single_type_kv_cache_manager.py) |
| Free hits counted in `allocate_slots`, cached-block counts | [`get_num_blocks_to_allocate`, `num_cached_block`, `cache_blocks`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/single_type_kv_cache_manager.py) |
| `SchedulerConfig.enable_prefix_caching`, `KVCacheManager.enable_caching` | [`CacheConfig.enable_prefix_caching`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/config/cache.py), [`KVCacheManager.enable_caching`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py) |
| `prefix_cache_queries`, `prefix_cache_hits` | [`PrefixCacheStats.record`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/metrics/stats.py), [`record_prefix_cache_stats`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py) |
| `BlockPool.cache_full_blocks`, `get_cached_block`, `reset_prefix_cache`, eviction in `allocate` | [`cache_full_blocks`, `get_cached_block`, `reset_prefix_cache`, `_maybe_evict_cached_block`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/block_pool.py) |
| `BlockHash`, `NONE_HASH`, `hash_block_tokens` | [Same names in `kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_utils.py) |
| `hash_full_blocks`, `Request.update_block_hashes` | [`get_request_block_hasher`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_utils.py), [`Request.update_block_hashes`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/request.py) |
| Discarded tokens of partial prefills | [`discard_request_mask` in the GPU model runner](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/worker/gpu_model_runner.py) |

The lab uses one token budget; this vLLM revision can distinguish scheduled tokens from the input budget. vLLM advances computed-token counts in `_update_after_schedule`; the lab advances them when completing the batch.

The lab allocates every block, including block zero. vLLM reserves a null block.

Released blocks follow vLLM's split: unhashed to the head, hashed to the tail. vLLM's [design doc](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/docs/design/prefix_caching.md) still says every freed block is appended to the tail. The split came with [#42656](https://github.com/vllm-project/vllm/pull/42656): unhashed partial blocks at the tail aged like cache entries and evicted real cached prefixes, and the PR measured lost hits restored. With caching off, vLLM's order went from append to prepend and back twice in 2026 ([#48017](https://github.com/vllm-project/vllm/pull/48017), [#51482](https://github.com/vllm-project/vllm/pull/51482)), for CPU overhead and a TPU runtime fault rather than for caching. In the lab that order only changes which block IDs are reused. The lab applies the same rule whatever the configuration, so the pool needs no caching flag. vLLM prepends one call's unhashed blocks as a group in the supplied order; the lab pushes them one at a time, which reverses their order among themselves and changes nothing else.

`get_computed_blocks` returns only the hit blocks; vLLM also returns the hit token count, which differs from `len(blocks) * block_size` only with external or partial hits the lab does not have. vLLM records prefix-cache counters at admission, as the lab does, but keeps resumed preempted requests in a separate bucket; the lab counts them with every other admission.

vLLM's map can hold several blocks under one hash, and both stay cached until evicted. The lab keeps one block per hash and leaves the duplicate unhashed, so it is reused first once freed. A later identical request hits only the registered block: once that block is evicted the prefix misses, even while a duplicate is still allocated, where vLLM would hit the duplicate.

vLLM guarantees that one request of `max_model_len` tokens fits: startup fails when the cache cannot hold one, the budget must reach `max_model_len` when chunking is off, and requests stop at that length. The lab has no `max_model_len`; it checks each request's own longest sequence when the request is added, then admits it against currently free blocks. With chunking on, admission only needs blocks for the first chunk; no space is reserved for future output. vLLM's allocator also supports full-sequence admission checks. The lab supports FCFS and stops on EOS, the output limit or an abort.

Preemption follows vLLM's FCFS policy: the victim is the last running request, and the waiting loop is skipped after a preemption. vLLM's scheduler keeps no preemption total: requests carry `num_preemptions`, and the metrics loggers sum per-iteration counts into `vllm:num_preemptions_total`. The lab keeps `Scheduler.num_preemptions` because finished requests leave the registry and there is no metrics layer yet.

vLLM hashes the pickled or CBOR-encoded tuple `(parent, token_ids, extra_keys)`, and derives `NONE_HASH` from a seed: a fixed default with SHA-256, `PYTHONHASHSEED` when set, a random one for the non-cryptographic hashes. The lab hashes a fixed binary encoding, has no extra keys (LoRA, multimodal inputs, cache salt) and uses 32 zero bytes as `NONE_HASH`. vLLM's hashing functions take `None` for the first block's parent; the lab passes `NONE_HASH` itself. vLLM gives each request a hasher at construction and extends `block_hashes` whenever an output token is appended; the lab's request does not know the block size, so its owner calls `update_block_hashes`.

vLLM's `finish_requests` takes one ID, an iterable of IDs, or `None` for every request; the lab takes IDs as separate arguments, so a bare string cannot be mistaken for a sequence of IDs. vLLM's `update_from_output` skips a request that is missing or already finished; the lab also skips one that is not running, which covers a request re-added under an aborted ID while the batch was pending.

vLLM accepts `long_prefill_token_threshold` with chunked prefill off and still caps admitted prompts by it; the lab rejects that configuration. The lab has no adaptive threshold (`long_prefill_token_threshold_adaptive`).

## Planned features

- **Workloads and metrics:** add deterministic workload replay, traces and an explicit duration model. The planned sampler must produce the same tokens for a request across preemption and different schedules. Reports must distinguish simulator estimates from engine measurements.

Sliding windows, hybrid allocators, multimodal inputs, LoRA, speculative decoding and async scheduling remain outside the simulator's scope.
