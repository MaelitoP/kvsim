# Architecture

kvsim models request scheduling and paged KV-cache allocation for a full-attention inference engine. It does not run a model or store KV tensors. Given the same request arrivals and sampled tokens, it produces the same batches and block assignments. It has no runtime dependencies and does not read the wall clock.

## Components and ownership

| Module | Responsibility | Owned state |
| --- | --- | --- |
| `block_pool.py` | Allocate, retain and release fixed-size blocks | Block objects, reference counts and free queue |
| `block_hash.py` | Hash full blocks of tokens into a chain of prefix fingerprints | None |
| `request.py` | Store request inputs, progress and outcome | Prompt, generated tokens, status, timeline and block hashes |
| `kv_cache_manager.py` | Allocate blocks for each request | Block pool and request block tables |
| `scheduler.py` | Build batches, preempt under memory pressure and apply sampled tokens | Waiting queue, running list, live request registry, pending batch and preemption total |

`BlockPool`, `KVCacheManager` and `Scheduler` use explicit constructors. Queues, block storage and request tables are private. Callers use the existing counts and `usage` to inspect capacity. Tests inspect private state through the invariant helpers; there are no test-only public accessors.

`Request` and `KVCacheBlock` are mutable dataclasses with identity-based equality. `SchedulerConfig` and the returned `ScheduledBatch` are frozen dataclasses.

### Block allocation

`BlockPool(num_blocks)` creates blocks in ID order. `allocate(n)` takes blocks from the head of the free queue with a reference count of one. A negative count or insufficient capacity raises before changing state. `retain(blocks)` adds references and removes any newly referenced blocks from the free queue. `release(blocks)` removes references; blocks that reach zero join the tail in the supplied order. Releasing an already free block raises.

The free queue uses an `OrderedDict` for constant-time removal from the head, insertion at the tail and removal by block ID. `block_hash` is reserved for prefix caching; no current operation sets or indexes it.

`KVCacheManager(num_blocks, block_size)` requires both sizes to be positive. `allocate_slots(request, num_new_tokens)` reserves enough blocks for `request.num_computed_tokens + num_new_tokens`, rounded up by block size. The caller supplies a nonnegative count and advances computed tokens before allocating the next step. Existing block tables only grow until freed.

An allocation returns the new blocks, possibly an empty list. It returns `None` without changing the pool or tables when capacity is insufficient. `free(request)` releases the entire table in reverse order, keeping earlier prompt blocks later in the free queue. Freeing a request without a table does nothing. `num_blocks_for`, `num_free_blocks` and `usage` report block counts and the fraction of allocated blocks.

### Block hashes

`hash_block_tokens(parent, token_ids)` returns the SHA-256 digest of the parent's 32-byte hash followed by the block's token IDs as little-endian 64-bit integers. The first block of a sequence uses `NONE_HASH`, 32 zero bytes, as its parent. A hash therefore fingerprints the whole prefix that ends with its block: for one block size, and barring SHA-256 collisions, two sequences share the hash of block `i` exactly when their first `i + 1` blocks hold the same tokens. `BlockHash` is a `NewType` over the digest bytes; a parent must be a real digest or `NONE_HASH`. The encoding is fixed, so the same tokens give the same hashes in every process and on every platform. Token IDs must fit in a signed 64-bit integer.

`hash_full_blocks(block_size, token_ids, parent=NONE_HASH)` hashes every full block in order and skips a partial tail; `parent` continues an existing chain. `Request.block_hashes` holds the chain for the request's full blocks. `Request.update_block_hashes(block_size)` hashes only the blocks completed since the last call, chaining from the last known hash. Tokens are only ever appended, so a hash is never recomputed and the list only grows. Every call must pass the same block size, the cache's; nothing checks it, and a request that already has hashes is not fresh, so `add_request` rejects it. Nothing calls the method during scheduling yet; prefix caching will use the chain to find cached blocks.

### Request lifecycle

Callers construct a `Request` with its ID, prompt token tuple, output limit and arrival step. Progress fields cannot be passed to the constructor. The scheduler updates them after admission. `arrival_step` is metadata: the caller decides when to submit a request, and submission order determines queue order.

Requests move from `WAITING` to `RUNNING`, may alternate between `PREEMPTED` and `RUNNING`, then finish on EOS or the output limit. `finish(status, step)` rejects live statuses and a second finish. EOS takes precedence when both stop conditions apply. Completion releases the block table but keeps tokens, computed-token counts and timestamps on the returned request. `FINISHED_ABORTED` exists as a status; the scheduler does not use it yet.

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

When a running request cannot get its blocks, the scheduler preempts the last request in the running list, the most recently admitted one, and retries. Preemption frees the victim's blocks, sets `PREEMPTED`, resets `num_computed_tokens` to zero, increments the request's `num_preemptions` and the scheduler's `num_preemptions`, and pushes the victim to the front of the waiting queue. Victims always come after the request being scheduled, so no request already scheduled in the step is preempted. If the request being scheduled becomes the victim itself, the running loop stops: every request behind it is already preempted. Successive victims are prepended one by one, so they return to the waiting queue in their running order, ahead of every fresh request. A step that preempts admits no waiting request, even when blocks and budget remain.

A preempted request keeps its output tokens and first-token step. On readmission it recomputes its prompt and generated tokens as one prefill, chunked like any other, and samples its next token only when that recomputation completes. The head of the running list is never preempted, because every other request can be preempted for it and the `add_request` check guarantees it then fits. It therefore makes progress every step, so every request finishes. Chunked admission reserves blocks for the first chunk only, so partial prefills can exhaust the cache and cause preemptions that full-prompt admission would have avoided.

The returned batch records a step number, token counts, `scheduled_running`, `scheduled_resumed`, `scheduled_new` and `preempted`. Resumed requests are preempted requests readmitted in this step; new requests are admitted for the first time. The token-count mapping is a read-only copy in running, resumed, then new order; the ID sequences are tuples. `preempted` lists victims in the order they were preempted, youngest first. Step numbers start at zero, and `num_steps` counts returned batches, including empty ones. Scheduling reserves blocks without advancing computed tokens.

`complete_step(batch, sampled_token_ids)` accepts only the exact pending batch returned by this scheduler. A second `schedule()` before completion raises `RuntimeError`. A foreign, copied or replayed batch raises `ValueError`. Missing sampled IDs also raise `ValueError` before any state changes; extra IDs are ignored. After rejection, callers can correct the input and retry the pending batch.

The caller supplies one sampled token per scheduled request, as a model samples every sequence of its batch. Completion advances computed-token counts. It discards the token of a request whose prompt is still incomplete; every other request appends its token and records the first-token timestamp. It finishes requests, releases their blocks, removes them from the live registry and returns them in batch order. Only then does it clear the pending batch. Empty batches also need completion. Requests added while a batch is pending wait for the next step.

## Current invariants and tests

- After `update_block_hashes(block_size)` with the same block size on every call, a request's `block_hashes` equals `hash_full_blocks(block_size, all_token_ids)`: one hash per full block, none for a partial tail. Earlier hashes are the same objects as before the call.
- The pool contains the same block objects throughout its lifetime. A block is free exactly when its reference count is zero and it appears in the free queue.
- A block's reference count matches the request tables holding it. Requests do not share blocks yet.
- A request table holds exactly enough blocks for its computed tokens plus the tokens reserved by the pending batch. Waiting, preempted and finished requests hold no blocks.
- Each live request belongs to exactly one queue with the matching status and object identity. Finished requests leave the registry.
- The waiting queue holds preempted requests first, then fresh ones. A preempted request has been preempted at least once; a fresh one has never been scheduled and has no output.
- A pending batch schedules every running request, at least one token each, within the token budget. A batch that preempts admits nothing, and its victims are in the waiting queue.
- Between steps, a running request has at least one computed token and at least one uncomputed token. A running request that was never preempted and has output has `num_computed_tokens == num_tokens - 1`; the sampled token gets computed in the next step. Without output, it is in partial prefill. A resumed request may have output and still be recomputing.

`tests/invariants.py` checks pool, table and queue state. A pinned digest guards the hash encoding against accidental changes. Example tests cover allocation order, failure behavior, admission, chunking, preemption order, resumption, stopping and batch ownership. The scheduler test helper also checks every step against the expected transitions. The head of the running list is scheduled. Victims leave from the tail and return to the waiting head. A request appends exactly the sampled token in the step its computation completes, and no other request's output changes. Hypothesis exercises block reference operations, request-table growth, and scheduling with varied arrivals, EOS positions, budgets and chunking settings. Its pools are sized from the drawn prompts so that preemption is frequent. The scheduler property samples each token from the request's output position, checks that every request ends with the output that sampler implies, and has a step bound so stalled progress fails the test.

## vLLM references and differences

These comparisons use vLLM revision `8122776fba2dcfae2973a0659db2da6ac688306b`.

| Lab name | vLLM counterpart |
| --- | --- |
| `BlockPool.allocate`, `retain`, `release` | [`get_new_blocks`, `touch`, `free_blocks`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/block_pool.py) |
| `KVCacheManager.allocate_slots` | [`KVCacheManager.allocate_slots`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py) |
| `max_running_requests`, `max_tokens_per_step` | [`max_num_seqs`, `max_num_batched_tokens`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `enable_chunked_prefill`, `long_prefill_token_threshold` | [Same names in `SchedulerConfig`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/config/scheduler.py) |
| `ScheduledBatch`, `complete_step` | [`SchedulerOutput`, `update_from_output`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `Scheduler._preempt`, `ScheduledBatch.preempted`, `scheduled_resumed` | [`_preempt_request`, `preempted_req_ids`, `scheduled_resumed_reqs`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `add_request` longest-sequence check | [`verify_max_model_len`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/config/scheduler.py), [`_check_enough_kv_cache_memory`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_planning.py) |
| `BlockHash`, `NONE_HASH`, `hash_block_tokens` | [Same names in `kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_utils.py) |
| `hash_full_blocks`, `Request.update_block_hashes` | [`get_request_block_hasher`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_utils.py), [`Request.update_block_hashes`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/request.py) |
| Discarded tokens of partial prefills | [`discard_request_mask` in the GPU model runner](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/worker/gpu_model_runner.py) |

The lab uses one token budget; this vLLM revision can distinguish scheduled tokens from the input budget. vLLM advances computed-token counts in `_update_after_schedule`; the lab advances them when completing the batch.

The lab allocates every block, including block zero. vLLM reserves a null block. The lab always appends released blocks to the free queue; this vLLM revision distinguishes reuse order for cached and uncached blocks.

vLLM guarantees that one request of `max_model_len` tokens fits: startup fails when the cache cannot hold one, the budget must reach `max_model_len` when chunking is off, and requests stop at that length. The lab has no `max_model_len`; it checks each request's own longest sequence when the request is added, then admits it against currently free blocks. With chunking on, admission only needs blocks for the first chunk; no space is reserved for future output. vLLM's allocator also supports full-sequence admission checks. The lab supports FCFS and stops on EOS or the output limit.

Preemption follows vLLM's FCFS policy: the victim is the last running request, and the waiting loop is skipped after a preemption. vLLM's scheduler keeps no preemption total: requests carry `num_preemptions`, and the metrics loggers sum per-iteration counts into `vllm:num_preemptions_total`. The lab keeps `Scheduler.num_preemptions` because finished requests leave the registry and there is no metrics layer yet.

vLLM hashes the pickled or CBOR-encoded tuple `(parent, token_ids, extra_keys)`, and derives `NONE_HASH` from a seed: a fixed default with SHA-256, `PYTHONHASHSEED` when set, a random one for the non-cryptographic hashes. The lab hashes a fixed binary encoding, has no extra keys (LoRA, multimodal inputs, cache salt) and uses 32 zero bytes as `NONE_HASH`. vLLM's hashing functions take `None` for the first block's parent; the lab passes `NONE_HASH` itself. vLLM gives each request a hasher at construction and extends `block_hashes` whenever an output token is appended; the lab's request does not know the block size, so its owner calls `update_block_hashes`.

vLLM accepts `long_prefill_token_threshold` with chunked prefill off and still caps admitted prompts by it; the lab rejects that configuration. The lab has no adaptive threshold (`long_prefill_token_threshold_adaptive`).

## Planned features

- **Prefix caching:** register hashed full blocks in the pool, evict the least recently freed, share cached prefixes on admission, and remove hash entries on eviction. Admission lookup covers at most `num_tokens - 1` tokens so some computation remains.
- **Abort, workloads and metrics:** add request cancellation, deterministic workload replay, traces and an explicit duration model. The planned sampler must produce the same tokens for a request across preemption and different schedules. Reports must distinguish simulator estimates from engine measurements.

Sliding windows, hybrid allocators, multimodal inputs, LoRA, speculative decoding and async scheduling remain outside the simulator's scope.
