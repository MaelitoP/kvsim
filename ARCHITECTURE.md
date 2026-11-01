# Architecture

kvsim models request scheduling and paged KV-cache allocation for a full-attention inference engine. It does not run a model or store KV tensors. Given the same request arrivals and sampled tokens, it produces the same batches and block assignments. It has no runtime dependencies and does not read the wall clock.

## Components and ownership

| Module | Responsibility | Owned state |
| --- | --- | --- |
| `block_pool.py` | Allocate, retain and release fixed-size blocks | Block objects, reference counts and free queue |
| `request.py` | Store request inputs, progress and outcome | Prompt, generated tokens, status and timeline |
| `kv_cache_manager.py` | Allocate blocks for each request | Block pool and request block tables |
| `scheduler.py` | Build batches and apply sampled tokens | Waiting queue, running list, live request registry and pending batch |

`BlockPool`, `KVCacheManager` and `Scheduler` use explicit constructors. Queues, block storage and request tables are private. Callers use the existing counts and `usage` to inspect capacity. Tests inspect private state through the invariant helpers; there are no test-only public accessors.

`Request` and `KVCacheBlock` are mutable dataclasses with identity-based equality. `SchedulerConfig` and the returned `ScheduledBatch` are frozen dataclasses.

### Block allocation

`BlockPool(num_blocks)` creates blocks in ID order. `allocate(n)` takes blocks from the head of the free queue with a reference count of one. A negative count or insufficient capacity raises before changing state. `retain(blocks)` adds references and removes any newly referenced blocks from the free queue. `release(blocks)` removes references; blocks that reach zero join the tail in the supplied order. Releasing an already free block raises.

The free queue uses an `OrderedDict` for constant-time removal from the head, insertion at the tail and removal by block ID. `block_hash` is reserved for prefix caching; no current operation sets or indexes it.

`KVCacheManager(num_blocks, block_size)` requires both sizes to be positive. `allocate_slots(request, num_new_tokens)` reserves enough blocks for `request.num_computed_tokens + num_new_tokens`, rounded up by block size. The caller supplies a nonnegative count and advances computed tokens before allocating the next step. Existing block tables only grow until freed.

An allocation returns the new blocks, possibly an empty list. It returns `None` without changing the pool or tables when capacity is insufficient. `free(request)` releases the entire table in reverse order, keeping earlier prompt blocks later in the free queue. Freeing a request without a table does nothing. `num_blocks_for`, `num_free_blocks` and `usage` report block counts and the fraction of allocated blocks.

### Request lifecycle

Callers construct a `Request` with its ID, prompt token tuple, output limit and arrival step. Progress fields cannot be passed to the constructor. The scheduler updates them after admission. `arrival_step` is metadata: the caller decides when to submit a request, and submission order determines queue order.

Requests move from `WAITING` to `RUNNING`, then finish on EOS or the output limit. `finish(status, step)` rejects live statuses and a second finish. EOS takes precedence when both stop conditions apply. Completion releases the block table but keeps tokens, computed-token counts and timestamps on the returned request. `PREEMPTED` and `FINISHED_ABORTED` exist as statuses; the scheduler does not use them yet.

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

`add_request` rejects a live ID, a request that is not waiting with zero progress and no output, an empty prompt, or an output limit below one. A prompt must fit the whole cache and, with chunked prefill off, the whole token budget. Rejection leaves the queues unchanged.

`schedule()` serves running requests in admission order, then waiting requests in submission order. Prefill and decode share `max_tokens_per_step`. At most `max_running_requests` requests run at once. If the head of the waiting queue cannot get its tokens or blocks, admission stops. Later requests do not skip ahead.

With `enable_chunked_prefill` (the default), each request gets its remaining tokens capped by the remaining budget, so a long prompt is computed over several steps. A request in partial prefill stays running, keeps its blocks and continues in the running loop of the next step. When `long_prefill_token_threshold` is positive and more than one request is live (running or waiting, admissible or not) at the start of the step, it also caps each request's tokens, so one long prompt cannot take the whole budget. The threshold requires chunked prefill. With chunking off, a waiting prompt must fit in the remaining budget, otherwise admission stops.

Running requests always fit the remaining budget. Only the last request scheduled in a step can have been cut by the budget, and nothing is admitted after it. Every earlier request needs no more tokens than it got: a decode needs one, and a chunk capped by the threshold stays capped while more than one request is live. A lone request has the whole budget.

Running-request cache exhaustion raises `RuntimeError` and ends the run. The scheduler does not roll back earlier allocations in that failed call; recovery requires preemption. Chunked admission reserves blocks for the first chunk only, so partial prefills can exhaust the cache in runs that full-prompt admission would have completed.

The returned batch records a step number, token counts, `scheduled_running` and `scheduled_new`. The token-count mapping is a read-only copy in running-then-new order; the ID sequences are tuples. Step numbers start at zero, and `num_steps` counts returned batches, including empty ones. Scheduling reserves blocks without advancing computed tokens.

`complete_step(batch, sampled_token_ids)` accepts only the exact pending batch returned by this scheduler. A second `schedule()` before completion raises `RuntimeError`. A foreign, copied or replayed batch raises `ValueError`. Missing sampled IDs also raise `ValueError` before any state changes; extra IDs are ignored. After rejection, callers can correct the input and retry the pending batch.

The caller supplies one sampled token per scheduled request, as a model samples every sequence of its batch. Completion advances computed-token counts. It discards the token of a request whose prompt is still incomplete; every other request appends its token and records the first-token timestamp. It finishes requests, releases their blocks, removes them from the live registry and returns them in batch order. Only then does it clear the pending batch. Empty batches also need completion. Requests added while a batch is pending wait for the next step.

## Current invariants and tests

- The pool contains the same block objects throughout its lifetime. A block is free exactly when its reference count is zero and it appears in the free queue.
- A block's reference count matches the request tables holding it. Requests do not share blocks yet.
- A request table holds exactly enough blocks for its computed tokens plus the tokens reserved by the pending batch. Waiting and finished requests hold no blocks. A running-request `RuntimeError` voids this, since the run ends with the step's earlier allocations kept.
- Each live request belongs to exactly one queue with the matching status and object identity. Finished requests leave the registry.
- A pending batch schedules every running request, at least one token each, within the token budget.
- Between steps, a running request with output has `num_computed_tokens == num_tokens - 1`; the sampled token gets computed in the next step. A running request without output is in partial prefill: part of its prompt is still uncomputed.

`tests/invariants.py` checks pool, table and queue state. Example tests cover allocation order, failure behavior, admission, chunking, stopping and batch ownership. Hypothesis exercises block reference operations, request-table growth, and scheduling with varied arrivals, EOS tokens, budgets and chunking settings. The scheduler property has a step bound so stalled progress fails the test.

## vLLM references and differences

These comparisons use vLLM revision `8122776fba2dcfae2973a0659db2da6ac688306b`.

| Lab name | vLLM counterpart |
| --- | --- |
| `BlockPool.allocate`, `retain`, `release` | [`get_new_blocks`, `touch`, `free_blocks`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/block_pool.py) |
| `KVCacheManager.allocate_slots` | [`KVCacheManager.allocate_slots`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py) |
| `max_running_requests`, `max_tokens_per_step` | [`max_num_seqs`, `max_num_batched_tokens`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `enable_chunked_prefill`, `long_prefill_token_threshold` | [Same names in `SchedulerConfig`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/config/scheduler.py) |
| `ScheduledBatch`, `complete_step` | [`SchedulerOutput`, `update_from_output`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| Discarded tokens of partial prefills | [`discard_request_mask` in the GPU model runner](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/worker/gpu_model_runner.py) |

The lab uses one token budget; this vLLM revision can distinguish scheduled tokens from the input budget. vLLM advances computed-token counts in `_update_after_schedule`; the lab advances them when completing the batch.

The lab allocates every block, including block zero. vLLM reserves a null block. The lab always appends released blocks to the free queue; this vLLM revision distinguishes reuse order for cached and uncached blocks.

The lab rejects prompts larger than its total cache capacity, or than its budget when chunking is off, then admits them against currently free blocks. With chunking on, admission only needs blocks for the first chunk. It does not reserve space for future output or enforce `max_model_len`. vLLM's allocator also supports full-sequence admission checks. The lab supports FCFS and stops on EOS or the output limit.

vLLM accepts `long_prefill_token_threshold` with chunked prefill off and still caps admitted prompts by it; the lab rejects that configuration. The lab has no adaptive threshold (`long_prefill_token_threshold_adaptive`).

## Planned features

- **M7, preemption:** release blocks in LIFO request order, return preempted requests to the waiting head, and recompute their prompt and generated tokens. A step that preempts admits no waiting requests. M7 will decide how resumed requests appear in the batch.
- **M10, prefix caching:** hash full blocks with chained parent hashes, share cached prefixes, and remove hash entries on eviction. Admission lookup covers at most `num_tokens - 1` tokens so some computation remains.
- **Abort, workloads and metrics:** add request cancellation, deterministic workload replay, traces and an explicit duration model. The planned sampler must produce the same tokens for a request across preemption and different schedules. Reports must distinguish simulator estimates from engine measurements.

Sliding windows, hybrid allocators, multimodal inputs, LoRA, speculative decoding and async scheduling remain outside the simulator's scope.
