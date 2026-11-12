# Architecture

The simulator keeps request state, token IDs, and block metadata. It does not
store KV tensors or execute a model. A driver submits requests, asks the scheduler
for a batch, then supplies the tokens that batch produced.

## Code layout

| File | Role |
| --- | --- |
| [scheduler.py](src/kvsim/scheduler.py) | Waiting and running requests, token budgets, preemption, batch completion |
| [kv_cache_manager.py](src/kvsim/kv_cache_manager.py) | Each request's block table, prefix lookup, capacity checks |
| [block_pool.py](src/kvsim/block_pool.py) | Blocks, reference counts, free queue, hash lookup |
| [request.py](src/kvsim/request.py), [block_hash.py](src/kvsim/block_hash.py) | Request progress and hashes of token prefixes |
| [workload.py](src/kvsim/workload.py), [simulator.py](src/kvsim/simulator.py) | Request generation, arrivals, synthetic output tokens, simulated time |
| [metrics.py](src/kvsim/metrics.py), [__main__.py](src/kvsim/__main__.py) | Result summaries and the command line |

The scripts in `benchmarks/` run real engines and compare their results with the
simulator. Their setup and measurement rules are in
[benchmarks/README.md](benchmarks/README.md).

## A scheduling step

Prefill processes a request's input. Once that is done, each decode step processes
one more token and produces the next output token.

`schedule()` serves running requests in admission order, then takes waiting
requests from the front of the queue. Both use the same token budget. A long
prefill can use the remaining budget and continue in the next step. Running
requests can still be in prefill; the running list is not a separate decode queue.

Admission stops when the next request cannot get its tokens or blocks, or when
the running limit is reached. A shorter request cannot skip ahead. A positive
`long_prefill_token_threshold` caps each request's work while more than one request
is live, so several prefills can advance in the same step.

Scheduling reserves blocks. `complete_step()` advances progress, appends output
tokens, and frees completed requests. The caller must complete the exact returned
batch before scheduling again, including an empty batch.

```python
from kvsim.request import Request
from kvsim.scheduler import Scheduler, SchedulerConfig

scheduler = Scheduler(SchedulerConfig(num_blocks=8, block_size=4))
request = Request("r0", prompt_token_ids=(1, 2, 3, 4), max_output_tokens=2, arrival_step=0)
scheduler.add_request(request)

batch = scheduler.schedule()
assert batch.num_scheduled_tokens == {"r0": 4}
assert scheduler.complete_step(batch, {"r0": 7}) == []

batch = scheduler.schedule()
assert batch.num_scheduled_tokens == {"r0": 1}
assert scheduler.complete_step(batch, {"r0": 8}) == [request]
```

The caller supplies one sampled token per scheduled request. Completion discards
that token if the request still has input to compute. Otherwise it appends the
token and checks EOS and the output limit. `finish_requests()` also allows a caller
to abort requests, including while a batch is pending.

## Blocks and prefix reuse

The cache is a fixed pool of blocks. Each request has a block table, which grows
as it computes more tokens. With blocks of four tokens, a request needs one block
for four computed tokens and two for five.

A block's reference count is the number of request tables holding it. A count of
zero puts it in the free queue, but a free block can still hold a cached prefix.
The pool removes its hash only when it reuses that block or resets the cache.

The free queue has two parts. Unhashed blocks go first; cached blocks follow in
the order they were freed. Allocation consumes unhashed blocks before evicting
cached ones. The manager frees a request's table in reverse order, so its later
cached blocks become eviction candidates before its earlier ones.

Each full block has a hash of its tokens and the previous block's hash. A matching
hash identifies the tokens from the start of the request through that block.
Lookup starts at the first block and stops at the first miss. Partial blocks are
not cached.

A hit must leave at least one input token to compute before sampling an output.
For a block size of four, an eight-token input can reuse at most four tokens; a
nine-token input can reuse eight. The scheduler recomputes the last block when a
cached input ends on a block boundary.

The manager retains hit blocks before allocating new ones. Otherwise allocation
could evict one of the hits it is about to use. Free hit blocks also count against
available capacity. If an allocation cannot fit, it changes no state.

Full blocks enter the cache during scheduling. A second request admitted in the
same step can reuse the first request's prefix. Blocks completed by decode also
enter the cache. Prefix-hit counters count each request's first admission;
resuming after preemption can reuse blocks without adding to those counters.

The hash map keeps one block per hash. If another block has the same hash, the
second copy stays unhashed. Later blocks can then depend on a cached block their
request does not hold. Evicting that earlier block makes them unreachable through
prefix lookup. The analysis script counts those blocks as orphaned.
[tests/test_analyze.py](tests/test_analyze.py) covers both ways duplicates arise:
recomputing a cached last block and computing a shared prefix in parallel chunks.

## Memory pressure

If a running request needs blocks the pool cannot supply, the scheduler preempts
the most recently admitted running request and retries. It can preempt several
requests in one step. A step that preempts does not admit waiting requests.

Preemption frees the victim's blocks and resets its computed-token count. It keeps
the generated tokens and first-token timestamp. The victim goes to the front of
the waiting queue, ahead of fresh requests. On resumption it looks up its prefix
again and recomputes whatever is missing before generating another token.

By default, admission requires space for the first chunk. With
`enable_full_input_admission=True`, the scheduler first checks whether the whole
input fits. It still allocates only the scheduled chunk and reserves no space for
future output. The benchmark analysis enables this check for vLLM replays unless
the recorded launch command disabled it. The CLI has no flag for this setting.

`add_request()` rejects a request whose longest possible computed sequence cannot
fit in the cache: `prompt_tokens + max_output_tokens - 1`. This lets the oldest
running request make progress after the others have been preempted. Without
chunked prefill, that sequence must also fit the token budget.

## Workloads and time

A workload fixes the seed, prompt and output lengths, shared prefixes, and arrival
model. `prompt_len` counts tokens after the shared prefix. Requests take prompt
ranges and prefix groups in turn. The project pins Python 3.12 and tests the
generated workloads against fixed digests.

A closed loop submits a new request when one finishes, keeping the requested
concurrency while requests remain. Poisson arrivals use gaps drawn from an
exponential distribution. The simulator submits those requests at the next step
boundary; when idle, it jumps to the next arrival.

Synthetic output tokens depend on the request ID and output position. They stay
the same after preemption and across schedules. The simulation driver avoids EOS,
so requests finish at their workload's output length.

The estimated duration of a step is:

```text
step overhead + prefill cost * prefill tokens + decode cost * decode sequences
```

A request scheduled for one token counts as a decode sequence in this formula,
even if that token finishes a prefill. The default coefficients are placeholders.
The analysis script fits separate coefficients for each measured engine.

For closed-loop workloads, changing those coefficients changes times but leaves
the schedule and counters unchanged. For Poisson workloads, it can change which
requests arrive during a step, so it can change the schedule too. Identical
workloads, configurations, and coefficients produce identical results.

## Results and checks

A result records its configuration, workload, request timestamps, and cache
samples. The CLI adds a summary. Simulator times are estimates; engine times are
measurements. The data structures live in
[simulator.py](src/kvsim/simulator.py), and the shared summary calculation lives
in [metrics.py](src/kvsim/metrics.py).

The simulator samples cache usage after scheduling and before completion. Its
sample includes blocks held by requests that finish in that step. The benchmark
client samples engine gauges on a timer, so it can miss short peaks. Missing
metrics remain `null` and print as `-`.

[tests/invariants.py](tests/invariants.py) checks block ownership, reference counts,
queue membership, token budgets, and allocation failures. The stateful tests mix
arrivals, steps, preemptions, and aborts, then drain and replay the scheduler. Tests
also pin generated workloads, check output determinism across processes, and
verify that the committed benchmark tables match the result files.

## Reading vLLM alongside kvsim

The original source comparison uses vLLM revision
`8122776fba2dcfae2973a0659db2da6ac688306b`. The main files to read alongside this repo
are its [scheduler](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py),
[KV cache manager](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_manager.py),
[block pool](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/block_pool.py),
and [hashing code](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/kv_cache_utils.py).

kvsim uses synchronous steps, one full-attention cache, and one cached block per
hash. Its hash encoding differs from vLLM's. It has no reserved null block or
admission watermark. The benchmark scripts account for vLLM's null block when
matching usable cache sizes.

GPU execution, async scheduling, sliding windows, hybrid caches, multimodal
inputs, LoRA, and speculative decoding are outside the model. The timing limits
show up in the [benchmark findings](docs/findings.md).
