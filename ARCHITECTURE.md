# Architecture

kvsim models request scheduling and paged KV-cache allocation for a full-attention inference engine. It does not run a model or store KV tensors. Given the same request arrivals and sampled tokens, it produces the same batches and block assignments. It has no runtime dependencies and does not read the wall clock.

A load client under `benchmarks/` submits the same workloads to a real engine and writes its measurements in the simulator's result format. It is a script outside the package, with its own dependencies, and the only code here that reads a clock.

## Components and ownership

| Module | Responsibility | Owned state |
| --- | --- | --- |
| `block_pool.py` | Allocate, retain and release fixed-size blocks; register and evict cached blocks | Block objects, reference counts, free queue and hash-to-block map |
| `block_hash.py` | Hash full blocks of tokens into a chain of prefix fingerprints | None |
| `request.py` | Store request inputs, progress and outcome | Prompt, generated tokens, status, timeline and block hashes |
| `kv_cache_manager.py` | Allocate blocks for each request; find cached prefixes and cache full blocks | Block pool, request block tables and per-request cached-block counts |
| `scheduler.py` | Build batches, preempt under memory pressure, apply sampled tokens and abort requests | Waiting queue, running list, live request registry, pending batch, preemption total and prefix-cache counters |
| `workload.py` | Read a workload spec from JSON and generate its requests deterministically | None |
| `simulator.py` | Replay a workload through a scheduler, estimate step durations and build the result | None |
| `metrics.py` | Summarize a result: latencies, throughput, cache usage, preemptions and prefix hit rate | None |
| `__main__.py` | Command line: `kvsim simulate` and `kvsim summarize` | None |
| `benchmarks/client.py` | Measure a running engine under a workload: text prompts, streamed completions, metric scrapes | None |

`BlockPool`, `KVCacheManager` and `Scheduler` use explicit constructors. Queues, block storage and request tables are private. Callers use the existing counts and `usage` to inspect capacity. Tests inspect private state through the invariant helpers; there are no test-only public accessors.

`Request` and `KVCacheBlock` are mutable dataclasses with identity-based equality. `SchedulerConfig`, the returned `ScheduledBatch`, the workload types `WorkloadSpec`, `ClosedLoop`, `Poisson` and `RequestSpec`, and `CostModel` are frozen dataclasses. A simulation result and its summary are `TypedDict`s: they are the content of a JSON file, written and read back as plain dictionaries.

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

## Workloads

`workload.py` describes a benchmark scenario as a `WorkloadSpec` and expands it into the requests a driver submits, a list of `RequestSpec(request_id, prompt_token_ids, output_len, arrival_s)`. The simulator generates from the spec, and the load client generates from the same one, so both submit the same requests; the client turns the token IDs into text first. Request IDs are `r0`, `r1` and so on, in generation order.

A spec names its scenario and fixes a seed, a vocabulary size, a request count, length ranges, a shared prefix and an arrival model. Ranges are inclusive and every length is at least one. `prompt_len` holds one or more ranges, and request `i` draws from range `i % len(prompt_len)`, so two ranges alternate short and long prompts. `output_len` is one range. With a positive `shared_prefix_len`, requests are dealt round robin to `num_prefix_groups` groups, each with its own random prefix of that length, and `prompt_len` sets the number of tokens after the prefix. Several groups without a shared prefix is rejected. Every token is drawn uniformly from `range(vocab_size)`, so past its group prefix a prompt matches another prompt's token at the same position only by chance, with probability `1 / vocab_size`. For two prompts to share one cached block beyond the prefix, that coincidence would have to repeat for `block_size` tokens in a row.

The arrival model is `ClosedLoop(concurrency)` or `Poisson(rate)`. In a closed loop every request has `arrival_s == 0.0`: all are ready at the start, and the driver keeps at most `concurrency` of them in flight, submitting the next one when one finishes. With Poisson arrivals the first request arrives at 0.0, and the gaps between arrivals are independent exponential draws with a mean of `1 / rate` seconds. Arrival times are simulated seconds for the simulator and wall-clock seconds for the client.

`generate_requests(spec)` draws everything from one `random.Random(spec.seed)` in a fixed order: the group prefixes; then, for each request, its prompt length, its tokens and its output length; then the arrival gaps. The arrival model therefore never changes the prompts. Across Python versions, only `Random.random()` is guaranteed to produce the same sequence. The project pins Python 3.12, and tests pin a digest of the requests generated for each benchmark scenario and for a Poisson workload, so any change in generation, deliberate or not, fails a test. `Random` ignores the sign of an integer seed, so negative seeds are rejected rather than producing the same workload as their positive counterpart.

`load_spec(path)` reads a UTF-8 JSON file. `parse_spec(data)` reads an already decoded object, such as the `workload` entry of a result file. A length can be written as a number or a `[low, high]` range; `prompt_len` can also be a list of ranges. The arrival is an object whose `kind` is `closed_loop` or `poisson`. Every key is required and unknown keys are rejected. Integer fields must be JSON integers, not booleans, floats or strings, so a misspelt key or a quoted number raises `ValueError` instead of changing the workload. `dataclasses.asdict(spec)` gives the spec as plain data, with ranges as tuples and the arrival with its `kind`. `parse_spec` reads it back to an equal spec, directly or after a trip through JSON.

`benchmarks/workloads/` holds the five scenarios of the benchmark matrix. All of them use a closed loop and seed 0:

| File | Prompt tokens | Output tokens | Concurrency | Requests | Studies |
| --- | --- | --- | --- | --- | --- |
| `short_low.json` | 128 to 256 | 128 | 4 | 200 | Baseline |
| `long_high.json` | 1536 to 2048 | 128 | 32 | 200 | Saturation, time to first token |
| `shared_prefix.json` | 1024 shared in 8 groups, then 128 to 256 | 128 | 16 | 256 | Prefix caching |
| `mixed.json` | 64 to 128 and 1536 to 2048, alternating | 128 | 16 | 200 | Chunked prefill |
| `saturation.json` | 1024 | 512 | 48 | 192 | Preemption under a small cache |

Their vocabulary size, 151,643, is the number of regular tokens of the Qwen2.5-3B-Instruct tokenizer, the model the benchmarks will run. Its special tokens start at ID 151,643, so every generated ID decodes to ordinary text.

## Simulation

`simulate(spec, config, cost_model)` replays a workload through a fresh `Scheduler` and returns the result as plain data. It reads no clock and draws nothing at random beyond the spec's seed. Time is the scheduler's step counter plus a clock of estimated seconds, and the result reports both.

Each turn of the loop is one step. The requests that have arrived are submitted first, in spec order. In a closed loop the next request arrives as soon as fewer than `concurrency` requests are in flight, submitted and not finished: at 0.0 for the first ones, then at the end of a step in which a request finished. Its recorded `arrival_s` is the clock at that moment. With Poisson arrivals a request is submitted at the first step boundary at or after its `arrival_s`, and that `arrival_s` is recorded, so the wait for the boundary counts in its time to first token. When nothing is in flight the clock jumps to the next arrival and no step runs. The simulator then calls `schedule()`, records a sample, samples one token for each scheduled request, calls `complete_step` and advances the clock by the estimated duration of the step.

A sampled token depends only on the request's ID and the number of tokens the request has generated, through CRC-32. A request's output is therefore the same whatever the schedule, after a preemption and in every process; Python's `hash` of a string changes from one process to the next. The token is never the EOS token. A request ends at its output limit, the `output_len` of its spec, which is what the load client asks of an engine with `ignore_eos` and `max_tokens`.

`CostModel(step_overhead_s, prefill_s_per_token, decode_s_per_seq)` estimates a step as `step_overhead_s + prefill_s_per_token * num_prefill_tokens + decode_s_per_seq * num_decode_seqs`. A request scheduled for exactly one token counts as a decode sequence; a request scheduled for more adds its tokens to the prefill tokens. The split follows what the step computes, not the request's history: the one-token tail of a chunked prompt, or a prompt whose cache hit leaves one token, costs one position like a decode. The step overhead must be positive, so every step takes time and sample times strictly increase. The other two coefficients must be at least zero, and all three finite. `step_duration` takes the token counts of a batch and makes the split itself. The defaults, 20 ms per step, 0.2 ms per prefill token and 0.5 ms per decode sequence, are placeholders, not measurements. Until coefficients are fitted to an engine, the seconds of a result only compare runs made with the same cost model; its step numbers and scheduler counters do not depend on the cost model at all.

### Result

The result has the shape the load client writes for engine measurements, with the simulator's own keys added:

| Key | Content |
| --- | --- |
| `source`, `measured` | `"simulator"` and `false`: every time in the file is an estimate |
| `engine_version`, `model`, `gpu` | The kvsim version; `null` for the other two |
| `config` | `block_size`, `num_blocks`, `kv_tokens` (their product), `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, `enable_chunked_prefill`, `long_prefill_token_threshold`, and `command`, `null` for the simulator |
| `cost_model` | The three coefficients behind every time in the file; simulator only |
| `workload` | The spec as `dataclasses.asdict` gives it; `parse_spec` reads it back |
| `requests` | One record per request, in spec order |
| `samples` | One record per step; an engine result has one per scrape of its metrics |
| `summary` | The metrics of the run, added by the command line; see Metrics |

A request record holds `request_id`, `arrival_s`, `first_token_s`, `finish_s`, `first_token_step`, `finish_step`, `prompt_tokens`, `output_tokens`, and `server_ttft_ms` and `server_queue_ms`, which only an engine reports and are `null` here. `first_token_s` and `finish_s` are the clock at the end of the steps that sampled the request's first and last tokens; the two step fields are simulator only.

A sample holds `step`, `t_s`, `kv_usage`, `running`, `waiting`, `preemptions_total`, `prefix_queries_total` and `prefix_hits_total`. It is taken after `schedule()` and before `complete_step`, and `t_s` is the start of the step: the sample describes the batch that runs from `t_s` until the step ends, which is the next sample's `t_s` unless the simulator then goes idle. The last step ends at the latest `finish_s`. `running` is the batch size, since every running request is scheduled. `waiting` counts the other requests in flight. `kv_usage` includes the blocks of requests that finish in the step, so its maximum over a run is the highest usage any step runs with. Inside `schedule()` the cache can briefly be fuller, before a preemption frees blocks. The counters are cumulative and include the step's own preemptions and admissions. Idle time between Poisson arrivals has no sample.

The same spec, configuration and cost model give the same result, and the same bytes once written: the clock adds the same floating-point durations in the same order, and no ordering depends on the string hash seed.

### Metrics

`summarize(result)` reduces a result to the figures that tables compare. It reads `requests` and `samples` and nothing else, so one function serves a simulator result and an engine measurement, in memory or decoded from a file. It takes plain data and checks what it reads, as `parse_spec` does. There must be at least one request. Each request needs finite times with `arrival_s <= first_token_s <= finish_s` and at least one output token. Each sample needs `kv_usage`, `preemptions_total`, `prefix_queries_total` and `prefix_hits_total`, any of which may be `null`. A usage must lie between 0 and 1, so a usage above 1, such as a percentage, is refused; the counters must be integers of at least zero, with no more hits than queries. A boolean is not a number. The run must last longer than zero seconds, and every figure of the summary must be finite: times so large that a sum or a percentile overflows, or so close together that a rate does, are refused. Anything else raises `ValueError`.

Three metrics are computed per request:

| Metric | Definition |
| --- | --- |
| Time to first token (TTFT) | `first_token_s - arrival_s`: queueing, prefill and any recomputation after a preemption that came before the first token |
| Time per output token (TPOT) | `(finish_s - first_token_s) / (output_tokens - 1)`, the mean gap between tokens after the first; undefined for a one-token output |
| End-to-end latency (E2E) | `finish_s - arrival_s` |

The summary holds:

| Key | Content |
| --- | --- |
| `num_requests`, `output_tokens` | Totals over the requests |
| `duration_s` | Latest `finish_s` minus earliest `arrival_s` |
| `requests_per_s`, `output_tokens_per_s` | The totals divided by the duration |
| `ttft_s`, `tpot_s`, `e2e_s` | `mean`, `p50`, `p95` and `p99` over the requests; `tpot_s` leaves out one-token outputs and is `null` when no request generated a second token |
| `kv_usage_mean`, `kv_usage_max` | Over the samples that have a usage; `null` when none has |
| `preemptions` | The last `preemptions_total` reading; `null` without one |
| `prefix_hit_rate` | `prefix_hits_total / prefix_queries_total` from the last sample that has both and at least one query, a ratio of tokens; `null` otherwise, which includes every run without prefix caching |

Percentiles come from `statistics.quantiles` with `method="inclusive"`. It interpolates linearly between the sorted values and treats them as the whole population, which a run's requests are: the 0th percentile is the minimum, the 100th the maximum, and no percentile leaves the observed range by more than floating-point rounding, which also applies to the mean. The default exclusive method assumes a sample from a larger population and can extrapolate past the largest value. A single value is its own mean and percentiles.

The usage mean is not weighted by time. A simulator result has one sample per step, whatever the step's estimated duration, so a short decode step counts as much as a long prefill step. Maximum usage does not depend on the sampling and is the figure to compare between results. The counters are read as totals of the run: the summary takes the last reading, not a difference, because a simulator's first sample already counts the admissions of step 0. A result must therefore hold counters that start at zero with the run, as the simulator's do. A writer whose engine counts since it started must subtract the reading it took before the run.

The seconds of a summary are what the result holds: estimates for a simulator result, measurements for an engine's. The summary does not repeat the `measured` flag of its result; whatever prints a summary must show it.

### Command line

`kvsim simulate <spec.json> --num-blocks N --out <result.json>`, or `python -m kvsim` with the same arguments, runs one simulation and writes the result with its summary as indented UTF-8 JSON, creating the parent directories of the output. `--block-size`, `--max-num-seqs`, `--max-num-batched-tokens`, `--long-prefill-token-threshold`, `--enable-prefix-caching` and `--enable-chunked-prefill`, each of the last two with its `--no-enable-...` form, set the scheduler configuration and default to `SchedulerConfig`'s defaults. `--cost-step-overhead-s`, `--cost-prefill-s-per-token` and `--cost-decode-s-per-seq` set the cost model. A spec that cannot be read, an invalid configuration or cost model, a request the cache could never hold, a result with times too large or too close together to summarize and an output path that cannot be written are reported as usage errors with exit status 2, and no result is written. JSON has no infinity, so a non-finite time is refused rather than written.

`kvsim summarize <result.json>...` prints a table with one row per file: the file, its `source`, whether its times are `measured` or `estimated`, the number of requests, the median and 95th percentile of TTFT and TPOT in milliseconds, output tokens and requests per second, mean and maximum cache usage, preemptions and the prefix hit rate. Columns that print a summary value as it is carry its key. A value the summary does not have prints as `-`. The summary is computed from the file's requests and samples, not read from its `summary`, so a file without one works and a stale one is ignored. When a row is estimated, a line under the table says that its seconds come from the cost model of the run and only compare between runs made with the same cost model. A file that cannot be read, is not a result or cannot be summarized is a usage error with exit status 2 that names the file.

## Load client

`benchmarks/client.py` measures an engine that serves the OpenAI completions API, vLLM or LMDeploy, under a workload spec. It needs `httpx` and `tokenizers` from the `bench` dependency group, which `uv sync` installs with the development tools so that the client is linted, type-checked and tested like the package. It never starts an engine: the engine's version, launch command and cache configuration are arguments, copied into the result.

### Prompts

Both engines take text on this endpoint, and LMDeploy takes nothing else, so `build_prompts(spec, tokenizer)` turns each request's token IDs into text with the model's tokenizer. Decoding random token IDs and tokenizing the text again does not give the same tokens back: with the Qwen2.5 tokenizer the sequence grows by 5 to 7%, and a 1024-token prefix comes back as 1070 to 1100 tokens. Each part is therefore fitted to its own length. The text of a prefix group is the decoding of the first `k` tokens of the group's prefix, with `k` chosen so that the text tokenizes to `shared_prefix_len` tokens. A request's suffix is fitted the same way so that prefix and suffix together tokenize to the request's prompt length. When more tokens are needed than the source holds, it is repeated from its start.

The search for `k` moves by the current excess while that gets closer to the target, then tries every count between its last two attempts and keeps the closest. It returns the closest text it found and does not fail on a remainder: a token whose text tokenizes to several tokens can make the exact length unreachable. On the five scenarios with seeds 0 and 1, 2,008 of 2,096 prompts get their exact length, all but one are within two tokens and that one is three tokens off, and every group prefix is exact. A token at the end of the prefix can merge with the start of a suffix: 12 of 496 requests share 1023 tokens with their group instead of 1024, which costs them the last of 16 shared blocks. Every record carries the prompt length the engine reports, so each deviation is visible in the result. A spec whose `vocab_size` exceeds the tokenizer's vocabulary is rejected: its token IDs would decode to nothing and the prompts would come out short.

### Requests and times

`measure(client, engine, model, spec, tokenizer)` first sends a warm-up pass, then the measured one. The warm-up is the same spec with the next seed: the same lengths, groups and arrival model with different tokens. A warm-up with the measured prompts would leave them in the engine's prefix cache, and the measured run would then hit almost every token, which neither describes the workload nor matches a simulation that starts with an empty cache. Nothing of the warm-up is recorded.

A request is `POST /v1/completions` with the text prompt, `max_tokens` set to the request's `output_len`, `ignore_eos`, `temperature: 0`, `stream` and `stream_options: {"include_usage": true}`. In a closed loop at most `concurrency` requests are open, and the next one is sent when one ends. With Poisson arrivals each request waits for its `arrival_s`, with no limit on open requests. The clock and the wait are arguments of `measure`, `time.perf_counter` and `asyncio.sleep` by default, so tests run on a fake time.

Times are seconds since the measured pass started. `arrival_s` is read just before the request is sent. `first_token_s` and `finish_s` are the arrivals of the first and the last stream event with a non-empty `choices` list. Such an event carries at least one generated token on vLLM; an engine may put several tokens in one event, so tokens are never counted from events. LMDeploy ends a stream with one more event that has an empty text and the finish reason, so its `finish_s` is that event's arrival. `prompt_tokens` and `output_tokens` come from the usage event that closes the stream. `server_ttft_ms` and `server_queue_ms` are vLLM's `time_to_first_token_ms` and `queue_time_ms`, present when the server runs with `--enable-per-request-metrics`. vLLM measures that time to first token from the moment the request is scheduled, so it excludes the queue time, which the client's TTFT includes. LMDeploy reports neither, and both are `null`.

A request that fails ends the run with an error and no result: an HTTP error status, an `error` event in the stream, which both engines send under status 200, an event that is not JSON, a stream that ends without a token or without usage, or a usage whose `completion_tokens` differs from the request's `output_len`. Both engines shorten an output that would pass their context length instead of refusing it, so only that last check tells a truncated request from a complete one. A failed scrape ends the run too. A result that lacks requests, or holds shorter ones, would not measure the workload. The other requests are cancelled and `measure` raises an `ExceptionGroup`. A request times out after 600 seconds without an event and a scrape after 10, so a stalled engine ends the run instead of hanging it.

### Samples

A sample is one scrape of `GET /metrics`. The first is taken after the warm-up and before the measured pass, at `t_s` 0.0; then about one per second, the interval running from one scrape's answer to the next scrape, and a last one after the last request ended, so that it holds the totals of the run. A sample name is matched exactly and summed over its label sets: `vllm:num_requests_waiting` is also the beginning of another metric's name.

| Sample key | vLLM | LMDeploy |
| --- | --- | --- |
| `kv_usage` | `vllm:kv_cache_usage_perc` | `lmdeploy:gpu_cache_usage_perc` |
| `running` | `vllm:num_requests_running` | `lmdeploy:num_requests_running` |
| `waiting` | `vllm:num_requests_waiting` | `lmdeploy:num_requests_waiting` |
| `preemptions_total` | `vllm:num_preemptions_total` | none, `null` |
| `prefix_queries_total` | `vllm:prefix_cache_queries_total` | `lmdeploy:prompt_tokens_total` |
| `prefix_hits_total` | `vllm:prefix_cache_hits_total` | `lmdeploy:cached_tokens_total` |

A metric the engine does not publish is `null`, never zero. The three counters are stored as the difference from the first scrape, so they count the measured pass only, as `summarize` requires; a counter that appears after the first scrape started from zero. Both usage gauges are fractions between 0 and 1 despite their names. A usage outside that range, or a counter below its first reading, ends the run at that scrape: `summarize` would refuse the result, and failing early names the metric.

The engines do not count the same things. vLLM's usage is the share of blocks that are allocated, cached free blocks excluded. LMDeploy's is the share of its cache memory held by live objects, which includes blocks its prefix cache keeps after a request ended, so with prefix caching it stays high until an eviction. LMDeploy refreshes its usage and request gauges every 10 seconds, so ten consecutive samples can repeat one reading and a short peak can be missed. vLLM counts prefix queries and hits in tokens when a request is first admitted and leaves out a request resumed after a preemption. LMDeploy counts a request's prompt tokens when its first token is produced and its cached tokens when it finishes; the two totals agree once every request has finished, which the last sample guarantees. With prefix caching off, vLLM's queries stay at zero and the hit rate is `null`, while LMDeploy's prompt tokens still count and the rate is 0.0. LMDeploy has no preemption counter.

### Engine result

`python benchmarks/client.py <spec.json> --engine vllm --base-url <url> --model <name> --out <result.json>` with the engine's `--engine-version`, `--gpu`, `--command`, `--num-blocks`, `--block-size`, `--max-num-seqs`, `--max-num-batched-tokens`, `--[no-]enable-prefix-caching` and `--[no-]enable-chunked-prefill`, all required, and `--long-prefill-token-threshold`. `--tokenizer` names the model's `tokenizer.json`; without it the tokenizer of `--model` is fetched from the Hugging Face Hub. The result is written with its summary, like a simulator result. A spec that cannot be read, a tokenizer that lacks the spec's token IDs and an output path that cannot be written are usage errors with exit status 2; the output directory is created before the engine receives a request.

An engine result has `source` set to the engine, `measured` set to `true`, the engine's version, model and GPU, and the launch command in `config.command`. It has no `cost_model`, its requests have no `first_token_step` and `finish_step`, and its samples have no `step`: an engine does not expose its steps. `summarize` and `kvsim summarize` read it like a simulator result.

## Invariants and where they are checked

The helpers live in `tests/invariants.py`, except the machines. `tests/strategies.py` holds the hypothesis strategies for workload specs and scheduler configurations that several test modules share. `check_scheduler` calls `check_manager`, which calls `check_pool`. `run_checked_step` drives one step and calls `check_scheduler` after its `schedule()`, `finish_requests` and `complete_step`; the scheduler tests run their steps through it, and some also call the scheduler directly to test rejected calls. The machines call their checker after every rule.

| Invariant | Checked by |
| --- | --- |
| A block is either free, with a reference count of zero and a place in the free queue, or allocated, with a positive count and no place in the queue. Block IDs are `0` to `num_blocks - 1`. | `check_pool` |
| The pool holds the same block objects throughout its life. | `BlockPoolMachine` |
| A block has a hash exactly when the map holds it under that hash; a cached block may be free or allocated. The free queue holds every unhashed free block before every cached one. | `check_pool` |
| A block's reference count equals the number of request tables holding it, which implies that the table lengths sum to the reference counts. A table never holds a block twice. Without prefix caching no block is shared and none has a hash. | `check_manager` |
| Blocks a table has not cached carry no hash, and only tables have cached-block counts. | `check_manager` |
| A failed allocation changes nothing: no reference count, hash, queue position, table or cached-block count. | `CheckedKVCacheManager`, which wraps every allocation made through `checked_scheduler` |
| A running request's table holds exactly enough blocks for its computed tokens plus the tokens the pending batch reserves for it. Waiting and preempted requests hold no blocks, and only running requests have tables. | `check_scheduler` |
| A finished or aborted request is no longer live, so it holds no blocks. | `check_finished`, with `check_scheduler` |
| Each live request is in exactly one queue, with the matching status and object identity. The waiting queue holds preempted requests first, then fresh ones; a preempted request has been preempted at least once, a fresh one was never scheduled and has no output. | `check_scheduler` |
| A pending batch schedules every running request, at least one token each, within the token budget and the running limit, in running, resumed, new order. Its other IDs belong to requests aborted since, gone or re-added and waiting. A batch that preempts admits nothing, and its live victims are in the waiting queue. | `check_scheduler` |
| Between steps, a running request has at least one computed and one uncomputed token; one that was never preempted and has output has `num_computed_tokens == num_tokens - 1`. A cache hit therefore leaves at least one token to compute. | `check_scheduler` |
| With prefix caching, a running request's cached-block count is its reserved tokens divided by the block size, rounded down; those blocks carry the request's hash for their position or no hash, and the request's hashes are a prefix of its token chain. Hits never exceed queries and are whole blocks; both stay zero without caching. | `check_scheduler` |
| Running requests are served first, the head of the running list is always scheduled, and waiting requests are admitted in queue order. Victims leave from the end of the running list, youngest first, and return to the head of the waiting queue in running order. | `run_checked_step` |
| Each `schedule()` returns the next step number. A step with an empty running list admits the head of a non-empty waiting queue. Each preemption adds one to the victim's count and to the scheduler's total. | `run_checked_step` |
| A request appends exactly the sampled token in the step its computation completes, so partial prefills and aborted requests produce nothing and no other output changes. Exactly the completing requests that sampled EOS or reached their output limit finish, as `FINISHED_STOPPED` when the last token is EOS and `FINISHED_LENGTH_CAPPED` otherwise; aborted requests finish as `FINISHED_ABORTED`. | `run_checked_step` |
| A request's output does not depend on the schedule: with a sampler that derives each token from the output position, every request ends with the output that sampler implies, whatever the configuration, pool size, preemptions or aborts. | `SchedulerMachine.teardown` in `tests/test_stateful.py` |
| Every request finishes: draining the scheduler takes at most the sum of the live requests' prompt and output lengths in steps, and leaves no block allocated. | `SchedulerMachine.teardown` |
| Within one process, the same inputs give the same trace: replaying every operation on a fresh scheduler yields equal batches, finished requests and abort results. This catches state-dependent nondeterminism, such as ordering by object identity or unseeded randomness, but not a dependence on the process's string hash seed, which the simulator's command-line test covers. | `SchedulerMachine.teardown` |
| After `update_block_hashes` with one block size, a request's hashes are the hash chain of its full blocks, and earlier hashes are never recomputed. | `tests/test_request.py`; a pinned digest in `tests/test_block_hash.py` guards the encoding |
| A workload spec generates the same requests in every process: a digest of each benchmark scenario's requests and of a Poisson workload is pinned. Requests follow their spec: lengths within the range their position picks, the prefix of their group, tokens within the vocabulary, the spec's arrival model. A spec survives a round trip through `asdict`, with or without JSON. | `tests/test_workload.py` |
| A simulation follows its spec: one record per request in spec order, with the spec's prompt and output lengths. A request generates at most one token per step and arrives no later than the start of its first-token step. A step number maps to one end time, after that step's start and no later than the next step's start. | `check_result` |
| Samples are numbered from zero, one per step, at strictly increasing times that start at 0.0. Each schedules at least one request, within the running limit, and uses part of the cache. Counters never decrease, and prefix hits never exceed queries. `running + waiting` equals the number of requests that have arrived and not finished at the sample's time. | `check_result` |
| A closed loop keeps exactly `concurrency` requests in flight while requests remain: the first `concurrency` arrive at 0.0 and each later one arrives at the next finish time, in order. Poisson requests keep the arrival times of their spec. | `check_result` |
| A summary agrees with its result: the request and token counts, the duration, the maximum usage, the last preemption total and the last sample's ratio of prefix hits to queries, `null` without prefix caching. For TTFT, E2E and TPOT, the minimum, `p50`, `p95`, `p99` and the maximum are in ascending order and the mean lies within the range, up to rounding; so does the mean usage. Each E2E statistic is at least its TTFT counterpart. TPOT statistics exist exactly when a request generated a second token. Every simulator result can be summarized, and the file the command line writes stores exactly that summary. | `check_summary`, `tests/test_simulator.py` |
| Summary values match a hand computation on three requests, one of them with a single output token. A gauge without a reading gives `null`, and the last reading wins. A result with a wrong value, or with times too large or too close together to summarize, is refused. | `tests/test_metrics.py` |
| A text prompt tokenizes to within one token of its request's prompt length, and the prompts of a group share a text that tokenizes to within one token of the prefix length. The bound belongs to the test tokenizer, whose decoded tokens merge or split by one token at a time; the Qwen2.5 figures are in the Load client section. | `tests/test_client.py` |
| A measurement follows its spec: one record per request in spec order with the requested output length and `arrival_s <= first_token_s <= finish_s`. Samples start at 0.0 with zeroed counters, at strictly increasing times, the last one no earlier than the last finish; a counter that has a reading keeps one and never decreases. In a closed loop, each request after the first `concurrency` arrives no earlier than a finish; a Poisson request is sent no earlier than its arrival. Every measurement can be summarized. | `check_measurement` |
| A closed loop opens exactly `concurrency` requests at once while that many remain, and Poisson arrivals are not limited: a fake engine holds each stream until the expected number of requests are open. A Poisson request is sent at its arrival time on a fake clock. The warm-up sends the next seed's prompts before the measured ones and is not recorded. Counters count the measured pass only, and one that appears late counts from zero. A sample reads each engine's metrics under the names pinned in a literal exposition. A missing metric is `null`. A failed, truncated or shortened request fails the run, in the warm-up or in the measured pass. The file the client's command line writes is summarized by `kvsim summarize` as measured. | `tests/test_client.py` |
| The same spec, configuration and cost model give the same result in one process, and byte-identical files from two processes with different string hash seeds. No result holds token IDs, so the files guard orderings; pinned values guard the sampled token itself, a fixed function of the request ID and the output position that is never EOS. | `tests/test_simulator.py` |

Two hypothesis machines drive these helpers. `BlockPoolMachine` in `tests/test_block_pool.py` mixes allocate, release, retain, cache and reset on one pool and checks each rule's effect on allocation order, eviction and the map against the state before the rule. `SchedulerMachine` in `tests/test_stateful.py` draws a configuration, then mixes arrivals of one to three requests, steps that may abort a request while the batch is pending, and aborts of live, finished and unknown IDs. The configuration covers prefix caching on and off, chunked prefill on and off with or without a long-prefill threshold, block sizes 2 and 4, and pools between one request's maximum and two blocks more, so preemption is frequent. Prompts take a block-aligned slice of one of two shared prefixes, so caching examples hit. Example tests cover the cases a machine reaches too rarely to rely on: allocation order, admission, chunking, preemption order, resumption, prefix sharing, the lookup cap and its first-miss stop, hits at the head of the free queue, aborts in every state and a same-ID re-add during a pending batch.

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
| `WorkloadSpec` lengths and shared prefix | [`vllm bench serve` dataset flags](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/benchmarks/datasets/datasets.py): `--random-input-len`, `--random-prefix-len`, `--prefix-repetition-prefix-len`, `--prefix-repetition-num-prefixes` |
| `ClosedLoop`, `Poisson` | [`--max-concurrency`, `--request-rate`, `--burstiness`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/benchmarks/serve.py) |
| Sample fields `kv_usage`, `running`, `waiting` and the prefix counters | [`SchedulerStats`, `PrefixCacheStats`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/metrics/stats.py), built by [`Scheduler.make_stats`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/v1/core/sched/scheduler.py) |
| `summarize`: TTFT, TPOT, E2E, throughput and percentiles | [`calculate_metrics` of `vllm bench serve`](https://github.com/vllm-project/vllm/blob/8122776fba2dcfae2973a0659db2da6ac688306b/vllm/benchmarks/serve.py) |
| `prefix_hit_rate`, `preemptions` | Ratio of `vllm:prefix_cache_hits_total` to `vllm:prefix_cache_queries_total`, and `vllm:num_preemptions_total` |
| `--max-num-seqs`, `--max-num-batched-tokens`, result keys `max_num_seqs`, `max_num_batched_tokens` | vLLM's flags of the same names, so an engine's configuration can be copied to a replay |

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

`vllm bench serve` builds similar workloads from dataset flags. Its random dataset also counts `--random-prefix-len` outside the sampled input length, but gives every request the same prefix. Its prefix repetition dataset generates `num_requests // num_prefixes` prompts per prefix, dropping the remainder, then shuffles them. The lab deals requests round robin, so it generates exactly the requested count and each request's group follows from its position. vLLM also regenerates token sequences until their decoded text re-encodes to the target length. The lab's workload holds token IDs only. Its load client decodes them once and searches for the number of source tokens whose text has the target length; vLLM's tool decodes, tokenizes again, cuts the tokens to the target or appends random ones, and repeats, up to ten times. Both get the measured length from the tokenizer rather than assuming that a decoded token tokenizes back to itself.

For arrivals, `--max-concurrency` with an infinite `--request-rate` is the lab's closed loop. A finite rate draws gamma-distributed gaps, which are exponential with `--burstiness 1`. vLLM delays even the first request by its own gap, then rescales every arrival time so that the last one lands at exactly `num_requests / rate`. The lab's first request arrives at 0.0, since an idle interval before it measures nothing. The lab keeps the gaps unscaled, so the arrivals remain a Poisson process, and the realised rate differs from `rate` only by sampling noise, about `1 / sqrt(num_requests)` relative.

vLLM builds its `SchedulerStats` at the end of `update_from_output`, once finished requests have left the running list and released their blocks, so its gauges show the state between two steps. The simulator samples after `schedule()`, the state a step runs with. Sampled between steps, the blocks and the batch slot of a request would disappear from the samples in the very step that finishes it, and the maximum usage of a run would miss its true peak whenever a request finishes in the fullest step. A simulator sample therefore reads at least as high as a between-steps reading of the same step. vLLM has no cost model: its step durations are measured, the simulator's are estimated from `CostModel`.

`vllm bench serve` defines TTFT, TPOT and end-to-end latency as the lab does, also leaving one-token outputs out of TPOT, and computes percentiles with NumPy's default linear interpolation, which is the inclusive method. A summary of an engine measurement is therefore comparable with that tool's report. It divides throughput by the wall time of the whole benchmark; the lab divides by the span from the first arrival to the last finish, which is the same interval for a run that starts with its first request. It also reports inter-token latencies and standard deviations. A result has no per-token times, so the lab reports TPOT only. The load client takes its three times as that tool does: TTFT at the first stream event with a non-empty `choices` list, the end at the last such event, and token counts from the usage event. Its warm-up differs: `--num-warmups` sends copies of the first request, which warms the engine and, with prefix caching, also caches that request's prompt; the lab sends a whole pass of other prompts.

## Planned features

- **Benchmarks:** scripts that launch vLLM and LMDeploy with comparable configurations and run the load client over a matrix of scenarios, then an analysis that tabulates the results and compares each measurement with a simulator replay.

Sliding windows, hybrid allocators, multimodal inputs, LoRA, speculative decoding and async scheduling remain outside the simulator's scope.
