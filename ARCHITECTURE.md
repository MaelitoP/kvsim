# Architecture

kv-cache-lab is a deterministic simulator of the scheduler and the paged KV-cache manager of an
LLM inference engine, modelled on vLLM V1. It makes the memory and scheduling mechanisms of such
an engine explicit and testable: a fixed pool of KV-cache blocks, per-request block tables, prefix
caching by block hash, and a scheduler that spends one token budget on prefill and decode and
preempts requests under memory pressure. There is no model execution, no GPU code and no tokenizer.
The package has no runtime dependency.

## Modules

Target layout under `src/kv_cache_lab/`, in build order. A module is marked planned until it
exists.

| Module | Role | Status |
| --- | --- | --- |
| `block_pool.py` | `KVCacheBlock`, `FreeBlockQueue`, `BlockPool`: the fixed pool of blocks, reference counts, the free queue in eviction order, block hashes and eviction | planned |
| `requests.py` | `Request`, `RequestStatus`: one request's tokens, computed-token counter, lifecycle status and timeline | planned |
| `kv_cache_manager.py` | `KVCacheManager`: per-request block tables on top of the pool, prefix-cache lookup on admission | planned |
| `prefix_cache.py` | block hashing of full blocks with chained parent hashes | planned |
| `scheduler.py` | `SchedulerConfig`, `SchedulerOutput`, `Scheduler`: one `schedule()` call per step, token budget, chunked prefill, preemption, abort | planned |
| `workload.py` | deterministic request workloads generated from a JSON spec | planned |
| `simulator.py` | step-by-step replay of a workload with an explicit cost model | planned |
| `metrics.py` | per-request and aggregate metrics, result summary | planned |
| `__main__.py` | CLI: `simulate`, `summarize` | planned |

Tests live in `tests/`, one module per source module, plus `tests/invariants.py`, which holds the
shared `check_pool`, `check_manager` and `check_scheduler` helpers that every test calls after an
operation, and `tests/test_stateful.py` for the hypothesis state machines.

## Invariants

These are the contract of the simulator. They hold after every public operation and each one maps
to an assertion in `tests/invariants.py`.

### Determinism

The simulator never reads the wall clock and never uses unseeded randomness. The same workload
spec, configuration and seed produce a byte-identical trace. Time inside the simulator is a step
counter plus an estimated duration from an explicit cost model; both are reported and the estimate
is always labelled as such.

### Memory model

The KV cache is a fixed pool of `num_blocks` blocks of `block_size` tokens. A request whose first
`n` tokens are computed or scheduled needs exactly `ceil(n / block_size)` blocks. Block tables are
append-only lists per request. There is one KV-cache type, full attention: no hybrid allocators,
no sliding window, no multimodal inputs, no LoRA, no speculative decoding, no async scheduling.

- Every block is in exactly one of two states: free (`ref_cnt == 0`, present in the free queue)
  or allocated (`ref_cnt > 0`, absent from the free queue). The total number of blocks never
  changes.
- `ref_cnt` of a block equals the number of request block tables that contain it.
- A block with a hash is present in the hash map under that hash and the map entry points to
  that block; a block that is not in the map has no hash. A cached block may be free (reusable
  until evicted) or allocated (shared).
- A failed allocation has no side effects: no block touched, no ref count changed, no hash cached.
- A finished, aborted or preempted request references zero blocks.
- The sum over requests of block-table lengths equals the sum of ref counts.

### Scheduling model

One `schedule()` call per step. Running requests are served before waiting ones; admission is
FCFS. One token budget (`max_num_batched_tokens`) is spent by prefill and decode tokens alike, and
at most `max_num_seqs` requests run at once. Chunked prefill caps a request's tokens by the
remaining budget. When a running request cannot get its blocks, running requests are preempted
LIFO; a preempted request goes back to the head of the waiting queue and is recomputed from
scratch, keeping its generated tokens. A step in which a preemption happened admits no waiting
request. A request in partial prefill produces no output token. Between steps a decoding request
has `num_computed_tokens == num_tokens - 1`: the sampled token is appended to its tokens and
computed in the next step. On admission with prefix caching, the cache lookup covers at most
`num_tokens - 1` tokens, so a fully cached prompt recomputes its whole last block and at least one
token is always computed. The output of a request never depends on scheduling decisions: a
preempted and resumed request produces the same token ids as an uninterrupted one.

### Measurement honesty

Measured numbers come from the engines and the load client; estimated numbers come from the
simulator; every table, plot and sentence says which is which.

## Differences from vLLM

Deliberate simplifications, listed as the modules that carry them appear.
