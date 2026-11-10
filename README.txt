kvsim is a Python simulator for learning how vLLM schedules requests and manages a paged KV cache. It does not run a model or require a GPU, tokenizer or runtime dependencies.

Implemented: block allocation and reference counts, request tracking, per-request block tables, and FCFS batching under a shared prefill and decode token budget, with long prompts chunked across steps. When the cache runs out, the most recently admitted request is preempted and later recomputed, keeping the tokens it already generated and reusing any of its blocks still cached. With prefix caching on, the default, full blocks are hashed into a chain of prefix fingerprints and stay cached after their request ends, evicted least recently freed first; a new request reuses the cached blocks of its longest matching prefix and computes only the rest. Requests can be aborted in any state, and every block they held is released. The scheduler's caller supplies the sampled tokens. A workload is described by a JSON spec, from which the same request list is generated every time; the five benchmark scenarios live in benchmarks/workloads. The simulator replays a workload step by step and writes one JSON result: per request, the steps and estimated times of its first token and its end; per step, cache usage, batch size, queue length, preemptions and prefix cache hits. Its times are estimates from an explicit cost model whose default coefficients are placeholders, not measurements. The result file ends with a summary: time to first token, time per output token and end-to-end latency as mean and percentiles, throughput, cache usage, preemptions and the prefix cache hit rate.

Simulate a scenario with a cache of 2048 blocks of 64 tokens:

    uv run kvsim simulate benchmarks/workloads/shared_prefix.json --num-blocks 2048 --block-size 64 --out results/simulator/shared_prefix.json

Print the summaries of one or more results as a table, one row per file:

    uv run kvsim summarize results/simulator/shared_prefix.json

See ARCHITECTURE.md for the scheduling example, state ownership, contracts, the result format and vLLM references.

Development requires Python 3.12 and uv. Sync the environment and run lint, type checks and tests with:

    make check

MIT licensed. See LICENSE.
