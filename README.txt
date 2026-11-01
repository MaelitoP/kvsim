kvsim is a Python simulator for learning how vLLM schedules requests and manages a paged KV cache. It does not run a model or require a GPU, tokenizer or runtime dependencies.

Implemented: block allocation and reference counts, request tracking, per-request block tables, and FCFS batching under a shared prefill and decode token budget, with long prompts chunked across steps. When the cache runs out, the most recently admitted request is preempted and later recomputed, keeping the tokens it already generated and reusing any of its blocks still cached. With prefix caching on, the default, full blocks are hashed into a chain of prefix fingerprints and stay cached after their request ends, evicted least recently freed first; a new request reuses the cached blocks of its longest matching prefix and computes only the rest. Requests can be aborted in any state, and every block they held is released. The caller supplies sampled tokens. A workload is described by a JSON spec, from which the same request list is generated every time; the five benchmark scenarios live in benchmarks/workloads. Simulation and metrics are planned.

See ARCHITECTURE.md for the scheduling example, state ownership, contracts and vLLM references.

Development requires Python 3.12 and uv. Sync the environment and run lint, type checks and tests with:

    make check

MIT licensed. See LICENSE.
