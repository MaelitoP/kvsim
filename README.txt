kv-cache-lab simulates how an LLM inference engine schedules requests and manages a paged KV cache, based on vLLM V1. I'm building it to understand block allocation, prefix caching, prefill and decode batching, and preemption under memory pressure.

The simulator is deterministic, with tests checking memory invariants after each operation. It does not execute a model or require a GPU or tokenizer. The package has no runtime dependencies.

Only the block pool is implemented. Next come the request model, cache manager, scheduler, prefix caching, workloads and metrics. GPU benchmarks will follow, running the same workloads against vLLM and LMDeploy to compare measured results with the simulator's estimates. Nothing has been benchmarked yet.

Architecture and behavioral contracts are documented in ARCHITECTURE.md.

Development requires Python 3.12 and uv. To sync the environment and run lint, type checks and tests:

    make check

MIT licensed. See LICENSE.
