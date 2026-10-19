kv-cache-lab is a deterministic simulator of the scheduler and the paged KV-cache manager of an LLM inference engine, modelled on vLLM V1: a fixed pool of KV-cache blocks with reference counts and a free queue, per-request block tables, prefix caching by block hash, and a scheduler that batches prefill and decode tokens under one budget and preempts requests under memory pressure. A test suite checks the memory invariants after every operation. The same request workloads are meant to run against vLLM and LMDeploy on one GPU, so that the simulator's estimates can be set next to measured numbers and the differences explained. It is a learning project, not a product.

The project is early and simulator only. It is built in order, the block pool and the request model first, then the KV-cache manager, the scheduler, prefix caching, the workload generator and the metrics, and the GPU benchmarks last. Nothing here executes a model, touches a GPU or needs a tokenizer, and nothing has been measured yet; the package has no runtime dependency.

Running it needs Python 3.12 and uv. One command syncs the environment and runs the lint, type and test gate:

    make check

Licensed under the MIT License; see LICENSE.
