kvsim is a Python simulator for learning how vLLM schedules requests and manages a paged KV cache. It does not run a model or require a GPU, tokenizer or runtime dependencies.

Implemented: block allocation and reference counts, request tracking, per-request block tables, and FCFS batching under a shared prefill and decode token budget. The caller supplies sampled tokens. Chunked prefill, preemption, prefix caching, workloads and metrics are planned.

See ARCHITECTURE.md for the scheduling example, state ownership, contracts and vLLM references.

Development requires Python 3.12 and uv. Sync the environment and run lint, type checks and tests with:

    make check

MIT licensed. See LICENSE.
