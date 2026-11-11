kvsim is a Python simulator for learning how vLLM schedules requests and manages a paged KV cache. It does not run a model or require a GPU, tokenizer or runtime dependencies.

Implemented: block allocation and reference counts, request tracking, per-request block tables, and FCFS batching under a shared prefill and decode token budget, with long prompts chunked across steps. When the cache runs out, the most recently admitted request is preempted and later recomputed, keeping the tokens it already generated and reusing any of its blocks still cached. With prefix caching on, the default, full blocks are hashed into a chain of prefix fingerprints and stay cached after their request ends, evicted least recently freed first; a new request reuses the cached blocks of its longest matching prefix and computes only the rest. Requests can be aborted in any state, and every block they held is released. The scheduler's caller supplies the sampled tokens. A workload is described by a JSON spec, from which the same request list is generated every time; the five benchmark scenarios live in benchmarks/workloads. The simulator replays a workload step by step and writes one JSON result: per request, the steps and estimated times of its first token and its end; per step, cache usage, batch size, queue length, preemptions and prefix cache hits. Its times are estimates from an explicit cost model whose default coefficients are placeholders, not measurements. The result file ends with a summary: time to first token, time per output token and end-to-end latency as mean and percentiles, throughput, cache usage, preemptions and the prefix cache hit rate.

Simulate a scenario with a cache of 2048 blocks of 64 tokens:

    uv run kvsim simulate benchmarks/workloads/shared_prefix.json --num-blocks 2048 --block-size 64 --out results/simulator/shared_prefix.json

Print the summaries of one or more results as a table, one row per file:

    uv run kvsim summarize results/simulator/shared_prefix.json

benchmarks/client.py measures a running vLLM or LMDeploy server under the same workloads and writes its measurements in the same result format, so the summary table reads both. It turns the token IDs of a workload into text prompts of the same length, sends the scenario once with other prompts to warm the engine up, then once more for the record, and reads the engine's metrics every second. The engine's version, launch command and cache configuration are arguments, since the client does not start the engine; the scripts that do are not written yet. With vLLM listening on port 8000:

    uv run python benchmarks/client.py benchmarks/workloads/short_low.json --engine vllm --base-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct --engine-version 0.32.0 --gpu "NVIDIA L4" --command "vllm serve Qwen/Qwen2.5-3B-Instruct" --num-blocks 2048 --block-size 64 --max-num-seqs 64 --max-num-batched-tokens 2048 --enable-prefix-caching --enable-chunked-prefill --out results/vllm/short_low.json

See ARCHITECTURE.md for the scheduling example, state ownership, contracts, the result format, the load client and vLLM references.

Development requires Python 3.12 and uv. Sync the environment and run lint, type checks and tests with:

    make check

MIT licensed. See LICENSE.
