I built kvsim to understand how vLLM schedules requests and manages its KV cache.

It simulates batching, chunked prefill, prefix caching, and preemption when the
cache runs out of blocks. It uses token IDs and block counts, with no model or
GPU. The simulator uses only the Python standard library.

To run a simulation, install Python 3.12 and uv, then run from the repo root:

    uv sync --locked
    uv run kvsim simulate benchmarks/workloads/shared_prefix.json \
      --num-blocks 2044 --block-size 64 \
      --out results/simulator/shared_prefix.json
    uv run kvsim summarize results/simulator/shared_prefix.json

The summary shows latency, throughput, cache usage, preemptions, and prefix hits.
Times come from a cost model. Its default coefficients are placeholders; use the
same coefficients when comparing simulations.

Try --no-enable-prefix-caching to disable prefix reuse, or reduce --num-blocks
to 144 to see what happens when the cache cannot keep the shared prefixes. Use a
different --out path to keep both results.

The five workloads in benchmarks/workloads/ cover short prompts, long prompts,
shared prefixes, mixed prompt lengths, and memory pressure. Run
uv run kvsim simulate --help for the other settings.

To read the code, start with Scheduler.schedule() in src/kvsim/scheduler.py.
It chooses the tokens for each step and asks KVCacheManager in
src/kvsim/kv_cache_manager.py for blocks. BlockPool in src/kvsim/block_pool.py
handles allocation, reference counts, and cache eviction. ARCHITECTURE.md
explains how these parts fit together and where the simulator differs from vLLM.

The repo includes 63 runs of vLLM and LMDeploy with Qwen2.5-3B-Instruct on an
NVIDIA L4. See docs/findings.md for comparisons with simulator replays.
benchmarks/README.md covers running the engines and regenerating results/tables.md
and the plots.

For development:

    make check

This runs dependency sync, lint, type checks, and tests.

MIT licensed. See LICENSE.
