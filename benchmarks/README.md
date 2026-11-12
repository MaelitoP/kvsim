# Benchmarks

These scripts run the workloads against vLLM and LMDeploy, then replay them in
kvsim. The committed results contain 36 vLLM runs and 27 LMDeploy runs from one
NVIDIA L4, using Qwen2.5-3B-Instruct.

Read [the findings](../docs/findings.md) for the results and
[the generated tables](../results/tables.md) for all values.

## Run the engines

You need a Linux host with one NVIDIA GPU, Docker with the NVIDIA container
toolkit, Python 3.12, and uv. The original host had a 24 GB L4.

From the repo root:

```sh
uv sync --locked
export HF_HOME="$HOME/hf"
benchmarks/run_matrix.sh vllm short_low
```

`HF_HOME` must be an absolute path to the Hugging Face cache mounted into the
container. The engine downloads the pinned model there if needed.

The runner skips existing result files. The repo already contains the full
matrix. To repeat a cell, move its existing `.json` and matching `.log` files out
of `results/<engine>/` first.

Omit the scenario to run the full matrix for an engine:

```sh
benchmarks/run_matrix.sh vllm
benchmarks/run_matrix.sh lmdeploy
```

The original matrix took 5 hours 46 minutes once the images and model were on the
host. Each run starts a fresh engine, checks its usable cache size from the log,
sends a warm-up workload with a different seed, then measures the requested
workload. Restarting prevents earlier runs from populating the prefix cache for
later ones.

Results go to `results/<engine>/<scenario>.<cell>.run<N>.json`, with the engine log
under the same name. A failed run leaves `failure.log` and stops the matrix. An
interrupted matrix resumes by skipping completed results.

The per-engine `README.txt` files record the GPU, driver, image digest, and model
revision. The runner generates them; edit `write_readme()` in
[run_matrix.sh](run_matrix.sh) if their wording needs to change.

## Workloads and settings

All committed workloads use seed 0 and closed-loop arrivals: a new request starts
when one finishes, up to the workload's concurrency.

| Workload | Prompt tokens | Output tokens | Concurrency | Requests |
| --- | --- | --- | --- | --- |
| `short_low` | 128 to 256 | 128 | 4 | 200 |
| `long_high` | 1536 to 2048 | 128 | 32 | 200 |
| `shared_prefix` | One of eight 1024-token prefixes, then 128 to 256 tokens | 128 | 16 | 256 |
| `mixed` | Alternating 64 to 128 and 1536 to 2048 | 128 | 16 | 200 |
| `saturation` | 1024 | 512 | 48 | 192 |

The engine scripts pin vLLM 0.32.0 and LMDeploy 0.18.0 by image digest and pin the
model revision. Both use bfloat16, 64-token blocks, a 4096-token sequence limit,
and a limit of 64 running requests.

A cell is one workload and configuration. Each cell has three runs.

| Cell | Change from baseline | Workload |
| --- | --- | --- |
| `baseline` | 2044 usable blocks, prefix caching and chunked prefill on | All |
| `no_prefix_caching` | Prefix caching off | `shared_prefix` |
| `tight` | 252 usable blocks | `long_high`, `saturation` |
| `tight_no_prefix_caching` | 252 blocks, prefix caching off | `saturation` |
| `no_chunked_prefill` | Chunking off, token budget raised from 2048 to 4096 | `mixed`, vLLM only |
| `long_prefill_512` | Each prefill chunk capped at 512 while requests compete | `mixed`, vLLM only |
| `tight_sim_aligned` | 252 blocks, full-input admission and async scheduling off | `saturation`, vLLM only |

The baseline token budget is 2048. vLLM shares it between prefill and decode;
TurboMind applies it to prefill, with decode tokens added on top.

Each token needs 36,864 bytes of KV cache for this model. vLLM's launch reserves
one extra block for its null block. TurboMind allocates this model's blocks in
groups of 14, so the common budgets are 2044 and 252 blocks. The LMDeploy script
calibrates its memory ratio from a startup log. The runner checks the resulting
block count before sending requests and stops if it differs from the budget.

## Measurements

[client.py](client.py) sends text prompts to the completions endpoint and reads
streamed output. It requests fixed output lengths with `ignore_eos` and rejects
shortened or failed requests.

The simulator and client use the same workload spec. The client decodes the token
IDs to text and adjusts the text to match the requested lengths. Tokenizing it
again can change the length or the shared prefix at its boundary. The result
records the prompt and output token counts reported by the engine.

TTFT is the time from sending a request to its first token event. TPOT is the time
from its first to last token event divided by `output_tokens - 1`; it is undefined
for a single output token. Throughput uses the span from the first request's
arrival to the last request's finish.

The client scrapes engine metrics once per second and subtracts the counters
recorded after warm-up. It takes a final scrape after all requests finish. An
unpublished metric stays `null`.

The usage gauges differ. vLLM counts allocated blocks; TurboMind includes cached
blocks that no request holds. LMDeploy also refreshes its gauges every 10 seconds
and publishes no preemption counter or per-request queue times. Its cache usage
therefore has a separate table.

## Analysis

```sh
uv run python benchmarks/analyze.py
uv run python benchmarks/analyze.py --check
```

The first command writes `results/tables.md` and the plots in `results/plots/`.
The second checks the table without writing files. A test runs the same comparison.
Edit table descriptions in `render_tables()` in [analyze.py](analyze.py), then
regenerate the output.

Measured table values are the median of each statistic across the three runs.
Simulated counters come from a replay. Estimated times use a cost model fitted
on all baseline runs for that engine; the other cells are outside the fit.

`replay_config()` maps recorded engine settings to the simulator. It enables
full-input admission for vLLM unless the launch command disabled it. For
LMDeploy, it approximates a prefill-only budget by adding the running-request
limit to the simulator's shared budget. This uses the maximum possible decode
count. The active count can be lower.

The fit uses arrival-to-first-token and first-token-to-finish intervals. Long
decode intervals dominate it, so the tables report separate fit scores for TTFT
and decode. Fitting and memory-pressure analysis support closed-loop workloads
only. The generator and simulator also support Poisson arrivals.
