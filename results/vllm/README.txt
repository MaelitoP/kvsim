Measurements of vllm 0.32.0 written by benchmarks/run_matrix.sh, last run on 2026-11-11 (UTC).

Every time in these files is measured: wall-clock seconds read by benchmarks/client.py. One file per run, named scenario.cell.run. The engine was restarted before every run, and its log is next to each result.

GPU: NVIDIA L4, driver 580.178.04, CUDA 13.0.
Image: vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34.
Model: Qwen/Qwen2.5-3B-Instruct at revision aa8e72537993ba99e69dfaafa59ed015b17504d1, bfloat16. Prompts were built with the tokenizer.json of that revision.
KV cache: blocks of 64 tokens, 36864 bytes per token; 2044 blocks in the large budget and 252 in the tight one, read back from the engine log before every run.
Engine metrics were scraped every second.
