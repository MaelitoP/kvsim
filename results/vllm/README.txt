Benchmark results: vllm 0.32.0.
Run metadata written by benchmarks/run_matrix.sh on 2026-11-11 (UTC).

Each <scenario>.<cell>.run<N>.json contains one measured run, with a matching .log file.
Each run starts a fresh engine and warms it up with different prompts.
benchmarks/client.py records times in seconds and scrapes metrics once per second.

GPU: NVIDIA L4, driver 580.178.04, CUDA 13.0.
Image: vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34.
Model: Qwen/Qwen2.5-3B-Instruct at revision aa8e72537993ba99e69dfaafa59ed015b17504d1, bfloat16.
Tokenizer: tokenizer.json from the same model revision.
KV cache: 64 tokens per block, 36864 bytes per token.
Usable blocks: 2044 (large), 252 (tight), checked against each engine log.
