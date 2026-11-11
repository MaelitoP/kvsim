# shellcheck shell=bash disable=SC2034,SC2154

ENGINE_VERSION=0.32.0
ENGINE_IMAGE=vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

engine_supports() {
    return 0
}

engine_command() {
    local args=(
        docker run --detach --name "$1" --gpus all --ipc host
        --publish "127.0.0.1:$PORT:8000"
        --volume "$HF_HOME:/root/.cache/huggingface"
        "$ENGINE_IMAGE"
        "$MODEL" --revision "$MODEL_REVISION" --dtype bfloat16 --seed 0
        --max-model-len "$MAX_MODEL_LEN"
        --block-size "$BLOCK_SIZE"
        # vLLM keeps one block of the cache as its null block.
        --kv-cache-memory-bytes "$(((num_blocks + 1) * BLOCK_SIZE * KV_BYTES_PER_TOKEN))"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-num-batched-tokens "$max_num_batched_tokens"
        "$(bool_flag enable-prefix-caching "$prefix_caching")"
        "$(bool_flag enable-chunked-prefill "$chunked_prefill")"
        --long-prefill-token-threshold "$long_prefill_token_threshold"
        --enable-per-request-metrics
    )
    if [[ $sim_aligned == true ]]; then
        args+=(--no-scheduler-reserve-full-isl --no-async-scheduling)
    fi
    printf '%s\n' "${args[@]}"
}

engine_num_blocks() {
    local tokens
    tokens=$(sed -n '/GPU KV cache size: /{s/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p;q;}')
    tokens=${tokens//,/}
    if [[ -z $tokens ]] || ((tokens % BLOCK_SIZE != 0)); then
        echo "vllm: no 'GPU KV cache size' of whole blocks in the engine log" >&2
        return 1
    fi
    echo $((tokens / BLOCK_SIZE - 1))
}
