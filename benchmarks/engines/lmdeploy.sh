# shellcheck shell=bash disable=SC2034,SC2154

ENGINE_VERSION=0.18.0
ENGINE_IMAGE=openmmlab/lmdeploy:v0.18.0-cu12.8@sha256:90bd451a49b56f68d953f0b49d6011b0419e043ececa0265c1e3cdb920748f4f

# TurboMind sizes its cache as a share of the GPU memory left once the weights are loaded,
# cut into 32 MiB pages; each page is a slab that holds 14 blocks of this model.
readonly PAGE_MIB=32
readonly PAGE_BLOCKS=14
readonly SLAB_GEOMETRY="slab obj=2\.25MiB slab=$PAGE_MIB\.00MiB cap=$PAGE_BLOCKS "

free_mib=

engine_supports() {
    case $1 in
        no_chunked_prefill | long_prefill_512 | tight_sim_aligned) return 1 ;;
    esac
}

engine_calibrate() {
    free_mib=$(sed -n '/Object cache budget: /{s/.* from free \([0-9.]*\) MB .*/\1/p;q;}')
    if [[ -z $free_mib ]]; then
        echo "lmdeploy: no 'Object cache budget' in the engine log" >&2
        return 1
    fi
}

engine_command() {
    local cache_ratio=0.5 log_level=INFO pages
    if [[ -n $free_mib ]]; then
        if ((num_blocks % PAGE_BLOCKS != 0)); then
            echo "lmdeploy: $num_blocks blocks is not a whole number of pages" >&2
            return 1
        fi
        pages=$((num_blocks / PAGE_BLOCKS))
        # Half a page more than the blocks need: free memory may move by half a page either way.
        if ! cache_ratio=$(LC_ALL=C awk -v free_mib="$free_mib" \
            -v budget_mib=$((pages * PAGE_MIB + PAGE_MIB / 2)) \
            'BEGIN { if (budget_mib >= free_mib) exit 1; printf "%.6f", budget_mib / free_mib }'); then
            echo "lmdeploy: $num_blocks blocks do not fit in $free_mib MiB of free memory" >&2
            return 1
        fi
        log_level=WARNING
    fi
    local args=(
        docker run --detach --name "$1" --gpus all --ipc host
        --publish "127.0.0.1:$PORT:8000"
        --volume "$HF_HOME:/root/.cache/huggingface"
        # Makes TurboMind log its cache geometry at startup; the interval is never reached.
        --env TM_CACHE_LOG_INTERVAL=1000000000
        "$ENGINE_IMAGE"
        lmdeploy serve api_server "$MODEL" --revision "$MODEL_REVISION" --model-name "$MODEL"
        --backend turbomind --server-port 8000 --dtype bfloat16 --tp 1 --quant-policy 0
        --session-len "$MAX_MODEL_LEN"
        --cache-block-seq-len "$BLOCK_SIZE"
        --cache-max-entry-count "$cache_ratio"
        --max-batch-size "$MAX_NUM_SEQS"
        --max-prefill-token-num "$max_num_batched_tokens"
        --log-level "$log_level"
    )
    if [[ $prefix_caching == true ]]; then
        args+=(--enable-prefix-caching)
    fi
    printf '%s\n' "${args[@]}"
}

engine_num_blocks() {
    local log pages
    log=$(cat)
    pages=$(sed -n "/\[cache\] region=/{s/.* pages [0-9]*\/\([0-9]*\) used .* page_size=$PAGE_MIB\.00MiB.*/\1/p;q;}" <<< "$log")
    if [[ -z $pages ]] || ! grep -q "$SLAB_GEOMETRY" <<< "$log"; then
        echo "lmdeploy: the engine log does not show the expected cache geometry" >&2
        return 1
    fi
    echo $((pages * PAGE_BLOCKS))
}
