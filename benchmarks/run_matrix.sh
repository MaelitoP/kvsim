#!/usr/bin/env bash
set -euo pipefail

readonly MODEL=Qwen/Qwen2.5-3B-Instruct
readonly MODEL_REVISION=aa8e72537993ba99e69dfaafa59ed015b17504d1
readonly KV_BYTES_PER_TOKEN=36864
readonly BLOCK_SIZE=64
readonly MAX_MODEL_LEN=4096
readonly MAX_NUM_SEQS=64
readonly LARGE_NUM_BLOCKS=2044
readonly TIGHT_NUM_BLOCKS=252
readonly RUNS=3
readonly PORT=${PORT:-8000}
readonly STARTUP_TIMEOUT_S=900

die() {
    echo "run_matrix: $*" >&2
    exit 1
}

bool_flag() {
    if [[ $2 == true ]]; then echo "--$1"; else echo "--no-$1"; fi
}

scenario_cells() {
    case $1 in
        short_low) echo baseline ;;
        long_high) echo baseline tight ;;
        shared_prefix) echo baseline no_prefix_caching ;;
        mixed) echo baseline no_chunked_prefill long_prefill_512 ;;
        saturation) echo baseline tight tight_no_prefix_caching tight_sim_aligned ;;
        *) die "unknown scenario $1" ;;
    esac
}

# shellcheck disable=SC2034
set_cell() {
    num_blocks=$LARGE_NUM_BLOCKS
    max_num_batched_tokens=2048
    prefix_caching=true
    chunked_prefill=true
    long_prefill_token_threshold=0
    sim_aligned=false
    case $1 in
        baseline) ;;
        no_prefix_caching) prefix_caching=false ;;
        no_chunked_prefill)
            chunked_prefill=false
            max_num_batched_tokens=$MAX_MODEL_LEN
            ;;
        long_prefill_512) long_prefill_token_threshold=512 ;;
        tight) num_blocks=$TIGHT_NUM_BLOCKS ;;
        tight_no_prefix_caching)
            num_blocks=$TIGHT_NUM_BLOCKS
            prefix_caching=false
            ;;
        tight_sim_aligned)
            num_blocks=$TIGHT_NUM_BLOCKS
            sim_aligned=true
            ;;
        *) die "unknown cell $1" ;;
    esac
}

start_engine() {
    local lines line deadline
    lines=$(engine_command "$container")
    engine_argv=()
    while IFS= read -r line; do
        engine_argv+=("$line")
    done <<< "$lines"
    docker rm --force "$container" > /dev/null 2>&1 || true
    "${engine_argv[@]}" > /dev/null
    deadline=$((SECONDS + STARTUP_TIMEOUT_S))
    until curl --silent --fail --max-time 10 --output /dev/null "http://127.0.0.1:$PORT/health"; do
        [[ $(docker inspect --format '{{.State.Running}}' "$container") == true ]] ||
            die "the engine exited during startup"
        ((SECONDS < deadline)) || die "no answer from /health after $STARTUP_TIMEOUT_S s"
        sleep 2
    done
}

container_log() {
    docker logs "$container" 2>&1
}

stop_engine() {
    local status=$?
    if ((status != 0)) && docker inspect "$container" > /dev/null 2>&1; then
        container_log > "$results_dir/failure.log" || true
        echo "run_matrix: engine log saved to $results_dir/failure.log" >&2
    fi
    docker rm --force "$container" > /dev/null 2>&1 || true
}

run_cell() {
    local scenario=$1 cell=$2 run=$3 out log reported command_line
    out=$results_dir/$scenario.$cell.run$run.json
    if [[ -e $out ]]; then
        echo "run_matrix: $out exists, skipped"
        return
    fi
    echo "run_matrix: $engine $scenario $cell run $run"
    set_cell "$cell"
    start_engine
    log=$(container_log)
    reported=$(engine_num_blocks <<< "$log")
    [[ $reported == "$num_blocks" ]] ||
        die "the engine holds $reported blocks, expected $num_blocks"
    printf -v command_line '%q ' "${engine_argv[@]}"
    uv run python benchmarks/client.py "benchmarks/workloads/$scenario.json" \
        --engine "$engine" --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
        --tokenizer "$tokenizer" --engine-version "$ENGINE_VERSION" --gpu "$gpu" \
        --command "${command_line% }" --num-blocks "$reported" --block-size "$BLOCK_SIZE" \
        --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$max_num_batched_tokens" \
        "$(bool_flag enable-prefix-caching "$prefix_caching")" \
        "$(bool_flag enable-chunked-prefill "$chunked_prefill")" \
        --long-prefill-token-threshold "$long_prefill_token_threshold" \
        --out "$out.tmp"
    container_log > "${out%.json}.log"
    mv "$out.tmp" "$out"
    docker rm --force "$container" > /dev/null
}

fetch_tokenizer() {
    tokenizer=${XDG_CACHE_HOME:-$HOME/.cache}/kvsim/tokenizer-$MODEL_REVISION.json
    [[ -e $tokenizer ]] && return
    curl --silent --show-error --fail --location --create-dirs --output "$tokenizer.tmp" \
        "https://huggingface.co/$MODEL/resolve/$MODEL_REVISION/tokenizer.json"
    mv "$tokenizer.tmp" "$tokenizer"
}

write_readme() {
    local driver cuda
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p)
    cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')
    cat > "$results_dir/README.txt" << EOF
Measurements of $engine $ENGINE_VERSION written by benchmarks/run_matrix.sh, last run on $(date -u +%F) (UTC).

Every time in these files is measured: wall-clock seconds read by benchmarks/client.py. One file per run, named scenario.cell.run. The engine was restarted before every run, and its log is next to each result.

GPU: $gpu, driver $driver, CUDA $cuda.
Image: $ENGINE_IMAGE.
Model: $MODEL at revision $MODEL_REVISION, bfloat16. Prompts were built with the tokenizer.json of that revision.
KV cache: blocks of $BLOCK_SIZE tokens, $KV_BYTES_PER_TOKEN bytes per token; $LARGE_NUM_BLOCKS blocks in the large budget and $TIGHT_NUM_BLOCKS in the tight one, read back from the engine log before every run.
Engine metrics were scraped every second.
EOF
}

main() {
    (($# >= 1)) || die "usage: run_matrix.sh <vllm|lmdeploy> [scenario ...]"
    engine=$1
    shift
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    [[ -e benchmarks/engines/$engine.sh ]] || die "unknown engine $engine"
    [[ ${HF_HOME:-} == /* ]] ||
        die "export HF_HOME, the absolute path of the Hugging Face cache mounted into the engine"
    # shellcheck source=/dev/null
    source "benchmarks/engines/$engine.sh"

    local scenarios=("$@") scenario cell run log
    ((${#scenarios[@]} > 0)) || scenarios=(short_low long_high shared_prefix mixed saturation)
    for scenario in "${scenarios[@]}"; do
        scenario_cells "$scenario" > /dev/null
    done

    container=kvsim-$engine
    results_dir=results/$engine
    mkdir -p "$results_dir"
    rm -f "$results_dir/failure.log"
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p)
    fetch_tokenizer
    write_readme
    trap stop_engine EXIT

    if declare -F engine_calibrate > /dev/null; then
        echo "run_matrix: calibration launch of $engine"
        set_cell baseline
        start_engine
        log=$(container_log)
        engine_calibrate <<< "$log"
    fi

    for scenario in "${scenarios[@]}"; do
        for cell in $(scenario_cells "$scenario"); do
            engine_supports "$cell" || continue
            for ((run = 1; run <= RUNS; run++)); do
                run_cell "$scenario" "$cell" "$run"
            done
        done
    done
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
