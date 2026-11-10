import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from kvsim.scheduler import SchedulerConfig
from kvsim.simulator import CostModel, simulate
from kvsim.workload import load_spec


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kvsim", description="Simulate an LLM scheduler and its paged KV cache."
    )
    commands = parser.add_subparsers(required=True)
    simulate_parser = commands.add_parser(
        "simulate", help="replay a workload and write the estimated result as JSON"
    )
    simulate_parser.add_argument("spec", type=Path, help="workload spec, a JSON file")
    simulate_parser.add_argument("--out", type=Path, required=True, help="result file to write")
    simulate_parser.add_argument("--num-blocks", type=int, required=True)
    simulate_parser.add_argument("--block-size", type=int, default=SchedulerConfig.block_size)
    simulate_parser.add_argument(
        "--max-num-seqs", type=int, default=SchedulerConfig.max_running_requests
    )
    simulate_parser.add_argument(
        "--max-num-batched-tokens", type=int, default=SchedulerConfig.max_tokens_per_step
    )
    simulate_parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=SchedulerConfig.enable_prefix_caching,
    )
    simulate_parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=SchedulerConfig.enable_chunked_prefill,
    )
    simulate_parser.add_argument(
        "--long-prefill-token-threshold",
        type=int,
        default=SchedulerConfig.long_prefill_token_threshold,
    )
    simulate_parser.add_argument(
        "--cost-step-overhead-s", type=float, default=CostModel.step_overhead_s
    )
    simulate_parser.add_argument(
        "--cost-prefill-s-per-token", type=float, default=CostModel.prefill_s_per_token
    )
    simulate_parser.add_argument(
        "--cost-decode-s-per-seq", type=float, default=CostModel.decode_s_per_seq
    )
    args = parser.parse_args(argv)
    try:
        result = simulate(
            load_spec(args.spec),
            SchedulerConfig(
                num_blocks=args.num_blocks,
                block_size=args.block_size,
                max_running_requests=args.max_num_seqs,
                max_tokens_per_step=args.max_num_batched_tokens,
                enable_chunked_prefill=args.enable_chunked_prefill,
                long_prefill_token_threshold=args.long_prefill_token_threshold,
                enable_prefix_caching=args.enable_prefix_caching,
            ),
            CostModel(
                step_overhead_s=args.cost_step_overhead_s,
                prefill_s_per_token=args.cost_prefill_s_per_token,
                decode_s_per_seq=args.cost_decode_s_per_seq,
            ),
        )
        text = json.dumps(result, indent=2, allow_nan=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        simulate_parser.error(str(error))


if __name__ == "__main__":
    main()
