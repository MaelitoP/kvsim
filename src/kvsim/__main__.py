import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from kvsim.metrics import LatencyStats, Summary, summarize
from kvsim.scheduler import SchedulerConfig
from kvsim.simulator import CostModel, simulate
from kvsim.workload import load_spec

_ESTIMATED_NOTE = (
    "estimated: simulated seconds from the cost model of the run; "
    "compare them only between runs made with the same cost model"
)


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
    simulate_parser.set_defaults(run=_simulate, parser=simulate_parser)
    summarize_parser = commands.add_parser(
        "summarize", help="print one table row of metrics per result file"
    )
    summarize_parser.add_argument("results", type=Path, nargs="+", help="result files, JSON")
    summarize_parser.set_defaults(run=_summarize, parser=summarize_parser)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        args.parser.error(str(error))


def _simulate(args: argparse.Namespace) -> None:
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
    text = json.dumps({**result, "summary": summarize(result)}, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")


def _summarize(args: argparse.Namespace) -> None:
    results = [_read_result(path) for path in args.results]
    rows = [_summary_row(path, *result) for path, result in zip(args.results, results, strict=True)]
    table = [list(rows[0]), *(list(row.values()) for row in rows)]
    widths = [max(len(cell) for cell in column) for column in zip(*table, strict=True)]
    for cells in table:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()
        )
    if not all(measured for _, measured, _ in results):
        print(_ESTIMATED_NOTE)


def _read_result(path: Path) -> tuple[str, bool, Summary]:
    try:
        result = json.loads(path.read_bytes())
        match result:
            case {"source": str() as source, "measured": bool() as measured}:
                return source, measured, summarize(result)
        raise ValueError("a result needs a source and a measured flag")
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _summary_row(path: Path, source: str, measured: bool, summary: Summary) -> dict[str, str]:
    return {
        "file": str(path),
        "source": source,
        "times": "measured" if measured else "estimated",
        "num_requests": str(summary["num_requests"]),
        "ttft_p50_ms": _ms(summary["ttft_s"], "p50"),
        "ttft_p95_ms": _ms(summary["ttft_s"], "p95"),
        "tpot_p50_ms": _ms(summary["tpot_s"], "p50"),
        "tpot_p95_ms": _ms(summary["tpot_s"], "p95"),
        "output_tokens_per_s": f"{summary['output_tokens_per_s']:.1f}",
        "requests_per_s": f"{summary['requests_per_s']:.2f}",
        "kv_usage_mean": _cell(summary["kv_usage_mean"], ".3f"),
        "kv_usage_max": _cell(summary["kv_usage_max"], ".3f"),
        "preemptions": _cell(summary["preemptions"]),
        "prefix_hit_rate": _cell(summary["prefix_hit_rate"], ".3f"),
    }


def _ms(stats: LatencyStats | None, percentile: Literal["p50", "p95"]) -> str:
    return "-" if stats is None else f"{stats[percentile] * 1000:.1f}"


def _cell(value: float | None, format_spec: str = "") -> str:
    return "-" if value is None else format(value, format_spec)


if __name__ == "__main__":
    main()
