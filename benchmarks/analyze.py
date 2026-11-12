import argparse
import functools
import json
import statistics
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib
import numpy as np

from kvsim.block_hash import BlockHash
from kvsim.metrics import Summary, summarize
from kvsim.request import Request
from kvsim.scheduler import Scheduler, SchedulerConfig
from kvsim.simulator import CostModel, RequestRecord, Result, simulate
from kvsim.workload import ClosedLoop, WorkloadSpec, generate_requests, parse_spec

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

type Metric = Callable[[Summary], float | None]
type Event = Literal["arrival_s", "first_token_s", "finish_s"]

SCENARIOS = ("short_low", "long_high", "shared_prefix", "mixed", "saturation")
CELLS = (
    "baseline",
    "no_prefix_caching",
    "no_chunked_prefill",
    "long_prefill_512",
    "tight",
    "tight_no_prefix_caching",
    "tight_sim_aligned",
)
ENGINES = ("vllm", "lmdeploy")
FITTED_CELL = "baseline"
ADMISSION_SWEEP_CELLS = ("tight", "tight_no_prefix_caching")
SWEPT_SCENARIO = "shared_prefix"
SWEPT_NUM_BLOCKS = (252, 192, 160, 144, 128)

_STEPS = CostModel(1, 0, 0)
_STEPS_AND_PREFILL_TOKENS = CostModel(1, 1, 0)
_STEPS_AND_DECODE_SEQS = CostModel(1, 0, 1)
_INTERVALS: tuple[tuple[Event, Event], ...] = (
    ("arrival_s", "first_token_s"),
    ("first_token_s", "finish_s"),
)

_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e6e5e0"
_ENGINE_COLORS = {"vllm": "#2a78d6", "lmdeploy": "#eb6834"}
_SIMULATOR_COLOR = "#1baf7a"
_ENGINE_LABELS = {"vllm": "vLLM", "lmdeploy": "LMDeploy"}


@dataclass(frozen=True)
class Cell:
    """The runs of one scenario, cell and engine, with the summary of each."""

    scenario: str
    name: str
    engine: str
    runs: tuple[Result, ...]
    summaries: tuple[Summary, ...]

    def median(self, metric: Metric) -> float | None:
        """`None` unless every run has the metric."""
        return _median(metric(summary) for summary in self.summaries)


@dataclass(frozen=True)
class Fit:
    cost_model: CostModel
    scenarios: tuple[str, ...]
    ttft_r_squared: float
    decode_r_squared: float


class TtftSplit(NamedTuple):
    """Medians of the two times an engine reports, and percentiles of the rest of the TTFT."""

    queue_p50_ms: float
    scheduling_to_first_token_p50_ms: float
    outside_p10_ms: float
    outside_p50_ms: float
    outside_p90_ms: float


@dataclass(frozen=True, kw_only=True)
class Pressure:
    """Counters of one replay that its result does not hold.

    `discarded_tokens` are the tokens in the victims' block tables when they were
    preempted, computed or taken from the cache; `recovered_tokens` those that resumed
    requests took from the cache. Their difference is what the replay computes twice.
    """

    preemptions: int
    scheduled_tokens: int
    discarded_tokens: int
    recovered_tokens: int
    max_orphaned_blocks: int


def load_cells(results_dir: Path) -> list[Cell]:
    """The cells of `<engine>/<scenario>.<cell>.run<k>.json` files, in matrix order.

    Raises `ValueError` naming the file for a name outside the matrix, a result whose
    `source` is not its directory, and a run whose workload or configuration differs from
    the other runs of its cell.
    """
    runs_by_cell: dict[tuple[int, int, int], list[Result]] = {}
    for path in sorted(results_dir.glob("*/*.run*.json")):
        try:
            scenario, name, _run, _suffix = path.name.split(".")
            key = (
                SCENARIOS.index(scenario),
                CELLS.index(name),
                ENGINES.index(path.parent.name),
            )
        except ValueError as error:
            raise ValueError(f"{path}: not a scenario, cell and engine of the matrix") from error
        run: Result = json.loads(path.read_bytes())
        runs = runs_by_cell.setdefault(key, [])
        if run["source"] != path.parent.name:
            raise ValueError(f"{path}: its source is {run['source']!r}")
        if runs and (run["workload"], run["config"]) != (runs[0]["workload"], runs[0]["config"]):
            raise ValueError(f"{path}: differs from the other runs of its cell")
        runs.append(run)
    return [
        Cell(
            SCENARIOS[scenario],
            CELLS[name],
            ENGINES[engine],
            tuple(runs),
            tuple(summarize(run) for run in runs),
        )
        for (scenario, name, engine), runs in sorted(runs_by_cell.items())
    ]


def replay_config(measurement: Result) -> SchedulerConfig:
    """The simulator configuration that stands for the engine of a measured result.

    LMDeploy's token limit bounds prefill only, so its decode tokens are added to the
    budget. vLLM checks the whole input at admission unless launched without it; what
    LMDeploy does is not observable, and its replay keeps the simulator's default.
    """
    recorded = measurement["config"]
    is_vllm = measurement["source"] == "vllm"
    config = SchedulerConfig(
        num_blocks=recorded["num_blocks"],
        block_size=recorded["block_size"],
        max_running_requests=recorded["max_num_seqs"],
        max_tokens_per_step=recorded["max_num_batched_tokens"],
        enable_chunked_prefill=recorded["enable_chunked_prefill"],
        long_prefill_token_threshold=recorded["long_prefill_token_threshold"],
        enable_prefix_caching=recorded["enable_prefix_caching"],
        enable_full_input_admission=(
            is_vllm and "--no-scheduler-reserve-full-isl" not in (recorded["command"] or "")
        ),
    )
    return config if is_vllm else _with_prefill_only_budget(config)


def _with_prefill_only_budget(config: SchedulerConfig) -> SchedulerConfig:
    return replace(
        config, max_tokens_per_step=config.max_tokens_per_step + config.max_running_requests
    )


_simulate: Callable[[WorkloadSpec, SchedulerConfig, CostModel], Result] = functools.cache(simulate)


def replay(measurement: Result, cost_model: CostModel) -> Result:
    """The simulation of a measured result's workload; shared between calls, not to be changed."""
    return _simulate(parse_spec(measurement["workload"]), replay_config(measurement), cost_model)


def fit_cost_model(measurements: Iterable[Result]) -> tuple[CostModel, float, float]:
    """The least-squares cost model, and its R² on the TTFT and on the decode intervals.

    A simulated time is linear in the three coefficients, and in a closed loop the steps do
    not depend on them, so three replays with unit cost models give each interval's number
    of steps, prefill tokens and decode sequences. Raises `ValueError` for another arrival
    model, for a result that does not hold its replay's requests, and when the results do
    not determine a valid cost model.
    """
    features: list[tuple[float, float, float]] = []
    durations: list[float] = []
    for measurement in measurements:
        if not isinstance(parse_spec(measurement["workload"]).arrival, ClosedLoop):
            raise ValueError("the cost model is fitted on closed-loop workloads only")
        requests = measurement["requests"]
        steps = replay(measurement, _STEPS)["requests"]
        prefill = replay(measurement, _STEPS_AND_PREFILL_TOKENS)["requests"]
        decode = replay(measurement, _STEPS_AND_DECODE_SEQS)["requests"]
        if [request["request_id"] for request in requests] != [
            request["request_id"] for request in steps
        ]:
            raise ValueError("a result does not hold the requests of its replay, in order")
        for measured, in_steps, with_prefill, with_decode in zip(
            requests, steps, prefill, decode, strict=True
        ):
            for start, end in _INTERVALS:
                num_steps = in_steps[end] - in_steps[start]
                features.append(
                    (
                        num_steps,
                        with_prefill[end] - with_prefill[start] - num_steps,
                        with_decode[end] - with_decode[start] - num_steps,
                    )
                )
                durations.append(measured[end] - measured[start])
    design = np.array(features).reshape(-1, 3)
    target = np.array(durations)
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("the results do not separate the three coefficients of a cost model")
    try:
        cost_model = CostModel(*(float(coefficient) for coefficient in coefficients))
    except ValueError as error:
        raise ValueError(f"the fit gives no valid cost model: {coefficients}") from error
    residual = target - design @ coefficients
    ttft_r_squared, decode_r_squared = (
        1 - float((residual[rows] ** 2).sum() / ((target[rows] - target[rows].mean()) ** 2).sum())
        for rows in (slice(0, None, 2), slice(1, None, 2))
    )
    return cost_model, ttft_r_squared, decode_r_squared


def fit_engines(cells: Sequence[Cell]) -> dict[str, Fit]:
    """One fit per engine that has a fitted cell, over every run of those cells."""
    fits: dict[str, Fit] = {}
    for engine in ENGINES:
        fitted = [cell for cell in cells if cell.engine == engine and cell.name == FITTED_CELL]
        if not fitted:
            continue
        try:
            cost_model, ttft_r_squared, decode_r_squared = fit_cost_model(
                run for cell in fitted for run in cell.runs
            )
        except ValueError as error:
            raise ValueError(f"{engine}: {error}") from error
        fits[engine] = Fit(
            cost_model,
            tuple(cell.scenario for cell in fitted),
            ttft_r_squared,
            decode_r_squared,
        )
    return fits


def ttft_split_ms(requests: Iterable[RequestRecord]) -> TtftSplit | None:
    """How the client's TTFT divides, over the requests whose engine reported its own times.

    The rest of the TTFT is what the engine's queue time and its time from scheduling to
    the first token leave. `None` when fewer than two requests carry the engine's times.
    """
    reported = [
        (
            request["server_queue_ms"],
            request["server_ttft_ms"],
            (request["first_token_s"] - request["arrival_s"]) * 1000,
        )
        for request in requests
        if request["server_queue_ms"] is not None and request["server_ttft_ms"] is not None
    ]
    if len(reported) < 2:
        return None
    deciles = statistics.quantiles(
        [ttft_ms - queue_ms - engine_ms for queue_ms, engine_ms, ttft_ms in reported],
        n=10,
        method="inclusive",
    )
    return TtftSplit(
        queue_p50_ms=statistics.median(queue_ms for queue_ms, _, _ in reported),
        scheduling_to_first_token_p50_ms=statistics.median(
            engine_ms for _, engine_ms, _ in reported
        ),
        outside_p10_ms=deciles[0],
        outside_p50_ms=deciles[4],
        outside_p90_ms=deciles[8],
    )


def simulate_pressure(spec: WorkloadSpec, config: SchedulerConfig) -> Pressure:
    """What preemptions cost in a closed-loop replay, and the cached blocks no lookup reaches.

    Drives a scheduler as `simulate` does. Each request repeats one output token of its
    own, outside the prompt vocabulary: a schedule depends on which sequences share a
    prefix, not on the token values. A cached block is orphaned when an earlier block of
    its chain is not cached, since a lookup stops at the first miss; it is counted
    whether it is free or a request still holds it. Raises `ValueError` for another
    arrival model.
    """
    if not isinstance(spec.arrival, ClosedLoop):
        raise ValueError("memory pressure is simulated on closed-loop workloads only")
    scheduler = Scheduler(config)
    block_pool = scheduler.kv_cache_manager.block_pool
    first_output_token_id = max(spec.vocab_size, config.eos_token_id + 1)
    pending = deque(generate_requests(spec))
    in_flight: dict[str, Request] = {}
    output_token_ids: dict[str, int] = {}
    tracked: list[Request] = []
    scheduled_tokens = discarded_tokens = recovered_tokens = max_orphaned_blocks = 0
    while pending or in_flight:
        while pending and len(in_flight) < spec.arrival.concurrency:
            request_spec = pending.popleft()
            request = Request(
                request_id=request_spec.request_id,
                prompt_token_ids=request_spec.prompt_token_ids,
                max_output_tokens=request_spec.output_len,
                arrival_step=scheduler.num_steps,
            )
            scheduler.add_request(request)
            in_flight[request.request_id] = request
            output_token_ids[request.request_id] = first_output_token_id + len(output_token_ids)
            tracked.append(request)
        num_held_tokens = {
            request_id: request.num_computed_tokens for request_id, request in in_flight.items()
        }
        batch = scheduler.schedule()
        scheduled_tokens += batch.total_tokens
        discarded_tokens += sum(num_held_tokens[request_id] for request_id in batch.preempted)
        recovered_tokens += sum(
            in_flight[request_id].num_computed_tokens for request_id in batch.scheduled_resumed
        )
        orphaned: set[BlockHash] = set()
        still_tracked: list[Request] = []
        for request in tracked:
            cached = [
                block_pool.get_cached_block(block_hash) is not None
                for block_hash in request.block_hashes
            ]
            if not request.is_finished or any(cached):
                still_tracked.append(request)
            reachable = True
            for block_hash, is_cached in zip(request.block_hashes, cached, strict=True):
                reachable = reachable and is_cached
                if is_cached and not reachable:
                    orphaned.add(block_hash)
        tracked = still_tracked
        max_orphaned_blocks = max(max_orphaned_blocks, len(orphaned))
        sampled_token_ids = {
            request_id: output_token_ids[request_id] for request_id in batch.num_scheduled_tokens
        }
        for request in scheduler.complete_step(batch, sampled_token_ids):
            del in_flight[request.request_id]
    return Pressure(
        preemptions=scheduler.num_preemptions,
        scheduled_tokens=scheduled_tokens,
        discarded_tokens=discarded_tokens,
        recovered_tokens=recovered_tokens,
        max_orphaned_blocks=max_orphaned_blocks,
    )


def render_tables(cells: Sequence[Cell], fits: Mapping[str, Fit]) -> str:
    budget_rows: list[list[str]] = []
    pressure_rows: list[list[str]] = []
    if "vllm" in fits:
        cost_model = fits["vllm"].cost_model
        budget_rows = [
            _budget_row(cell, cells, cost_model)
            for cell in cells
            if (cell.engine, cell.name) == ("vllm", FITTED_CELL)
        ]
        pressure_rows = [
            _pressure_row(scenario, spec, config, cost_model)
            for scenario, spec, config in _pressure_replays(cells)
        ]
    sections = [
        "# Benchmark tables\n\n"
        "Generated by `benchmarks/analyze.py` from the result files of this directory; do not "
        "edit. **Measured** values come from the engines and the load client: the median over "
        "the runs of a cell. **Simulated** counters and **estimated** times come from a kvsim "
        "replay of the same workload with the engine's cache size and settings; the replay is "
        "regenerated, not stored. A dash is a value that is not published, not defined, such "
        "as a hit rate when nothing was looked up, or missing from one of the runs.",
        "## Engines, measured\n\n"
        "Spread is the range of the TTFT median over the runs, relative to its median.\n\n"
        + _table(
            [
                "Scenario",
                "Cell",
                "Engine",
                "TTFT p50 ms",
                "TTFT p95 ms",
                "TPOT p50 ms",
                "TPOT p95 ms",
                "Output tokens/s",
                "Prefix hit rate",
                "Spread",
            ],
            [_engine_row(cell) for cell in cells],
        ),
        "## vLLM against kvsim, scheduler counters\n\n"
        "The simulated counters do not depend on the cost model. vLLM's usage is read from a "
        "gauge once per second, between steps; the simulator's is the highest usage a step "
        "runs with.\n\n"
        + _table(
            [
                "Scenario",
                "Cell",
                "Preemptions, measured",
                "Preemptions, simulated",
                "Max KV usage, measured",
                "Max KV usage, simulated",
                "Hit rate, measured",
                "Hit rate, simulated",
            ],
            [_counter_row(cell) for cell in cells if cell.engine == "vllm"],
        ),
        "## Estimated against measured times\n\n"
        "Estimates use the cost model fitted for the engine, below. An error is the estimate's "
        f"distance from the measurement. The `{FITTED_CELL}` cells were used for the fit; the "
        "others were not.\n\n"
        + _table(
            [
                "Scenario",
                "Cell",
                "Engine",
                "Fitted on",
                "TTFT p50 ms, measured",
                "TTFT p50 ms, estimated",
                "TTFT error",
                "TPOT p50 ms, measured",
                "TPOT p50 ms, estimated",
                "TPOT error",
                "Output tokens/s, measured",
                "Output tokens/s, estimated",
                "Throughput error",
            ],
            [
                _estimate_row(cell, fits[cell.engine].cost_model)
                for cell in cells
                if cell.engine in fits
            ],
        ),
        "## Time to first token by arrival\n\n"
        "The first requests of a closed loop, as many as its concurrency, arrive together "
        "and queue behind each other; each later request arrives when one finishes. The TTFT "
        "columns are medians over the requests of each kind, measured or estimated. The last "
        "five columns are measured on the later requests of vLLM, which reports for each "
        "request its time in the queue and its time from scheduling to the first token: "
        "their medians, then the 10th, 50th and 90th percentiles of the rest of the client's "
        "TTFT, which passes before the scheduler has the request or after the token is "
        "sampled.\n\n"
        + _table(
            [
                "Scenario",
                "Cell",
                "Engine",
                "First requests, TTFT p50 ms, measured",
                "First requests, estimated",
                "Later requests, TTFT p50 ms, measured",
                "Later requests, estimated",
                "Queue p50 ms",
                "Scheduling to first token p50 ms",
                "Outside the scheduler p10 ms",
                "Outside p50 ms",
                "Outside p90 ms",
            ],
            [
                _arrival_row(cell, fits[cell.engine].cost_model)
                for cell in cells
                if cell.engine in fits
            ],
        ),
        "## Time to first token by prompt length\n\n"
        "A scenario with several ranges of prompt lengths gives them to its requests in "
        "turn. The median TTFT over the requests of each range, measured, and estimated with "
        "the engine's fitted cost model.\n\n"
        + _table(
            [
                "Scenario",
                "Cell",
                "Engine",
                "Prompt tokens",
                "TTFT p50 ms, measured",
                "TTFT p50 ms, estimated",
            ],
            [
                row
                for cell in cells
                if cell.engine in fits
                for row in _prompt_length_rows(cell, fits[cell.engine].cost_model)
            ],
        ),
        "## Token budget rule, estimated\n\n"
        f"Each vLLM `{FITTED_CELL}` cell replayed twice with vLLM's fitted cost model: with "
        "its own rule, one token budget that decode tokens are taken from first, and with "
        "TurboMind's, where the budget bounds prefill only and decode tokens come on top, "
        "replayed as the budget plus the limit on running requests. The measured columns are "
        "the two engines.\n\n"
        + _table(
            [
                "Scenario",
                "TTFT p50 ms, estimated, shared budget",
                "TTFT p50 ms, estimated, prefill-only budget",
                "TTFT p50 ms, measured, vllm",
                "TTFT p50 ms, measured, lmdeploy",
            ],
            budget_rows,
        ),
        "## Memory pressure, simulated\n\n"
        "Replays of the vLLM cells with the small cache under both admission rules, vLLM's "
        "default, which admits a request when its whole input fits, and the first chunk "
        f"only; then `{SWEPT_SCENARIO}` with vLLM's `{FITTED_CELL}` settings and smaller "
        "caches, which no engine was measured with on that scenario. Tokens discarded are "
        "those in the victims' block tables when they were preempted, computed or taken from "
        "the cache; the next column counts the tokens resumed requests took from the cache, "
        "and the difference is what the replay computes twice. A cached block is orphaned "
        "when an earlier block of its chain is not cached, so that no lookup can reach it; "
        "the column is the highest count at any step, free and allocated blocks alike. The "
        "hit rate counts first admissions only. The counters do not depend on the cost "
        "model; the last two columns are estimated with vLLM's.\n\n"
        + _table(
            [
                "Scenario",
                "Blocks",
                "Prefix caching",
                "Admission",
                "Preemptions",
                "Tokens computed",
                "Tokens discarded",
                "Found cached on resumption",
                "Orphaned blocks",
                "Hit rate",
                "TTFT p50 ms, estimated",
                "Output tokens/s, estimated",
            ],
            pressure_rows,
        ),
        "## LMDeploy cache usage, measured\n\n"
        "LMDeploy's gauge counts the memory of live cache objects, blocks kept by its prefix "
        "cache included, so it is not comparable with vLLM's or the simulator's usage. It "
        "publishes no preemption counter.\n\n"
        + _table(
            ["Scenario", "Cell", "Max cache usage"],
            [
                [
                    cell.scenario,
                    cell.name,
                    _number(cell.median(lambda summary: summary["kv_usage_max"]), 3),
                ]
                for cell in cells
                if cell.engine == "lmdeploy"
            ],
        ),
        "## Cost model, fitted\n\n"
        "Least squares on two intervals of every request, arrival to first token and first "
        f"token to finish, over all runs of the `{FITTED_CELL}` cell of the scenarios listed. "
        "The decode intervals are long and dominate the fit, which the two R² show. These are "
        "estimates of one model on one GPU, not constants of the simulator.\n\n"
        + _table(
            [
                "Engine",
                "Step overhead ms",
                "Prefill ms per token",
                "Decode ms per sequence",
                "R², TTFT intervals",
                "R², decode intervals",
                "Fitted on",
            ],
            [
                [
                    engine,
                    _number(fit.cost_model.step_overhead_s * 1000, 2),
                    _number(fit.cost_model.prefill_s_per_token * 1000, 4),
                    _number(fit.cost_model.decode_s_per_seq * 1000, 3),
                    _number(fit.ttft_r_squared, 2),
                    _number(fit.decode_r_squared, 4),
                    ", ".join(fit.scenarios),
                ]
                for engine, fit in fits.items()
            ],
        ),
    ]
    return "\n\n".join(sections) + "\n"


def _engine_row(cell: Cell) -> list[str]:
    ttft_p50s = [summary["ttft_s"]["p50"] for summary in cell.summaries]
    spread = (max(ttft_p50s) - min(ttft_p50s)) / statistics.median(ttft_p50s)
    return [
        cell.scenario,
        cell.name,
        cell.engine,
        _ms(cell.median(lambda summary: summary["ttft_s"]["p50"])),
        _ms(cell.median(lambda summary: summary["ttft_s"]["p95"])),
        _ms(cell.median(lambda summary: _tpot(summary, "p50"))),
        _ms(cell.median(lambda summary: _tpot(summary, "p95"))),
        _number(cell.median(lambda summary: summary["output_tokens_per_s"]), 1),
        _number(cell.median(lambda summary: summary["prefix_hit_rate"]), 3),
        _percent(spread) if len(ttft_p50s) > 1 else "-",
    ]


def _counter_row(cell: Cell) -> list[str]:
    simulated = summarize(replay(cell.runs[0], _STEPS))
    return [
        cell.scenario,
        cell.name,
        _number(cell.median(lambda summary: summary["preemptions"]), 0),
        _number(simulated["preemptions"], 0),
        _number(cell.median(lambda summary: summary["kv_usage_max"]), 3),
        _number(simulated["kv_usage_max"], 3),
        _number(cell.median(lambda summary: summary["prefix_hit_rate"]), 3),
        _number(simulated["prefix_hit_rate"], 3),
    ]


def _estimate_row(cell: Cell, cost_model: CostModel) -> list[str]:
    estimated = summarize(replay(cell.runs[0], cost_model))
    row = [cell.scenario, cell.name, cell.engine, "yes" if cell.name == FITTED_CELL else "no"]
    metrics: tuple[tuple[Metric, int], ...] = (
        (lambda summary: summary["ttft_s"]["p50"], 1000),
        (lambda summary: _tpot(summary, "p50"), 1000),
        (lambda summary: summary["output_tokens_per_s"], 1),
    )
    for metric, scale in metrics:
        measured_value = cell.median(metric)
        estimated_value = metric(estimated)
        if measured_value is None or estimated_value is None:
            row += ["-", "-", "-"]
        else:
            row += [
                _number(measured_value * scale, 1),
                _number(estimated_value * scale, 1),
                _percent(estimated_value / measured_value - 1, signed=True),
            ]
    return row


def _arrival_row(cell: Cell, cost_model: CostModel) -> list[str]:
    arrival = parse_spec(cell.runs[0]["workload"]).arrival
    if not isinstance(arrival, ClosedLoop):
        raise ValueError("requests are told apart by arrival on closed-loop workloads only")
    first, later = slice(None, arrival.concurrency), slice(arrival.concurrency, None)
    estimated = replay(cell.runs[0], cost_model)["requests"]
    row = [cell.scenario, cell.name, cell.engine]
    for kind in (first, later):
        row += [
            _ms(_median(_ttft_p50(run["requests"][kind]) for run in cell.runs)),
            _ms(_ttft_p50(estimated[kind])),
        ]
    splits = [ttft_split_ms(run["requests"][later]) for run in cell.runs]
    row += [
        _number(_median(None if split is None else split[index] for split in splits), 1)
        for index in range(len(TtftSplit._fields))
    ]
    return row


def _prompt_length_rows(cell: Cell, cost_model: CostModel) -> list[list[str]]:
    ranges = parse_spec(cell.runs[0]["workload"]).prompt_len
    if len(ranges) == 1:
        return []
    estimated = replay(cell.runs[0], cost_model)["requests"]
    return [
        [
            cell.scenario,
            cell.name,
            cell.engine,
            f"{low} to {high}",
            _ms(_median(_ttft_p50(run["requests"][index :: len(ranges)]) for run in cell.runs)),
            _ms(_ttft_p50(estimated[index :: len(ranges)])),
        ]
        for index, (low, high) in enumerate(ranges)
    ]


def _ttft_p50(requests: Sequence[RequestRecord]) -> float | None:
    if not requests:
        return None
    return statistics.median(
        request["first_token_s"] - request["arrival_s"] for request in requests
    )


def _budget_row(cell: Cell, cells: Sequence[Cell], cost_model: CostModel) -> list[str]:
    measurement = cell.runs[0]
    prefill_only = _simulate(
        parse_spec(measurement["workload"]),
        _with_prefill_only_budget(replay_config(measurement)),
        cost_model,
    )
    lmdeploy = next(
        (
            other
            for other in cells
            if (other.scenario, other.name, other.engine) == (cell.scenario, cell.name, "lmdeploy")
        ),
        None,
    )
    return [
        cell.scenario,
        _ms(summarize(replay(measurement, cost_model))["ttft_s"]["p50"]),
        _ms(summarize(prefill_only)["ttft_s"]["p50"]),
        _ms(cell.median(_summary_ttft_p50)),
        _ms(None if lmdeploy is None else lmdeploy.median(_summary_ttft_p50)),
    ]


def _summary_ttft_p50(summary: Summary) -> float:
    return summary["ttft_s"]["p50"]


def _pressure_replays(
    cells: Sequence[Cell],
) -> Iterator[tuple[str, WorkloadSpec, SchedulerConfig]]:
    for cell in cells:
        if cell.engine != "vllm":
            continue
        spec = parse_spec(cell.runs[0]["workload"])
        config = replay_config(cell.runs[0])
        if cell.name in ADMISSION_SWEEP_CELLS:
            for whole_input in (True, False):
                yield cell.scenario, spec, replace(config, enable_full_input_admission=whole_input)
        elif (cell.scenario, cell.name) == (SWEPT_SCENARIO, FITTED_CELL):
            for num_blocks in SWEPT_NUM_BLOCKS:
                yield cell.scenario, spec, replace(config, num_blocks=num_blocks)


def _pressure_row(
    scenario: str, spec: WorkloadSpec, config: SchedulerConfig, cost_model: CostModel
) -> list[str]:
    pressure = simulate_pressure(spec, config)
    estimated = summarize(_simulate(spec, config, cost_model))
    return [
        scenario,
        _number(config.num_blocks, 0),
        "on" if config.enable_prefix_caching else "off",
        "whole input" if config.enable_full_input_admission else "first chunk",
        _number(pressure.preemptions, 0),
        _number(pressure.scheduled_tokens, 0),
        _number(pressure.discarded_tokens, 0),
        _number(pressure.recovered_tokens, 0),
        _number(pressure.max_orphaned_blocks, 0),
        _number(estimated["prefix_hit_rate"], 3),
        _ms(estimated["ttft_s"]["p50"]),
        _number(estimated["output_tokens_per_s"], 1),
    ]


def _median(values: Iterable[float | None]) -> float | None:
    """`None` unless every value is present."""
    every = list(values)
    present = [value for value in every if value is not None]
    return statistics.median(present) if len(present) == len(every) else None


def _tpot(summary: Summary, percentile: Literal["p50", "p95"]) -> float | None:
    stats = summary["tpot_s"]
    return None if stats is None else stats[percentile]


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def _number(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{_rounded(value, digits):.{digits}f}"


def _ms(seconds: float | None) -> str:
    return _number(None if seconds is None else seconds * 1000, 1)


def _percent(ratio: float, *, signed: bool = False) -> str:
    return f"{_rounded(ratio * 100, 1):{'+' if signed else ''}.1f}%"


def _rounded(value: float, digits: int) -> float:
    # Adding 0.0 turns a negative zero into zero, which would print with a minus sign.
    return round(value, digits) + 0.0


def write_plots(cells: Sequence[Cell], fits: Mapping[str, Fit], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(
        {
            "figure.facecolor": _SURFACE,
            "axes.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
            "text.color": _TEXT,
            "axes.labelcolor": _MUTED,
            "axes.edgecolor": _GRID,
            "xtick.color": _MUTED,
            "ytick.color": _MUTED,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "font.size": 9,
        }
    ):
        _plot_bars(
            cells,
            lambda summary: summary["ttft_s"]["p50"],
            "Time to first token, median, seconds (measured)",
            plots_dir / "ttft.png",
        )
        _plot_bars(
            cells,
            lambda summary: summary["output_tokens_per_s"],
            "Output tokens per second (measured)",
            plots_dir / "throughput.png",
        )
        if "vllm" in fits:
            _plot_kv_usage(cells, fits["vllm"].cost_model, plots_dir / "kv_usage_saturation.png")


def _plot_bars(cells: Sequence[Cell], metric: Metric, title: str, path: Path) -> None:
    scenarios = list(dict.fromkeys(cell.scenario for cell in cells))
    heights = [
        len({cell.name for cell in cells if cell.scenario == scenario}) + 0.6
        for scenario in scenarios
    ]
    figure, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(8, 0.63 * sum(heights) + 0.6),
        squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )
    for axis, scenario in zip(axes[:, 0], scenarios, strict=True):
        names = list(dict.fromkeys(cell.name for cell in cells if cell.scenario == scenario))
        height = 0.36
        longest = 0.0
        for cell in cells:
            measured = cell.median(metric)
            if cell.scenario != scenario or measured is None:
                continue
            engines = [
                other.engine
                for other in cells
                if (other.scenario, other.name) == (cell.scenario, cell.name)
            ]
            offset = engines.index(cell.engine) - (len(engines) - 1) / 2
            position = names.index(cell.name) + offset * (height + 0.04)
            axis.barh(position, measured, height=height, color=_ENGINE_COLORS[cell.engine])
            label = f" {measured:,.0f}" if measured >= 100 else f" {measured:.3g}"
            axis.text(measured, position, label, va="center", color=_TEXT, fontsize=8)
            longest = max(longest, measured)
        axis.set_xlim(0, longest * 1.14)
        axis.set_yticks(range(len(names)), names)
        axis.invert_yaxis()
        axis.set_title(scenario)
        axis.grid(axis="x", color=_GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
    figure.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    figure.legend(
        handles=[
            Patch(color=_ENGINE_COLORS[engine], label=_ENGINE_LABELS[engine]) for engine in ENGINES
        ],
        loc="upper right",
        ncols=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_kv_usage(cells: Sequence[Cell], cost_model: CostModel, path: Path) -> None:
    vllm_cells = {
        cell.name: cell for cell in cells if cell.scenario == "saturation" and cell.engine == "vllm"
    }
    names = [name for name in ("baseline", "tight") if name in vllm_cells]
    if not names:
        return
    figure, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), squeeze=False)
    for axis, name in zip(axes[:, 0], names, strict=True):
        measured = vllm_cells[name].runs[0]
        simulated = replay(measured, cost_model)
        axis.plot(
            [sample["t_s"] for sample in simulated["samples"]],
            [sample["kv_usage"] for sample in simulated["samples"]],
            color=_SIMULATOR_COLOR,
            linewidth=1.2,
            label="kvsim, simulated usage over estimated time",
        )
        axis.plot(
            [sample["t_s"] for sample in measured["samples"]],
            [sample["kv_usage"] for sample in measured["samples"]],
            color=_ENGINE_COLORS["vllm"],
            linewidth=2,
            label="vLLM, measured, first run",
        )
        axis.set_ylim(0, 1.05)
        axis.set_title(f"saturation, {name}")
        axis.set_xlabel("seconds since the run started")
        axis.set_ylabel("share of the KV cache allocated")
        axis.grid(color=_GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(length=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="upper right", ncols=2, frameon=False
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tabulate and plot measured results and compare them with kvsim replays."
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; fail when tables.md differs from the tables the results give",
    )
    args = parser.parse_args(argv)
    tables_path = args.results / "tables.md"
    try:
        cells = load_cells(args.results)
        if not cells:
            raise ValueError(f"no result file under {args.results}")
        fits = fit_engines(cells)
        tables = render_tables(cells, fits)
        if args.check:
            if not tables_path.exists() or tables_path.read_text(encoding="utf-8") != tables:
                parser.exit(1, f"{tables_path} is not what the results give; run analyze.py\n")
            return
        tables_path.write_text(tables, encoding="utf-8")
        write_plots(cells, fits, args.results / "plots")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
