import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from analyze import (
    Pressure,
    TtftSplit,
    fit_cost_model,
    fit_engines,
    load_cells,
    main,
    render_tables,
    replay_config,
    simulate_pressure,
    ttft_split_ms,
)
from client import Engine
from kvsim.scheduler import SchedulerConfig
from kvsim.simulator import CostModel, Result, simulate
from kvsim.workload import ClosedLoop, Poisson, WorkloadSpec

ROOT = Path(__file__).parent.parent
COST_MODEL = CostModel(step_overhead_s=0.02, prefill_s_per_token=0.0002, decode_s_per_seq=0.0005)
MAX_NUM_SEQS = 8
MAX_NUM_BATCHED_TOKENS = 64
SPECS = {
    "short_low": WorkloadSpec(
        name="short",
        seed=0,
        vocab_size=1000,
        num_requests=12,
        prompt_len=((8, 40),),
        output_len=(6, 6),
        shared_prefix_len=0,
        num_prefix_groups=1,
        arrival=ClosedLoop(concurrency=2),
    ),
    "saturation": WorkloadSpec(
        name="long",
        seed=1,
        vocab_size=1000,
        num_requests=12,
        prompt_len=((60, 90),),
        output_len=(10, 10),
        shared_prefix_len=0,
        num_prefix_groups=1,
        arrival=ClosedLoop(concurrency=5),
    ),
}


def measurement(
    engine: Engine,
    spec: WorkloadSpec,
    *,
    num_blocks: int = 64,
    enable_prefix_caching: bool = True,
    cost_model: CostModel = COST_MODEL,
) -> Result:
    """An engine-shaped result whose times a simulation with `cost_model` produced."""
    config = SchedulerConfig(
        num_blocks=num_blocks,
        block_size=8,
        max_running_requests=MAX_NUM_SEQS,
        max_tokens_per_step=MAX_NUM_BATCHED_TOKENS + (MAX_NUM_SEQS if engine == "lmdeploy" else 0),
        enable_prefix_caching=enable_prefix_caching,
        enable_full_input_admission=engine == "vllm",
    )
    result = simulate(spec, config, cost_model)
    return cast(
        Result,
        {
            **result,
            "source": engine,
            "measured": True,
            "config": {
                **result["config"],
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "command": "docker run engine",
            },
        },
    )


def write_runs(results_dir: Path, name: str, runs: list[Result]) -> None:
    engine_dir = results_dir / runs[0]["source"]
    engine_dir.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(runs, start=1):
        (engine_dir / f"{name}.run{index}.json").write_text(json.dumps(run), encoding="utf-8")


def tables_of(results_dir: Path) -> str:
    cells = load_cells(results_dir)
    return render_tables(cells, fit_engines(cells))


def table_rows(tables: str, heading: str) -> list[list[str]]:
    section = tables.split(f"## {heading}\n")[1].split("\n## ")[0]
    lines = [line for line in section.splitlines() if line.startswith("| ")]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


@pytest.mark.parametrize(
    ("engine", "recorded", "replayed"),
    [
        ("vllm", {}, {"enable_full_input_admission": True}),
        (
            "vllm",
            {"command": "docker run vllm --no-scheduler-reserve-full-isl"},
            {"enable_full_input_admission": False},
        ),
        (
            "vllm",
            {"long_prefill_token_threshold": 512},
            {"long_prefill_token_threshold": 512, "enable_full_input_admission": True},
        ),
        (
            "vllm",
            {"enable_chunked_prefill": False},
            {"enable_chunked_prefill": False, "enable_full_input_admission": True},
        ),
        ("lmdeploy", {}, {"max_tokens_per_step": MAX_NUM_BATCHED_TOKENS + MAX_NUM_SEQS}),
    ],
)
def test_a_replay_takes_the_engine_configuration_of_its_result(
    engine: Engine, recorded: dict[str, object], replayed: dict[str, object]
) -> None:
    result = measurement(engine, SPECS["short_low"], enable_prefix_caching=False)
    result["config"].update(recorded)  # type: ignore[typeddict-item]

    assert replay_config(result) == replace(
        SchedulerConfig(
            num_blocks=64,
            block_size=8,
            max_running_requests=MAX_NUM_SEQS,
            max_tokens_per_step=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=False,
        ),
        **replayed,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("engine", ["vllm", "lmdeploy"])
def test_the_fit_recovers_the_cost_model_that_produced_the_times(engine: Engine) -> None:
    cost_model, ttft_r_squared, decode_r_squared = fit_cost_model(
        measurement(engine, spec) for spec in SPECS.values()
    )

    assert cost_model.step_overhead_s == pytest.approx(COST_MODEL.step_overhead_s)
    assert cost_model.prefill_s_per_token == pytest.approx(COST_MODEL.prefill_s_per_token)
    assert cost_model.decode_s_per_seq == pytest.approx(COST_MODEL.decode_s_per_seq)
    assert (ttft_r_squared, decode_r_squared) == pytest.approx((1.0, 1.0))


def test_the_two_r_squared_tell_the_ttft_intervals_from_the_decode_intervals() -> None:
    results = [measurement("vllm", spec) for spec in SPECS.values()]
    for result in results:
        for index, request in enumerate(result["requests"]):
            request["arrival_s"] -= 0.02 * (index % 2)

    _, ttft_r_squared, decode_r_squared = fit_cost_model(results)

    assert ttft_r_squared < 0.95 < 0.99 < decode_r_squared


def reversed_times(result: Result) -> Result:
    for request in result["requests"]:
        for event in ("arrival_s", "first_token_s", "finish_s"):
            request[event] = -request[event]
    return result


def without_its_last_request(result: Result) -> Result:
    del result["requests"][-1]
    return result


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            measurement("vllm", replace(SPECS["short_low"], arrival=Poisson(rate=5.0))),
            "closed-loop",
        ),
        (
            measurement(
                "vllm",
                replace(SPECS["short_low"], prompt_len=((8, 8),), arrival=ClosedLoop(1)),
            ),
            "do not separate",
        ),
        (reversed_times(measurement("vllm", SPECS["saturation"])), "no valid cost model"),
        (without_its_last_request(measurement("vllm", SPECS["saturation"])), "requests"),
    ],
    ids=["poisson", "one-request-at-a-time", "negative-coefficients", "missing-request"],
)
def test_the_fit_refuses_results_that_determine_no_cost_model(result: Result, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_cost_model([result])


def test_the_ttft_split_sets_the_engine_times_apart_from_the_rest_of_the_client_ttft() -> None:
    requests = measurement("vllm", SPECS["short_low"])["requests"]
    assert ttft_split_ms(requests) is None

    for request, outside_ms in zip(requests, range(12), strict=True):
        ttft_ms = (request["first_token_s"] - request["arrival_s"]) * 1000
        request["server_queue_ms"] = 5.0
        request["server_ttft_ms"] = ttft_ms - 5.0 - outside_ms
    requests[0]["server_queue_ms"] = None
    engine_ms = sorted(cast(float, request["server_ttft_ms"]) for request in requests[1:])

    assert ttft_split_ms(requests) == pytest.approx(TtftSplit(5.0, engine_ms[5], 2.0, 6.0, 10.0))
    assert ttft_split_ms(requests[:2]) is None


TWO_REQUESTS = WorkloadSpec(
    name="two",
    seed=0,
    vocab_size=1000,
    num_requests=2,
    prompt_len=((8, 8),),
    output_len=(5, 5),
    shared_prefix_len=0,
    num_prefix_groups=1,
    arrival=ClosedLoop(concurrency=2),
)
SHARED_FIRST_BLOCK = replace(TWO_REQUESTS, shared_prefix_len=4, prompt_len=((4, 4),))


def pressure(
    scheduled_tokens: int, discarded_tokens: int, recovered_tokens: int, preemptions: int = 1
) -> Pressure:
    return Pressure(
        preemptions=preemptions,
        scheduled_tokens=scheduled_tokens,
        discarded_tokens=discarded_tokens,
        recovered_tokens=recovered_tokens,
        max_orphaned_blocks=0,
    )


@pytest.mark.parametrize(
    ("spec", "config", "expected"),
    [
        (
            TWO_REQUESTS,
            SchedulerConfig(num_blocks=5, block_size=4),
            pressure(scheduled_tokens=24, discarded_tokens=8, recovered_tokens=8),
        ),
        (
            TWO_REQUESTS,
            SchedulerConfig(num_blocks=5, block_size=4, enable_prefix_caching=False),
            pressure(scheduled_tokens=32, discarded_tokens=8, recovered_tokens=0),
        ),
        (
            SHARED_FIRST_BLOCK,
            SchedulerConfig(num_blocks=4, block_size=4),
            pressure(scheduled_tokens=20, discarded_tokens=8, recovered_tokens=8),
        ),
    ],
    ids=["resumed-from-the-cache", "resumed-without-a-cache", "victim-admitted-with-a-hit"],
)
def test_pressure_counts_the_tokens_a_victim_held_and_those_it_finds_cached_on_resumption(
    spec: WorkloadSpec, config: SchedulerConfig, expected: Pressure
) -> None:
    assert simulate_pressure(spec, config) == expected


@pytest.mark.parametrize(
    ("spec", "config"),
    [
        (SPECS["saturation"], SchedulerConfig(13, block_size=8, max_tokens_per_step=64)),
        (SPECS["saturation"], SchedulerConfig(16, block_size=8, max_tokens_per_step=64)),
        (
            replace(
                TWO_REQUESTS,
                seed=992,
                vocab_size=3,
                num_requests=7,
                prompt_len=((2, 8),),
                output_len=(1, 3),
            ),
            SchedulerConfig(
                4, block_size=4, max_tokens_per_step=8, enable_full_input_admission=True
            ),
        ),
    ],
    ids=["one-request-fits", "two-requests-fit", "outputs-within-the-prompt-vocabulary"],
)
def test_pressure_is_counted_on_the_schedule_the_simulator_replays(
    spec: WorkloadSpec, config: SchedulerConfig
) -> None:
    steps = simulate(spec, config, CostModel(1, 0, 0))
    steps_and_tokens = simulate(spec, config, CostModel(1, 1, 1))

    counted = simulate_pressure(spec, config)

    assert counted.preemptions == steps["samples"][-1]["preemptions_total"] > 0
    assert counted.scheduled_tokens == max(
        request["finish_s"] for request in steps_and_tokens["requests"]
    ) - max(request["finish_s"] for request in steps["requests"])


SAME_PROMPTS = replace(TWO_REQUESTS, vocab_size=1, num_requests=4, prompt_len=((4, 4),))
SHARED_TWO_BLOCKS = replace(TWO_REQUESTS, shared_prefix_len=8, prompt_len=((2, 2),))


@pytest.mark.parametrize(
    ("spec", "config", "num_orphaned_blocks"),
    [
        (SAME_PROMPTS, SchedulerConfig(5, block_size=4), 3),
        (replace(SAME_PROMPTS, vocab_size=1000), SchedulerConfig(5, block_size=4), 0),
        (SHARED_TWO_BLOCKS, SchedulerConfig(4, block_size=4, long_prefill_token_threshold=4), 1),
        (SHARED_TWO_BLOCKS, SchedulerConfig(4, block_size=4), 0),
    ],
    ids=[
        "same-block-aligned-prompts",
        "distinct-prompts",
        "shared-prefix-computed-in-parallel",
        "shared-prefix-computed-once",
    ],
)
def test_pressure_counts_the_cached_blocks_chained_to_a_duplicate_of_an_evicted_block(
    spec: WorkloadSpec, config: SchedulerConfig, num_orphaned_blocks: int
) -> None:
    assert simulate_pressure(spec, config).max_orphaned_blocks == num_orphaned_blocks


def test_pressure_is_simulated_on_closed_loops_only() -> None:
    with pytest.raises(ValueError, match="closed-loop"):
        simulate_pressure(replace(TWO_REQUESTS, arrival=Poisson(rate=5.0)), SchedulerConfig(8))


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    for engine in ("vllm", "lmdeploy"):
        for scenario, spec in SPECS.items():
            runs = [measurement(engine, spec) for _ in range(3)]
            if (engine, scenario) == ("vllm", "saturation"):
                for sample in runs[1]["samples"]:
                    sample["prefix_queries_total"] = None  # type: ignore[typeddict-item]
                    sample["prefix_hits_total"] = None  # type: ignore[typeddict-item]
            if (engine, scenario) == ("vllm", "short_low"):
                for run in runs:
                    run["samples"][-1]["preemptions_total"] = 3
                    for outside_ms, request in enumerate(run["requests"]):
                        ttft_ms = (request["first_token_s"] - request["arrival_s"]) * 1000
                        request["server_queue_ms"] = 1.0
                        request["server_ttft_ms"] = ttft_ms - 1.0 - outside_ms
            write_runs(tmp_path, f"{scenario}.baseline", runs)
    write_runs(
        tmp_path,
        "short_low.no_prefix_caching",
        [
            measurement(
                "vllm",
                SPECS["short_low"],
                enable_prefix_caching=False,
                cost_model=replace(COST_MODEL, step_overhead_s=step_overhead_s),
            )
            for step_overhead_s in (0.02, 0.04, 0.08)
        ],
    )
    write_runs(tmp_path, "saturation.tight", [measurement("lmdeploy", SPECS["saturation"])])
    return tmp_path


def test_tables_hold_the_median_of_the_runs_and_a_dash_for_a_missing_value(
    results_dir: Path,
) -> None:
    engines = table_rows(tables_of(results_dir), "Engines, measured")

    assert [row[:3] for row in engines] == [
        ["short_low", "baseline", "vllm"],
        ["short_low", "baseline", "lmdeploy"],
        ["short_low", "no_prefix_caching", "vllm"],
        ["saturation", "baseline", "vllm"],
        ["saturation", "baseline", "lmdeploy"],
        ["saturation", "tight", "lmdeploy"],
    ]
    assert engines[0][3:] == ["24.5", "42.6", "21.4", "21.7", "87.9", "0.000", "0.0%"]
    assert engines[2][3:] == ["44.5", "71.6", "41.4", "41.7", "46.2", "-", "134.8%"]
    assert engines[3][8] == "-"
    assert engines[5][9] == "-"


def test_tables_set_vllm_against_its_replay_and_estimates_against_measurements(
    results_dir: Path,
) -> None:
    tables = tables_of(results_dir)

    assert table_rows(tables, "vLLM against kvsim, scheduler counters") == [
        ["short_low", "baseline", "3", "0", "0.172", "0.172", "0.000", "0.000"],
        ["short_low", "no_prefix_caching", "0", "0", "0.172", "0.172", "-", "-"],
        ["saturation", "baseline", "0", "0", "0.812", "0.812", "-", "0.000"],
    ]
    estimates = table_rows(tables, "Estimated against measured times")
    assert [row[2:4] for row in estimates] == [
        ["vllm", "yes"],
        ["lmdeploy", "yes"],
        ["vllm", "no"],
        ["vllm", "yes"],
        ["lmdeploy", "yes"],
        ["lmdeploy", "no"],
    ]
    assert estimates[0][4:] == [
        "24.5", "24.5", "+0.0%", "21.4", "21.4", "+0.0%", "87.9", "87.9", "+0.0%"
    ]  # fmt: skip
    assert estimates[2][4:] == [
        "44.5", "24.5", "-44.9%", "41.4", "21.4", "-48.3%", "46.2", "87.9", "+90.4%"
    ]  # fmt: skip
    assert table_rows(tables, "LMDeploy cache usage, measured") == [
        ["short_low", "baseline", "0.172"],
        ["saturation", "baseline", "0.797"],
        ["saturation", "tight", "0.797"],
    ]
    assert table_rows(tables, "Cost model, fitted") == [
        ["vllm", "20.00", "0.2000", "0.500", "1.00", "1.0000", "short_low, saturation"],
        ["lmdeploy", "20.00", "0.2000", "0.500", "1.00", "1.0000", "short_low, saturation"],
    ]


def test_tables_split_the_ttft_by_arrival_and_replay_the_other_budget_and_admission_rules(
    results_dir: Path,
) -> None:
    tight = measurement("vllm", SPECS["saturation"], num_blocks=16)
    write_runs(results_dir, "saturation.tight", [tight])
    tables = tables_of(results_dir)

    arrivals = table_rows(tables, "Time to first token by arrival")
    assert arrivals[0][3:] == ["29.0", "29.0", "24.3", "24.3", "1.0", "15.4", "2.9", "6.5", "10.1"]
    assert arrivals[1][3:] == ["29.0", "29.0", "27.6", "27.6", "-", "-", "-", "-", "-"]
    assert arrivals[2][3:7] == ["49.0", "29.0", "44.3", "24.3"]
    assert table_rows(tables, "Token budget rule, estimated") == [
        ["short_low", "24.5", "27.9", "24.5", "27.9"],
        ["saturation", "67.4", "64.2", "67.4", "64.2"],
    ]
    pressure_rows = table_rows(tables, "Memory pressure, simulated")
    assert [row[:4] for row in pressure_rows] == [
        ["saturation", "16", "on", "whole input"],
        ["saturation", "16", "on", "first chunk"],
    ]
    assert pressure_rows[0][4:] == ["0", "955", "0", "0", "0", "0.000", "967.7", "43.1"]
    assert pressure_rows[1][4:] == ["8", "1002", "255", "208", "0", "0.000", "929.8", "44.2"]


def test_tables_split_the_ttft_by_prompt_range_and_sweep_the_cache_of_shared_prefixes(
    results_dir: Path,
) -> None:
    two_ranges = replace(SPECS["short_low"], prompt_len=((8, 8), (56, 56)), arrival=ClosedLoop(1))
    shared_prefix = replace(SPECS["short_low"], shared_prefix_len=16, num_prefix_groups=2)
    write_runs(results_dir, "mixed.baseline", [measurement("vllm", two_ranges)])
    write_runs(results_dir, "shared_prefix.baseline", [measurement("vllm", shared_prefix)])
    tables = tables_of(results_dir)

    assert table_rows(tables, "Time to first token by prompt length") == [
        ["mixed", "baseline", "vllm", "8 to 8", "21.6", "21.6"],
        ["mixed", "baseline", "vllm", "56 to 56", "31.2", "31.2"],
    ]
    assert [row[:4] for row in table_rows(tables, "Memory pressure, simulated")] == [
        ["shared_prefix", str(num_blocks), "on", "whole input"]
        for num_blocks in (252, 192, 160, 144, 128)
    ]


def test_tables_replay_no_other_setting_of_an_engine_without_a_fit(tmp_path: Path) -> None:
    tight = measurement("vllm", SPECS["saturation"], num_blocks=16)
    write_runs(tmp_path, "saturation.tight", [tight])
    tables = tables_of(tmp_path)

    assert table_rows(tables, "Token budget rule, estimated") == []
    assert table_rows(tables, "Memory pressure, simulated") == []


def test_the_command_writes_tables_and_plots_and_checks_them(
    results_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--results", str(results_dir)]
    tables_path = results_dir / "tables.md"
    with pytest.raises(SystemExit) as missing:
        main([*argv, "--check"])
    assert missing.value.code == 1

    main(argv)

    assert tables_path.read_text(encoding="utf-8") == tables_of(results_dir)
    assert sorted(path.name for path in (results_dir / "plots").iterdir()) == [
        "kv_usage_saturation.png",
        "throughput.png",
        "ttft.png",
    ]
    main([*argv, "--check"])
    tables_path.write_text(tables_path.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    with pytest.raises(SystemExit) as edited:
        main([*argv, "--check"])
    assert edited.value.code == 1
    assert "is not what the results give" in capsys.readouterr().err


def misplaced(results_dir: Path) -> str:
    write_runs(results_dir, "saturation.baseline", [measurement("vllm", SPECS["saturation"])])
    (results_dir / "vllm").rename(results_dir / "lmdeploy")
    return "saturation.baseline.run1.json: its source is 'vllm'"


def unknown_cell(results_dir: Path) -> str:
    write_runs(results_dir, "saturation.warm", [measurement("vllm", SPECS["saturation"])])
    return "saturation.warm.run1.json: not a scenario, cell and engine of the matrix"


def mixed_runs(results_dir: Path) -> str:
    runs = [measurement("vllm", SPECS["saturation"]), measurement("vllm", SPECS["short_low"])]
    write_runs(results_dir, "saturation.baseline", runs)
    return "saturation.baseline.run2.json: differs from the other runs of its cell"


def unfittable(results_dir: Path) -> str:
    one_at_a_time = replace(SPECS["short_low"], prompt_len=((8, 8),), arrival=ClosedLoop(1))
    write_runs(results_dir, "short_low.baseline", [measurement("vllm", one_at_a_time)])
    return "vllm: the results do not separate the three coefficients of a cost model"


def empty(results_dir: Path) -> str:
    return f"no result file under {results_dir}"


@pytest.mark.parametrize("arrange", [misplaced, unknown_cell, mixed_runs, unfittable, empty])
def test_the_command_refuses_results_it_cannot_tabulate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arrange: Callable[[Path], str]
) -> None:
    message = arrange(tmp_path)

    with pytest.raises(SystemExit) as failure:
        main(["--results", str(tmp_path)])

    assert failure.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "tables.md").exists()


def test_the_committed_tables_are_what_the_committed_results_give() -> None:
    results_dir = ROOT / "results"

    assert (results_dir / "tables.md").read_text(encoding="utf-8") == tables_of(results_dir)
