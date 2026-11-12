import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from analyze import (
    client_overhead_ms,
    fit_cost_model,
    fit_engines,
    load_cells,
    main,
    render_tables,
    replay_config,
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
    enable_prefix_caching: bool = True,
    cost_model: CostModel = COST_MODEL,
) -> Result:
    """An engine-shaped result whose times a simulation with `cost_model` produced."""
    config = SchedulerConfig(
        num_blocks=64,
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


def test_the_client_overhead_is_the_ttft_the_engine_does_not_account_for() -> None:
    result = measurement("vllm", SPECS["short_low"])
    assert client_overhead_ms(result) is None

    for request, unaccounted_ms in zip(result["requests"], range(12), strict=True):
        ttft_ms = (request["first_token_s"] - request["arrival_s"]) * 1000
        request["server_queue_ms"] = 5.0
        request["server_ttft_ms"] = ttft_ms - 5.0 - unaccounted_ms
    result["requests"][0]["server_queue_ms"] = None

    assert client_overhead_ms(result) == pytest.approx(6.0)


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
                for request in runs[0]["requests"]:
                    ttft_ms = (request["first_token_s"] - request["arrival_s"]) * 1000
                    request["server_queue_ms"] = 1.0
                    request["server_ttft_ms"] = ttft_ms - 4.0
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
        ["short_low", "baseline", "3", "0", "0.172", "0.172", "0.000", "0.000", "3.0"],
        ["short_low", "no_prefix_caching", "0", "0", "0.172", "0.172", "-", "-", "-"],
        ["saturation", "baseline", "0", "0", "0.812", "0.812", "-", "0.000", "-"],
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
