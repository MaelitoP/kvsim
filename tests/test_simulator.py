import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariants import check_result, check_summary, check_usage_error
from kvsim import __version__
from kvsim.__main__ import main
from kvsim.metrics import summarize
from kvsim.request import Request
from kvsim.scheduler import SchedulerConfig
from kvsim.simulator import CostModel, _sample_token, simulate
from kvsim.workload import ClosedLoop, Poisson, WorkloadSpec, generate_requests, parse_spec
from strategies import configs, specs

COST = CostModel(step_overhead_s=1.0, prefill_s_per_token=0.5, decode_s_per_seq=0.25)

SPEC = WorkloadSpec(
    name="test",
    seed=0,
    vocab_size=1_000_000,
    num_requests=3,
    prompt_len=((4, 4),),
    output_len=(2, 2),
    shared_prefix_len=0,
    num_prefix_groups=1,
    arrival=ClosedLoop(concurrency=2),
)

CONFIG = SchedulerConfig(
    num_blocks=8, block_size=4, max_tokens_per_step=5, enable_prefix_caching=False
)


def simulate_argv(directory: Path, spec: WorkloadSpec, out_path: Path, flags: str) -> list[str]:
    spec_path = directory / "spec.json"
    spec_path.write_text(json.dumps(asdict(spec)), encoding="utf-8")
    return ["simulate", str(spec_path), "--out", str(out_path), *flags.split()]


def test_a_closed_loop_run_matches_hand_computed_steps_and_times() -> None:
    result = simulate(SPEC, CONFIG, COST)
    check_result(result)
    assert [
        (
            record["request_id"],
            record["arrival_s"],
            record["first_token_s"],
            record["finish_s"],
            record["first_token_step"],
            record["finish_step"],
        )
        for record in result["requests"]
    ] == [
        ("r0", 0.0, 3.25, 6.0, 0, 1),
        ("r1", 0.0, 6.0, 9.25, 1, 2),
        ("r2", 6.0, 9.25, 10.5, 2, 3),
    ]
    assert [
        (sample["t_s"], sample["kv_usage"], sample["running"], sample["waiting"])
        for sample in result["samples"]
    ] == [
        (0.0, 0.25, 2, 0),
        (3.25, 0.375, 2, 0),
        (6.0, 0.375, 2, 0),
        (9.25, 0.25, 1, 0),
    ]


def test_an_idle_simulator_jumps_to_the_next_poisson_arrival() -> None:
    spec = replace(SPEC, num_requests=2, output_len=(1, 1), arrival=Poisson(rate=0.001))
    result = simulate(spec, CONFIG, COST)
    check_result(result)
    arrival_s = generate_requests(spec)[1].arrival_s
    first, second = result["requests"]
    assert first["finish_s"] == 3.0
    assert result["samples"][1]["t_s"] == arrival_s
    assert second["finish_s"] == arrival_s + 3.0


def test_a_poisson_arrival_during_a_step_waits_for_the_next_one() -> None:
    spec = replace(SPEC, arrival=Poisson(rate=1000.0))
    result = simulate(spec, CONFIG, COST)
    check_result(result)
    arrival_s = generate_requests(spec)[1].arrival_s
    assert 0.0 < arrival_s < 3.0
    assert result["requests"][1]["arrival_s"] == arrival_s
    assert result["requests"][1]["first_token_s"] == 6.25
    busy = result["samples"][1]
    assert (busy["t_s"], busy["running"], busy["waiting"]) == (3.0, 2, 1)


def test_a_cached_prefix_is_counted_and_shortens_the_step() -> None:
    spec = replace(
        SPEC,
        num_requests=2,
        prompt_len=((2, 2),),
        output_len=(1, 1),
        shared_prefix_len=8,
        arrival=ClosedLoop(concurrency=1),
    )
    result = simulate(
        spec, replace(CONFIG, max_tokens_per_step=16, enable_prefix_caching=True), COST
    )
    check_result(result)
    assert [
        (sample["t_s"], sample["prefix_queries_total"], sample["prefix_hits_total"])
        for sample in result["samples"]
    ] == [(0.0, 10, 0), (6.0, 20, 8)]
    assert result["requests"][1]["finish_s"] == 8.0


@pytest.mark.parametrize(
    ("num_blocks", "num_preemptions", "finish_steps"),
    [(6, 0, [5, 5]), (3, 1, [5, 10])],
)
def test_a_small_cache_preempts_and_delays_the_youngest_request(
    num_blocks: int, num_preemptions: int, finish_steps: list[int]
) -> None:
    spec = replace(SPEC, num_requests=2, output_len=(6, 6))
    result = simulate(spec, replace(CONFIG, num_blocks=num_blocks, max_tokens_per_step=16), COST)
    check_result(result)
    assert result["samples"][-1]["preemptions_total"] == num_preemptions
    assert [record["finish_step"] for record in result["requests"]] == finish_steps


@given(spec=specs(), data=st.data())
def test_every_simulation_follows_its_spec(spec: WorkloadSpec, data: st.DataObject) -> None:
    max_prompt_len = spec.shared_prefix_len + max(high for _, high in spec.prompt_len)
    config = data.draw(configs(max_prompt_len + spec.output_len[1] - 1))
    result = simulate(spec, config, COST)
    check_result(result)
    check_summary(result, summarize(result))
    assert parse_spec(result["workload"]) == spec
    assert simulate(spec, config, COST) == result


def test_a_sampled_token_depends_only_on_the_request_id_and_the_output_position() -> None:
    request = Request("r0", prompt_token_ids=(1,), max_output_tokens=4, arrival_step=0)
    assert _sample_token(request, eos_token_id=0) == 3345595471
    assert _sample_token(request, eos_token_id=5) == 3345595476
    request.append_output_token(7)
    assert _sample_token(request, eos_token_id=0) == 2960035033


@pytest.mark.parametrize(
    ("coefficients", "message"),
    [
        ({"step_overhead_s": 0.0}, "step_overhead_s must be > 0 and finite, got 0.0"),
        ({"step_overhead_s": math.inf}, "step_overhead_s must be > 0 and finite, got inf"),
        ({"step_overhead_s": math.nan}, "step_overhead_s must be > 0 and finite, got nan"),
        ({"prefill_s_per_token": -0.001}, "must be >= 0 and finite, got -0.001, 0.0005"),
        ({"prefill_s_per_token": math.inf}, "must be >= 0 and finite, got inf, 0.0005"),
        ({"decode_s_per_seq": -0.001}, "must be >= 0 and finite, got 0.0002, -0.001"),
        ({"decode_s_per_seq": math.nan}, "must be >= 0 and finite, got 0.0002, nan"),
    ],
)
def test_a_cost_model_needs_finite_coefficients_and_a_positive_step_overhead(
    coefficients: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CostModel(**coefficients)


def test_the_cli_writes_the_same_bytes_whatever_the_string_hash_seed(tmp_path: Path) -> None:
    spec = replace(
        SPEC, num_requests=12, output_len=(1, 6), shared_prefix_len=8, num_prefix_groups=2
    )
    outputs: list[bytes] = []
    for hash_seed in ("1", "2"):
        out_path = tmp_path / f"result{hash_seed}.json"
        argv = simulate_argv(tmp_path, spec, out_path, "--num-blocks 6 --block-size 4")
        subprocess.run(
            [sys.executable, "-m", "kvsim", *argv],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            timeout=60,
        )
        outputs.append(out_path.read_bytes())
    assert outputs[0] == outputs[1]
    result = json.loads(outputs[0])
    check_result(result)
    assert result["samples"][-1]["preemptions_total"] > 0
    assert result["samples"][-1]["prefix_hits_total"] > 0


@pytest.mark.parametrize(
    ("flags", "config", "cost_model"),
    [
        (
            "--num-blocks 8 --no-enable-chunked-prefill",
            {
                "block_size": 16,
                "num_blocks": 8,
                "kv_tokens": 128,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 2048,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": False,
                "long_prefill_token_threshold": 0,
                "command": None,
            },
            asdict(CostModel()),
        ),
        (
            "--num-blocks 8 --block-size 4 --max-num-seqs 3 --max-num-batched-tokens 64 "
            "--long-prefill-token-threshold 32 --no-enable-prefix-caching "
            "--enable-chunked-prefill --cost-step-overhead-s 1 "
            "--cost-prefill-s-per-token 0.5 --cost-decode-s-per-seq 0.25",
            {
                "block_size": 4,
                "num_blocks": 8,
                "kv_tokens": 32,
                "max_num_seqs": 3,
                "max_num_batched_tokens": 64,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "long_prefill_token_threshold": 32,
                "command": None,
            },
            asdict(COST),
        ),
    ],
)
def test_cli_flags_set_the_scheduler_config_and_the_cost_model(
    tmp_path: Path, flags: str, config: dict[str, object], cost_model: dict[str, float]
) -> None:
    out_path = tmp_path / "results" / "simulator" / "result.json"
    main(simulate_argv(tmp_path, SPEC, out_path, flags))
    result = json.loads(out_path.read_text(encoding="utf-8"))
    check_result(result)
    assert result.pop("summary") == summarize(result)
    del result["requests"], result["samples"]
    assert result == {
        "source": "simulator",
        "measured": False,
        "engine_version": __version__,
        "model": None,
        "gpu": None,
        "config": config,
        "cost_model": cost_model,
        "workload": json.loads(json.dumps(asdict(SPEC))),
    }


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ("--num-blocks 1 --block-size 4", "request r0 needs 2 blocks, the cache holds 1"),
        ("--num-blocks 0", "num_blocks and block_size must be >= 1"),
        (
            "--num-blocks 8 --no-enable-chunked-prefill --long-prefill-token-threshold 4",
            "long_prefill_token_threshold requires enable_chunked_prefill",
        ),
        ("--num-blocks 8 --cost-step-overhead-s 0", "step_overhead_s must be > 0"),
        ("--num-blocks 8 --cost-step-overhead-s 1e308", "'finish_s': inf"),
        ("--num-blocks 8 --cost-step-overhead-s 4e306", "too large or too close together"),
    ],
)
def test_the_cli_rejects_a_run_it_cannot_simulate_or_write_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: str, message: str
) -> None:
    out_path = tmp_path / "result.json"
    check_usage_error(simulate_argv(tmp_path, SPEC, out_path, flags), capsys, message)
    assert not out_path.exists()


def test_the_cli_reports_a_missing_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec_path = str(tmp_path / "missing.json")
    argv = ["simulate", spec_path, "--out", str(tmp_path / "result.json"), "--num-blocks", "8"]
    check_usage_error(argv, capsys, spec_path)


def test_the_cli_reports_an_output_path_it_cannot_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = simulate_argv(tmp_path, SPEC, tmp_path, "--num-blocks 8")
    check_usage_error(argv, capsys, str(tmp_path))
