import json
from pathlib import Path

import pytest

from invariants import check_usage_error
from kvsim.__main__ import main
from kvsim.metrics import summarize

REQUESTS = [
    {"arrival_s": 0.0, "first_token_s": 0.5, "finish_s": 3.5, "output_tokens": 7},
    {"arrival_s": 0.0, "first_token_s": 1.0, "finish_s": 4.0, "output_tokens": 4},
    {"arrival_s": 1.0, "first_token_s": 2.5, "finish_s": 2.5, "output_tokens": 1},
]

SAMPLES = [
    {"kv_usage": 0.25, "preemptions_total": 0, "prefix_queries_total": 0, "prefix_hits_total": 0},
    {"kv_usage": 0.5, "preemptions_total": 1, "prefix_queries_total": 64, "prefix_hits_total": 16},
    {
        "kv_usage": 0.75,
        "preemptions_total": 2,
        "prefix_queries_total": 128,
        "prefix_hits_total": 48,
    },
]

NO_GAUGES = {
    "kv_usage": None,
    "preemptions_total": None,
    "prefix_queries_total": None,
    "prefix_hits_total": None,
}

ENGINE_RESULT = {"source": "vllm", "measured": True, "requests": REQUESTS, "samples": SAMPLES}

SIMULATOR_RESULT = {
    "source": "simulator",
    "measured": False,
    "requests": REQUESTS[2:],
    "samples": [NO_GAUGES],
}


def test_a_summary_matches_hand_computed_values() -> None:
    assert summarize({"requests": REQUESTS, "samples": SAMPLES}) == {
        "num_requests": 3,
        "output_tokens": 12,
        "duration_s": 4.0,
        "requests_per_s": 0.75,
        "output_tokens_per_s": 3.0,
        "ttft_s": {"mean": 1.0, "p50": 1.0, "p95": 1.45, "p99": 1.49},
        "tpot_s": {"mean": 0.75, "p50": 0.75, "p95": 0.975, "p99": 0.995},
        "e2e_s": {"mean": 3.0, "p50": 3.5, "p95": 3.95, "p99": 3.99},
        "kv_usage_mean": 0.5,
        "kv_usage_max": 0.75,
        "preemptions": 2,
        "prefix_hit_rate": 0.375,
    }


def test_a_lone_request_with_one_output_token_has_no_tpot() -> None:
    summary = summarize({"requests": REQUESTS[2:], "samples": SAMPLES})
    assert summary["tpot_s"] is None
    assert summary["ttft_s"] == {"mean": 1.5, "p50": 1.5, "p95": 1.5, "p99": 1.5}
    assert summary["e2e_s"] == summary["ttft_s"]
    assert summary["output_tokens_per_s"] == 1 / 1.5


@pytest.mark.parametrize(
    ("samples", "kv_usage_mean", "kv_usage_max", "preemptions", "prefix_hit_rate"),
    [
        ([], None, None, None, None),
        ([NO_GAUGES], None, None, None, None),
        ([SAMPLES[0]], 0.25, 0.25, 0, None),
        ([*SAMPLES[:2], NO_GAUGES], 0.375, 0.5, 1, 0.25),
        ([{**SAMPLES[1], "prefix_hits_total": None}], 0.5, 0.5, 1, None),
    ],
)
def test_gauges_take_their_last_reading_and_none_without_one(
    samples: list[dict[str, float | None]],
    kv_usage_mean: float | None,
    kv_usage_max: float | None,
    preemptions: int | None,
    prefix_hit_rate: float | None,
) -> None:
    summary = summarize({"requests": REQUESTS, "samples": samples})
    assert summary["kv_usage_mean"] == kv_usage_mean
    assert summary["kv_usage_max"] == kv_usage_max
    assert summary["preemptions"] == preemptions
    assert summary["prefix_hit_rate"] == prefix_hit_rate


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([], "a result needs a non-empty list of requests and a list of samples"),
        ({"requests": [], "samples": SAMPLES}, "a result needs a non-empty list of requests"),
        ({"requests": REQUESTS}, "a result needs a non-empty list of requests"),
        ({"requests": "r0", "samples": SAMPLES}, "a result needs a non-empty list of requests"),
        ({"requests": [{"arrival_s": 0.0}], "samples": []}, "got {'arrival_s': 0.0}"),
        (
            {"requests": [{**REQUESTS[2], "arrival_s": 2.5}], "samples": []},
            "a result must span a positive duration",
        ),
        ({"requests": REQUESTS, "samples": [{"kv_usage": 0.5}]}, "got {'kv_usage': 0.5}"),
    ],
)
def test_a_result_without_requests_samples_or_duration_is_rejected(
    result: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize(result)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("first_token_s", "0.5"),
        ("first_token_s", True),
        ("first_token_s", float("nan")),
        ("arrival_s", 1.0),
        ("finish_s", 0.25),
        ("finish_s", float("inf")),
        ("output_tokens", 0),
        ("output_tokens", 7.0),
    ],
)
def test_a_request_with_a_wrong_value_is_rejected(key: str, value: object) -> None:
    result = {"requests": [{**REQUESTS[0], key: value}], "samples": []}
    with pytest.raises(ValueError, match=f"a request needs .*'{key}': {value!r}"):
        summarize(result)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kv_usage", 25.0),
        ("kv_usage", -0.25),
        ("kv_usage", True),
        ("preemptions_total", 1.5),
        ("preemptions_total", -1),
        ("prefix_queries_total", -1),
        ("prefix_hits_total", -1),
        ("prefix_hits_total", 65),
    ],
)
def test_a_sample_with_a_wrong_value_is_rejected(key: str, value: object) -> None:
    result = {"requests": REQUESTS, "samples": [{**SAMPLES[1], key: value}]}
    with pytest.raises(ValueError, match=f"a sample needs .*'{key}': {value!r}"):
        summarize(result)


@pytest.mark.parametrize(
    "requests",
    [
        [{**REQUESTS[0], "finish_s": 10**400}],
        [{**REQUESTS[0], "output_tokens": 10**400}],
        [
            {"arrival_s": 0.0, "first_token_s": 2e306, "finish_s": 2e306, "output_tokens": 1},
            {"arrival_s": 0.0, "first_token_s": 3e306, "finish_s": 3e306, "output_tokens": 1},
        ],
        [{"arrival_s": -1.7e308, "first_token_s": 0.0, "finish_s": 1.7e308, "output_tokens": 2}],
        [{"arrival_s": 0.0, "first_token_s": 0.0, "finish_s": 5e-324, "output_tokens": 1}],
    ],
)
def test_times_too_large_or_too_close_to_summarize_are_rejected(
    requests: list[dict[str, float]],
) -> None:
    with pytest.raises(ValueError, match="too large or too close together to summarize"):
        summarize({"requests": requests, "samples": []})


def test_the_cli_prints_one_row_per_result_and_labels_estimated_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("engine.json").write_text(json.dumps(ENGINE_RESULT), encoding="utf-8")
    Path("simulator.json").write_text(json.dumps(SIMULATOR_RESULT), encoding="utf-8")
    main(["summarize", "engine.json"])
    assert capsys.readouterr().out.splitlines() == [
        "file         source  times     num_requests  ttft_p50_ms  ttft_p95_ms  tpot_p50_ms  "
        "tpot_p95_ms  output_tokens_per_s  requests_per_s  kv_usage_mean  kv_usage_max  "
        "preemptions  prefix_hit_rate",
        "engine.json  vllm    measured  3             1000.0       1450.0       750.0        "
        "975.0        3.0                  0.75            0.500          0.750         "
        "2            0.375",
    ]
    main(["summarize", "engine.json", "simulator.json"])
    assert capsys.readouterr().out.splitlines() == [
        "file            source     times      num_requests  ttft_p50_ms  ttft_p95_ms  "
        "tpot_p50_ms  tpot_p95_ms  output_tokens_per_s  requests_per_s  kv_usage_mean  "
        "kv_usage_max  preemptions  prefix_hit_rate",
        "engine.json     vllm       measured   3             1000.0       1450.0       "
        "750.0        975.0        3.0                  0.75            0.500          "
        "0.750         2            0.375",
        "simulator.json  simulator  estimated  1             1500.0       1500.0       "
        "-            -            0.7                  0.67            -              "
        "-             -            -",
        "estimated: simulated seconds from the cost model of the run; "
        "compare them only between runs made with the same cost model",
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "result.json: Expecting property name"),
        ("[]", "result.json: a result needs a source and a measured flag"),
        (json.dumps({**ENGINE_RESULT, "measured": "yes"}), "result.json: a result needs a source"),
        (
            json.dumps({**ENGINE_RESULT, "requests": []}),
            "result.json: a result needs a non-empty list of requests",
        ),
    ],
)
def test_the_cli_names_the_result_file_it_cannot_summarize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    path = tmp_path / "result.json"
    path.write_text(content, encoding="utf-8")
    check_usage_error(["summarize", str(path)], capsys, message)


def test_the_cli_reports_a_missing_result_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = str(tmp_path / "missing.json")
    check_usage_error(["summarize", path], capsys, path)
