import hashlib
import json
import math
from dataclasses import asdict, replace
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kvsim.workload import (
    ClosedLoop,
    LengthRange,
    Poisson,
    RequestSpec,
    WorkloadSpec,
    generate_requests,
    load_spec,
    parse_spec,
)

WORKLOADS = Path(__file__).parents[1] / "benchmarks" / "workloads"

SPEC = WorkloadSpec(
    name="test",
    seed=0,
    vocab_size=1_000_000,
    num_requests=12,
    prompt_len=((1, 4),),
    output_len=(1, 4),
    shared_prefix_len=8,
    num_prefix_groups=3,
    arrival=ClosedLoop(concurrency=4),
)

SPEC_DATA: dict[str, Any] = {
    "name": "test",
    "seed": 0,
    "vocab_size": 100,
    "num_requests": 4,
    "prompt_len": [1, 4],
    "output_len": [1, 4],
    "shared_prefix_len": 0,
    "num_prefix_groups": 1,
    "arrival": {"kind": "closed_loop", "concurrency": 2},
}

LENGTH_RANGES = st.tuples(st.integers(1, 6), st.integers(0, 6)).map(
    lambda bounds: (bounds[0], bounds[0] + bounds[1])
)


@st.composite
def specs(draw: st.DrawFn) -> WorkloadSpec:
    shared_prefix_len = draw(st.integers(0, 6))
    prompt_len: list[LengthRange] = draw(st.lists(LENGTH_RANGES, min_size=1, max_size=3))
    return WorkloadSpec(
        name=draw(st.text()),
        seed=draw(st.integers(min_value=0)),
        vocab_size=draw(st.integers(1, 50)),
        num_requests=draw(st.integers(1, 12)),
        prompt_len=tuple(prompt_len),
        output_len=draw(LENGTH_RANGES),
        shared_prefix_len=shared_prefix_len,
        num_prefix_groups=draw(st.integers(1, 3)) if shared_prefix_len else 1,
        arrival=draw(
            st.one_of(
                st.builds(ClosedLoop, concurrency=st.integers(1, 8)),
                st.builds(Poisson, rate=st.floats(0.01, 100)),
            )
        ),
    )


def spec_data(**changes: object) -> dict[str, object]:
    return {**SPEC_DATA, **changes}


def digest(requests: list[RequestSpec]) -> str:
    return hashlib.sha256(
        json.dumps([asdict(request) for request in requests]).encode()
    ).hexdigest()


def common_prefix_len(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(first, second, strict=False)) if a != b),
        min(len(first), len(second)),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("short_low", "6d48f07e49f5d8fbc199abe215c887cf5b0f6b10c411a54b85271c660139f58d"),
        ("long_high", "42767544c192febac1df8e1a59272614c556d7cc53033cf90a22eedf72941c60"),
        ("shared_prefix", "1e1abab179892e13999a71fcddeeb98c52d24fc4d6e3bf394def0632d2ce369b"),
        ("mixed", "6a8dd9da14e4e425ca1b5c2b4e618bfc52c26b6bb2c8b4ae048ba0771053888d"),
        ("saturation", "6616511eb0fd7a75998de39c4244c64d5488c7a7b0ead900edd83bfd41ed8d14"),
    ],
)
def test_each_scenario_generates_its_pinned_workload(name: str, expected: str) -> None:
    spec = load_spec(WORKLOADS / f"{name}.json")
    assert spec.name == name
    assert digest(generate_requests(spec)) == expected


def test_poisson_generation_matches_its_pinned_workload() -> None:
    spec = replace(
        SPEC,
        vocab_size=100,
        num_requests=6,
        prompt_len=((1, 4), (6, 8)),
        arrival=Poisson(rate=2.0),
    )
    assert digest(generate_requests(spec)) == (
        "128ad327c12c7947c872fa748cf5258b4e2dca530b2efa0d3deaf486a4dfd131"
    )


def test_different_seeds_draw_different_workloads() -> None:
    assert generate_requests(replace(SPEC, seed=1)) != generate_requests(SPEC)


@given(spec=specs())
def test_requests_follow_their_spec(spec: WorkloadSpec) -> None:
    requests = generate_requests(spec)
    prefix_len = spec.shared_prefix_len
    assert [request.request_id for request in requests] == [
        f"r{index}" for index in range(spec.num_requests)
    ]
    for index, request in enumerate(requests):
        low, high = spec.prompt_len[index % len(spec.prompt_len)]
        assert low <= len(request.prompt_token_ids) - prefix_len <= high
        assert spec.output_len[0] <= request.output_len <= spec.output_len[1]
        assert all(0 <= token < spec.vocab_size for token in request.prompt_token_ids)
        if index >= spec.num_prefix_groups:
            same_group = requests[index - spec.num_prefix_groups]
            assert request.prompt_token_ids[:prefix_len] == same_group.prompt_token_ids[:prefix_len]
    arrivals = [request.arrival_s for request in requests]
    if isinstance(spec.arrival, ClosedLoop):
        assert arrivals == [0.0] * spec.num_requests
    else:
        assert arrivals[0] == 0.0
        assert all(earlier < later for earlier, later in pairwise(arrivals))


def test_prefix_groups_share_exactly_their_prefix() -> None:
    requests = generate_requests(SPEC)
    for (first_index, first), (second_index, second) in combinations(enumerate(requests), 2):
        same_group = first_index % SPEC.num_prefix_groups == second_index % SPEC.num_prefix_groups
        assert common_prefix_len(first.prompt_token_ids, second.prompt_token_ids) == (
            SPEC.shared_prefix_len if same_group else 0
        )


def test_poisson_rate_counts_requests_per_second() -> None:
    requests = generate_requests(replace(SPEC, num_requests=4001, arrival=Poisson(rate=4.0)))
    assert requests[-1].arrival_s / 4000 == pytest.approx(0.25, rel=0.05)


@given(spec=specs())
def test_a_spec_survives_a_round_trip(spec: WorkloadSpec) -> None:
    assert parse_spec(asdict(spec)) == spec
    assert parse_spec(json.loads(json.dumps(asdict(spec)))) == spec


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("prompt_len", 1024, ((1024, 1024),)),
        ("prompt_len", [128, 256], ((128, 256),)),
        ("prompt_len", [[64, 128], [1536, 2048]], ((64, 128), (1536, 2048))),
        ("output_len", 128, (128, 128)),
        ("output_len", [1, 4], (1, 4)),
    ],
)
def test_a_length_is_a_number_a_range_or_a_list_of_ranges(
    key: str, value: object, expected: object
) -> None:
    assert getattr(parse_spec(spec_data(**{key: value})), key) == expected


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], r"workload must be an object, got \[\]"),
        (spec_data(prompt_length=[1, 4]), "workload keys must be"),
        ({k: v for k, v in SPEC_DATA.items() if k != "seed"}, "workload keys must be"),
        (spec_data(name=5), "name must be str, got 5"),
        *(
            (spec_data(**{key: True}), f"{key} must be int, got True")
            for key in ("seed", "vocab_size", "num_requests", "shared_prefix_len")
        ),
        (spec_data(num_prefix_groups=1.0), "num_prefix_groups must be int, got 1.0"),
        (spec_data(seed="0"), "seed must be int, got '0'"),
        (spec_data(prompt_len=[1.5, 4]), "prompt_len must be int, got 1.5"),
        (spec_data(prompt_len=[1, 4.5]), "prompt_len must be int, got 4.5"),
        (spec_data(seed=-1), "seed must be >= 0"),
        (spec_data(vocab_size=0), "vocab_size and num_requests must be >= 1"),
        (spec_data(num_requests=0), "vocab_size and num_requests must be >= 1"),
        (spec_data(prompt_len=[]), "prompt_len needs at least one range"),
        (spec_data(prompt_len=[[1, 4], [4, 1]]), r"length range \(4, 1\)"),
        (spec_data(output_len=0), r"length range \(0, 0\)"),
        (spec_data(shared_prefix_len=-1), "shared_prefix_len must be >= 0"),
        (spec_data(num_prefix_groups=0), "num_prefix_groups >= 1"),
        (spec_data(num_prefix_groups=2), "requires a shared prefix"),
        (spec_data(arrival={"kind": "closed_loop", "concurrency": 0}), "concurrency must be >= 1"),
        (
            spec_data(arrival={"kind": "closed_loop", "concurrency": True}),
            "concurrency must be int",
        ),
        (spec_data(arrival={"kind": "poisson", "rate": True}), "rate must be int or float"),
        (spec_data(arrival={"kind": "poisson", "rate": 0}), "rate must be positive and finite"),
        (spec_data(arrival={"kind": "poisson", "rate": math.inf}), "rate must be positive"),
        (spec_data(arrival={"kind": "poisson", "rate": 10**400}), "rate must be positive"),
        (spec_data(arrival={"kind": "closed_loop", "concurrency": 2, "rate": 1}), "arrival must"),
        (spec_data(arrival={"kind": "poisson", "rate": 1, "concurrency": 2}), "arrival must"),
        (spec_data(arrival={"kind": "burst", "rate": 1}), "arrival must"),
        (spec_data(arrival="poisson"), "arrival must"),
    ],
)
def test_an_invalid_spec_is_rejected(data: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_spec(data)
