import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from itertools import accumulate
from pathlib import Path
from typing import Literal

type LengthRange = tuple[int, int]


@dataclass(frozen=True)
class ClosedLoop:
    """Every request arrives at 0.0; the driver keeps at most `concurrency` of them in flight."""

    kind: Literal["closed_loop"] = field(default="closed_loop", init=False)
    concurrency: int

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")


@dataclass(frozen=True)
class Poisson:
    """`rate` counts requests per second; the first request arrives at 0.0."""

    kind: Literal["poisson"] = field(default="poisson", init=False)
    rate: float

    def __post_init__(self) -> None:
        if not 0 < self.rate < math.inf:
            raise ValueError(f"rate must be positive and finite, got {self.rate}")


type Arrival = ClosedLoop | Poisson


@dataclass(frozen=True)
class WorkloadSpec:
    """Ranges are inclusive; requests pick `prompt_len` ranges and prefix groups round robin."""

    name: str
    seed: int
    vocab_size: int
    num_requests: int
    prompt_len: tuple[LengthRange, ...]
    output_len: LengthRange
    shared_prefix_len: int
    num_prefix_groups: int
    arrival: Arrival

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.vocab_size < 1 or self.num_requests < 1:
            raise ValueError(
                "vocab_size and num_requests must be >= 1, "
                f"got {self.vocab_size}, {self.num_requests}"
            )
        if not self.prompt_len:
            raise ValueError("prompt_len needs at least one range")
        for low, high in (*self.prompt_len, self.output_len):
            if not 1 <= low <= high:
                raise ValueError(f"length range ({low}, {high}) must have 1 <= low <= high")
        if self.shared_prefix_len < 0 or self.num_prefix_groups < 1:
            raise ValueError(
                "shared_prefix_len must be >= 0 and num_prefix_groups >= 1, "
                f"got {self.shared_prefix_len}, {self.num_prefix_groups}"
            )
        if self.num_prefix_groups > 1 and self.shared_prefix_len == 0:
            raise ValueError("num_prefix_groups > 1 requires a shared prefix")


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    output_len: int
    arrival_s: float


def generate_requests(spec: WorkloadSpec) -> list[RequestSpec]:
    rng = random.Random(spec.seed)
    vocabulary = range(spec.vocab_size)
    prefixes = [
        tuple(rng.choices(vocabulary, k=spec.shared_prefix_len))
        for _ in range(spec.num_prefix_groups)
    ]
    prompts: list[tuple[int, ...]] = []
    output_lens: list[int] = []
    for index in range(spec.num_requests):
        num_suffix_tokens = rng.randint(*spec.prompt_len[index % len(spec.prompt_len)])
        suffix = rng.choices(vocabulary, k=num_suffix_tokens)
        prompts.append(prefixes[index % spec.num_prefix_groups] + tuple(suffix))
        output_lens.append(rng.randint(*spec.output_len))
    if isinstance(spec.arrival, Poisson):
        gaps = [rng.expovariate(spec.arrival.rate) for _ in range(spec.num_requests - 1)]
    else:
        gaps = [0.0] * (spec.num_requests - 1)
    arrivals = [0.0, *accumulate(gaps)]
    return [
        RequestSpec(f"r{index}", prompt, output_len, arrival_s)
        for index, (prompt, output_len, arrival_s) in enumerate(
            zip(prompts, output_lens, arrivals, strict=True)
        )
    ]


def load_spec(path: Path) -> WorkloadSpec:
    return parse_spec(json.loads(path.read_bytes()))


def parse_spec(data: object) -> WorkloadSpec:
    if not isinstance(data, Mapping):
        raise ValueError(f"workload must be an object, got {data!r}")
    keys = {spec_field.name for spec_field in fields(WorkloadSpec)}
    if data.keys() != keys:
        raise ValueError(f"workload keys must be {sorted(keys)}, got {sorted(data)}")
    return WorkloadSpec(
        name=_typed(data["name"], "name", str),
        seed=_typed(data["seed"], "seed", int),
        vocab_size=_typed(data["vocab_size"], "vocab_size", int),
        num_requests=_typed(data["num_requests"], "num_requests", int),
        prompt_len=_length_ranges(data["prompt_len"], "prompt_len"),
        output_len=_length_range(data["output_len"], "output_len"),
        shared_prefix_len=_typed(data["shared_prefix_len"], "shared_prefix_len", int),
        num_prefix_groups=_typed(data["num_prefix_groups"], "num_prefix_groups", int),
        arrival=_arrival(data["arrival"]),
    )


def _typed[T](value: object, key: str, *types: type[T]) -> T:
    if not isinstance(value, types) or isinstance(value, bool):
        names = " or ".join(expected.__name__ for expected in types)
        raise ValueError(f"{key} must be {names}, got {value!r}")
    return value


def _length_range(value: object, key: str) -> LengthRange:
    match value:
        case [low, high]:
            return (_typed(low, key, int), _typed(high, key, int))
        case _:
            length = _typed(value, key, int)
            return (length, length)


def _length_ranges(value: object, key: str) -> tuple[LengthRange, ...]:
    match value:
        case [] | [[*_], *_]:
            return tuple(_length_range(item, key) for item in value)
        case _:
            return (_length_range(value, key),)


def _arrival(value: object) -> Arrival:
    match value:
        case {"kind": "closed_loop", "concurrency": concurrency, **rest} if not rest:
            return ClosedLoop(_typed(concurrency, "concurrency", int))
        case {"kind": "poisson", "rate": rate, **rest} if not rest:
            return Poisson(_typed(rate, "rate", int, float))
    raise ValueError(f"arrival must be a closed_loop or poisson object, got {value!r}")
