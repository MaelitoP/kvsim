import math
from statistics import fmean, quantiles
from typing import NamedTuple, TypedDict


class LatencyStats(TypedDict):
    mean: float
    p50: float
    p95: float
    p99: float


class Summary(TypedDict):
    """Seconds are measured for an engine result and estimated for a simulator result."""

    num_requests: int
    output_tokens: int
    duration_s: float
    requests_per_s: float
    output_tokens_per_s: float
    ttft_s: LatencyStats
    tpot_s: LatencyStats | None
    e2e_s: LatencyStats
    kv_usage_mean: float | None
    kv_usage_max: float | None
    preemptions: int | None
    prefix_hit_rate: float | None


class _RequestRecord(NamedTuple):
    arrival_s: float
    first_token_s: float
    finish_s: float
    output_tokens: int


class _Sample(NamedTuple):
    kv_usage: float | None
    preemptions_total: int | None
    prefix_queries_total: int | None
    prefix_hits_total: int | None


def summarize(result: object) -> Summary:
    match result:
        case {"requests": [_, *_] as requests, "samples": [*samples]}:
            try:
                return _summarize(
                    [_read_request(record) for record in requests],
                    [_read_sample(sample) for sample in samples],
                )
            except OverflowError as error:
                raise ValueError(
                    "the times of a result are too large or too close together to summarize"
                ) from error
    raise ValueError("a result needs a non-empty list of requests and a list of samples")


def _summarize(requests: list[_RequestRecord], samples: list[_Sample]) -> Summary:
    duration_s = _finite(
        max(request.finish_s for request in requests)
        - min(request.arrival_s for request in requests)
    )
    if duration_s == 0:
        raise ValueError("a result must span a positive duration")
    num_output_tokens = sum(request.output_tokens for request in requests)
    tpots_s = [
        (request.finish_s - request.first_token_s) / (request.output_tokens - 1)
        for request in requests
        if request.output_tokens > 1
    ]
    kv_usages = [sample.kv_usage for sample in samples if sample.kv_usage is not None]
    preemptions = [
        sample.preemptions_total for sample in samples if sample.preemptions_total is not None
    ]
    prefix_hit_rates = [
        sample.prefix_hits_total / sample.prefix_queries_total
        for sample in samples
        if sample.prefix_hits_total is not None
        and sample.prefix_queries_total is not None
        and sample.prefix_queries_total > 0
    ]
    return {
        "num_requests": len(requests),
        "output_tokens": num_output_tokens,
        "duration_s": duration_s,
        "requests_per_s": _finite(len(requests) / duration_s),
        "output_tokens_per_s": _finite(num_output_tokens / duration_s),
        "ttft_s": _latency_stats(
            [request.first_token_s - request.arrival_s for request in requests]
        ),
        "tpot_s": _latency_stats(tpots_s) if tpots_s else None,
        "e2e_s": _latency_stats([request.finish_s - request.arrival_s for request in requests]),
        "kv_usage_mean": fmean(kv_usages) if kv_usages else None,
        "kv_usage_max": max(kv_usages) if kv_usages else None,
        "preemptions": preemptions[-1] if preemptions else None,
        "prefix_hit_rate": prefix_hit_rates[-1] if prefix_hit_rates else None,
    }


def _read_request(record: object) -> _RequestRecord:
    match record:
        case {
            "arrival_s": int() | float() as arrival_s,
            "first_token_s": int() | float() as first_token_s,
            "finish_s": int() | float() as finish_s,
            "output_tokens": int() as output_tokens,
        } if (
            not _any_bool(arrival_s, first_token_s, finish_s, output_tokens)
            and -math.inf < arrival_s <= first_token_s <= finish_s < math.inf
            and output_tokens >= 1
        ):
            return _RequestRecord(
                float(arrival_s), float(first_token_s), float(finish_s), output_tokens
            )
    raise ValueError(
        "a request needs finite arrival_s <= first_token_s <= finish_s "
        f"and output_tokens >= 1, got {record!r}"
    )


def _read_sample(sample: object) -> _Sample:
    match sample:
        case {
            "kv_usage": int() | float() | None as kv_usage,
            "preemptions_total": int() | None as preemptions_total,
            "prefix_queries_total": int() | None as prefix_queries_total,
            "prefix_hits_total": int() | None as prefix_hits_total,
        } if (
            not _any_bool(kv_usage, preemptions_total, prefix_queries_total, prefix_hits_total)
            and (kv_usage is None or 0 <= kv_usage <= 1)
            and (preemptions_total is None or preemptions_total >= 0)
            and (prefix_queries_total is None or prefix_queries_total >= 0)
            and (prefix_hits_total is None or prefix_hits_total >= 0)
            and (
                prefix_queries_total is None
                or prefix_hits_total is None
                or prefix_hits_total <= prefix_queries_total
            )
        ):
            return _Sample(kv_usage, preemptions_total, prefix_queries_total, prefix_hits_total)
    raise ValueError(
        "a sample needs a kv_usage within 0 and 1, counters that are integers >= 0 and "
        f"prefix_hits_total <= prefix_queries_total, each possibly null, got {sample!r}"
    )


def _any_bool(*values: object) -> bool:
    return any(isinstance(value, bool) for value in values)


def _latency_stats(values: list[float]) -> LatencyStats:
    mean = fmean(values)
    if len(values) == 1:
        return {"mean": mean, "p50": mean, "p95": mean, "p99": mean}
    percentiles = quantiles(values, n=100, method="inclusive")
    return {
        "mean": mean,
        "p50": _finite(percentiles[49]),
        "p95": _finite(percentiles[94]),
        "p99": _finite(percentiles[98]),
    }


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise OverflowError(value)
    return value
