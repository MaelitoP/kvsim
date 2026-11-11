import argparse
import asyncio
import contextlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, replace
from itertools import cycle, islice
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

import httpx
from tokenizers import Tokenizer

from kvsim.metrics import summarize
from kvsim.simulator import ResultConfig
from kvsim.workload import (
    Arrival,
    ClosedLoop,
    RequestSpec,
    WorkloadSpec,
    generate_requests,
    load_spec,
)

type Engine = Literal["vllm", "lmdeploy"]


class MetricNames(NamedTuple):
    """Prometheus sample names of one engine; `None` where the engine publishes nothing."""

    kv_usage: str
    running: str
    waiting: str
    preemptions_total: str | None
    prefix_queries_total: str
    prefix_hits_total: str


METRIC_NAMES: Final[Mapping[Engine, MetricNames]] = {
    "vllm": MetricNames(
        kv_usage="vllm:kv_cache_usage_perc",
        running="vllm:num_requests_running",
        waiting="vllm:num_requests_waiting",
        preemptions_total="vllm:num_preemptions_total",
        prefix_queries_total="vllm:prefix_cache_queries_total",
        prefix_hits_total="vllm:prefix_cache_hits_total",
    ),
    "lmdeploy": MetricNames(
        kv_usage="lmdeploy:gpu_cache_usage_perc",
        running="lmdeploy:num_requests_running",
        waiting="lmdeploy:num_requests_waiting",
        preemptions_total=None,
        prefix_queries_total="lmdeploy:prompt_tokens_total",
        prefix_hits_total="lmdeploy:cached_tokens_total",
    ),
}

_METRIC_LINE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(\S+)(?:\s+-?\d+)?")

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class RequestRecord(TypedDict):
    """Times are wall-clock seconds since the measured pass started."""

    request_id: str
    arrival_s: float
    first_token_s: float
    finish_s: float
    prompt_tokens: int
    output_tokens: int
    server_ttft_ms: float | None
    server_queue_ms: float | None


class Sample(TypedDict):
    """One scrape of the engine's metrics; counters count from the start of the measured pass."""

    t_s: float
    kv_usage: float | None
    running: int | None
    waiting: int | None
    preemptions_total: int | None
    prefix_queries_total: int | None
    prefix_hits_total: int | None


def build_prompts(spec: WorkloadSpec, tokenizer: Tokenizer) -> list[str]:
    """One text prompt per request of the spec, each tokenizing to about the spec's length.

    The requests of a prefix group start with the same text, which tokenizes to about
    `shared_prefix_len` tokens. Raises `ValueError` when the spec draws token IDs the
    tokenizer does not have.
    """
    if spec.vocab_size > tokenizer.get_vocab_size():
        raise ValueError(
            f"vocab_size {spec.vocab_size} exceeds the tokenizer's {tokenizer.get_vocab_size()}"
        )
    requests = generate_requests(spec)
    prefix_len = spec.shared_prefix_len
    prefixes = [
        _fitted_text(tokenizer, "", request.prompt_token_ids[:prefix_len], prefix_len)
        for request in requests[: spec.num_prefix_groups]
    ]
    prompts: list[str] = []
    for index, request in enumerate(requests):
        prefix = prefixes[index % spec.num_prefix_groups]
        suffix_ids = request.prompt_token_ids[prefix_len:]
        prompts.append(
            prefix + _fitted_text(tokenizer, prefix, suffix_ids, len(request.prompt_token_ids))
        )
    return prompts


def _fitted_text(tokenizer: Tokenizer, prefix: str, token_ids: Sequence[int], target: int) -> str:
    def attempt(num_ids: int) -> tuple[int, str]:
        text = tokenizer.decode(list(islice(cycle(token_ids), num_ids)))
        return len(tokenizer.encode(prefix + text).ids) - target, text

    num_ids = len(token_ids)
    excess, text = attempt(num_ids)
    while excess:
        next_num_ids = max(num_ids - excess, 0)
        next_excess, next_text = attempt(next_num_ids)
        if abs(next_excess) >= abs(excess):
            between = range(min(num_ids, next_num_ids) + 1, max(num_ids, next_num_ids))
            attempts = [(excess, text), *(attempt(count) for count in between)]
            return min(attempts, key=lambda attempt: abs(attempt[0]))[1]
        num_ids, excess, text = next_num_ids, next_excess, next_text
    return text


def parse_metrics(text: str) -> dict[str, float]:
    """The value of every sample name of a Prometheus exposition, summed over its labels."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if sample_line := _METRIC_LINE.fullmatch(line):
            name, value = sample_line.groups()
            totals[name] = totals.get(name, 0.0) + float(value)
    return totals


async def measure(
    client: httpx.AsyncClient,
    engine: Engine,
    model: str,
    spec: WorkloadSpec,
    tokenizer: Tokenizer,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    scrape_interval_s: float = 1.0,
) -> tuple[list[RequestRecord], list[Sample]]:
    """Warm the engine up with the spec's next seed, then measure the spec itself.

    The first sample is the scrape taken before the measured pass, at 0.0 with zeroed
    counters; the last one is taken after the last request finished. A request or a scrape
    that fails in either pass raises an `ExceptionGroup` holding a `RuntimeError` or an
    `httpx.HTTPError`, after the other requests are cancelled.
    """
    names = METRIC_NAMES[engine]
    warm_up_spec = replace(spec, seed=spec.seed + 1)
    warm_up_requests = generate_requests(warm_up_spec)
    warm_up_prompts = build_prompts(warm_up_spec, tokenizer)
    requests = generate_requests(spec)
    prompts = build_prompts(spec, tokenizer)
    await _send_all(
        client, model, spec.arrival, warm_up_requests, warm_up_prompts, _stopwatch(clock), sleep
    )
    baseline = await _scrape(client)
    elapsed = _stopwatch(clock)
    finished = asyncio.Event()
    async with asyncio.TaskGroup() as group:
        scrapes = group.create_task(
            _scrape_until(finished, client, names, baseline, elapsed, scrape_interval_s)
        )
        records = await _send_all(client, model, spec.arrival, requests, prompts, elapsed, sleep)
        finished.set()
    return records, [_sample(names, baseline, baseline, 0.0), *scrapes.result()]


def _stopwatch(clock: Callable[[], float]) -> Callable[[], float]:
    start_s = clock()
    return lambda: clock() - start_s


async def _send_all(
    client: httpx.AsyncClient,
    model: str,
    arrival: Arrival,
    requests: list[RequestSpec],
    prompts: list[str],
    elapsed: Callable[[], float],
    sleep: Callable[[float], Awaitable[object]],
) -> list[RequestRecord]:
    slots = asyncio.Semaphore(
        arrival.concurrency if isinstance(arrival, ClosedLoop) else len(requests)
    )
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(_send(client, model, request, prompt, elapsed, sleep, slots))
            for request, prompt in zip(requests, prompts, strict=True)
        ]
    return [task.result() for task in tasks]


async def _send(
    client: httpx.AsyncClient,
    model: str,
    request: RequestSpec,
    prompt: str,
    elapsed: Callable[[], float],
    sleep: Callable[[float], Awaitable[object]],
    slots: asyncio.Semaphore,
) -> RequestRecord:
    request_id = request.request_id
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": request.output_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    token_times_s: list[float] = []
    usage: tuple[int, int] | None = None
    server_metrics: object = None
    async with slots:
        await sleep(max(request.arrival_s - elapsed(), 0.0))
        arrival_s = elapsed()
        async with client.stream("POST", "/v1/completions", json=body) as response:
            if response.is_error:
                reason = (await response.aread()).decode(errors="replace")
                raise RuntimeError(f"{request_id}: HTTP {response.status_code}: {reason}")
            async for line in response.aiter_lines():
                try:
                    event = _parse_event(line)
                except ValueError as error:
                    raise RuntimeError(f"{request_id}: unreadable event {line!r}") from error
                if event.get("error") is not None:
                    raise RuntimeError(f"{request_id}: {event['error']}")
                if event.get("choices"):
                    token_times_s.append(elapsed())
                match event.get("usage"):
                    case {
                        "prompt_tokens": int() as prompt_tokens,
                        "completion_tokens": int() as output_tokens,
                    }:
                        usage = (prompt_tokens, output_tokens)
                        server_metrics = event.get("metrics")
    if not token_times_s or usage is None:
        raise RuntimeError(f"{request_id}: the stream ended without a token or its usage")
    if usage[1] != request.output_len:
        raise RuntimeError(
            f"{request_id}: the engine generated {usage[1]} tokens, not {request.output_len}"
        )
    return {
        "request_id": request_id,
        "arrival_s": arrival_s,
        "first_token_s": token_times_s[0],
        "finish_s": token_times_s[-1],
        "prompt_tokens": usage[0],
        "output_tokens": usage[1],
        "server_ttft_ms": _server_ms(server_metrics, "time_to_first_token_ms"),
        "server_queue_ms": _server_ms(server_metrics, "queue_time_ms"),
    }


def _parse_event(line: str) -> dict[str, object]:
    if not line.startswith("data:"):
        return {}
    payload = line.removeprefix("data:").strip()
    event = {} if payload == "[DONE]" else json.loads(payload)
    return event if isinstance(event, dict) else {}


def _server_ms(server_metrics: object, key: str) -> float | None:
    if isinstance(server_metrics, Mapping):
        value = server_metrics.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


async def _scrape(client: httpx.AsyncClient) -> dict[str, float]:
    response = await client.get("/metrics", timeout=10.0)
    response.raise_for_status()
    return parse_metrics(response.text)


async def _scrape_until(
    finished: asyncio.Event,
    client: httpx.AsyncClient,
    names: MetricNames,
    baseline: Mapping[str, float],
    elapsed: Callable[[], float],
    interval_s: float,
) -> list[Sample]:
    samples: list[Sample] = []
    while True:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(interval_s):
                await finished.wait()
        is_last = finished.is_set()
        samples.append(_sample(names, baseline, await _scrape(client), elapsed()))
        if is_last:
            return samples


def _sample(
    names: MetricNames, baseline: Mapping[str, float], metrics: Mapping[str, float], t_s: float
) -> Sample:
    kv_usage = metrics.get(names.kv_usage)
    if kv_usage is not None and not 0 <= kv_usage <= 1:
        raise RuntimeError(f"{names.kv_usage} must be within 0 and 1, got {kv_usage}")

    def since_baseline(name: str | None) -> int | None:
        if name is None or name not in metrics:
            return None
        increase = round(metrics[name] - baseline.get(name, 0.0))
        if increase < 0:
            raise RuntimeError(f"{name} fell below its value at the start of the measured pass")
        return increase

    return {
        "t_s": t_s,
        "kv_usage": kv_usage,
        "running": _rounded(metrics.get(names.running)),
        "waiting": _rounded(metrics.get(names.waiting)),
        "preemptions_total": since_baseline(names.preemptions_total),
        "prefix_queries_total": since_baseline(names.prefix_queries_total),
        "prefix_hits_total": since_baseline(names.prefix_hits_total),
    }


def _rounded(value: float | None) -> int | None:
    return None if value is None else round(value)


def main(
    argv: Sequence[str] | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> None:
    parser = argparse.ArgumentParser(
        description="Measure an OpenAI-compatible engine under a kvsim workload."
    )
    parser.add_argument("spec", type=Path, help="workload spec, a JSON file")
    parser.add_argument("--out", type=Path, required=True, help="result file to write")
    parser.add_argument("--engine", choices=sorted(METRIC_NAMES), required=True)
    parser.add_argument("--base-url", required=True, help="for example http://localhost:8000")
    parser.add_argument("--model", required=True, help="model name the engine serves")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer.json of the model; fetched from the Hugging Face Hub when omitted",
    )
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--command", required=True, help="command that launched the engine")
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument(
        "--enable-chunked-prefill", action=argparse.BooleanOptionalAction, required=True
    )
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        tokenizer = (
            Tokenizer.from_pretrained(args.model)
            if args.tokenizer is None
            else Tokenizer.from_file(str(args.tokenizer))
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        records, samples = asyncio.run(
            _measure_engine(args.base_url, transport, args.engine, args.model, spec, tokenizer)
        )
        config: ResultConfig = {
            "block_size": args.block_size,
            "num_blocks": args.num_blocks,
            "kv_tokens": args.num_blocks * args.block_size,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enable_chunked_prefill": args.enable_chunked_prefill,
            "long_prefill_token_threshold": args.long_prefill_token_threshold,
            "command": args.command,
        }
        result = {
            "source": args.engine,
            "measured": True,
            "engine_version": args.engine_version,
            "model": args.model,
            "gpu": args.gpu,
            "config": config,
            "workload": asdict(spec),
            "requests": records,
            "samples": samples,
        }
        text = json.dumps({**result, "summary": summarize(result)}, indent=2, allow_nan=False)
        args.out.write_text(text + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))


async def _measure_engine(
    base_url: str,
    transport: httpx.AsyncBaseTransport | None,
    engine: Engine,
    model: str,
    spec: WorkloadSpec,
    tokenizer: Tokenizer,
) -> tuple[list[RequestRecord], list[Sample]]:
    async with httpx.AsyncClient(
        base_url=base_url,
        transport=transport,
        timeout=_TIMEOUT,
        limits=httpx.Limits(max_connections=None),
    ) as client:
        return await measure(client, engine, model, spec, tokenizer)


if __name__ == "__main__":
    main()
