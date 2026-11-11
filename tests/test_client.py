import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given
from tokenizers import Tokenizer, decoders, models

from client import (
    METRIC_NAMES,
    Engine,
    RequestRecord,
    Sample,
    _sample,
    build_prompts,
    measure,
    parse_metrics,
)
from client import main as client_main
from invariants import check_measurement
from kvsim.__main__ import main
from kvsim.metrics import summarize
from kvsim.workload import ClosedLoop, Poisson, WorkloadSpec, generate_requests
from strategies import specs

VOCAB = {"a": 0, "b": 1, "c": 2, "d": 3, "ab": 4, "cd": 5}

SPEC = WorkloadSpec(
    name="test",
    seed=0,
    vocab_size=len(VOCAB),
    num_requests=4,
    prompt_len=((6, 10),),
    output_len=(2, 3),
    shared_prefix_len=8,
    num_prefix_groups=2,
    arrival=ClosedLoop(concurrency=1),
)

VLLM_METRICS = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 0.375
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 3.0
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 2.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 9.0
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 5.0
vllm:prefix_cache_queries_total{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 4096.0
vllm:prefix_cache_hits_total{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct"} 1024.0
"""

LMDEPLOY_METRICS = """\
lmdeploy:gpu_cache_usage_perc{engine="0",model_name="qwen"} 0.625
lmdeploy:num_requests_running{engine="0",model_name="qwen"} 4.0
lmdeploy:num_requests_waiting{engine="0",model_name="qwen"} 1.0
lmdeploy:num_api_requests_waiting{engine="0",model_name="qwen"} 9.0
lmdeploy:prompt_tokens_total{engine="0",model_name="qwen"} 2048.0
lmdeploy:generation_tokens_total{engine="0",model_name="qwen"} 512.0
lmdeploy:cached_tokens_total{engine="0",model_name="qwen"} 256.0
"""


def make_tokenizer() -> Tokenizer:
    """Decoded tokens do not tokenize back to themselves: `a b` merges and `cd` splits."""
    tokenizer = Tokenizer(models.BPE(vocab=VOCAB, merges=[("a", "b")]))
    tokenizer.decoder = decoders.Fuse()
    return tokenizer


TOKENIZER = make_tokenizer()


def num_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text).ids)


def event(payload: object) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class FakeTime:
    """Every reading of the clock takes one second and a sleep takes exactly its delay."""

    def __init__(self) -> None:
        self.now_s = 0.0

    def clock(self) -> float:
        self.now_s += 1.0
        return self.now_s

    async def sleep(self, delay_s: float) -> None:
        self.now_s += delay_s


class FakeEngine:
    """Streams one event per requested token, then the usage event, in the engine's shape.

    vLLM starts with a keep-alive comment and reports per-request metrics; LMDeploy adds a
    null usage to every event and an empty event that carries the finish reason. Stream `i`
    of a pass of `pass_size` requests waits until `hold - 1` later requests of its pass have
    arrived, so `hold` requests overlap whenever the client sends that many. Completions
    after the first `num_served` get a 503.
    """

    def __init__(
        self,
        engine: Engine = "vllm",
        *,
        hold: int = 1,
        pass_size: int = SPEC.num_requests,
        num_served: int | None = None,
        publishes_kv_usage: bool = True,
    ) -> None:
        self.engine = engine
        self.hold = hold
        self.pass_size = pass_size
        self.num_served = num_served
        self.publishes_kv_usage = publishes_kv_usage
        self.arrived = asyncio.Condition()
        self.bodies: list[dict[str, Any]] = []
        self.num_in_flight = 0
        self.max_in_flight = 0
        self.prompt_tokens_total = 0
        self.cached_tokens_total = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=self.metrics())
        assert (request.method, request.url.path) == ("POST", "/v1/completions")
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.num_served is not None and len(self.bodies) > self.num_served:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, content=self.stream(len(self.bodies) - 1, body))

    async def stream(self, index: int, body: dict[str, Any]) -> AsyncIterator[bytes]:
        position = index % self.pass_size
        num_needed = index - position + min(position + self.hold, self.pass_size)
        self.num_in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.num_in_flight)
        try:
            async with self.arrived:
                self.arrived.notify_all()
                await self.arrived.wait_for(lambda: len(self.bodies) >= num_needed)
            usage = {
                "prompt_tokens": num_tokens(body["prompt"]),
                "completion_tokens": body["max_tokens"],
            }
            self.prompt_tokens_total += usage["prompt_tokens"]
            self.cached_tokens_total += usage["prompt_tokens"] // 2
            if self.engine == "vllm":
                yield b": keep-alive\n\n"
                for _ in range(body["max_tokens"]):
                    yield event({"choices": [{"index": 0, "text": "x", "finish_reason": None}]})
                metrics = {"time_to_first_token_ms": 12.5, "queue_time_ms": 3}
                yield event({"choices": [], "usage": usage, "metrics": metrics})
            else:
                for _ in range(body["max_tokens"]):
                    yield event({"choices": [{"index": 0, "text": "x"}], "usage": None})
                final_choice = {"index": 0, "text": "", "finish_reason": "length"}
                yield event({"choices": [final_choice], "usage": None})
                yield event({"choices": [], "usage": usage})
            yield b"data: [DONE]\n\n"
        finally:
            self.num_in_flight -= 1

    def metrics(self) -> str:
        names = METRIC_NAMES[self.engine]
        lines = [
            f'{names.running}{{engine="0"}} {self.num_in_flight}.0',
            f'{names.waiting}{{engine="0"}} 0.0',
            f'{names.prefix_queries_total}{{engine="0"}} {self.prompt_tokens_total}.0',
            f'{names.prefix_hits_total}{{engine="0"}} {self.cached_tokens_total}.0',
        ]
        if self.publishes_kv_usage:
            lines.append(f'{names.kv_usage}{{engine="0"}} 0.25')
        if names.preemptions_total is not None:
            lines.append(f'{names.preemptions_total}{{engine="0"}} 7.0')
        return "\n".join(lines) + "\n"


def run_measure(
    transport: httpx.MockTransport,
    spec: WorkloadSpec,
    engine: Engine = "vllm",
    scrape_interval_s: float = 3600.0,
) -> tuple[list[RequestRecord], list[Sample]]:
    fake_time = FakeTime()

    async def measure_with_deadline() -> tuple[list[RequestRecord], list[Sample]]:
        async with (
            asyncio.timeout(10),
            httpx.AsyncClient(transport=transport, base_url="http://engine") as client,
        ):
            return await measure(
                client,
                engine,
                "model",
                spec,
                TOKENIZER,
                clock=fake_time.clock,
                sleep=fake_time.sleep,
                scrape_interval_s=scrape_interval_s,
            )

    return asyncio.run(measure_with_deadline())


@given(spec=specs().map(lambda spec: replace(spec, vocab_size=len(VOCAB))))
def test_prompts_tokenize_to_their_spec_lengths_and_share_their_group_prefix(
    spec: WorkloadSpec,
) -> None:
    requests = generate_requests(spec)
    prompts = build_prompts(spec, TOKENIZER)
    assert len(prompts) == len(requests)
    for index, (request, prompt) in enumerate(zip(requests, prompts, strict=True)):
        assert abs(num_tokens(prompt) - len(request.prompt_token_ids)) <= 1
        prefix = prompts[index % spec.num_prefix_groups]
        shared_text = next(
            prompt[:end] for end in range(len(prompt), -1, -1) if prefix.startswith(prompt[:end])
        )
        assert num_tokens(shared_text) >= spec.shared_prefix_len - 1


@pytest.mark.parametrize(("seed", "is_trimmed"), [(3, True), (39, False)])
def test_a_prompt_is_trimmed_or_extended_from_its_source_tokens(
    seed: int, is_trimmed: bool
) -> None:
    spec = replace(SPEC, num_requests=1, shared_prefix_len=0, num_prefix_groups=1, seed=seed)
    token_ids = generate_requests(spec)[0].prompt_token_ids
    decoded = TOKENIZER.decode(list(token_ids))
    assert (num_tokens(decoded) > len(token_ids)) == is_trimmed
    (prompt,) = build_prompts(spec, TOKENIZER)
    assert num_tokens(prompt) == len(token_ids)
    assert prompt != decoded
    assert decoded.startswith(prompt) if is_trimmed else prompt.startswith(decoded)


def test_a_spec_with_token_ids_the_tokenizer_lacks_is_rejected() -> None:
    with pytest.raises(ValueError, match="vocab_size 7 exceeds the tokenizer's 6"):
        build_prompts(replace(SPEC, vocab_size=7), TOKENIZER)


@pytest.mark.parametrize(
    ("engine", "ticks_after_last_token", "server_ttft_ms", "server_queue_ms"),
    [("vllm", 0, 12.5, 3.0), ("lmdeploy", 1, None, None)],
)
def test_a_record_holds_the_times_of_its_token_events_and_the_engine_usage(
    engine: Engine,
    ticks_after_last_token: int,
    server_ttft_ms: float | None,
    server_queue_ms: float | None,
) -> None:
    records, samples = run_measure(FakeEngine(engine).transport(), SPEC, engine)
    check_measurement(SPEC, records, samples)
    prompts = build_prompts(SPEC, TOKENIZER)
    for record, request, prompt in zip(records, generate_requests(SPEC), prompts, strict=True):
        assert record["first_token_s"] - record["arrival_s"] == 1.0
        assert (
            record["finish_s"] - record["first_token_s"]
            == request.output_len - 1 + ticks_after_last_token
        )
        assert record["prompt_tokens"] == num_tokens(prompt)
        assert record["server_ttft_ms"] == server_ttft_ms
        assert record["server_queue_ms"] == server_queue_ms
    assert [sample["t_s"] for sample in samples] == [0.0, records[-1]["finish_s"] + 1.0]


def test_the_warm_up_pass_sends_the_next_seed_and_is_not_recorded() -> None:
    engine = FakeEngine()
    records, _ = run_measure(engine.transport(), SPEC)
    warm_up_spec = replace(SPEC, seed=SPEC.seed + 1)
    prompts = build_prompts(SPEC, TOKENIZER)
    requests = generate_requests(SPEC)
    assert [body["prompt"] for body in engine.bodies] == [
        *build_prompts(warm_up_spec, TOKENIZER),
        *prompts,
    ]
    assert [body["max_tokens"] for body in engine.bodies] == [
        request.output_len for request in generate_requests(warm_up_spec) + requests
    ]
    assert engine.bodies[-1] == {
        "model": "model",
        "prompt": prompts[-1],
        "max_tokens": requests[-1].output_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert [record["request_id"] for record in records] == ["r0", "r1", "r2", "r3"]


@pytest.mark.parametrize(("engine", "preemptions"), [("vllm", 0), ("lmdeploy", None)])
def test_counters_count_from_the_measured_pass_and_a_missing_metric_is_null(
    engine: Engine, preemptions: int | None
) -> None:
    fake = FakeEngine(engine, publishes_kv_usage=engine == "vllm")
    records, samples = run_measure(fake.transport(), SPEC, engine)
    check_measurement(SPEC, records, samples)
    prompt_lens = [record["prompt_tokens"] for record in records]
    assert samples == [
        {
            "t_s": 0.0,
            "kv_usage": 0.25 if engine == "vllm" else None,
            "running": 0,
            "waiting": 0,
            "preemptions_total": preemptions,
            "prefix_queries_total": 0,
            "prefix_hits_total": 0,
        },
        {
            "t_s": samples[-1]["t_s"],
            "kv_usage": 0.25 if engine == "vllm" else None,
            "running": 0,
            "waiting": 0,
            "preemptions_total": preemptions,
            "prefix_queries_total": sum(prompt_lens),
            "prefix_hits_total": sum(prompt_len // 2 for prompt_len in prompt_lens),
        },
    ]


@pytest.mark.parametrize(
    ("engine", "text", "sample"),
    [
        (
            "vllm",
            VLLM_METRICS,
            {
                "t_s": 2.5,
                "kv_usage": 0.375,
                "running": 3,
                "waiting": 2,
                "preemptions_total": 5,
                "prefix_queries_total": 4096,
                "prefix_hits_total": 1024,
            },
        ),
        (
            "lmdeploy",
            LMDEPLOY_METRICS,
            {
                "t_s": 2.5,
                "kv_usage": 0.625,
                "running": 4,
                "waiting": 1,
                "preemptions_total": None,
                "prefix_queries_total": 2048,
                "prefix_hits_total": 256,
            },
        ),
    ],
)
def test_a_sample_reads_the_metrics_each_engine_publishes(
    engine: Engine, text: str, sample: Sample
) -> None:
    assert _sample(METRIC_NAMES[engine], {}, parse_metrics(text), 2.5) == sample


@pytest.mark.parametrize(
    ("baseline", "metrics", "message"),
    [
        ({}, {"vllm:kv_cache_usage_perc": 37.5}, "vllm:kv_cache_usage_perc must be within 0"),
        ({}, {"vllm:kv_cache_usage_perc": float("nan")}, "must be within 0 and 1, got nan"),
        (
            {"vllm:num_preemptions_total": 5.0},
            {"vllm:num_preemptions_total": 4.0},
            "vllm:num_preemptions_total fell below its value at the start",
        ),
    ],
)
def test_a_reading_no_summary_accepts_fails_at_its_scrape(
    baseline: dict[str, float], metrics: dict[str, float], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _sample(METRIC_NAMES["vllm"], baseline, metrics, 1.0)


@pytest.mark.parametrize(
    ("arrival", "max_in_flight"),
    [
        (ClosedLoop(concurrency=1), 1),
        (ClosedLoop(concurrency=3), 3),
        (ClosedLoop(concurrency=8), 6),
        (Poisson(rate=1e9), 6),
    ],
)
def test_the_arrival_model_bounds_the_requests_in_flight(
    arrival: ClosedLoop | Poisson, max_in_flight: int
) -> None:
    spec = replace(SPEC, num_requests=6, arrival=arrival)
    engine = FakeEngine(hold=max_in_flight, pass_size=spec.num_requests)
    records, samples = run_measure(engine.transport(), spec, scrape_interval_s=0.0)
    check_measurement(spec, records, samples)
    assert engine.max_in_flight == max_in_flight
    assert len(samples) > 2


def test_a_poisson_request_is_sent_at_its_arrival_time() -> None:
    spec = replace(SPEC, arrival=Poisson(rate=0.01))
    records, samples = run_measure(FakeEngine().transport(), spec)
    check_measurement(spec, records, samples)
    requests = generate_requests(spec)
    assert all(later.arrival_s - earlier.arrival_s > 10 for earlier, later in pairwise(requests))
    assert [record["arrival_s"] for record in records[1:]] == pytest.approx(
        [request.arrival_s + 1.0 for request in requests[1:]]
    )


def scripted_transport(status: int, *events: bytes) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(status, content=b"".join(events))

    return httpx.MockTransport(handle)


TOKEN = event({"choices": [{"index": 0, "text": "x"}]})
USAGE = event({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})


@pytest.mark.parametrize(
    ("status", "events", "message"),
    [
        (400, [b'{"error": {"message": "too long"}}'], "r0: HTTP 400: .*too long"),
        (200, [TOKEN, event({"error": {"message": "out of memory"}})], "r0: .*out of memory"),
        (200, [TOKEN, b"data: {not json\n\n"], "r0: unreadable event 'data: {not json'"),
        (200, [TOKEN, b"data: [DONE]\n\n"], "r0: the stream ended without a token or its usage"),
        (200, [USAGE, b"data: [DONE]\n\n"], "r0: the stream ended without a token or its usage"),
        (200, [TOKEN, USAGE], "r0: the engine generated 1 tokens, not [23]"),
    ],
)
def test_a_request_the_engine_fails_or_cuts_short_fails_the_run(
    status: int, events: list[bytes], message: str
) -> None:
    with pytest.raises(ExceptionGroup) as raised:
        run_measure(scripted_transport(status, *events), SPEC)
    assert raised.group_contains(RuntimeError, match=message)


def test_a_failure_in_the_measured_pass_cancels_the_other_requests_and_the_scrapes() -> None:
    spec = replace(SPEC, arrival=ClosedLoop(concurrency=2))
    engine = FakeEngine(hold=2, num_served=spec.num_requests + 1)
    with pytest.raises(ExceptionGroup) as raised:
        run_measure(engine.transport(), spec, scrape_interval_s=0.0)
    assert raised.group_contains(RuntimeError, match="r1: HTTP 503: overloaded")
    assert engine.num_in_flight == 0


def test_metric_samples_are_summed_over_their_labels_under_their_exact_name() -> None:
    text = "\n".join(
        [
            "# HELP vllm:num_requests_waiting Number of requests waiting.",
            "# TYPE vllm:num_requests_waiting gauge",
            'vllm:num_requests_waiting{engine="0",model_name="a b}c"} 2.0',
            'vllm:num_requests_waiting{engine="1",model_name="a b}c"} 3.0',
            'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 4.0',
            "vllm:kv_cache_usage_perc 0.5 1700000000000",
            "process_open_fds 1e+02",
            "",
        ]
    )
    assert parse_metrics(text) == {
        "vllm:num_requests_waiting": 5.0,
        "vllm:num_requests_waiting_by_reason": 4.0,
        "vllm:kv_cache_usage_perc": 0.5,
        "process_open_fds": 100.0,
    }


def client_argv(directory: Path, out_path: Path) -> list[str]:
    spec_path = directory / "spec.json"
    spec_path.write_text(json.dumps(asdict(SPEC)), encoding="utf-8")
    tokenizer_path = directory / "tokenizer.json"
    TOKENIZER.save(str(tokenizer_path))
    flags = (
        "--engine vllm --base-url http://engine --model model --engine-version 0.32.0 "
        "--num-blocks 256 --block-size 64 --max-num-seqs 64 --max-num-batched-tokens 2048 "
        "--enable-prefix-caching --no-enable-chunked-prefill --long-prefill-token-threshold 512"
    )
    return [
        str(spec_path),
        "--out",
        str(out_path),
        "--tokenizer",
        str(tokenizer_path),
        "--gpu",
        "NVIDIA L4",
        "--command",
        "vllm serve model",
        *flags.split(),
    ]


def test_the_cli_writes_a_measured_result_that_kvsim_summarizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "results" / "vllm" / "test.json"
    client_main(client_argv(tmp_path, out_path), transport=FakeEngine().transport())
    result = json.loads(out_path.read_text(encoding="utf-8"))
    check_measurement(SPEC, result["requests"], result["samples"])
    assert result.pop("summary") == summarize(result)
    del result["requests"], result["samples"]
    assert result == {
        "source": "vllm",
        "measured": True,
        "engine_version": "0.32.0",
        "model": "model",
        "gpu": "NVIDIA L4",
        "config": {
            "block_size": 64,
            "num_blocks": 256,
            "kv_tokens": 16384,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": False,
            "long_prefill_token_threshold": 512,
            "command": "vllm serve model",
        },
        "workload": json.loads(json.dumps(asdict(SPEC))),
    }
    main(["summarize", str(out_path)])
    header, row = capsys.readouterr().out.splitlines()
    cells = dict(zip(header.split(), row.split(), strict=True))
    assert (cells["source"], cells["times"], cells["num_requests"]) == ("vllm", "measured", "4")
    assert (cells["kv_usage_max"], cells["preemptions"]) == ("0.250", "0")


def test_the_cli_rejects_an_output_path_it_cannot_create_before_measuring(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    engine = FakeEngine()
    with pytest.raises(SystemExit) as raised:
        client_main(client_argv(tmp_path, blocker / "result.json"), transport=engine.transport())
    assert raised.value.code == 2
    assert str(blocker) in capsys.readouterr().err
    assert engine.bodies == []
