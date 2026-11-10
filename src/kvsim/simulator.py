import math
import zlib
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TypedDict

from kvsim import __version__
from kvsim.request import Request
from kvsim.scheduler import Scheduler, SchedulerConfig
from kvsim.workload import (
    Arrival,
    ClosedLoop,
    Poisson,
    RequestSpec,
    WorkloadSpec,
    generate_requests,
)


@dataclass(frozen=True)
class CostModel:
    """A request scheduled for one token is a decode sequence; other tokens are prefill tokens."""

    step_overhead_s: float = 0.02
    prefill_s_per_token: float = 0.0002
    decode_s_per_seq: float = 0.0005

    def __post_init__(self) -> None:
        if not 0 < self.step_overhead_s < math.inf:
            raise ValueError(f"step_overhead_s must be > 0 and finite, got {self.step_overhead_s}")
        if not (0 <= self.prefill_s_per_token < math.inf and 0 <= self.decode_s_per_seq < math.inf):
            raise ValueError(
                "prefill_s_per_token and decode_s_per_seq must be >= 0 and finite, "
                f"got {self.prefill_s_per_token}, {self.decode_s_per_seq}"
            )

    def step_duration(self, num_scheduled_tokens: Iterable[int]) -> float:
        token_counts = list(num_scheduled_tokens)
        num_decode_seqs = token_counts.count(1)
        return (
            self.step_overhead_s
            + self.prefill_s_per_token * (sum(token_counts) - num_decode_seqs)
            + self.decode_s_per_seq * num_decode_seqs
        )


class ResultConfig(TypedDict):
    block_size: int
    num_blocks: int
    kv_tokens: int
    max_num_seqs: int
    max_num_batched_tokens: int
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    long_prefill_token_threshold: int
    command: str | None


class RequestRecord(TypedDict):
    request_id: str
    arrival_s: float
    first_token_s: float
    finish_s: float
    first_token_step: int
    finish_step: int
    prompt_tokens: int
    output_tokens: int
    server_ttft_ms: float | None
    server_queue_ms: float | None


class Sample(TypedDict):
    """The state a step runs with, taken after `schedule()`; `t_s` is the step's start."""

    step: int
    t_s: float
    kv_usage: float
    running: int
    waiting: int
    preemptions_total: int
    prefix_queries_total: int
    prefix_hits_total: int


class Result(TypedDict):
    source: str
    measured: bool
    engine_version: str
    model: str | None
    gpu: str | None
    config: ResultConfig
    cost_model: dict[str, float]
    workload: dict[str, object]
    requests: list[RequestRecord]
    samples: list[Sample]


def simulate(spec: WorkloadSpec, config: SchedulerConfig, cost_model: CostModel) -> Result:
    scheduler = Scheduler(config)
    pending = deque(generate_requests(spec))
    in_flight: dict[str, Request] = {}
    submitted: list[tuple[Request, float]] = []
    step_ends_s: list[float] = []
    samples: list[Sample] = []
    clock_s = 0.0
    while pending or in_flight:
        while pending:
            arrival_s = _arrival_s(spec.arrival, pending[0], clock_s, len(in_flight))
            if arrival_s is None:
                break
            request_spec = pending.popleft()
            request = Request(
                request_id=request_spec.request_id,
                prompt_token_ids=request_spec.prompt_token_ids,
                max_output_tokens=request_spec.output_len,
                arrival_step=scheduler.num_steps,
            )
            scheduler.add_request(request)
            in_flight[request.request_id] = request
            submitted.append((request, arrival_s))
        if not in_flight:
            clock_s = pending[0].arrival_s
            continue
        batch = scheduler.schedule()
        num_scheduled_tokens = batch.num_scheduled_tokens
        samples.append(
            {
                "step": batch.step,
                "t_s": clock_s,
                "kv_usage": scheduler.kv_cache_manager.usage,
                "running": len(num_scheduled_tokens),
                "waiting": len(in_flight) - len(num_scheduled_tokens),
                "preemptions_total": scheduler.num_preemptions,
                "prefix_queries_total": scheduler.prefix_cache_queries,
                "prefix_hits_total": scheduler.prefix_cache_hits,
            }
        )
        sampled_token_ids = {
            request_id: _sample_token(in_flight[request_id], config.eos_token_id)
            for request_id in num_scheduled_tokens
        }
        for request in scheduler.complete_step(batch, sampled_token_ids):
            del in_flight[request.request_id]
        clock_s += cost_model.step_duration(num_scheduled_tokens.values())
        step_ends_s.append(clock_s)
    return {
        "source": "simulator",
        "measured": False,
        "engine_version": __version__,
        "model": None,
        "gpu": None,
        "config": {
            "block_size": config.block_size,
            "num_blocks": config.num_blocks,
            "kv_tokens": config.num_blocks * config.block_size,
            "max_num_seqs": config.max_running_requests,
            "max_num_batched_tokens": config.max_tokens_per_step,
            "enable_prefix_caching": config.enable_prefix_caching,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "long_prefill_token_threshold": config.long_prefill_token_threshold,
            "command": None,
        },
        "cost_model": asdict(cost_model),
        "workload": asdict(spec),
        "requests": [
            _request_record(request, arrival_s, step_ends_s) for request, arrival_s in submitted
        ],
        "samples": samples,
    }


def _arrival_s(
    arrival: Arrival, request: RequestSpec, clock_s: float, num_in_flight: int
) -> float | None:
    match arrival:
        case ClosedLoop(concurrency=concurrency):
            return clock_s if num_in_flight < concurrency else None
        case Poisson():
            return request.arrival_s if request.arrival_s <= clock_s else None


def _sample_token(request: Request, eos_token_id: int) -> int:
    position = f"{request.request_id}:{request.num_output_tokens}"
    return eos_token_id + 1 + zlib.crc32(position.encode())


def _request_record(request: Request, arrival_s: float, step_ends_s: list[float]) -> RequestRecord:
    assert request.first_token_step is not None
    assert request.finished_step is not None
    return {
        "request_id": request.request_id,
        "arrival_s": arrival_s,
        "first_token_s": step_ends_s[request.first_token_step],
        "finish_s": step_ends_s[request.finished_step],
        "first_token_step": request.first_token_step,
        "finish_step": request.finished_step,
        "prompt_tokens": request.num_prompt_tokens,
        "output_tokens": request.num_output_tokens,
        "server_ttft_ms": None,
        "server_queue_ms": None,
    }
