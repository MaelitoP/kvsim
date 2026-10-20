from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from kvsim.kv_cache_manager import KVCacheManager
from kvsim.request import Request, RequestStatus


@dataclass(frozen=True)
class SchedulerConfig:
    num_blocks: int
    block_size: int = 16
    max_running_requests: int = 256
    max_tokens_per_step: int = 2048
    eos_token_id: int = 0

    def __post_init__(self) -> None:
        if self.max_running_requests < 1 or self.max_tokens_per_step < 1:
            raise ValueError(
                "max_running_requests and max_tokens_per_step must be >= 1, "
                f"got {self.max_running_requests}, {self.max_tokens_per_step}"
            )


@dataclass(frozen=True)
class ScheduledBatch:
    """Lists running requests before requests admitted in this step."""

    step: int
    num_scheduled_tokens: Mapping[str, int]
    scheduled_running: tuple[str, ...]
    scheduled_new: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._requests: dict[str, Request] = {}
        self.kv_cache_manager = KVCacheManager(config.num_blocks, config.block_size)
        self.num_steps = 0
        self._pending_batch: ScheduledBatch | None = None

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self._requests)

    def add_request(self, request: Request) -> None:
        live = self._requests.get(request.request_id)
        if live is not None:
            raise ValueError(f"request {request.request_id} is already {live.status.name}")
        if (
            request.status is not RequestStatus.WAITING
            or request.num_computed_tokens != 0
            or request.output_token_ids
        ):
            raise ValueError(f"request {request.request_id} is not a fresh waiting request")
        if request.num_prompt_tokens < 1 or request.max_output_tokens < 1:
            raise ValueError(f"request {request.request_id} needs a prompt and an output token")
        if request.num_prompt_tokens > self.config.max_tokens_per_step:
            raise ValueError(
                f"request {request.request_id} has {request.num_prompt_tokens} prompt tokens, "
                f"more than the budget of {self.config.max_tokens_per_step}"
            )
        num_blocks = -(-request.num_prompt_tokens // self.config.block_size)
        if num_blocks > self.config.num_blocks:
            raise ValueError(
                f"request {request.request_id} needs {num_blocks} blocks, "
                f"the cache holds {self.config.num_blocks}"
            )
        self._requests[request.request_id] = request
        self._waiting.append(request)

    def schedule(self) -> ScheduledBatch:
        if self._pending_batch is not None:
            raise RuntimeError(f"step {self._pending_batch.step} has not been completed")
        step = self.num_steps
        token_budget = self.config.max_tokens_per_step
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running: list[str] = []
        scheduled_new: list[str] = []

        for request in self._running:
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            if self.kv_cache_manager.allocate_slots(request, num_new_tokens) is None:
                raise RuntimeError(
                    f"out of KV cache blocks for running request {request.request_id}"
                )
            num_scheduled_tokens[request.request_id] = num_new_tokens
            scheduled_running.append(request.request_id)
            token_budget -= num_new_tokens

        while (
            self._waiting
            and token_budget > 0
            and len(self._running) < self.config.max_running_requests
        ):
            request = self._waiting[0]
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            if num_new_tokens > token_budget:
                break
            if self.kv_cache_manager.allocate_slots(request, num_new_tokens) is None:
                break
            self._waiting.popleft()
            self._running.append(request)
            request.status = RequestStatus.RUNNING
            if request.first_scheduled_step is None:
                request.first_scheduled_step = step
            num_scheduled_tokens[request.request_id] = num_new_tokens
            scheduled_new.append(request.request_id)
            token_budget -= num_new_tokens

        self.num_steps += 1
        batch = ScheduledBatch(
            step=step,
            num_scheduled_tokens=MappingProxyType(num_scheduled_tokens.copy()),
            scheduled_running=tuple(scheduled_running),
            scheduled_new=tuple(scheduled_new),
        )
        self._pending_batch = batch
        return batch

    def complete_step(
        self, batch: ScheduledBatch, sampled_token_ids: Mapping[str, int]
    ) -> list[Request]:
        if batch is not self._pending_batch:
            raise ValueError(f"batch for step {batch.step} is not the pending batch")
        missing = batch.num_scheduled_tokens.keys() - sampled_token_ids.keys()
        if missing:
            raise ValueError(f"no sampled token for scheduled requests {sorted(missing)}")
        finished: list[Request] = []
        for request_id, num_tokens in batch.num_scheduled_tokens.items():
            request = self._requests[request_id]
            request.num_computed_tokens += num_tokens
            token_id = sampled_token_ids[request_id]
            request.append_output_token(token_id)
            if request.first_token_step is None:
                request.first_token_step = batch.step
            if token_id == self.config.eos_token_id:
                request.finish(RequestStatus.FINISHED_STOPPED, batch.step)
            elif request.num_output_tokens >= request.max_output_tokens:
                request.finish(RequestStatus.FINISHED_LENGTH_CAPPED, batch.step)
            if request.is_finished:
                self.kv_cache_manager.free(request)
                del self._requests[request_id]
                finished.append(request)
        if finished:
            self._running = [request for request in self._running if not request.is_finished]
        self._pending_batch = None
        return finished
