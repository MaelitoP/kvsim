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
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0
    enable_prefix_caching: bool = True
    eos_token_id: int = 0

    def __post_init__(self) -> None:
        if self.max_running_requests < 1 or self.max_tokens_per_step < 1:
            raise ValueError(
                "max_running_requests and max_tokens_per_step must be >= 1, "
                f"got {self.max_running_requests}, {self.max_tokens_per_step}"
            )
        if self.long_prefill_token_threshold < 0:
            raise ValueError(
                "long_prefill_token_threshold must be >= 0, "
                f"got {self.long_prefill_token_threshold}"
            )
        if self.long_prefill_token_threshold > 0 and not self.enable_chunked_prefill:
            raise ValueError("long_prefill_token_threshold requires enable_chunked_prefill")


@dataclass(frozen=True)
class ScheduledBatch:
    """Token counts list running, then resumed, then new requests; `preempted` is youngest first."""

    step: int
    num_scheduled_tokens: Mapping[str, int]
    scheduled_running: tuple[str, ...]
    scheduled_resumed: tuple[str, ...]
    scheduled_new: tuple[str, ...]
    preempted: tuple[str, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._waiting: deque[Request] = deque()
        self._running: list[Request] = []
        self._requests: dict[str, Request] = {}
        self.kv_cache_manager = KVCacheManager(
            config.num_blocks, config.block_size, enable_caching=config.enable_prefix_caching
        )
        self.num_steps = 0
        self.num_preemptions = 0
        self.prefix_cache_queries = 0
        self.prefix_cache_hits = 0
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
            or request.block_hashes
        ):
            raise ValueError(f"request {request.request_id} is not a fresh waiting request")
        if request.num_prompt_tokens < 1 or request.max_output_tokens < 1:
            raise ValueError(f"request {request.request_id} needs a prompt and an output token")
        max_num_computed_tokens = request.num_prompt_tokens + request.max_output_tokens - 1
        if (
            not self.config.enable_chunked_prefill
            and max_num_computed_tokens > self.config.max_tokens_per_step
        ):
            raise ValueError(
                f"request {request.request_id} can compute {max_num_computed_tokens} tokens, "
                f"more than the budget of {self.config.max_tokens_per_step}"
            )
        num_blocks = -(-max_num_computed_tokens // self.config.block_size)
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
        max_tokens_per_request = (
            self.config.long_prefill_token_threshold
            if self.config.long_prefill_token_threshold > 0 and len(self._requests) > 1
            else self.config.max_tokens_per_step
        )
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_running: list[str] = []
        scheduled_resumed: list[str] = []
        scheduled_new: list[str] = []
        preempted: list[str] = []

        index = 0
        while index < len(self._running):
            request = self._running[index]
            num_new_tokens = min(
                request.num_tokens - request.num_computed_tokens,
                max_tokens_per_request,
                token_budget,
            )
            while (
                request.status is RequestStatus.RUNNING
                and self.kv_cache_manager.allocate_slots(request, num_new_tokens) is None
            ):
                victim = self._running.pop()
                self._preempt(victim)
                preempted.append(victim.request_id)
            if request.status is RequestStatus.PREEMPTED:
                break
            num_scheduled_tokens[request.request_id] = num_new_tokens
            scheduled_running.append(request.request_id)
            token_budget -= num_new_tokens
            index += 1

        while (
            not preempted
            and self._waiting
            and token_budget > 0
            and len(self._running) < self.config.max_running_requests
        ):
            request = self._waiting[0]
            computed_blocks = self.kv_cache_manager.get_computed_blocks(request)
            num_computed_tokens = len(computed_blocks) * self.config.block_size
            num_new_tokens = request.num_tokens - num_computed_tokens
            if not self.config.enable_chunked_prefill and num_new_tokens > token_budget:
                break
            num_new_tokens = min(num_new_tokens, max_tokens_per_request, token_budget)
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, computed_blocks
            )
            if new_blocks is None:
                break
            self._waiting.popleft()
            self._running.append(request)
            if request.status is RequestStatus.PREEMPTED:
                scheduled_resumed.append(request.request_id)
            else:
                request.first_scheduled_step = step
                scheduled_new.append(request.request_id)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            if self.config.enable_prefix_caching:
                self.prefix_cache_queries += request.num_tokens
                self.prefix_cache_hits += num_computed_tokens
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        self.num_steps += 1
        batch = ScheduledBatch(
            step=step,
            num_scheduled_tokens=MappingProxyType(num_scheduled_tokens.copy()),
            scheduled_running=tuple(scheduled_running),
            scheduled_resumed=tuple(scheduled_resumed),
            scheduled_new=tuple(scheduled_new),
            preempted=tuple(preempted),
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
        for request_id, num_new_tokens in batch.num_scheduled_tokens.items():
            request = self._requests.get(request_id)
            if request is None or request.status is not RequestStatus.RUNNING:
                continue
            request.num_computed_tokens += num_new_tokens
            if request.num_computed_tokens < request.num_tokens:
                continue
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

    def finish_requests(
        self, *request_ids: str, status: RequestStatus = RequestStatus.FINISHED_ABORTED
    ) -> list[Request]:
        if not status.is_finished:
            raise ValueError(f"{status.name} is not a finished status")
        finished: list[Request] = []
        for request_id in request_ids:
            request = self._requests.pop(request_id, None)
            if request is None:
                continue
            if request.status is RequestStatus.RUNNING:
                self._running.remove(request)
            else:
                self._waiting.remove(request)
            request.finish(status, self.num_steps - 1)
            self.kv_cache_manager.free(request)
            finished.append(request)
        return finished

    def _preempt(self, request: Request) -> None:
        self.kv_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        self.num_preemptions += 1
        self._waiting.appendleft(request)
