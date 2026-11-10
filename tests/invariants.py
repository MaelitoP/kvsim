import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import NamedTuple, override

from kvsim.block_hash import BlockHash, hash_full_blocks
from kvsim.block_pool import BlockPool, KVCacheBlock
from kvsim.kv_cache_manager import KVCacheManager
from kvsim.request import Request, RequestStatus
from kvsim.scheduler import ScheduledBatch, Scheduler, SchedulerConfig
from kvsim.simulator import Result
from kvsim.workload import ClosedLoop, generate_requests, parse_spec


class ManagerState(NamedTuple):
    blocks: list[tuple[int, BlockHash | None]]
    free_queue: list[int]
    block_tables: dict[str, list[int]]
    num_cached_blocks: dict[str, int]


def block_ids(blocks: Iterable[KVCacheBlock]) -> list[int]:
    return [block.block_id for block in blocks]


def request_ids(requests: Iterable[Request]) -> list[str]:
    return [request.request_id for request in requests]


def manager_state(manager: KVCacheManager) -> ManagerState:
    return ManagerState(
        blocks=[(block.ref_count, block.block_hash) for block in manager.block_pool._blocks],
        free_queue=block_ids(manager.block_pool._free_queue),
        block_tables={
            request_id: block_ids(blocks)
            for request_id, blocks in manager._blocks_by_request.items()
        },
        num_cached_blocks=dict(manager._num_cached_blocks),
    )


class CheckedKVCacheManager(KVCacheManager):
    @override
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock] = (),
    ) -> list[KVCacheBlock] | None:
        before = manager_state(self)
        new_blocks = super().allocate_slots(request, num_new_tokens, new_computed_blocks)
        if new_blocks is None:
            assert manager_state(self) == before
        return new_blocks


def checked_scheduler(config: SchedulerConfig) -> Scheduler:
    scheduler = Scheduler(config)
    manager = scheduler.kv_cache_manager
    scheduler.kv_cache_manager = CheckedKVCacheManager(
        manager.num_blocks, manager.block_size, enable_caching=manager.enable_caching
    )
    return scheduler


def check_pool(pool: BlockPool) -> None:
    assert [block.block_id for block in pool._blocks] == list(range(pool.num_blocks))
    for block in pool._free_queue:
        assert block is pool._blocks[block.block_id]
    for block in pool._blocks:
        assert block.ref_count >= 0
        assert (block.ref_count == 0) == (block in pool._free_queue)
    for block_hash, block in pool._block_by_hash.items():
        assert block is pool._blocks[block.block_id]
        assert block.block_hash == block_hash
    assert sum(block.block_hash is not None for block in pool._blocks) == len(pool._block_by_hash)
    cached_flags = [block.block_hash is not None for block in pool._free_queue]
    assert cached_flags == sorted(cached_flags)


def check_manager(manager: KVCacheManager) -> None:
    pool = manager.block_pool
    check_pool(pool)
    assert pool.num_blocks == manager.num_blocks
    references = Counter(
        block.block_id for blocks in manager._blocks_by_request.values() for block in blocks
    )
    for request_id, blocks in manager._blocks_by_request.items():
        assert len(set(block_ids(blocks))) == len(blocks)
        for block in blocks:
            assert block is pool._blocks[block.block_id]
        num_cached_blocks = manager._num_cached_blocks.get(request_id, 0)
        assert num_cached_blocks <= len(blocks)
        assert all(block.block_hash is None for block in blocks[num_cached_blocks:])
    assert set(manager._num_cached_blocks) <= set(manager._blocks_by_request)
    for block in pool._blocks:
        assert block.ref_count == references[block.block_id]
    if not manager.enable_caching:
        assert not manager._num_cached_blocks
        assert all(block.ref_count <= 1 for block in pool._blocks)
        assert not pool._block_by_hash


def check_scheduler(scheduler: Scheduler) -> None:
    manager = scheduler.kv_cache_manager
    check_manager(manager)
    block_size = scheduler.config.block_size
    pending = scheduler._pending_batch
    scheduled: Mapping[str, int] = pending.num_scheduled_tokens if pending is not None else {}
    waiting = list(scheduler._waiting)
    running = scheduler._running
    if pending is not None:
        assert tuple(scheduled) == (
            *pending.scheduled_running,
            *pending.scheduled_resumed,
            *pending.scheduled_new,
        )
        running_ids = set(request_ids(running))
        assert running_ids <= set(scheduled)
        for request_id in set(scheduled) - running_ids:
            readded = scheduler._requests.get(request_id)
            assert readded is None or readded.status is RequestStatus.WAITING
        assert all(num_tokens >= 1 for num_tokens in scheduled.values())
        assert sum(scheduled.values()) <= scheduler.config.max_tokens_per_step
        if pending.preempted:
            assert pending.scheduled_resumed == pending.scheduled_new == ()
            live_victims = set(pending.preempted) & set(scheduler._requests)
            assert live_victims <= set(request_ids(waiting))
    assert len(running) <= scheduler.config.max_running_requests
    assert len(waiting) + len(running) == len(scheduler._requests)
    assert set(request_ids(waiting + running)) == set(scheduler._requests)
    assert set(manager._blocks_by_request) <= set(request_ids(running))
    assert scheduler.prefix_cache_hits <= scheduler.prefix_cache_queries
    assert scheduler.prefix_cache_hits % block_size == 0
    if not scheduler.config.enable_prefix_caching:
        assert scheduler.prefix_cache_queries == 0
        assert all(not request.block_hashes for request in waiting + running)
    preempted_flags = [request.status is RequestStatus.PREEMPTED for request in waiting]
    assert preempted_flags == sorted(preempted_flags, reverse=True)
    for request in waiting:
        assert scheduler._requests[request.request_id] is request
        assert request.num_computed_tokens == 0
        assert manager.num_blocks_for(request) == 0
        if request.status is RequestStatus.PREEMPTED:
            assert request.num_preemptions >= 1
            assert request.first_scheduled_step is not None
        else:
            assert request.status is RequestStatus.WAITING
            assert request.num_preemptions == 0
            assert request.first_scheduled_step is None
            assert not request.output_token_ids
    for request in running:
        assert scheduler._requests[request.request_id] is request
        assert request.status is RequestStatus.RUNNING
        num_reserved_tokens = request.num_computed_tokens + scheduled.get(request.request_id, 0)
        assert num_reserved_tokens <= request.num_tokens
        assert manager.num_blocks_for(request) == math.ceil(num_reserved_tokens / block_size)
        if scheduler.config.enable_prefix_caching:
            num_cached_blocks = manager._num_cached_blocks[request.request_id]
            assert num_cached_blocks == num_reserved_tokens // block_size
            block_hashes = request.block_hashes
            assert len(block_hashes) >= num_cached_blocks
            assert (
                block_hashes
                == hash_full_blocks(block_size, request.all_token_ids)[: len(block_hashes)]
            )
            blocks = manager._blocks_by_request[request.request_id]
            for block, block_hash in zip(
                blocks[:num_cached_blocks], block_hashes[:num_cached_blocks], strict=True
            ):
                assert block.block_hash in (None, block_hash)
        if pending is None:
            assert 1 <= request.num_computed_tokens < request.num_tokens
            if request.output_token_ids and request.num_preemptions == 0:
                assert request.num_computed_tokens == request.num_tokens - 1


def check_finished(scheduler: Scheduler, request: Request) -> None:
    assert request.is_finished
    assert request.finished_step is not None
    assert scheduler._requests.get(request.request_id) is not request


def run_checked_step(
    scheduler: Scheduler, sampled_token_ids: Mapping[str, int], aborted_ids: Sequence[str] = ()
) -> tuple[ScheduledBatch, list[Request]]:
    step = scheduler.num_steps
    num_preemptions = scheduler.num_preemptions
    running_before = request_ids(scheduler._running)
    waiting_before = request_ids(scheduler._waiting)
    requests = dict(scheduler._requests)
    outputs_before = {
        request_id: list(request.output_token_ids) for request_id, request in requests.items()
    }
    preemptions_before = {
        request_id: request.num_preemptions for request_id, request in requests.items()
    }
    batch = scheduler.schedule()
    check_scheduler(scheduler)
    assert batch.step == step
    assert scheduler.num_steps == step + 1
    assert scheduler.num_preemptions == num_preemptions + len(batch.preempted)
    for request_id, request in requests.items():
        assert request.num_preemptions == preemptions_before[request_id] + (
            request_id in batch.preempted
        )
    assert batch.scheduled_running[:1] == tuple(running_before[:1])
    assert (*batch.scheduled_running, *batch.preempted[::-1]) == tuple(running_before)
    admitted = (*batch.scheduled_resumed, *batch.scheduled_new)
    assert admitted == tuple(waiting_before[: len(admitted)])
    if waiting_before and not running_before:
        assert admitted
    assert request_ids(scheduler._waiting) == [
        *batch.preempted[::-1],
        *waiting_before[len(admitted) :],
    ]
    aborted = scheduler.finish_requests(*aborted_ids)
    check_scheduler(scheduler)
    completing = [
        request_id
        for request_id, num_new_tokens in batch.num_scheduled_tokens.items()
        if requests[request_id] not in aborted
        and requests[request_id].num_computed_tokens + num_new_tokens
        == requests[request_id].num_tokens
    ]
    finished = scheduler.complete_step(batch, sampled_token_ids)
    check_scheduler(scheduler)
    assert finished == [
        requests[request_id]
        for request_id in completing
        if sampled_token_ids[request_id] == scheduler.config.eos_token_id
        or len(outputs_before[request_id]) + 1 == requests[request_id].max_output_tokens
    ]
    for request in aborted:
        check_finished(scheduler, request)
        assert request.status is RequestStatus.FINISHED_ABORTED
    for request in finished:
        check_finished(scheduler, request)
        assert request.status is (
            RequestStatus.FINISHED_STOPPED
            if request.output_token_ids[-1] == scheduler.config.eos_token_id
            else RequestStatus.FINISHED_LENGTH_CAPPED
        )
    for request_id, request in requests.items():
        sampled = [sampled_token_ids[request_id]] if request_id in completing else []
        assert request.output_token_ids == outputs_before[request_id] + sampled
    return batch, finished


def check_result(result: Result) -> None:
    spec = parse_spec(result["workload"])
    request_specs = generate_requests(spec)
    config = result["config"]
    records = result["requests"]
    samples = result["samples"]
    assert result["source"] == "simulator"
    assert result["measured"] is False
    assert config["kv_tokens"] == config["num_blocks"] * config["block_size"]
    assert [sample["step"] for sample in samples] == list(range(len(samples)))
    assert samples[0]["t_s"] == 0.0
    for earlier, later in pairwise(samples):
        assert earlier["t_s"] < later["t_s"]
        assert earlier["preemptions_total"] <= later["preemptions_total"]
        assert earlier["prefix_queries_total"] <= later["prefix_queries_total"]
        assert earlier["prefix_hits_total"] <= later["prefix_hits_total"]
    for sample in samples:
        assert 0 < sample["kv_usage"] <= 1
        assert 1 <= sample["running"] <= config["max_num_seqs"]
        assert sample["waiting"] >= 0
        assert sample["prefix_hits_total"] <= sample["prefix_queries_total"]
        num_in_flight = sum(
            record["arrival_s"] <= sample["t_s"] < record["finish_s"] for record in records
        )
        assert sample["running"] + sample["waiting"] == num_in_flight
    if not config["enable_prefix_caching"]:
        assert samples[-1]["prefix_queries_total"] == 0
    step_ends_s: dict[int, float] = {}
    for record, request_spec in zip(records, request_specs, strict=True):
        assert record["request_id"] == request_spec.request_id
        assert record["prompt_tokens"] == len(request_spec.prompt_token_ids)
        assert record["output_tokens"] == request_spec.output_len
        assert record["finish_step"] - record["first_token_step"] >= request_spec.output_len - 1
        assert record["arrival_s"] <= samples[record["first_token_step"]]["t_s"]
        for step, end_s in (
            (record["first_token_step"], record["first_token_s"]),
            (record["finish_step"], record["finish_s"]),
        ):
            assert math.isfinite(end_s)
            assert step_ends_s.setdefault(step, end_s) == end_s
    for step, end_s in step_ends_s.items():
        assert samples[step]["t_s"] < end_s
        assert step + 1 == len(samples) or end_s <= samples[step + 1]["t_s"]
    arrivals_s = [record["arrival_s"] for record in records]
    if isinstance(spec.arrival, ClosedLoop):
        finishes_s = sorted(record["finish_s"] for record in records)
        assert arrivals_s == ([0.0] * spec.arrival.concurrency + finishes_s)[: len(records)]
    else:
        assert arrivals_s == [request_spec.arrival_s for request_spec in request_specs]
