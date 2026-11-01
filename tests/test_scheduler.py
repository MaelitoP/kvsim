import math
from collections import deque
from collections.abc import Callable, MutableMapping
from dataclasses import asdict, replace
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariants import block_ids, check_scheduler, request_ids
from kvsim.request import Request, RequestStatus
from kvsim.scheduler import ScheduledBatch, Scheduler, SchedulerConfig

EOS = 0
TOKEN = 1
BLOCK_SIZE = 4
MAX_PROMPT = 20
MAX_OUTPUT = 4
MAX_REQUESTS = 12
MAX_COMPUTED_TOKENS = MAX_PROMPT + MAX_OUTPUT - 1


def make_request(
    request_id: str,
    num_prompt_tokens: int = 4,
    max_output_tokens: int = MAX_OUTPUT,
    first_token_id: int = 1,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=tuple(range(first_token_id, first_token_id + num_prompt_tokens)),
        max_output_tokens=max_output_tokens,
        arrival_step=0,
    )


def make_scheduler(
    num_blocks: int = 64,
    max_running_requests: int = 256,
    max_tokens_per_step: int = 2048,
    enable_chunked_prefill: bool = True,
    long_prefill_token_threshold: int = 0,
    enable_prefix_caching: bool = False,
) -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            max_running_requests=max_running_requests,
            max_tokens_per_step=max_tokens_per_step,
            enable_chunked_prefill=enable_chunked_prefill,
            long_prefill_token_threshold=long_prefill_token_threshold,
            enable_prefix_caching=enable_prefix_caching,
            eos_token_id=EOS,
        )
    )


def snapshot(scheduler: Scheduler) -> dict[str, object]:
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    return {
        "num_steps": scheduler.num_steps,
        "pending_batch": id(scheduler._pending_batch),
        "waiting": tuple(scheduler._waiting),
        "running": tuple(scheduler._running),
        "requests": {
            request_id: (request, asdict(request))
            for request_id, request in scheduler._requests.items()
        },
        "blocks": [(block, block.ref_count, block.block_hash) for block in pool._blocks],
        "free_queue": block_ids(pool._free_queue),
        "block_tables": {
            request_id: block_ids(blocks)
            for request_id, blocks in manager._blocks_by_request.items()
        },
    }


def run_step(
    scheduler: Scheduler, sampled_token_ids: dict[str, int] | None = None
) -> tuple[ScheduledBatch, list[Request]]:
    running_before = request_ids(scheduler._running)
    waiting_before = request_ids(scheduler._waiting)
    requests = dict(scheduler._requests)
    outputs_before = {
        request_id: list(request.output_token_ids) for request_id, request in requests.items()
    }
    batch = scheduler.schedule()
    check_scheduler(scheduler)
    assert batch.scheduled_running[:1] == tuple(running_before[:1])
    assert (*batch.scheduled_running, *batch.preempted[::-1]) == tuple(running_before)
    admitted = (*batch.scheduled_resumed, *batch.scheduled_new)
    assert admitted == tuple(waiting_before[: len(admitted)])
    assert request_ids(scheduler._waiting) == [
        *batch.preempted[::-1],
        *waiting_before[len(admitted) :],
    ]
    completing = {
        request_id
        for request_id, num_new_tokens in batch.num_scheduled_tokens.items()
        if requests[request_id].num_computed_tokens + num_new_tokens
        == requests[request_id].num_tokens
    }
    if sampled_token_ids is None:
        sampled_token_ids = dict.fromkeys(batch.num_scheduled_tokens, TOKEN)
    finished = scheduler.complete_step(batch, sampled_token_ids)
    check_scheduler(scheduler)
    for request_id, request in requests.items():
        sampled = [sampled_token_ids[request_id]] if request_id in completing else []
        assert request.output_token_ids == outputs_before[request_id] + sampled
    return batch, finished


@pytest.mark.parametrize(
    ("make_config", "message"),
    [
        (lambda: SchedulerConfig(num_blocks=4, max_running_requests=0), "must be >= 1"),
        (lambda: SchedulerConfig(num_blocks=4, max_tokens_per_step=0), "must be >= 1"),
        (
            lambda: SchedulerConfig(num_blocks=4, long_prefill_token_threshold=-1),
            "must be >= 0, got -1",
        ),
        (
            lambda: SchedulerConfig(
                num_blocks=4, enable_chunked_prefill=False, long_prefill_token_threshold=8
            ),
            "requires enable_chunked_prefill",
        ),
    ],
)
def test_config_rejects_invalid_settings(
    make_config: Callable[[], SchedulerConfig], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_config()


def test_add_request_rejects_requests_that_could_never_run() -> None:
    scheduler = make_scheduler(num_blocks=1, max_tokens_per_step=8, enable_chunked_prefill=False)
    scheduler.add_request(make_request("r0", max_output_tokens=1))
    with pytest.raises(ValueError, match="r0 is already WAITING"):
        scheduler.add_request(make_request("r0"))
    with pytest.raises(ValueError, match="long can compute 9 tokens, more than the budget of 8"):
        scheduler.add_request(make_request("long", num_prompt_tokens=4, max_output_tokens=6))
    with pytest.raises(ValueError, match="grows needs 2 blocks, the cache holds 1"):
        scheduler.add_request(make_request("grows", num_prompt_tokens=4, max_output_tokens=2))
    with pytest.raises(ValueError, match="needs a prompt and an output token"):
        scheduler.add_request(make_request("empty", num_prompt_tokens=0))
    with pytest.raises(ValueError, match="needs a prompt and an output token"):
        scheduler.add_request(make_request("mute", max_output_tokens=0))
    stale = make_request("stale")
    stale.num_computed_tokens = 2
    with pytest.raises(ValueError, match="stale is not a fresh waiting request"):
        scheduler.add_request(stale)
    hashed = make_request("hashed", max_output_tokens=1)
    hashed.update_block_hashes(BLOCK_SIZE)
    with pytest.raises(ValueError, match="hashed is not a fresh waiting request"):
        scheduler.add_request(hashed)
    assert list(scheduler._requests) == ["r0"]
    check_scheduler(scheduler)


def test_schedule_on_an_empty_scheduler_advances_the_step() -> None:
    scheduler = make_scheduler()
    batch = scheduler.schedule()
    before = snapshot(scheduler)
    with pytest.raises(RuntimeError, match="step 0 has not been completed"):
        scheduler.schedule()
    assert snapshot(scheduler) == before
    assert batch.step == 0
    assert batch.total_tokens == 0
    assert scheduler.complete_step(batch, {}) == []
    assert scheduler.num_steps == 1
    batch, _ = run_step(scheduler)
    assert batch.step == 1


def test_prefill_then_one_decode_token_per_step_until_length_cap() -> None:
    scheduler = make_scheduler()
    request = make_request("r0", num_prompt_tokens=6, max_output_tokens=3)
    scheduler.add_request(request)
    batch, finished = run_step(scheduler)
    assert batch.step == 0
    assert batch.num_scheduled_tokens == {"r0": 6}
    assert batch.scheduled_new == ("r0",)
    assert finished == []
    assert scheduler._running == [request]
    assert request.first_scheduled_step == 0
    assert request.first_token_step == 0

    batch, finished = run_step(scheduler)
    assert batch.step == 1
    assert batch.num_scheduled_tokens == {"r0": 1}
    assert batch.scheduled_running == ("r0",)
    assert finished == []

    batch, finished = run_step(scheduler)
    assert finished == [request]
    assert request.status is RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.finished_step == 2
    assert request.output_token_ids == [TOKEN, TOKEN, TOKEN]
    assert not scheduler.has_unfinished_requests
    assert scheduler.kv_cache_manager.usage == 0.0


def test_eos_finishes_a_request_and_releases_every_block() -> None:
    scheduler = make_scheduler()
    first = make_request("first", num_prompt_tokens=9, max_output_tokens=100)
    second = make_request("second", num_prompt_tokens=9, max_output_tokens=100)
    scheduler.add_request(first)
    scheduler.add_request(second)
    run_step(scheduler)
    assert scheduler.kv_cache_manager.num_blocks_for(first) == 3
    _, finished = run_step(scheduler, {"first": EOS, "second": TOKEN})
    assert finished == [first]
    assert first.status is RequestStatus.FINISHED_STOPPED
    assert first.finished_step == 1
    assert first.output_token_ids == [TOKEN, EOS]
    assert scheduler._running == [second]
    assert scheduler.kv_cache_manager.num_blocks_for(first) == 0
    assert scheduler.kv_cache_manager.num_blocks_for(second) == 3


def test_a_request_admitted_mid_generation_shares_the_step_with_decoding_ones() -> None:
    scheduler = make_scheduler()
    old = make_request("old", num_prompt_tokens=5)
    new = make_request("new", num_prompt_tokens=7)
    scheduler.add_request(old)
    run_step(scheduler)
    scheduler.add_request(new)
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("old", 1), ("new", 7)]
    assert batch.total_tokens == 8
    assert old.first_scheduled_step == old.first_token_step == 0
    assert new.first_scheduled_step == new.first_token_step == 1


def test_running_requests_are_served_before_waiting_ones_in_fcfs_order() -> None:
    scheduler = make_scheduler(max_tokens_per_step=10, enable_chunked_prefill=False)
    scheduler.add_request(make_request("decoding", num_prompt_tokens=4))
    run_step(scheduler)
    scheduler.add_request(make_request("big", num_prompt_tokens=10, max_output_tokens=1))
    scheduler.add_request(make_request("small", num_prompt_tokens=2))
    batch, _ = run_step(scheduler, {"decoding": EOS})
    assert list(batch.num_scheduled_tokens.items()) == [("decoding", 1)]
    assert request_ids(scheduler._waiting) == ["big", "small"]
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("big", 10)]
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("small", 2)]


def test_a_long_prompt_is_chunked_behind_decodes_and_samples_after_its_last_chunk() -> None:
    scheduler = make_scheduler(num_blocks=256, max_tokens_per_step=256)
    decoding = make_request("decoding", max_output_tokens=100)
    long = make_request("long", num_prompt_tokens=1000)
    scheduler.add_request(decoding)
    run_step(scheduler)
    scheduler.add_request(long)
    for step, chunk in enumerate([255, 255, 255, 235], start=1):
        batch, _ = run_step(scheduler, {"decoding": TOKEN, "long": EOS if step < 4 else TOKEN})
        assert list(batch.num_scheduled_tokens.items()) == [("decoding", 1), ("long", chunk)]
        assert long.output_token_ids == ([TOKEN] if step == 4 else [])
    assert decoding.num_output_tokens == 5
    assert long.first_scheduled_step == 1
    assert long.first_token_step == 4


def test_long_prefill_threshold_caps_chunks_while_other_requests_exist() -> None:
    scheduler = make_scheduler(max_tokens_per_step=16, long_prefill_token_threshold=6)
    scheduler.add_request(make_request("long", num_prompt_tokens=20))
    scheduler.add_request(make_request("short", num_prompt_tokens=3, max_output_tokens=2))
    batches = [run_step(scheduler)[0] for _ in range(3)]
    assert [list(batch.num_scheduled_tokens.items()) for batch in batches] == [
        [("long", 6), ("short", 3)],
        [("long", 6), ("short", 1)],
        [("long", 8)],
    ]


@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
def test_the_youngest_running_request_is_preempted_first_and_resumes_at_the_queue_head(
    enable_chunked_prefill: bool,
) -> None:
    scheduler = make_scheduler(num_blocks=3, enable_chunked_prefill=enable_chunked_prefill)
    for request_id in ("old", "middle", "young"):
        scheduler.add_request(make_request(request_id))
    run_step(scheduler)
    scheduler.add_request(make_request("late", num_prompt_tokens=1, max_output_tokens=1))
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("old", 1)]
    assert batch.preempted == ("young", "middle")
    assert request_ids(scheduler._waiting) == ["middle", "young", "late"]
    for request in list(scheduler._waiting)[:2]:
        assert request.status is RequestStatus.PREEMPTED
        assert request.num_preemptions == 1
        assert request.output_token_ids == [TOKEN]
        assert request.first_token_step == 0
    assert scheduler.num_preemptions == 2
    batches = [run_step(scheduler)[0] for _ in range(6)]
    assert [list(batch.num_scheduled_tokens.items()) for batch in batches] == [
        [("old", 1)],
        [("old", 1)],
        [("middle", 5)],
        [("middle", 1)],
        [("middle", 1)],
        [("young", 5), ("late", 1)],
    ]
    assert batches[2].scheduled_resumed == ("middle",)
    assert batches[-1].scheduled_resumed == ("young",)
    assert batches[-1].scheduled_new == ("late",)


def test_a_request_in_partial_prefill_can_be_preempted_and_restarts_from_zero() -> None:
    scheduler = make_scheduler(num_blocks=4, max_tokens_per_step=4)
    scheduler.add_request(make_request("held", num_prompt_tokens=4))
    run_step(scheduler)
    long = make_request("long", num_prompt_tokens=16, max_output_tokens=1)
    scheduler.add_request(long)
    batches = [run_step(scheduler)[0] for _ in range(7)]
    assert [(list(batch.num_scheduled_tokens.items()), batch.preempted) for batch in batches] == [
        ([("held", 1), ("long", 3)], ()),
        ([("held", 1), ("long", 3)], ()),
        ([("held", 1)], ("long",)),
        ([("long", 4)], ()),
        ([("long", 4)], ()),
        ([("long", 4)], ()),
        ([("long", 4)], ()),
    ]
    assert batches[3].scheduled_resumed == ("long",)
    assert long.num_preemptions == 1
    assert long.first_scheduled_step == 1
    assert long.first_token_step == long.finished_step == 7


def test_a_resumed_request_keeps_its_output_and_samples_after_its_last_recompute_chunk() -> None:
    scheduler = make_scheduler(num_blocks=3, long_prefill_token_threshold=2)
    young = make_request("young")
    scheduler.add_request(make_request("old"))
    scheduler.add_request(young)
    batches: list[ScheduledBatch] = []
    outputs: list[list[int]] = []
    for _ in range(6):
        batches.append(run_step(scheduler)[0])
        outputs.append(list(young.output_token_ids))
    assert [(list(batch.num_scheduled_tokens.items()), batch.preempted) for batch in batches] == [
        ([("old", 2), ("young", 2)], ()),
        ([("old", 2), ("young", 2)], ()),
        ([("old", 1)], ("young",)),
        ([("old", 1), ("young", 2)], ()),
        ([("old", 1), ("young", 2)], ()),
        ([("young", 1)], ()),
    ]
    assert outputs[2:] == [[TOKEN], [TOKEN], [TOKEN], [TOKEN, TOKEN]]


def test_identical_prompts_admitted_in_the_same_step_share_blocks() -> None:
    scheduler = make_scheduler(num_blocks=4, enable_prefix_caching=True)
    scheduler.add_request(make_request("first", num_prompt_tokens=9))
    scheduler.add_request(make_request("second", num_prompt_tokens=9))
    batch, _ = run_step(scheduler)
    assert batch.num_scheduled_tokens == {"first": 9, "second": 1}
    tables = scheduler.kv_cache_manager._blocks_by_request
    assert tables["second"][:2] == tables["first"][:2]
    assert [block.ref_count for block in tables["second"]] == [2, 2, 1]
    assert (scheduler.prefix_cache_queries, scheduler.prefix_cache_hits) == (18, 8)


@pytest.mark.parametrize(
    ("num_prompt_tokens", "num_hit_tokens"), [(3, 0), (4, 0), (16, 12), (17, 16)]
)
def test_a_cache_hit_always_leaves_a_token_to_compute(
    num_prompt_tokens: int, num_hit_tokens: int
) -> None:
    scheduler = make_scheduler(enable_prefix_caching=True)
    scheduler.add_request(make_request("cached", num_prompt_tokens, max_output_tokens=1))
    run_step(scheduler)
    scheduler.add_request(make_request("hit", num_prompt_tokens))
    batch, _ = run_step(scheduler)
    assert batch.num_scheduled_tokens == {"hit": num_prompt_tokens - num_hit_tokens}
    assert scheduler.prefix_cache_hits == num_hit_tokens


@pytest.mark.parametrize(
    ("old_max_output_tokens", "resume_step", "num_hit_tokens"), [(4, 4, 4), (6, 6, 0)]
)
def test_a_resumed_request_hits_its_own_blocks_unless_they_were_evicted(
    old_max_output_tokens: int, resume_step: int, num_hit_tokens: int
) -> None:
    scheduler = make_scheduler(num_blocks=3, enable_prefix_caching=True)
    scheduler.add_request(
        make_request(
            "old", num_prompt_tokens=4, max_output_tokens=old_max_output_tokens, first_token_id=100
        )
    )
    scheduler.add_request(
        make_request("young", num_prompt_tokens=6, max_output_tokens=2, first_token_id=200)
    )
    batches = [run_step(scheduler)[0] for _ in range(resume_step + 1)]
    assert batches[1].preempted == ("young",)
    assert batches[-1].scheduled_resumed == ("young",)
    assert batches[-1].num_scheduled_tokens == {"young": 7 - num_hit_tokens}
    assert scheduler.prefix_cache_queries == 4 + 6 + 7
    assert scheduler.prefix_cache_hits == num_hit_tokens


def test_a_block_completed_by_decoding_is_cached() -> None:
    scheduler = make_scheduler(enable_prefix_caching=True)
    scheduler.add_request(make_request("first", num_prompt_tokens=3, max_output_tokens=2))
    run_step(scheduler)
    run_step(scheduler)
    scheduler.add_request(
        Request(
            request_id="hit",
            prompt_token_ids=(1, 2, 3, TOKEN, 5),
            max_output_tokens=1,
            arrival_step=0,
        )
    )
    batch, _ = run_step(scheduler)
    assert batch.num_scheduled_tokens == {"hit": 1}


def test_hits_let_a_prompt_fit_the_budget_with_chunked_prefill_off() -> None:
    scheduler = make_scheduler(
        max_tokens_per_step=8, enable_chunked_prefill=False, enable_prefix_caching=True
    )
    scheduler.add_request(make_request("cached", num_prompt_tokens=8, max_output_tokens=1))
    run_step(scheduler)
    scheduler.add_request(make_request("decoding", num_prompt_tokens=1, first_token_id=100))
    run_step(scheduler)
    scheduler.add_request(make_request("hit", num_prompt_tokens=8, max_output_tokens=1))
    batch, _ = run_step(scheduler)
    assert batch.num_scheduled_tokens == {"decoding": 1, "hit": 4}


def make_waiting() -> tuple[Scheduler, Request]:
    scheduler = make_scheduler(enable_prefix_caching=True)
    target = make_request("target")
    scheduler.add_request(target)
    return scheduler, target


def make_partial_prefill() -> tuple[Scheduler, Request]:
    scheduler = make_scheduler(max_tokens_per_step=4, enable_prefix_caching=True)
    target = make_request("target", num_prompt_tokens=9)
    scheduler.add_request(target)
    run_step(scheduler)
    assert 0 < target.num_computed_tokens < target.num_prompt_tokens
    return scheduler, target


def make_decoding() -> tuple[Scheduler, Request]:
    scheduler = make_scheduler(enable_prefix_caching=True)
    target = make_request("target", num_prompt_tokens=5)
    scheduler.add_request(target)
    run_step(scheduler)
    assert target.output_token_ids == [TOKEN]
    return scheduler, target


def make_preempted() -> tuple[Scheduler, Request]:
    scheduler = make_scheduler(num_blocks=3, enable_prefix_caching=True)
    scheduler.add_request(make_request("old", num_prompt_tokens=4, first_token_id=100))
    target = make_request("target", num_prompt_tokens=6, max_output_tokens=2, first_token_id=200)
    scheduler.add_request(target)
    for _ in range(4):
        run_step(scheduler)
    assert target.status is RequestStatus.PREEMPTED
    assert request_ids(scheduler._waiting) == ["target"]
    assert not scheduler._running
    return scheduler, target


@pytest.mark.parametrize(
    ("make_state", "finished_step"),
    [(make_waiting, -1), (make_partial_prefill, 0), (make_decoding, 0), (make_preempted, 3)],
)
def test_aborting_a_request_in_any_state_frees_the_pool_and_keeps_cached_blocks(
    make_state: Callable[[], tuple[Scheduler, Request]], finished_step: int
) -> None:
    scheduler, target = make_state()
    pool = scheduler.kv_cache_manager.block_pool
    cached_blocks = dict(pool._block_by_hash)
    assert scheduler.finish_requests("target") == [target]
    assert target.status is RequestStatus.FINISHED_ABORTED
    assert target.finished_step == finished_step
    assert not scheduler.has_unfinished_requests
    assert pool.num_free_blocks == pool.num_blocks
    assert pool._block_by_hash == cached_blocks
    check_scheduler(scheduler)


def test_requests_aborted_during_a_step_are_skipped_when_the_step_completes() -> None:
    scheduler = make_scheduler()
    kept, aborted, gone = (make_request(request_id) for request_id in ("kept", "aborted", "gone"))
    for request in (kept, aborted, gone):
        scheduler.add_request(request)
    batch = scheduler.schedule()
    assert scheduler.finish_requests("gone", "aborted") == [gone, aborted]
    check_scheduler(scheduler)
    assert aborted.finished_step == gone.finished_step == batch.step
    retry = make_request("aborted")
    scheduler.add_request(retry)
    assert scheduler.complete_step(batch, {"kept": TOKEN, "aborted": EOS, "gone": EOS}) == []
    check_scheduler(scheduler)
    assert aborted.output_token_ids == gone.output_token_ids == []
    assert kept.output_token_ids == [TOKEN]
    assert retry.num_computed_tokens == 0
    assert retry.status is RequestStatus.WAITING


def test_finishing_with_a_live_status_or_unknown_ids_changes_nothing() -> None:
    scheduler = make_scheduler(max_running_requests=1)
    stopped, live = make_request("stopped"), make_request("live")
    scheduler.add_request(stopped)
    run_step(scheduler, {"stopped": EOS})
    scheduler.add_request(live)
    scheduler.add_request(make_request("waiting"))
    run_step(scheduler)
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="RUNNING is not a finished status"):
        scheduler.finish_requests("live", "waiting", status=RequestStatus.RUNNING)
    assert scheduler.finish_requests("stopped", "unknown") == []
    assert snapshot(scheduler) == before
    assert scheduler.finish_requests("live", "live") == [live]


def test_max_running_requests_caps_admission() -> None:
    scheduler = make_scheduler(max_running_requests=2)
    for index in range(3):
        scheduler.add_request(make_request(f"r{index}", max_output_tokens=2))
    batch, _ = run_step(scheduler)
    assert batch.scheduled_new == ("r0", "r1")
    assert request_ids(scheduler._waiting) == ["r2"]
    batch, _ = run_step(scheduler)
    assert batch.scheduled_new == ()
    batch, _ = run_step(scheduler)
    assert batch.scheduled_new == ("r2",)


def test_a_waiting_request_that_does_not_fit_the_pool_blocks_the_queue() -> None:
    scheduler = make_scheduler(num_blocks=4)
    scheduler.add_request(make_request("held", num_prompt_tokens=8))
    run_step(scheduler)
    scheduler.add_request(make_request("big", num_prompt_tokens=5))
    scheduler.add_request(make_request("small", num_prompt_tokens=1))
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("held", 1)]
    assert request_ids(scheduler._waiting) == ["big", "small"]
    assert scheduler.kv_cache_manager.num_free_blocks == 1


def test_every_step_needs_exactly_one_update_with_every_sampled_token() -> None:
    scheduler = make_scheduler()
    first = make_request("first", max_output_tokens=1)
    second = make_request("second")
    scheduler.add_request(first)
    scheduler.add_request(second)
    batch = scheduler.schedule()
    scheduler.add_request(make_request("waiting"))
    before = snapshot(scheduler)
    with pytest.raises(RuntimeError, match="step 0 has not been completed"):
        scheduler.schedule()
    assert snapshot(scheduler) == before
    with pytest.raises(ValueError, match=r"no sampled token for scheduled requests \['second'\]"):
        scheduler.complete_step(batch, {"first": EOS})
    assert snapshot(scheduler) == before
    assert scheduler.complete_step(batch, {"first": EOS, "second": TOKEN}) == [first]
    assert first.status is RequestStatus.FINISHED_STOPPED
    assert first.num_computed_tokens == second.num_computed_tokens == 4
    before = snapshot(scheduler)
    with pytest.raises(ValueError, match="batch for step 0 is not the pending batch"):
        scheduler.complete_step(batch, {"first": EOS, "second": TOKEN})
    assert snapshot(scheduler) == before
    assert scheduler._pending_batch is None
    check_scheduler(scheduler)


def test_only_the_returned_batch_can_complete_a_step() -> None:
    scheduler = make_scheduler()
    other = make_scheduler()
    request = make_request("r0", num_prompt_tokens=4)
    scheduler.add_request(request)
    other.add_request(make_request("r0", num_prompt_tokens=8))
    batch = scheduler.schedule()
    foreign_batch = other.schedule()
    before = snapshot(scheduler)
    for rejected_batch in (foreign_batch, replace(batch)):
        with pytest.raises(ValueError, match="batch for step 0 is not the pending batch"):
            scheduler.complete_step(rejected_batch, {"r0": TOKEN})
        assert snapshot(scheduler) == before
    assert scheduler._pending_batch is batch
    scheduler.complete_step(batch, {"r0": TOKEN})
    assert request.num_computed_tokens == 4
    assert request.output_token_ids == [TOKEN]
    check_scheduler(scheduler)


def test_scheduled_token_counts_cannot_be_changed() -> None:
    scheduler = make_scheduler()
    request = make_request("r0")
    scheduler.add_request(request)
    batch = scheduler.schedule()
    before = snapshot(scheduler)
    with pytest.raises(TypeError, match="does not support item assignment"):
        cast(MutableMapping[str, int], batch.num_scheduled_tokens)["r0"] = 100
    assert batch.num_scheduled_tokens == {"r0": 4}
    assert snapshot(scheduler) == before
    scheduler.complete_step(batch, {"r0": TOKEN})
    assert request.num_computed_tokens == 4


def test_requests_added_during_a_pending_batch_wait_until_the_next_step() -> None:
    scheduler = make_scheduler()
    first = make_request("first", max_output_tokens=1)
    later = make_request("later")
    scheduler.add_request(first)
    batch = scheduler.schedule()
    scheduler.add_request(later)
    assert scheduler.complete_step(batch, {"first": TOKEN, "later": EOS}) == [first]
    assert later.status is RequestStatus.WAITING
    assert later.num_computed_tokens == 0
    assert later.output_token_ids == []
    assert scheduler.kv_cache_manager.num_blocks_for(later) == 0
    batch, _ = run_step(scheduler)
    assert batch.scheduled_new == ("later",)
    assert later.first_scheduled_step == 1


@given(
    prompts=st.lists(st.integers(1, MAX_PROMPT), min_size=1, max_size=MAX_REQUESTS),
    eos_positions=st.lists(
        st.integers(0, MAX_OUTPUT), min_size=MAX_REQUESTS, max_size=MAX_REQUESTS
    ),
    arrivals=st.lists(st.integers(0, 3), min_size=MAX_REQUESTS, max_size=MAX_REQUESTS),
    max_running_requests=st.integers(2, 4),
    num_spare_blocks=st.integers(0, 2),
    enable_prefix_caching=st.booleans(),
    prefill=st.tuples(st.just(False), st.integers(MAX_COMPUTED_TOKENS, 2 * MAX_PROMPT), st.just(0))
    | st.tuples(st.just(True), st.integers(1, 2 * MAX_PROMPT), st.integers(0, MAX_PROMPT)),
)
def test_limits_hold_and_every_request_finishes_with_its_expected_output(
    prompts: list[int],
    eos_positions: list[int],
    arrivals: list[int],
    max_running_requests: int,
    num_spare_blocks: int,
    enable_prefix_caching: bool,
    prefill: tuple[bool, int, int],
) -> None:
    enable_chunked_prefill, max_tokens_per_step, long_prefill_token_threshold = prefill
    max_computed_tokens = [prompt + MAX_OUTPUT - 1 for prompt in prompts]
    scheduler = make_scheduler(
        num_blocks=math.ceil(max(max_computed_tokens) / BLOCK_SIZE) + num_spare_blocks,
        max_running_requests=max_running_requests,
        max_tokens_per_step=max_tokens_per_step,
        enable_chunked_prefill=enable_chunked_prefill,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = [
        make_request(f"r{index}", num_prompt_tokens=prompt) for index, prompt in enumerate(prompts)
    ]
    eos_position = {f"r{index}": eos for index, eos in enumerate(eos_positions)}
    pending = deque(requests)
    batches: list[ScheduledBatch] = []
    while pending or scheduler.has_unfinished_requests:
        assert len(batches) < sum(max_computed_tokens)
        num_arrivals = arrivals[len(batches) % len(arrivals)]
        if not scheduler.has_unfinished_requests:
            num_arrivals = max(num_arrivals, 1)
        for _ in range(min(num_arrivals, len(pending))):
            scheduler.add_request(pending.popleft())
        check_scheduler(scheduler)
        batch, _ = run_step(
            scheduler,
            {
                request_id: EOS
                if request.num_output_tokens == eos_position[request_id]
                else TOKEN + request.num_output_tokens
                for request_id, request in scheduler._requests.items()
            },
        )
        assert batch.step == len(batches)
        batches.append(batch)
    assert scheduler.kv_cache_manager.usage == 0.0
    assert scheduler.num_preemptions == sum(request.num_preemptions for request in requests)
    assert scheduler.num_preemptions == sum(len(batch.preempted) for batch in batches)
    for request in requests:
        num_tokens = min(eos_position[request.request_id], MAX_OUTPUT)
        expected = [TOKEN + position for position in range(num_tokens)]
        if num_tokens < MAX_OUTPUT:
            expected.append(EOS)
        assert request.output_token_ids == expected
        assert request.num_computed_tokens == request.num_tokens - 1
        assert request.status is (
            RequestStatus.FINISHED_STOPPED
            if expected[-1] == EOS
            else RequestStatus.FINISHED_LENGTH_CAPPED
        )
