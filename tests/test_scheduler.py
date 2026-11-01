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


def make_request(
    request_id: str, num_prompt_tokens: int = 4, max_output_tokens: int = MAX_OUTPUT
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=tuple(range(1, num_prompt_tokens + 1)),
        max_output_tokens=max_output_tokens,
        arrival_step=0,
    )


def make_scheduler(
    num_blocks: int = 64,
    max_running_requests: int = 256,
    max_tokens_per_step: int = 2048,
    enable_chunked_prefill: bool = True,
    long_prefill_token_threshold: int = 0,
) -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            max_running_requests=max_running_requests,
            max_tokens_per_step=max_tokens_per_step,
            enable_chunked_prefill=enable_chunked_prefill,
            long_prefill_token_threshold=long_prefill_token_threshold,
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
    batch = scheduler.schedule()
    check_scheduler(scheduler)
    assert batch.scheduled_running == tuple(running_before)
    assert batch.scheduled_new == tuple(waiting_before[: len(batch.scheduled_new)])
    assert list(batch.num_scheduled_tokens) == [*batch.scheduled_running, *batch.scheduled_new]
    if sampled_token_ids is None:
        sampled_token_ids = dict.fromkeys(batch.num_scheduled_tokens, TOKEN)
    finished = scheduler.complete_step(batch, sampled_token_ids)
    check_scheduler(scheduler)
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
    scheduler.add_request(make_request("r0"))
    with pytest.raises(ValueError, match="r0 is already WAITING"):
        scheduler.add_request(make_request("r0"))
    with pytest.raises(ValueError, match="9 prompt tokens, more than the budget of 8"):
        scheduler.add_request(make_request("r1", num_prompt_tokens=9))
    with pytest.raises(ValueError, match="too_wide needs 2 blocks, the cache holds 1"):
        scheduler.add_request(make_request("too_wide", num_prompt_tokens=5))
    with pytest.raises(ValueError, match="needs a prompt and an output token"):
        scheduler.add_request(make_request("empty", num_prompt_tokens=0))
    with pytest.raises(ValueError, match="needs a prompt and an output token"):
        scheduler.add_request(make_request("mute", max_output_tokens=0))
    stale = make_request("stale")
    stale.num_computed_tokens = 2
    with pytest.raises(ValueError, match="stale is not a fresh waiting request"):
        scheduler.add_request(stale)
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
    scheduler.add_request(make_request("big", num_prompt_tokens=10))
    scheduler.add_request(make_request("small", num_prompt_tokens=2))
    batch, _ = run_step(scheduler, {"decoding": EOS})
    assert list(batch.num_scheduled_tokens.items()) == [("decoding", 1)]
    assert request_ids(scheduler._waiting) == ["big", "small"]
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("big", 10)]
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("big", 1), ("small", 2)]


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


def test_chunked_admission_reserves_only_the_first_chunk_and_can_exhaust_the_cache() -> None:
    scheduler = make_scheduler(num_blocks=4, max_tokens_per_step=4)
    scheduler.add_request(make_request("held", num_prompt_tokens=4, max_output_tokens=100))
    run_step(scheduler)
    scheduler.add_request(make_request("long", num_prompt_tokens=16))
    batches = [run_step(scheduler)[0] for _ in range(2)]
    assert [list(batch.num_scheduled_tokens.items()) for batch in batches] == [
        [("held", 1), ("long", 3)]
    ] * 2
    assert scheduler.kv_cache_manager.num_free_blocks == 0
    with pytest.raises(RuntimeError, match="out of KV cache blocks for running request long"):
        scheduler.schedule()


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
    scheduler.add_request(make_request("held", num_prompt_tokens=8, max_output_tokens=100))
    run_step(scheduler)
    scheduler.add_request(make_request("big", num_prompt_tokens=5))
    scheduler.add_request(make_request("small", num_prompt_tokens=1))
    batch, _ = run_step(scheduler)
    assert list(batch.num_scheduled_tokens.items()) == [("held", 1)]
    assert request_ids(scheduler._waiting) == ["big", "small"]
    assert scheduler.kv_cache_manager.num_free_blocks == 1


def test_a_running_request_that_cannot_grow_raises() -> None:
    scheduler = make_scheduler(num_blocks=1)
    scheduler.add_request(make_request("r0", num_prompt_tokens=4, max_output_tokens=100))
    run_step(scheduler)
    with pytest.raises(RuntimeError, match="out of KV cache blocks for running request r0"):
        scheduler.schedule()


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
    arrivals=st.lists(st.integers(0, 3), min_size=MAX_REQUESTS, max_size=MAX_REQUESTS),
    stops=st.lists(st.booleans(), min_size=MAX_REQUESTS, max_size=MAX_REQUESTS),
    max_running_requests=st.integers(1, 4),
    prefill=st.tuples(st.just(False), st.integers(MAX_PROMPT, 2 * MAX_PROMPT), st.just(0))
    | st.tuples(st.just(True), st.integers(1, 2 * MAX_PROMPT), st.integers(0, MAX_PROMPT)),
)
def test_limits_hold_and_every_request_finishes(
    prompts: list[int],
    arrivals: list[int],
    stops: list[bool],
    max_running_requests: int,
    prefill: tuple[bool, int, int],
) -> None:
    enable_chunked_prefill, max_tokens_per_step, long_prefill_token_threshold = prefill
    scheduler = make_scheduler(
        num_blocks=max_running_requests * math.ceil((MAX_PROMPT + MAX_OUTPUT) / BLOCK_SIZE),
        max_running_requests=max_running_requests,
        max_tokens_per_step=max_tokens_per_step,
        enable_chunked_prefill=enable_chunked_prefill,
        long_prefill_token_threshold=long_prefill_token_threshold,
    )
    requests = [
        make_request(f"r{index}", num_prompt_tokens=prompt) for index, prompt in enumerate(prompts)
    ]
    pending = deque(requests)
    step = 0
    while pending or scheduler.has_unfinished_requests:
        assert step < sum(prompts) + MAX_OUTPUT * len(requests)
        num_arrivals = arrivals[step % len(arrivals)]
        if not scheduler.has_unfinished_requests:
            num_arrivals = max(num_arrivals, 1)
        for _ in range(min(num_arrivals, len(pending))):
            scheduler.add_request(pending.popleft())
        check_scheduler(scheduler)
        batch, _ = (
            run_step(scheduler)
            if not stops[step % len(stops)]
            else run_step(
                scheduler,
                dict.fromkeys(request_ids(scheduler._running + list(scheduler._waiting)), EOS),
            )
        )
        assert batch.step == step
        step += 1
    assert scheduler.kv_cache_manager.usage == 0.0
    for request in requests:
        assert request.is_finished
        assert request.num_computed_tokens == request.num_tokens - 1
        assert request.output_token_ids[:-1] == [TOKEN] * (len(request.output_token_ids) - 1)
        if request.status is RequestStatus.FINISHED_LENGTH_CAPPED:
            assert request.output_token_ids == [TOKEN] * MAX_OUTPUT
        else:
            assert request.output_token_ids[-1] == EOS
