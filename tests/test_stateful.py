import sys
from collections.abc import Callable

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from invariants import (
    check_finished,
    check_scheduler,
    checked_scheduler,
    request_ids,
    run_checked_step,
)
from kvsim.request import Request
from kvsim.scheduler import ScheduledBatch, Scheduler, SchedulerConfig
from strategies import EOS, configs

TOKEN = 1
MAX_PREFIX = 8
MAX_SUFFIX = 4
MAX_OUTPUT = 6
MAX_COMPUTED_TOKENS = MAX_PREFIX + MAX_SUFFIX + MAX_OUTPUT - 1
PREFIXES = (tuple(range(100, 100 + MAX_PREFIX)), tuple(range(200, 200 + MAX_PREFIX)))


def expected_output(max_output_tokens: int, eos_position: int) -> list[int]:
    num_tokens = min(eos_position, max_output_tokens)
    output = [TOKEN + position for position in range(num_tokens)]
    if num_tokens < max_output_tokens:
        output.append(EOS)
    return output


type Operation = Callable[[Scheduler], object]


class SchedulerMachine(RuleBasedStateMachine):
    config: SchedulerConfig
    scheduler: Scheduler

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[Request] = []
        self.eos_position: dict[str, int] = {}
        self.aborted_ids: set[str] = set()
        self.operations: list[Operation] = []
        self.results: list[object] = []

    @initialize(config=configs(MAX_COMPUTED_TOKENS))
    def start(self, config: SchedulerConfig) -> None:
        self.config = config
        self.scheduler = checked_scheduler(config)

    def record(self, operation: Operation) -> object:
        result = operation(self.scheduler)
        self.operations.append(operation)
        self.results.append(result)
        return result

    def sampled_token_ids(self, scheduler: Scheduler) -> dict[str, int]:
        return {
            request_id: EOS
            if request.num_output_tokens == self.eos_position[request_id]
            else TOKEN + request.num_output_tokens
            for request_id, request in scheduler._requests.items()
        }

    def record_step(self, aborted_ids: list[str]) -> None:
        def step_operation(scheduler: Scheduler) -> tuple[ScheduledBatch, list[str]]:
            batch, finished = run_checked_step(
                scheduler, self.sampled_token_ids(scheduler), aborted_ids
            )
            return batch, request_ids(finished)

        self.record(step_operation)
        self.aborted_ids.update(aborted_ids)

    @rule(
        arrivals=st.lists(
            st.tuples(
                st.sampled_from(PREFIXES),
                st.sampled_from([0, MAX_PREFIX // 2, MAX_PREFIX]),
                st.lists(st.integers(1, 3), min_size=1, max_size=MAX_SUFFIX),
                st.integers(1, MAX_OUTPUT),
                st.integers(0, MAX_OUTPUT),
            ),
            min_size=1,
            max_size=3,
        )
    )
    def add_requests(
        self, arrivals: list[tuple[tuple[int, ...], int, list[int], int, int]]
    ) -> None:
        specs = [
            (
                f"r{len(self.requests) + index}",
                prefix[:prefix_len] + tuple(suffix),
                max_output_tokens,
                eos_position,
            )
            for index, (prefix, prefix_len, suffix, max_output_tokens, eos_position) in enumerate(
                arrivals
            )
        ]

        def add_operation(scheduler: Scheduler) -> None:
            for request_id, prompt_token_ids, max_output_tokens, _ in specs:
                scheduler.add_request(
                    Request(
                        request_id=request_id,
                        prompt_token_ids=prompt_token_ids,
                        max_output_tokens=max_output_tokens,
                        arrival_step=scheduler.num_steps,
                    )
                )

        self.record(add_operation)
        for request_id, _, _, eos_position in specs:
            self.requests.append(self.scheduler._requests[request_id])
            self.eos_position[request_id] = eos_position

    @rule(data=st.data())
    def step(self, data: st.DataObject) -> None:
        live_ids = sorted(self.scheduler._requests)
        aborted_ids = data.draw(st.lists(st.sampled_from(live_ids), max_size=1)) if live_ids else []
        self.record_step(aborted_ids)

    @rule(data=st.data())
    def abort(self, data: st.DataObject) -> None:
        candidate_ids = [*request_ids(self.requests), "unknown"]
        aborted_ids = data.draw(st.lists(st.sampled_from(candidate_ids), max_size=3))
        expected = list(
            dict.fromkeys(id_ for id_ in aborted_ids if id_ in self.scheduler._requests)
        )

        def abort_operation(scheduler: Scheduler) -> list[str]:
            return request_ids(scheduler.finish_requests(*aborted_ids))

        assert self.record(abort_operation) == expected
        self.aborted_ids.update(expected)

    @invariant()
    def scheduler_is_consistent(self) -> None:
        check_scheduler(self.scheduler)
        for request in self.requests:
            if request.is_finished:
                check_finished(self.scheduler, request)

    def teardown(self) -> None:
        if sys.exc_info()[1] is not None:
            return
        live = list(self.scheduler._requests.values())
        max_steps = sum(request.num_prompt_tokens + request.max_output_tokens for request in live)
        for _ in range(max_steps):
            if not self.scheduler.has_unfinished_requests:
                break
            self.record_step([])
        assert not self.scheduler.has_unfinished_requests
        assert self.scheduler.kv_cache_manager.usage == 0.0
        for request in self.requests:
            expected = expected_output(
                request.max_output_tokens, self.eos_position[request.request_id]
            )
            if request.request_id in self.aborted_ids:
                assert request.output_token_ids == expected[: request.num_output_tokens]
            else:
                assert request.output_token_ids == expected
                assert request.num_computed_tokens == request.num_tokens - 1
        replay = checked_scheduler(self.config)
        assert [operation(replay) for operation in self.operations] == self.results


TestScheduler = SchedulerMachine.TestCase
TestScheduler.settings = settings(max_examples=300, stateful_step_count=50, deadline=None)
