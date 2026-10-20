from inspect import signature

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kvsim.request import Request, RequestStatus


def make_request(prompt: tuple[int, ...] = (0, 1, 2, 3)) -> Request:
    return Request(request_id="r0", prompt_token_ids=prompt, max_output_tokens=8, arrival_step=0)


def test_constructor_accepts_only_request_inputs() -> None:
    assert tuple(signature(Request).parameters) == (
        "request_id",
        "prompt_token_ids",
        "max_output_tokens",
        "arrival_step",
    )


def test_new_request_is_waiting_with_nothing_computed() -> None:
    request = make_request()
    assert request.status is RequestStatus.WAITING
    assert not request.is_finished
    assert request.num_computed_tokens == 0
    assert request.num_preemptions == 0
    assert request.first_scheduled_step is None
    assert request.first_token_step is None
    assert request.finished_step is None


@given(prompt=st.lists(st.integers(), min_size=1), output=st.lists(st.integers()))
def test_token_counts_follow_prompt_and_output(prompt: list[int], output: list[int]) -> None:
    request = make_request(tuple(prompt))
    for token in output:
        request.append_output_token(token)
    assert request.all_token_ids == tuple(prompt + output)
    assert request.num_prompt_tokens == len(prompt)
    assert request.num_output_tokens == len(output)
    assert request.num_tokens == len(prompt) + len(output)


@pytest.mark.parametrize("status", list(RequestStatus))
def test_only_finished_statuses_are_finished(status: RequestStatus) -> None:
    assert status.is_finished == status.name.startswith("FINISHED_")


@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_ABORTED,
    ],
)
def test_finish_sets_status_and_step(status: RequestStatus) -> None:
    request = make_request()
    request.finish(status, step=7)
    assert request.status is status
    assert request.is_finished
    assert request.finished_step == 7


def test_finishing_twice_raises_and_keeps_the_first_outcome() -> None:
    request = make_request()
    request.finish(RequestStatus.FINISHED_STOPPED, step=3)
    with pytest.raises(ValueError, match="already FINISHED_STOPPED"):
        request.finish(RequestStatus.FINISHED_ABORTED, step=4)
    assert request.status is RequestStatus.FINISHED_STOPPED
    assert request.finished_step == 3


def test_finish_rejects_a_live_status() -> None:
    request = make_request()
    with pytest.raises(ValueError, match="RUNNING is not a finished status"):
        request.finish(RequestStatus.RUNNING, step=1)
    assert request.status is RequestStatus.WAITING
    assert request.finished_step is None
