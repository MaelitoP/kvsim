from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in {
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED,
        }


@dataclass(eq=False)
class Request:
    """One generation request: its tokens, how many are computed, its status and timeline.

    `num_computed_tokens` counts tokens of `all_token_ids` whose KV entries exist in the cache.
    A request is finished at most once, through `finish`.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    max_output_tokens: int
    arrival_step: int
    output_token_ids: list[int] = field(default_factory=list, repr=False)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    first_scheduled_step: int | None = None
    first_token_step: int | None = None
    finished_step: int | None = None
    num_preemptions: int = 0

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def finish(self, status: RequestStatus, step: int) -> None:
        if not status.is_finished:
            raise ValueError(f"{status.name} is not a finished status")
        if self.is_finished:
            raise ValueError(f"request {self.request_id} is already {self.status.name}")
        self.status = status
        self.finished_step = step
