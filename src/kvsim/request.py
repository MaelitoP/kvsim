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
    """Computed-token counts remain after completion releases the cache blocks."""

    request_id: str
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    max_output_tokens: int
    arrival_step: int
    output_token_ids: list[int] = field(init=False, default_factory=list, repr=False)
    num_computed_tokens: int = field(init=False, default=0)
    status: RequestStatus = field(init=False, default=RequestStatus.WAITING)
    first_scheduled_step: int | None = field(init=False, default=None)
    first_token_step: int | None = field(init=False, default=None)
    finished_step: int | None = field(init=False, default=None)
    num_preemptions: int = field(init=False, default=0)

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
