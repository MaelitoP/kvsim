import math

from hypothesis import strategies as st

from kvsim.scheduler import SchedulerConfig
from kvsim.workload import ClosedLoop, LengthRange, Poisson, WorkloadSpec

EOS = 0

LENGTH_RANGES = st.tuples(st.integers(1, 6), st.integers(0, 6)).map(
    lambda bounds: (bounds[0], bounds[0] + bounds[1])
)


@st.composite
def specs(draw: st.DrawFn) -> WorkloadSpec:
    shared_prefix_len = draw(st.integers(0, 6))
    prompt_len: list[LengthRange] = draw(st.lists(LENGTH_RANGES, min_size=1, max_size=3))
    return WorkloadSpec(
        name=draw(st.text()),
        seed=draw(st.integers(min_value=0)),
        vocab_size=draw(st.integers(1, 50)),
        num_requests=draw(st.integers(1, 12)),
        prompt_len=tuple(prompt_len),
        output_len=draw(LENGTH_RANGES),
        shared_prefix_len=shared_prefix_len,
        num_prefix_groups=draw(st.integers(1, 3)) if shared_prefix_len else 1,
        arrival=draw(
            st.one_of(
                st.builds(ClosedLoop, concurrency=st.integers(1, 8)),
                st.builds(Poisson, rate=st.floats(0.01, 100)),
            )
        ),
    )


@st.composite
def configs(draw: st.DrawFn, max_computed_tokens: int) -> SchedulerConfig:
    block_size = draw(st.sampled_from([2, 4]))
    max_blocks_per_request = math.ceil(max_computed_tokens / block_size)
    enable_chunked_prefill = draw(st.booleans())
    if enable_chunked_prefill:
        max_tokens_per_step = draw(st.integers(1, 2 * max_computed_tokens))
        long_prefill_token_threshold = draw(st.integers(0, max_computed_tokens))
    else:
        max_tokens_per_step = draw(st.integers(max_computed_tokens, 2 * max_computed_tokens))
        long_prefill_token_threshold = 0
    return SchedulerConfig(
        num_blocks=max_blocks_per_request + draw(st.integers(0, 2)),
        block_size=block_size,
        max_running_requests=draw(st.integers(2, 4)),
        max_tokens_per_step=max_tokens_per_step,
        enable_chunked_prefill=enable_chunked_prefill,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_prefix_caching=draw(st.booleans()),
        enable_full_input_admission=draw(st.booleans()),
        eos_token_id=EOS,
    )
