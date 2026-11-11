import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from client import Engine

ROOT = Path(__file__).parent.parent
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

VLLM_LOG = (
    "(EngineCore pid=71) INFO 11-12 09:14:03 [kv_cache_utils.py:2327] "
    "GPU KV cache size: {tokens:,} tokens, "
    "Maximum concurrency for 4,096 tokens per request: 31.95x\n"
)
LMDEPLOY_BUDGET = (
    "[TM][INFO] Object cache budget: 8000.00 MB from free 16000.00 MB and ratio 0.500\n"
)
LMDEPLOY_CACHE = (
    "[TM][WARN][1111.16:10:58.081141][engine.cc:262] dp0 cache stats:\n"
    "[cache] region=4.58GiB live=0.00B (0.00%) live_alloc=0 "
    "| pages 0/{pages} used (0.00%) page_size=32.00MiB\n"
    "[cache]   free pages by order: o1=1 o4=1 o7=1\n"
    "[cache]   slab obj=2.25MiB slab=32.00MiB cap=14 slabs=0(F0/P0/E0) obj 0/0 used (0.00%)\n"
)
ENGINE_LOGS: dict[Engine, str] = {
    "vllm": VLLM_LOG.format(tokens=2045 * 64),
    "lmdeploy": LMDEPLOY_BUDGET + LMDEPLOY_CACHE.format(pages=146),
}
ENGINE_VERSIONS: dict[Engine, str] = {"vllm": "0.32.0", "lmdeploy": "0.18.0"}

BASELINE_LAUNCHES: dict[Engine, str] = {
    "vllm": (
        "docker run --detach --name kvsim-vllm --gpus all --ipc host "
        "--publish 127.0.0.1:8000:8000 --volume /hf:/root/.cache/huggingface "
        "vllm/vllm-openai:v0.32.0"
        "@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34 "
        f"Qwen/Qwen2.5-3B-Instruct --revision {REVISION} --dtype bfloat16 --seed 0 "
        "--max-model-len 4096 --block-size 64 --kv-cache-memory-bytes 4824760320 "
        "--max-num-seqs 64 --max-num-batched-tokens 2048 "
        "--enable-prefix-caching --enable-chunked-prefill "
        "--long-prefill-token-threshold 0 --enable-per-request-metrics"
    ),
    "lmdeploy": (
        "docker run --detach --name kvsim-lmdeploy --gpus all --ipc host "
        "--publish 127.0.0.1:8000:8000 --volume /hf:/root/.cache/huggingface "
        "--env TM_CACHE_LOG_INTERVAL=1000000000 "
        "openmmlab/lmdeploy:v0.18.0-cu12.8"
        "@sha256:90bd451a49b56f68d953f0b49d6011b0419e043ececa0265c1e3cdb920748f4f "
        f"lmdeploy serve api_server Qwen/Qwen2.5-3B-Instruct --revision {REVISION} "
        "--model-name Qwen/Qwen2.5-3B-Instruct --backend turbomind --server-port 8000 "
        "--dtype bfloat16 --tp 1 --quant-policy 0 --session-len 4096 "
        "--cache-block-seq-len 64 --cache-max-entry-count 0.293000 --max-batch-size 64 "
        "--max-prefill-token-num 2048 --log-level WARNING --enable-prefix-caching"
    ),
}
CALIBRATION_ARGUMENTS = [
    ("--cache-max-entry-count 0.293000", "--cache-max-entry-count 0.5"),
    ("--log-level WARNING", "--log-level INFO"),
]

VLLM_TIGHT = ("--kv-cache-memory-bytes 4824760320", "--kv-cache-memory-bytes 596901888")
VLLM_NO_PREFIX_CACHING = ("--enable-prefix-caching", "--no-enable-prefix-caching")
LMDEPLOY_TIGHT = ("--cache-max-entry-count 0.293000", "--cache-max-entry-count 0.037000")
LMDEPLOY_NO_PREFIX_CACHING = (" --enable-prefix-caching", "")

FAKE_COMMANDS = {
    "docker": """
        case $1 in
            run) echo "$*" >> "$FAKE_CALLS/docker_run" ;;
            logs)
                launch=$(tail -n 1 "$FAKE_CALLS/docker_run")
                if [[ -n ${FAKE_TIGHT_ARGUMENT:-} && $launch == *"$FAKE_TIGHT_ARGUMENT"* ]]; then
                    cat "$FAKE_TIGHT_ENGINE_LOG"
                else
                    cat "$FAKE_ENGINE_LOG"
                fi
                ;;
            inspect) echo "${FAKE_ENGINE_RUNNING:-true}" ;;
        esac
    """,
    "curl": """
        if [[ $* == */health ]]; then exit "${FAKE_HEALTH_STATUS:-0}"; fi
        while (($# > 0)); do
            if [[ $1 == --output ]]; then mkdir -p "$(dirname "$2")" && echo '{}' > "$2"; fi
            shift
        done
    """,
    "nvidia-smi": """
        case "$*" in
            *name*) echo "NVIDIA L4" ;;
            *driver_version*) echo "580.65.06" ;;
            *) echo "| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0 |" ;;
        esac
    """,
    "uv": """
        echo "$*" >> "$FAKE_CALLS/client"
        if [[ -n ${FAKE_CLIENT_STATUS:-} ]]; then exit "$FAKE_CLIENT_STATUS"; fi
        while (($# > 0)); do
            if [[ $1 == --out ]]; then echo '{}' > "$2"; fi
            shift
        done
    """,
}


def replaced(text: str, *replacements: tuple[str, str]) -> str:
    for old, new in replacements:
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text


def run_sourced(engine: Engine, script: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    prelude = (
        "set -euo pipefail; HF_HOME=/hf; "
        f"source benchmarks/run_matrix.sh; source benchmarks/engines/{engine}.sh\n"
    )
    return subprocess.run(
        ["bash", "-c", prelude + script],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )


def calibrated_launch(engine: Engine, cell: str) -> subprocess.CompletedProcess[str]:
    calibrate = "engine_calibrate\n" if engine == "lmdeploy" else ""
    return run_sourced(
        engine, f"{calibrate}set_cell {cell}; engine_command kvsim-{engine}", LMDEPLOY_BUDGET
    )


@pytest.mark.parametrize(
    ("engine", "log", "num_blocks"),
    [
        pytest.param("vllm", VLLM_LOG.format(tokens=2045 * 64), 2044, id="vllm-large"),
        pytest.param("vllm", VLLM_LOG.format(tokens=253 * 64), 252, id="vllm-tight"),
        pytest.param("lmdeploy", LMDEPLOY_CACHE.format(pages=146), 2044, id="lmdeploy-large"),
        pytest.param(
            "lmdeploy", LMDEPLOY_CACHE.format(pages=147), 2058, id="lmdeploy-one-page-more"
        ),
        pytest.param("lmdeploy", LMDEPLOY_CACHE.format(pages=18), 252, id="lmdeploy-tight"),
    ],
)
def test_the_block_count_is_read_from_the_startup_log(
    engine: Engine, log: str, num_blocks: int
) -> None:
    result = run_sourced(engine, "engine_num_blocks", log)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{num_blocks}\n"


@pytest.mark.parametrize(
    ("engine", "function", "log"),
    [
        pytest.param(
            "vllm",
            "engine_num_blocks",
            "INFO Available KV cache memory: 4.49 GiB\n",
            id="vllm-no-cache-size",
        ),
        pytest.param(
            "vllm",
            "engine_num_blocks",
            VLLM_LOG.format(tokens=2045 * 64 + 16),
            id="vllm-partial-block",
        ),
        pytest.param("lmdeploy", "engine_num_blocks", LMDEPLOY_BUDGET, id="lmdeploy-no-report"),
        pytest.param(
            "lmdeploy",
            "engine_num_blocks",
            LMDEPLOY_CACHE.format(pages=146).replace("cap=14", "cap=28"),
            id="lmdeploy-slab-capacity",
        ),
        pytest.param(
            "lmdeploy",
            "engine_num_blocks",
            LMDEPLOY_CACHE.format(pages=146).replace("slab=32.00MiB", "slab=64.00MiB"),
            id="lmdeploy-slab-size",
        ),
        pytest.param(
            "lmdeploy",
            "engine_num_blocks",
            LMDEPLOY_CACHE.format(pages=146).replace("page_size=32.00MiB", "page_size=16.00MiB"),
            id="lmdeploy-page-size",
        ),
        pytest.param(
            "lmdeploy",
            "engine_calibrate",
            LMDEPLOY_CACHE.format(pages=146),
            id="lmdeploy-no-free-memory",
        ),
    ],
)
def test_a_log_without_the_expected_cache_report_is_rejected(
    engine: Engine, function: str, log: str
) -> None:
    result = run_sourced(engine, function, log)

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith(f"{engine}: ")


@pytest.mark.parametrize(
    ("engine", "cell", "arguments"),
    [
        ("vllm", "baseline", []),
        ("vllm", "no_prefix_caching", [VLLM_NO_PREFIX_CACHING]),
        (
            "vllm",
            "no_chunked_prefill",
            [
                ("--enable-chunked-prefill", "--no-enable-chunked-prefill"),
                ("--max-num-batched-tokens 2048", "--max-num-batched-tokens 4096"),
            ],
        ),
        (
            "vllm",
            "long_prefill_512",
            [("--long-prefill-token-threshold 0", "--long-prefill-token-threshold 512")],
        ),
        ("vllm", "tight", [VLLM_TIGHT]),
        ("vllm", "tight_no_prefix_caching", [VLLM_TIGHT, VLLM_NO_PREFIX_CACHING]),
        (
            "vllm",
            "tight_sim_aligned",
            [
                VLLM_TIGHT,
                (
                    "--enable-per-request-metrics",
                    "--enable-per-request-metrics "
                    "--no-scheduler-reserve-full-isl --no-async-scheduling",
                ),
            ],
        ),
        ("lmdeploy", "baseline", []),
        ("lmdeploy", "no_prefix_caching", [LMDEPLOY_NO_PREFIX_CACHING]),
        ("lmdeploy", "tight", [LMDEPLOY_TIGHT]),
        ("lmdeploy", "tight_no_prefix_caching", [LMDEPLOY_TIGHT, LMDEPLOY_NO_PREFIX_CACHING]),
    ],
)
def test_a_cell_launches_the_baseline_command_with_its_own_arguments(
    engine: Engine, cell: str, arguments: list[tuple[str, str]]
) -> None:
    result = calibrated_launch(engine, cell)

    assert result.returncode == 0, result.stderr
    assert " ".join(result.stdout.splitlines()) == replaced(BASELINE_LAUNCHES[engine], *arguments)


def test_a_budget_larger_than_the_free_memory_is_rejected() -> None:
    result = run_sourced(
        "lmdeploy",
        "engine_calibrate; set_cell baseline; engine_command kvsim-lmdeploy",
        LMDEPLOY_BUDGET.replace("16000.00", "4688.00"),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == "lmdeploy: 2044 blocks do not fit in 4688.00 MiB of free memory\n"


class FakeHost:
    def __init__(
        self, directory: Path, engine: Engine, engine_log: str, fakes: Mapping[str, str] = {}
    ) -> None:
        self.engine = engine
        self.results = directory / "results" / engine
        self.hf_home = directory / "hf"
        self.tokenizer = directory / "cache" / "kvsim" / f"tokenizer-{REVISION}.json"
        self._script = directory / "benchmarks" / "run_matrix.sh"
        self._calls = directory / "calls"
        bin_directory = directory / "bin"
        shutil.copytree(ROOT / "benchmarks", directory / "benchmarks")
        self._calls.mkdir()
        bin_directory.mkdir()
        for name, body in FAKE_COMMANDS.items():
            command = bin_directory / name
            command.write_text(f"#!/usr/bin/env bash\n{body}", encoding="utf-8")
            command.chmod(0o755)
        engine_log_path = directory / "engine.log"
        engine_log_path.write_text(engine_log, encoding="utf-8")
        self._env = {
            "PATH": f"{bin_directory}{os.pathsep}{os.environ['PATH']}",
            "HOME": str(directory),
            "HF_HOME": str(self.hf_home),
            "XDG_CACHE_HOME": str(directory / "cache"),
            "FAKE_CALLS": str(self._calls),
            "FAKE_ENGINE_LOG": str(engine_log_path),
            **fakes,
        }

    def run_matrix(self, *scenarios: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._script, self.engine, *scenarios],
            env=self._env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
        )

    def calls(self, name: str) -> list[str]:
        path = self._calls / name
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def result_names(self) -> list[str]:
        return sorted(path.name for path in self.results.iterdir())

    def launch(self, *arguments: tuple[str, str]) -> str:
        return replaced(BASELINE_LAUNCHES[self.engine], ("/hf:", f"{self.hf_home}:"), *arguments)

    def client_call(self, launch: str, scenario: str, cell: str, run: int, config: str) -> str:
        return (
            f"run python benchmarks/client.py benchmarks/workloads/{scenario}.json "
            f"--engine {self.engine} --base-url http://127.0.0.1:8000 "
            f"--model Qwen/Qwen2.5-3B-Instruct --tokenizer {self.tokenizer} "
            f"--engine-version {ENGINE_VERSIONS[self.engine]} --gpu NVIDIA L4 "
            f"--command {launch} --num-blocks 2044 --block-size 64 --max-num-seqs 64 "
            f"{config} --out results/{self.engine}/{scenario}.{cell}.run{run}.json.tmp"
        )


BASELINE_CONFIG = (
    "--max-num-batched-tokens 2048 --enable-prefix-caching --enable-chunked-prefill "
    "--long-prefill-token-threshold 0"
)


@pytest.mark.parametrize(
    ("engine", "calibration_arguments"),
    [("vllm", None), ("lmdeploy", CALIBRATION_ARGUMENTS)],
)
def test_a_scenario_restarts_the_engine_for_each_run_and_resumes_where_it_stopped(
    tmp_path: Path, engine: Engine, calibration_arguments: list[tuple[str, str]] | None
) -> None:
    host = FakeHost(tmp_path, engine, ENGINE_LOGS[engine])
    launch = host.launch()
    calibrations = [] if calibration_arguments is None else [host.launch(*calibration_arguments)]
    client_calls = [
        host.client_call(launch, "short_low", "baseline", run, BASELINE_CONFIG) for run in (1, 2, 3)
    ]

    first = host.run_matrix("short_low")

    assert first.returncode == 0, first.stderr
    assert host.result_names() == [
        "README.txt",
        "short_low.baseline.run1.json",
        "short_low.baseline.run1.log",
        "short_low.baseline.run2.json",
        "short_low.baseline.run2.log",
        "short_low.baseline.run3.json",
        "short_low.baseline.run3.log",
    ]
    log_path = host.results / "short_low.baseline.run1.log"
    assert log_path.read_text(encoding="utf-8") == ENGINE_LOGS[engine]
    assert [f"docker {call}" for call in host.calls("docker_run")] == [*calibrations, *[launch] * 3]
    assert host.calls("client") == client_calls

    second = host.run_matrix("short_low")

    assert second.returncode == 0, second.stderr
    assert len(host.calls("docker_run")) == 3 + 2 * len(calibrations)
    assert host.calls("client") == client_calls


def test_each_cell_passes_its_own_configuration_to_the_client(tmp_path: Path) -> None:
    host = FakeHost(tmp_path, "vllm", ENGINE_LOGS["vllm"])
    cells = {
        "baseline": (host.launch(), BASELINE_CONFIG),
        "no_chunked_prefill": (
            host.launch(
                ("--enable-chunked-prefill", "--no-enable-chunked-prefill"),
                ("--max-num-batched-tokens 2048", "--max-num-batched-tokens 4096"),
            ),
            "--max-num-batched-tokens 4096 --enable-prefix-caching --no-enable-chunked-prefill "
            "--long-prefill-token-threshold 0",
        ),
        "long_prefill_512": (
            host.launch(("--long-prefill-token-threshold 0", "--long-prefill-token-threshold 512")),
            "--max-num-batched-tokens 2048 --enable-prefix-caching --enable-chunked-prefill "
            "--long-prefill-token-threshold 512",
        ),
    }

    result = host.run_matrix("mixed")

    assert result.returncode == 0, result.stderr
    assert host.calls("client") == [
        host.client_call(launch, "mixed", cell, run, config)
        for cell, (launch, config) in cells.items()
        for run in (1, 2, 3)
    ]


@pytest.mark.parametrize(
    ("engine", "cells"),
    [
        pytest.param(
            "vllm",
            "long_high.baseline long_high.tight "
            "mixed.baseline mixed.long_prefill_512 mixed.no_chunked_prefill "
            "saturation.baseline saturation.tight saturation.tight_no_prefix_caching "
            "saturation.tight_sim_aligned "
            "shared_prefix.baseline shared_prefix.no_prefix_caching short_low.baseline",
            id="vllm",
        ),
        pytest.param(
            "lmdeploy",
            "long_high.baseline long_high.tight mixed.baseline "
            "saturation.baseline saturation.tight saturation.tight_no_prefix_caching "
            "shared_prefix.baseline shared_prefix.no_prefix_caching short_low.baseline",
            id="lmdeploy",
        ),
    ],
)
def test_the_whole_matrix_runs_the_cells_the_engine_supports_three_times(
    tmp_path: Path, engine: Engine, cells: str
) -> None:
    tight_argument, tight_log = {
        "vllm": (VLLM_TIGHT[1], VLLM_LOG.format(tokens=253 * 64)),
        "lmdeploy": (LMDEPLOY_TIGHT[1], LMDEPLOY_CACHE.format(pages=18)),
    }[engine]
    tight_log_path = tmp_path / "tight.log"
    tight_log_path.write_text(tight_log, encoding="utf-8")
    fakes = {"FAKE_TIGHT_ARGUMENT": tight_argument, "FAKE_TIGHT_ENGINE_LOG": str(tight_log_path)}
    host = FakeHost(tmp_path, engine, ENGINE_LOGS[engine], fakes)

    result = host.run_matrix()

    assert result.returncode == 0, result.stderr
    runs = [name.removesuffix(".json") for name in host.result_names() if name.endswith(".json")]
    assert runs == sorted(f"{cell}.run{run}" for cell in cells.split() for run in (1, 2, 3))


def test_an_engine_with_another_cache_size_is_stopped_before_the_first_request(
    tmp_path: Path,
) -> None:
    host = FakeHost(tmp_path, "vllm", VLLM_LOG.format(tokens=2048 * 64))

    result = host.run_matrix("short_low")

    assert result.returncode == 1
    assert "the engine holds 2047 blocks, expected 2044" in result.stderr
    assert host.calls("client") == []
    assert host.result_names() == ["README.txt", "failure.log"]


def test_a_later_run_removes_the_log_of_an_earlier_failure(tmp_path: Path) -> None:
    host = FakeHost(tmp_path, "vllm", VLLM_LOG.format(tokens=2048 * 64))
    host.run_matrix("short_low")
    (tmp_path / "engine.log").write_text(ENGINE_LOGS["vllm"], encoding="utf-8")

    result = host.run_matrix("short_low")

    assert result.returncode == 0, result.stderr
    assert "failure.log" not in host.result_names()


def test_an_engine_that_exits_during_startup_stops_the_matrix(tmp_path: Path) -> None:
    fakes = {"FAKE_HEALTH_STATUS": "7", "FAKE_ENGINE_RUNNING": "false"}
    host = FakeHost(tmp_path, "vllm", "ValueError: unrecognized arguments\n", fakes)

    result = host.run_matrix("short_low")

    assert result.returncode == 1
    assert "the engine exited during startup" in result.stderr
    assert host.calls("client") == []
    assert host.result_names() == ["README.txt", "failure.log"]
    failure_log = (host.results / "failure.log").read_text(encoding="utf-8")
    assert failure_log == "ValueError: unrecognized arguments\n"


def test_a_failed_client_leaves_no_result(tmp_path: Path) -> None:
    host = FakeHost(tmp_path, "vllm", ENGINE_LOGS["vllm"], {"FAKE_CLIENT_STATUS": "3"})

    result = host.run_matrix("short_low")

    assert result.returncode == 3
    assert len(host.calls("client")) == 1
    assert host.result_names() == ["README.txt", "failure.log"]


def test_an_unknown_scenario_is_rejected_before_any_launch(tmp_path: Path) -> None:
    host = FakeHost(tmp_path, "vllm", ENGINE_LOGS["vllm"])

    result = host.run_matrix("short_low", "shortlow")

    assert result.returncode == 1
    assert "unknown scenario shortlow" in result.stderr
    assert host.calls("docker_run") == []
