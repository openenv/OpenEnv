# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""data_analysis_env: tasks, grading, answer extraction and the MCP episode flow.

Network-free except the tests marked `network`, which download one small task anonymously.
"""

import concurrent.futures
import hashlib
import io
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("pandas", reason="pandas is not installed")
pytest.importorskip("fastmcp", reason="fastmcp is not installed")

from data_analysis_env import extract_answer  # noqa: E402
from data_analysis_env.models import DataAnalysisState  # noqa: E402
from data_analysis_env.server import grading, sandbox, tasks  # noqa: E402
from data_analysis_env.server.data_analysis_environment import (  # noqa: E402
    DataAnalysisEnvironment,
    MAX_TOOL_OUTPUT_CHARS,
)
from openenv.core.env_server.mcp_environment import MCPEnvironment  # noqa: E402
from openenv.core.env_server.mcp_types import (  # noqa: E402
    CallToolAction,
    ListToolsAction,
)

ENV_DIR = Path(__file__).resolve().parents[2] / "envs" / "data_analysis_env"
GOLD = "GOLD-SENTINEL-1234"
FAKE_TOKEN = "hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00"


def row(task_id="t1", answer=GOLD, files=("sales.csv",), **extra):
    return {
        "task_id": task_id,
        "question": f"Question {task_id}?",
        "answer": answer,
        "reward_mode": extra.pop("reward_mode", "exact"),
        "atol": extra.pop("atol", 1e-3),
        "rtol": extra.pop("rtol", 1e-3),
        "difficulty_tier": "easy",
        "hf_bucket": "someone/bucket",
        "bucket_prefix": f"data/{task_id}",
        "files": list(files),
        **extra,
    }


# --- Grader --------------------------------------------------------------------------------------


def test_vendored_grader_matches_upstream_hash():
    text = (ENV_DIR / "server" / "grader.py").read_text()
    upstream = text.split("# --- upstream grader.py below, unmodified ---\n", 1)[1]
    assert (
        hashlib.sha256(upstream.encode()).hexdigest()
        == "bbcbf4e52964404159a4abd3f2eb8c19c449f7dfbc4f9aab62e4fd7ac22ac84a"
    )


@pytest.mark.parametrize(
    "task,candidate,reward",
    [
        (row(answer="Yes"), "yes", 1.0),
        (row(answer="42.5", reward_mode="numeric"), "42.5004", 1.0),
        (row(answer="42.5", reward_mode="numeric"), "43", 0.0),
        (row(answer="96.00", reward_mode="numeric"), "0.96", 1.0),
        (row(answer="a, b, c", reward_mode="list"), "c,b,a", 1.0),
        (row(answer="Paris"), "London", 0.0),
        (
            row(answer="1/2"),
            "0.5",
            1.0,
        ),  # math-verify tier, in the helper's main thread
    ],
)
def test_grade_answer(task, candidate, reward):
    assert grading.grade_answer(task, candidate)[0] == reward


def test_grade_answer_edge_cases():
    assert grading.grade_answer(row(), None) == (0.0, "no_submission")
    assert grading.grade_answer(row(), "   ") == (0.0, "empty")
    assert grading.grade_answer(row(), "x" * 1001) == (0.0, "too_long")
    assert grading.grade_answer(row(atol=float("nan")), GOLD)[0] == 1.0


def test_model_text_is_never_executed(tmp_path):
    marker = tmp_path / "pwned"
    for candidate in (
        f"__import__('os').system('touch {marker}')",
        f"\\boxed{{__import__('os').system('touch {marker}')}}",
    ):
        assert grading.grade_answer(row(answer="3"), candidate)[0] == 0.0
    assert not marker.exists()


# --- Answer extraction (client side) -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The mean is 4.\nAnswer: 4.5", "4.5"),
        ("Final answer: **Paris**", "Paris"),
        ("so \\boxed{\\text{yes}}", "yes"),
        ("<think>Answer: 1</think>Answer: 2", "2"),
        ("<think>never closed Answer: 1", None),
        ("Answer:\n\n  `north, south`", "north, south"),
        ("42", "42"),
        ("", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


# --- Tasks -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,ok",
    [
        ("data.csv", True),
        ("My Data (2020).csv", True),
        ("../x.csv", False),
        ("a/b.csv", False),
        (".hidden.csv", False),
        ("x\x00.csv", False),
        ("..", False),
    ],
)
def test_safe_file_name(name, ok):
    assert (tasks.safe_file_name(name) is not None) == ok


def test_runnable_and_pick_task():
    rows = [
        row("a", files=["x.csv"]),
        row("b", files=["x.parquet"]),
        row("c", files=["../x.csv"]),
        row("d", files=[]),
    ]
    assert [r["task_id"] for r in rows if tasks.runnable(r)] == ["a"]
    assert tasks.pick_task(rows, random.Random(0))["task_id"] == "a"
    assert tasks.pick_task(rows, random.Random(0), "a")["task_id"] == "a"
    with pytest.raises(tasks.TaskError):
        tasks.pick_task(rows, random.Random(0), "b")
    with pytest.raises(tasks.TaskError):
        tasks.pick_task(rows, random.Random(0), "missing")


def test_prompt_never_contains_the_answer():
    prompt = tasks.build_prompt(row(), ["/data/t1/sales.csv"])
    assert GOLD not in prompt
    assert "/data/t1/sales.csv" in prompt and "Question t1?" in prompt


def test_unknown_split_and_bad_locations_are_refused(tmp_path):
    with pytest.raises(tasks.TaskError):
        tasks.load_split("../train")
    for bad in ({"hf_bucket": "x"}, {"bucket_prefix": "../etc"}, {"bucket_prefix": ""}):
        with pytest.raises(tasks.TaskError):
            tasks.fetch_task_files(
                {**row(), **bad}, "eval", str(tmp_path), 1024, lister=dict
            )


class _FakeResponse(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_downloads_are_anonymous_and_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResponse(b"a,b\n1,2\n" * (50 if "big" in req.full_url else 1))

    monkeypatch.setattr(tasks.urllib.request, "urlopen", fake_urlopen)
    sizes = {"sales.csv": 8, "big.csv": 400}
    task_dir, paths = tasks.fetch_task_files(
        row(), "eval", str(tmp_path), 100, lister=lambda b, p: sizes
    )
    assert Path(paths[0]).read_text() == "a,b\n1,2\n"
    assert os.stat(paths[0]).st_mode & 0o777 == 0o644
    # Other tasks can't be listed: the root and split directories are only traversable.
    assert os.stat(tmp_path).st_mode & 0o777 == 0o711
    assert os.stat(tmp_path / "eval").st_mode & 0o777 == 0o711
    assert os.stat(task_dir).st_mode & 0o777 == 0o755
    assert os.stat(task_dir).st_mode & 0o777 == 0o755
    assert seen and all(not r.has_header("Authorization") for r in seen)
    assert all(FAKE_TOKEN not in str(r.header_items()) for r in seen)
    # Over the cap by the listing, and over the cap while streaming (listing lies).
    with pytest.raises(tasks.TaskError):
        tasks.fetch_task_files(
            row("t2", files=["big.csv"]),
            "eval",
            str(tmp_path),
            100,
            lister=lambda b, p: sizes,
        )
    with pytest.raises(tasks.TaskError):
        tasks.fetch_task_files(
            row("t3", files=["big.csv"]),
            "eval",
            str(tmp_path),
            100,
            lister=lambda b, p: {"big.csv": 10},
        )
    assert not (tmp_path / "eval" / "t3" / "big.csv").exists()


def test_verified_splits_are_accepted(monkeypatch):
    import pandas as pd

    buf = io.BytesIO()
    pd.DataFrame([row()]).to_parquet(buf)
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        if "test_verified" in req.full_url:
            raise tasks.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        return _FakeResponse(buf.getvalue())

    monkeypatch.setattr(tasks.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tasks, "_splits", {})
    rows = tasks.load_split("eval_verified")
    assert [r["task_id"] for r in rows] == ["t1"]
    assert seen[0].endswith("/data/eval_verified-00000-of-00001.parquet")
    # A revision without the split refuses it with a clear error, not a raw HTTP one.
    with pytest.raises(tasks.TaskError, match="not in the pinned dataset revision"):
        tasks.load_split("test_verified")


def test_tasks_root_must_belong_to_the_server(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(tasks.TaskError):
        tasks.ensure_dir(str(link))


@pytest.mark.network
def test_real_task_download(tmp_path):
    rows = tasks.load_split("eval")
    assert len(rows) == 144 and all("answer" in r for r in rows)
    small = next(
        r for r in rows if tasks.runnable(r) and r["task_id"].startswith("0011_")
    )
    _, paths = tasks.fetch_task_files(small, "eval", str(tmp_path), 10 * 1024 * 1024)
    assert all(os.path.getsize(p) > 0 for p in paths)


# --- The environment -------------------------------------------------------------------------


@pytest.fixture()
def env(monkeypatch):
    # The sandbox worker preloads these. The env's own CI job installs them.
    pytest.importorskip("openpyxl", reason="openpyxl is not installed")
    pytest.importorskip("sklearn", reason="scikit-learn is not installed")
    # Not pytest's tmp_path: as root it is 0700, and session users must be able to read the data.
    tmp_path = Path(tempfile.mkdtemp(prefix="data_analysis_test_"))
    rows = [row("t1"), row("t2", answer="7", reward_mode="numeric"), row("t3")]
    for r in rows:
        d = tmp_path / "eval" / r["task_id"]
        d.mkdir(parents=True)
        (d / "sales.csv").write_text("region,amount\nnorth,3\nsouth,4\n")
    for path in [tmp_path, *tmp_path.rglob("*")]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    monkeypatch.setenv("DATA_ANALYSIS_TASKS_ROOT", str(tmp_path))
    monkeypatch.setenv("DATA_ANALYSIS_EXEC_TIMEOUT_S", "10")
    monkeypatch.setattr(tasks, "load_split", lambda split: rows)
    e = DataAnalysisEnvironment()
    yield e
    e.close()
    shutil.rmtree(tmp_path, ignore_errors=True)


def call(env, tool, **arguments):
    return env.step(CallToolAction(tool_name=tool, arguments=arguments))


def text(obs):
    return obs.result.data


def test_reset_gives_the_prompt_and_never_the_answer(env):
    obs = env.reset(task_id="t1", episode_id="ep-1")
    meta = obs.metadata
    assert obs.done is False and obs.reward is None
    assert meta["task_id"] == "t1" and meta["files"][0].endswith("/eval/t1/sales.csv")
    assert meta["tools"] == ["run_python", "submit_answer"]
    assert GOLD not in str(meta)
    state = env.state
    assert isinstance(state, DataAnalysisState) and state.episode_id == "ep-1"
    assert GOLD not in state.model_dump_json()


def test_seed_picks_the_same_task(env):
    first = [env.reset(seed=3).metadata["task_id"] for _ in range(1)]
    first += [env.reset().metadata["task_id"] for _ in range(4)]
    second = [env.reset(seed=3).metadata["task_id"] for _ in range(1)]
    second += [env.reset().metadata["task_id"] for _ in range(4)]
    assert first == second


def test_episode_run_then_submit(env):
    path = env.reset(task_id="t2").metadata["files"][0]
    obs = call(
        env,
        "run_python",
        code=f"import pandas as pd\nint(pd.read_csv({path!r}).amount.sum())",
    )
    assert text(obs) == "7\n" and obs.done is False and obs.reward is None
    obs = call(env, "run_python", code="1 / 0")
    assert "ZeroDivisionError" in text(obs)
    assert (env.state.python_calls, env.state.python_errors) == (2, 1)
    obs = call(env, "submit_answer", answer="7.0")
    assert obs.done is True and obs.reward == 1.0
    assert env.state.grade_method == "numeric" and env.state.submitted
    # The episode is over: no more code, no second answer.
    assert "over" in text(call(env, "run_python", code="1"))
    obs = call(env, "submit_answer", answer="8")
    assert "already submitted" in text(obs)


def test_error_after_a_large_print_reaches_the_model(env):
    env.reset(task_id="t2")
    obs = call(env, "run_python", code="print('x' * 30_000)\n1 / 0")
    out = text(obs)
    assert "ZeroDivisionError" in out and "(output truncated)" in out
    assert len(out) <= MAX_TOOL_OUTPUT_CHARS


def test_error_after_long_stderr_reaches_the_model(env):
    env.reset(task_id="t2")
    code = "import sys\nfor i in range(2000):\n    print('step', i, file=sys.stderr)\n1 / 0"
    out = text(call(env, "run_python", code=code))
    assert "ZeroDivisionError" in out and "(output truncated)" in out
    assert len(out) <= MAX_TOOL_OUTPUT_CHARS


def test_reward_is_emitted_only_on_the_submit_step(env):
    env.reset(task_id="t2")
    obs = call(env, "submit_answer", answer="7")
    assert (obs.done, obs.reward) == (True, 1.0)
    # Steps after the submit end the same way but don't pay the reward again.
    for obs in (
        call(env, "submit_answer", answer="7"),
        call(env, "run_python", code="1"),
        env.step(ListToolsAction()),
    ):
        assert (obs.done, obs.reward) == (True, None)
    assert env.state.reward == 1.0
    # A new episode pays its own reward.
    env.reset(task_id="t1")
    obs = call(env, "submit_answer", answer="nope")
    assert (obs.done, obs.reward) == (True, 0.0)


def test_wrong_answer_scores_zero(env):
    env.reset(task_id="t1")
    obs = call(env, "submit_answer", answer="nope")
    assert (obs.done, obs.reward) == (True, 0.0)


def test_python_call_budget(env):
    env.reset(task_id="t1", max_python_calls=2)
    assert text(call(env, "run_python", code="1")) == "1\n"
    assert text(call(env, "run_python", code="2")) == "2\n"
    assert "no run_python calls left" in text(call(env, "run_python", code="3"))
    with pytest.raises(ValueError):
        env.reset(task_id="t1", max_python_calls=1000)
    with pytest.raises(ValueError):
        env.reset(task_id="t1", max_python_calls=0)


def test_reset_starts_a_fresh_session(env):
    env.reset(task_id="t1")
    assert text(call(env, "run_python", code="x = 1\nx")) == "1\n"
    env.reset(task_id="t1")
    assert "NameError" in text(call(env, "run_python", code="x"))


def test_failed_reset_keeps_the_current_episode(env, monkeypatch):
    env.reset(task_id="t1")
    assert text(call(env, "run_python", code="x = 1\nx")) == "1\n"

    def unavailable(*args, **kwargs):
        raise tasks.TaskError("download failed")

    monkeypatch.setattr(tasks, "fetch_task_files", unavailable)
    with pytest.raises(tasks.TaskError):
        env.reset(task_id="t2")
    assert env.state.task_id == "t1"
    assert text(call(env, "run_python", code="x")) == "1\n"
    obs = call(env, "submit_answer", answer=GOLD)
    assert (obs.done, obs.reward) == (True, 1.0)


def test_random_reset_does_not_retry_a_task_that_failed(env, monkeypatch):
    fetch = tasks.fetch_task_files
    tried = []

    def first_fails(row, *args, **kwargs):
        tried.append(row["task_id"])
        if len(tried) == 1:
            raise tasks.TaskError("download failed")
        return fetch(row, *args, **kwargs)

    monkeypatch.setattr(tasks, "fetch_task_files", first_fails)
    env.reset(seed=0)
    assert len(tried) == 2 and tried[1] != tried[0]
    assert env.state.task_id == tried[1]


class _SlowSession:
    """Stands in for the sandbox: counts the calls that overlap."""

    def __init__(self):
        self.active = self.peak = 0
        self.closed = False

    def run(self, code):
        self.active += 1
        self.peak = max(self.peak, self.active)
        time.sleep(0.2)
        self.active -= 1
        if self.closed:
            return "", "closed during the call", 1
        return f"{code}\n", "", 0

    def close(self):
        self.closed = True


def test_tool_calls_never_share_the_session(env):
    """A call the MCP layer gave up on keeps running in its thread: the next one waits for it.

    Calls the tools' thread targets directly, as `asyncio.to_thread` does in the server.
    """
    env.reset(task_id="t1")
    env._session.close()
    env._session = slow = _SlowSession()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        outputs = list(pool.map(env._run_python, ["0", "1", "2", "3"]))
    assert outputs == ["0\n", "1\n", "2\n", "3\n"]
    assert slow.peak == 1 and env.state.python_calls == 4
    # A submit during a call waits for it, then closes the session.
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        running = pool.submit(env._run_python, "last")
        time.sleep(0.05)
        submitted = pool.submit(env._submit_answer, GOLD)
        assert running.result() == "last\n"
        assert "submitted" in submitted.result()
    assert slow.closed and env.state.reward == 1.0


def test_tool_timeout_covers_a_cold_start_and_a_call(env, monkeypatch):
    seen = []
    handle = MCPEnvironment._async_handle_call_tool

    async def spy(self, action, timeout_s=None):
        seen.append(timeout_s)
        return await handle(self, action, timeout_s=timeout_s)

    monkeypatch.setattr(MCPEnvironment, "_async_handle_call_tool", spy)
    env.reset(task_id="t1")
    call(env, "run_python", code="1")
    env.step(
        CallToolAction(tool_name="run_python", arguments={"code": "1"}), timeout_s=5
    )
    # A cold start (spawner, then worker) plus the fixture's 10 s call limit.
    cold_start = sandbox.SPAWNER_STARTUP_TIMEOUT_S + sandbox.WORKER_START_TIMEOUT_S
    assert seen[0] is not None and seen[0] >= cold_start + 10
    assert seen[1] == 5


def test_model_code_cannot_read_another_task(env):
    other = env.reset(task_id="t3").metadata["files"][0]
    env.reset(task_id="t1")
    assert "not allowed" in text(
        call(env, "run_python", code=f"open({other!r}).read()")
    )


def test_tools_before_reset_are_errors():
    e = DataAnalysisEnvironment()
    try:
        assert "no task" in text(call(e, "run_python", code="1"))
        assert "no task" in text(call(e, "submit_answer", answer="1"))
    finally:
        e.close()


def test_reset_ignores_unknown_arguments(env):
    """Unknown reset kwargs are dropped by the server, never echoed back."""
    import inspect

    params = inspect.signature(env.reset).parameters
    assert not any(p.kind is p.VAR_KEYWORD for p in params.values())
