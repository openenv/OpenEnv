# SPDX-License-Identifier: BSD-3-Clause

"""Each coding_env session runs in its own worker process.

Code from one episode must not change what another episode or another
session sees, server secrets must not reach model code, and a runaway step
must only stop its own session.
"""

from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("smolagents", reason="smolagents is not installed")

from coding_env.models import CodeAction
from coding_env.server.python_codeact_env import PythonCodeActEnv

# Each pair patches shared state of an imported module (not a session
# variable), then checks it from a clean interpreter.
PATCH_JSON = "import json\njson.JSONEncoder.item_separator = '!!'"
CHECK_JSON = "import json\njson.dumps([1, 2])"
PATCH_QUEUE = "import queue\nqueue.Queue.patched = lambda self: 'patched'"
CHECK_QUEUE = "import queue\nhasattr(queue.Queue, 'patched')"


def _run(env: PythonCodeActEnv, code: str):
    return env.step(CodeAction(code=code))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def make_env():
    envs = []

    def factory(**kwargs) -> PythonCodeActEnv:
        env = PythonCodeActEnv(**kwargs)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()


def test_patched_module_state_does_not_survive_reset(make_env):
    env = make_env()
    env.reset()
    assert _run(env, PATCH_JSON).exit_code == 0
    assert _run(env, PATCH_QUEUE).exit_code == 0
    assert "!!" in _run(env, CHECK_JSON).stdout

    env.reset()

    assert _run(env, CHECK_JSON).stdout == '"[1, 2]"'
    assert _run(env, CHECK_QUEUE).stdout == "false"


def test_reset_replaces_the_worker_process(make_env):
    env = make_env()
    env.reset()
    _run(env, "1")
    old_pid = env._executor.pid

    env.reset()
    _run(env, "1")

    assert env._executor.pid != old_pid
    assert not _alive(old_pid)


def test_concurrent_sessions_do_not_share_interpreter_state(make_env):
    env_a, env_b = make_env(), make_env()
    env_a.reset()
    env_b.reset()

    for code in (PATCH_JSON, PATCH_QUEUE, "x = 41"):
        assert _run(env_a, code).exit_code == 0

    def session_b():
        return [_run(env_b, code) for code in (CHECK_JSON, CHECK_QUEUE, "print(x)")]

    # Both sessions step at the same time, each on its own thread, the way
    # the server runs concurrent WebSocket sessions.
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(_run, env_a, "import time\ntime.sleep(1)\n" + CHECK_JSON)
        b_json, b_queue, b_var = pool.submit(session_b).result()
        assert "!!" in a.result().stdout

    assert b_json.stdout == '"[1, 2]"'
    assert b_queue.stdout == "false"
    assert b_var.exit_code == 1

    assert env_a._executor.pid != env_b._executor.pid
    # The server process itself is untouched too.
    import json
    import queue

    assert json.dumps([1, 2]) == "[1, 2]"
    assert not hasattr(queue.Queue, "patched")


def test_server_secrets_are_not_visible_to_model_code(make_env, monkeypatch):
    value = secrets.token_hex(16)
    names = [
        "CODING_ENV_TEST_TOKEN",
        "CODING_ENV_TEST_API_KEY",
        "CODING_ENV_TEST_SECRET",
    ]
    for name in names:
        monkeypatch.setenv(name, value)

    env = make_env(additional_imports=["os"])
    env.reset()
    observation = _run(env, "import os\nprint(sorted(os.environ))\ndict(os.environ)")

    assert observation.exit_code == 0, observation.stderr
    for name in names:
        assert name not in observation.stdout
    assert value not in observation.stdout


def test_timeout_kills_only_that_sessions_worker(make_env):
    hung = make_env(timeout_s=2)
    other = make_env()
    hung.reset()
    other.reset()
    assert _run(hung, "y = 1").exit_code == 0
    assert _run(other, "z = 7").exit_code == 0
    hung_pid, other_pid = hung._executor.pid, other._executor.pid

    with ThreadPoolExecutor(max_workers=1) as pool:
        loop = pool.submit(_run, hung, "for i in range(10**12):\n    pass")
        # The other session keeps working while the loop runs.
        assert _run(other, "z + 1").stdout == "8"
        observation = loop.result(timeout=30)

    assert observation.exit_code == 1
    assert "TimeoutError" in observation.stderr
    assert not _alive(hung_pid)
    assert _alive(other_pid)
    assert _run(other, "z").stdout == "7"

    # The next step starts a fresh worker, without the old variables.
    recovered = _run(hung, "print(y)")
    assert recovered.exit_code == 1
    assert _run(hung, "1 + 1").stdout == "2"
    assert hung._executor.pid != hung_pid


def test_sleeping_step_times_out(make_env):
    env = make_env(timeout_s=1)
    env.reset()
    observation = _run(env, "import time\ntime.sleep(60)")
    assert observation.exit_code == 1
    assert "TimeoutError" in observation.stderr
    assert env.state.last_exit_code == 1


def test_close_stops_the_worker(make_env):
    env = make_env()
    env.reset()
    _run(env, "1")
    pid = env._executor.pid

    env.close()

    assert not _alive(pid)


def test_env_declares_concurrent_sessions():
    assert PythonCodeActEnv.SUPPORTS_CONCURRENT_SESSIONS is True


def test_server_runs_isolated_concurrent_websocket_sessions():
    from coding_env.server.app import app
    from fastapi.testclient import TestClient

    def step(socket, code):
        socket.send_json({"type": "step", "data": {"code": code}})
        return socket.receive_json()["data"]["observation"]

    client = TestClient(app)
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        for socket in (a, b):
            socket.send_json({"type": "reset", "data": {}})
            assert socket.receive_json()["type"] == "observation"

        assert step(a, PATCH_JSON)["exit_code"] == 0
        assert step(b, CHECK_JSON)["stdout"] == '"[1, 2]"'

        for socket in (a, b):
            socket.send_json({"type": "close"})
