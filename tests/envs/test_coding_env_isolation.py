# SPDX-License-Identifier: BSD-3-Clause

"""Each coding_env session runs in its own worker process.

Code from one episode must not change what another episode or another
session sees, server secrets must not reach model code, and a runaway step
must only stop its own session.
"""

from __future__ import annotations

import gc
import os
import secrets
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("smolagents", reason="smolagents is not installed")

from coding_env.models import CodeAction
from coding_env.server import python_executor
from coding_env.server.python_codeact_env import PythonCodeActEnv

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux /proc and prctl"
)
root_only = pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.geteuid() != 0,
    reason="needs root on Linux",
)

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


def _uid(pid: int) -> int:
    with open(f"/proc/{pid}/status") as status:
        line = next(line for line in status if line.startswith("Uid:"))
    return int(line.split()[1])


def _read_as(uid: int, path: str) -> subprocess.CompletedProcess:
    """Read `path` from a new process running as `uid`."""
    user = {} if uid == os.geteuid() else dict(user=uid, group=uid, extra_groups=[])
    return subprocess.run(
        [sys.executable, "-c", "import sys; open(sys.argv[1], 'rb').read()", path],
        capture_output=True,
        text=True,
        cwd="/",
        **user,
    )


@pytest.fixture
def fresh_zygote(monkeypatch):
    """Start the next worker from a new zygote, and stop that zygote after."""
    monkeypatch.setattr(python_executor, "_zygote", None)
    yield
    if python_executor._zygote is not None:
        python_executor._zygote.close()


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


@root_only
def test_root_server_runs_workers_as_nobody(make_env):
    env = make_env()
    env.reset()
    _run(env, "1")  # the worker switches user right after the fork
    assert _uid(env._executor.pid) == python_executor.UNPRIVILEGED_UID
    # The zygote keeps the server's uid, so it can read the server's packages.
    assert _uid(python_executor._zygote._proc.pid) == 0


@root_only
def test_workers_cannot_signal_the_zygote_or_the_server(make_env):
    env = make_env(additional_imports=["os"])
    env.reset()
    for pid in (python_executor._zygote._proc.pid, os.getpid()):
        observation = _run(env, f"import os\nos.kill({pid}, 9)")
        assert observation.exit_code == 1
        assert "PermissionError" in observation.stderr
        assert _alive(pid)


def test_first_import_of_a_slow_module_does_not_stop_the_zygote(
    make_env, monkeypatch, fresh_zygote, tmp_path
):
    # Like the first session that authorizes torch: the zygote imports it
    # before it forks, which takes longer than an ordinary spawn.
    (tmp_path / "coding_env_slow_module.py").write_text(
        "import time\ntime.sleep(1.5)\nVALUE = 7\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(python_executor, "ZYGOTE_REPLY_TIMEOUT_S", 0.5)

    first = make_env()
    first.reset()
    zygote = python_executor._zygote
    slow = make_env(additional_imports=["coding_env_slow_module"])
    slow.reset()

    assert python_executor._zygote is zygote
    assert _run(first, "1 + 1").stdout == "2"
    observation = _run(
        slow, "import coding_env_slow_module\ncoding_env_slow_module.VALUE"
    )
    assert observation.stdout == "7", observation.stderr
    # Already imported in the zygote, so the next session gets the short budget.
    make_env(additional_imports=["coding_env_slow_module"]).reset()


def _running(pid: int) -> bool:
    """Alive and not a zombie waiting to be reaped."""
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_timeout_kill_does_not_wait_for_another_sessions_preload(
    make_env, monkeypatch, fresh_zygote, tmp_path
):
    (tmp_path / "coding_env_slower_module.py").write_text(
        "import time\ntime.sleep(4)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(python_executor, "ZYGOTE_REPLY_TIMEOUT_S", 0.5)
    runaway = make_env(timeout_s=1.0)
    runaway.reset()
    pid = runaway._executor.pid

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Holds the zygote lock for about 4 s while it imports the module.
        preload = pool.submit(
            make_env(additional_imports=["coding_env_slower_module"]).reset
        )
        time.sleep(0.5)
        started = time.monotonic()
        observation = _run(runaway, "for i in range(10**12):\n    pass")
        elapsed = time.monotonic() - started
        assert not _running(pid)
        preload.result()

    assert "TimeoutError" in observation.stderr, observation.stderr
    assert elapsed < 2.5, elapsed


@root_only
def test_authorized_modules_load_from_a_directory_nobody_cannot_read(
    make_env, monkeypatch, fresh_zygote, tmp_path
):
    # Like a virtual environment under /root/.cache/uv.
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (private / "coding_env_probe_module.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(private))

    env = make_env(additional_imports=["coding_env_probe_module"])
    env.reset()
    observation = _run(
        env, "import coding_env_probe_module\ncoding_env_probe_module.VALUE"
    )

    assert observation.stdout == "42", observation.stderr
    assert _uid(env._executor.pid) == python_executor.UNPRIVILEGED_UID


@linux_only
def test_workers_are_never_dumpable(make_env):
    env = make_env()
    env.reset()
    pid = env._executor.pid
    # Right after the fork, before the worker has run anything, read as the
    # user the worker runs as. A root server's worker is still root or already
    # nobody, non-dumpable either way.
    uid = python_executor.UNPRIVILEGED_UID if os.geteuid() == 0 else os.geteuid()
    for target in (pid, python_executor._zygote._proc.pid):
        for name in ("environ", "mem"):
            result = _read_as(uid, f"/proc/{target}/{name}")
            assert "PermissionError" in result.stderr, (target, name, result)


def test_temporary_files_go_to_the_session_home(make_env):
    env = make_env(additional_imports=["os", "tempfile"])
    env.reset()
    home = os.path.realpath(env._executor._home)

    observation = _run(
        env,
        "import os, tempfile\n"
        "print(tempfile.gettempdir(), os.environ['TMPDIR'], os.environ['TMP'], "
        "os.environ['TEMP'], os.environ['HOME'], os.getcwd())",
    )

    assert observation.exit_code == 0, observation.stderr
    paths = observation.stdout.split()
    assert len(paths) == 6
    assert {os.path.realpath(path) for path in paths} == {home}


def test_a_worker_that_cannot_start_is_reported_and_cleaned_up(make_env, monkeypatch):
    homes = []

    def mkdtemp(**kwargs):
        homes.append(real_mkdtemp(**kwargs))
        return homes[-1]

    def spawn(self, fd, config):
        raise OSError("no fork")

    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(python_executor.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(python_executor._Zygote, "spawn", spawn)
    env = make_env()

    observation = env.reset()
    assert observation.exit_code == 1
    assert observation.stderr == python_executor.START_FAILED_MESSAGE
    assert _run(env, "1").stderr == python_executor.START_FAILED_MESSAGE
    assert homes and not any(os.path.exists(home) for home in homes)

    monkeypatch.undo()
    assert _run(env, "1 + 1").stdout == "2"


def test_garbage_collected_env_stops_its_worker():
    env = PythonCodeActEnv()
    env.reset()
    pid, home = env._executor.pid, env._executor._home

    del env
    gc.collect()

    assert not _alive(pid)
    assert not os.path.exists(home)


def test_dead_zygote_is_replaced(make_env):
    env = make_env()
    env.reset()
    zygote = python_executor._zygote
    zygote._proc.kill()
    zygote._proc.wait()

    assert env.reset().exit_code == 0
    assert _run(env, "1 + 1").stdout == "2"
    assert python_executor._zygote is not zygote


@pytest.mark.skipif(
    bool(python_executor._socket_filter_unavailable()),
    reason="the socket filter exists for Linux on x86_64 and aarch64",
)
def test_workers_cannot_create_sockets(make_env):
    """Even with `socket` authorized, model code gets no socket of any family."""
    env = make_env(additional_imports=["socket"])
    env.reset()
    for family in ("AF_INET", "AF_INET6", "AF_UNIX"):
        result = _run(env, f"import socket\nsocket.socket(socket.{family})")
        assert result.exit_code != 0
        assert "not permitted" in result.stderr, result.stderr
    assert _run(env, "1 + 1").stdout == "2"
    with open(f"/proc/{env._executor.pid}/status") as status:
        assert "Seccomp:\t2" in status.read()
