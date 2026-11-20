# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The model's code runs in a worker process that holds no server secrets."""

import os
import sys

import pytest

pytest.importorskip("smolagents")

from repl_env.models import REPLAction  # noqa: E402
from repl_env.server.python_executor import PythonExecutor  # noqa: E402
from repl_env.server.repl_environment import REPLEnvironment  # noqa: E402

FAKE_SERVER_TOKEN = "hf_FAKEservertoken0000000000000000000"
FAKE_CLIENT_TOKEN = "hf_FAKEclienttoken0000000000000000000"

# `os` stands in for anything model code could reach in its own process.
SHOW_ENV = "import os\nprint(dict(os.environ))"


def test_model_code_does_not_see_server_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_SERVER_TOKEN)
    monkeypatch.setenv("MY_API_KEY", FAKE_SERVER_TOKEN)
    executor = PythonExecutor(allowed_imports=["os"])
    try:
        result = executor.execute(SHOW_ENV)
    finally:
        executor.close()
    assert result["success"], result["stderr"]
    assert "HF_TOKEN" not in result["stdout"]
    assert FAKE_SERVER_TOKEN not in result["stdout"]


def test_model_code_does_not_see_the_client_token(monkeypatch):
    import repl_env.server.python_executor as python_executor

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        python_executor, "DEFAULT_IMPORTS", [*python_executor.DEFAULT_IMPORTS, "os"]
    )
    env = REPLEnvironment()
    try:
        env.reset(hf_token=FAKE_CLIENT_TOKEN)
        assert "llm_query" in env.state.namespace_keys
        obs = env.step(REPLAction(code=f"{SHOW_ENV}\nprint(llm_query)"))
    finally:
        env.close()
    assert obs.result.success, obs.result.stderr
    assert FAKE_CLIENT_TOKEN not in obs.result.stdout


def test_injected_functions_run_in_the_server():
    calls = []

    def llm_query(prompt, model=None):
        calls.append((os.getpid(), prompt, model))
        return f"answer to {prompt}"

    executor = PythonExecutor()
    try:
        executor.inject_function("llm_query", llm_query)
        result = executor.execute("x = llm_query('hi', model='m')\nprint(x)")
        assert result["success"], result["stderr"]
        assert "answer to hi" in result["stdout"]
        assert executor.get_variable("x") == "answer to hi"
        assert calls == [(os.getpid(), "hi", "m")]

        failing = executor.execute("llm_query()")
        assert not failing["success"]
        assert "TypeError" in failing["stderr"]
    finally:
        executor.close()


def test_finalization_helpers_run_in_the_worker():
    env = REPLEnvironment()
    try:
        env.reset(context="hello")
        obs = env.step(REPLAction(code="n = len(context)\nprint(FINAL_VAR('n'))"))
        assert obs.done
        assert obs.metadata["final_answer"] == "5"
    finally:
        env.close()


def test_reset_and_close_stop_the_worker():
    executor = PythonExecutor()
    first = executor._proc
    executor.execute("x = 1")
    executor.reset()
    assert first.poll() is not None
    assert executor.get_variable("x") is None
    second = executor._proc
    executor.close()
    assert second.poll() is not None
    with pytest.raises(RuntimeError):
        executor.get_variable("x")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc")
def test_worker_cannot_read_the_server_process(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", FAKE_SERVER_TOKEN)
    executor = PythonExecutor(allowed_imports=["os"])
    try:
        for what in ("environ", "mem"):
            result = executor.execute(
                f"import os\nos.open('/proc/{os.getpid()}/{what}', os.O_RDONLY)"
            )
            assert not result["success"]
            assert "PermissionError" in result["stderr"]
        if os.geteuid() == 0:
            ids = executor.execute("import os\nprint(os.getuid(), os.getgroups())")
            assert ids["stdout"].strip() == "65534 []"
    finally:
        executor.close()
