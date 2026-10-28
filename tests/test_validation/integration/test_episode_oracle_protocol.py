"""Episode isolation and oracle containment over the real OpenEnv WebSocket endpoint."""

import importlib.util
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.graders.runtime.oracle import OracleContainmentGrader
from openenv.validation.runner import _get
from openenv.validation.runtime.collector import collect_runtime_evidence
from openenv.validation.runtime.contracts import (
    IsolationEvidence,
    RuntimePlan,
    RuntimeReset,
)

ROOT = Path(__file__).parents[2] / "fixtures/validation/runtime/served_probe"
spec = importlib.util.spec_from_file_location("served_probe_app", ROOT / "app.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

PLAN = RuntimePlan(
    plan_schema_version="1",
    reset=RuntimeReset(episode_id="validation-probe", seed=42),
    actions=[{"increment": 1}, {"increment": 1}],
)


@contextmanager
def served(mode, monkeypatch):
    monkeypatch.setenv("VALIDATION_FAULT", mode)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(fixture.make_app(mode), log_level="error", lifespan="on")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


def measure(base_url):
    collect = lambda **k: collect_runtime_evidence(  # noqa: E731
        base_url, PLAN, episode_timeout_s=10, **k
    )
    return SimpleNamespace(
        root=ROOT,
        manifest=SimpleNamespace(
            capabilities=SimpleNamespace(
                oracle=SimpleNamespace(location="oracle/solve.sh")
            )
        ),
        runtime_evidence=collect(),
        isolation_evidence=IsolationEvidence(
            sequential=collect(episodes=2), fresh=collect()
        ),
        oracle_http=(_get(base_url, "oracle/solve.sh"),),
    )


@pytest.mark.parametrize(
    "mode, isolation, oracle",
    [
        ("good", "pass", "pass"),
        ("sticky_episode", "fail", "pass"),
        ("sticky_file", "fail", "pass"),
        ("sticky_background", "fail", "pass"),
        ("exposed_oracle", "pass", "fail"),
    ],
)
def test_real_session_isolation_and_oracle(monkeypatch, mode, isolation, oracle):
    with served(mode, monkeypatch) as base_url:
        subject = measure(base_url)
    isolated = EpisodeIsolationGrader().run(subject)
    contained = OracleContainmentGrader().run(subject)
    assert isolated.status.value == isolation, isolated.evidence
    assert contained.status.value == oracle, contained.evidence
    if mode in {"sticky_episode", "sticky_file", "sticky_background"}:
        assert "exchange 2 (step)" in isolated.evidence[0]
    if mode == "exposed_oracle":
        assert contained.evidence == [
            "oracle content is readable through the API: GET /oracle/solve.sh"
        ]
