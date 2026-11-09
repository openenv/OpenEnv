"""Supported isolation violations survive unrelated limits in real wire evidence."""

import json
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.runner import run_validation
from openenv.validation.runtime.collector import (
    collect_runtime_evidence,
    RuntimeCollectionUnsupported,
)
from openenv.validation.runtime.contracts import IsolationEvidence, RuntimePlan
from openenv.validation.types import CheckStatus, Level
from support.runtime import FakeRuntimeProvider
from test_runtime_execution import FIXTURE

PLAN = RuntimePlan.model_validate(
    {
        "plan_schema_version": "1",
        "reset": {"episode_id": "validation-probe", "seed": 42},
        "actions": [{"increment": 1}] * 2,
    }
)


@contextmanager
def served(kind, leak, unknown_session="sequential"):
    app = FastAPI()
    sessions = 0

    @app.get("/schema")
    def schema():
        return {"observation": {"type": "object"}}

    @app.websocket("/ws")
    async def ws(connection: WebSocket):
        nonlocal sessions
        sessions += 1
        session = sessions
        await connection.accept()
        resets = steps = 0
        try:
            while (request := await connection.receive_json())["type"] != "close":
                operation = request["type"]
                if operation == "reset":
                    resets += 1
                    steps = 0
                elif operation == "step":
                    steps += 1
                if operation == "state":
                    response = {
                        "type": "state",
                        "data": {"episode_id": "validation-probe", "step_count": steps},
                    }
                else:
                    counter = (
                        1 if leak and resets == 2 and operation == "reset" else steps
                    )
                    response = {
                        "type": "observation",
                        "data": {
                            "observation": {"counter": counter},
                            "reward": 0,
                            "done": False,
                        },
                    }
                raw = json.dumps(response)
                unsupported = (
                    resets == 2 if unknown_session == "sequential" else session == 3
                )
                if unsupported and operation == "step" and steps == 1:
                    value = {
                        "numeric_range": "1e9999999999999999999",
                        "integer_digits": "7" * 5_000,
                        "depth": "[" * 10_000 + "0" + "]" * 10_000,
                    }[kind]
                    raw = raw.replace('"counter": 1', '"counter": ' + value)
                await connection.send_text(raw)
        except WebSocketDisconnect:
            # The collector closes sessions after complete or unsupported evidence.
            pass

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("kind", ["numeric_range", "depth", "integer_digits"])
@pytest.mark.parametrize("leak", [False, True])
@pytest.mark.parametrize("unknown_session", ["sequential", "fresh"])
def test_real_collector_preserves_supported_reset_verdict(kind, leak, unknown_session):
    with served(kind, leak, unknown_session) as base_url:

        def collect(**kwargs):
            return collect_runtime_evidence(
                base_url, PLAN, episode_timeout_s=10, **kwargs
            )

        def limited_collect(**kwargs):
            offset = 6 if unknown_session == "sequential" else 0
            if kind == "depth":
                # The production collector stops at depth exhaustion and carries
                # the raw failed exchange as well as the supported earlier reset.
                with pytest.raises(RuntimeCollectionUnsupported) as caught:
                    collect(**kwargs)
                measured = caught.value.evidence
                assert measured.failure_phase == "step"
                assert "json_depth" in measured.failure_reason
                assert len(measured.exchanges) == offset + 3
            elif kind == "integer_digits":
                measured = collect(**kwargs)
                assert measured.failure_reason == "step failed (ValueError)"
                assert len(measured.exchanges) == offset + 3
            else:
                # Stdlib envelope parsing accepts this number. Only the grader's
                # exact Decimal parser discovers the unsupported range.
                measured = collect(**kwargs)
                assert measured.failure_reason is None
                assert len(measured.exchanges) == offset + 6
            return measured

        reference = collect()
        sequential = (
            limited_collect(episodes=2)
            if unknown_session == "sequential"
            else collect(episodes=2)
        )
        fresh = limited_collect() if unknown_session == "fresh" else collect()
        subject = SimpleNamespace(
            runtime_evidence=reference,
            isolation_evidence=IsolationEvidence(sequential=sequential, fresh=fresh),
        )
    result = EpisodeIsolationGrader().run(subject)
    assert result.status is (CheckStatus.FAIL if leak else CheckStatus.SKIP), (
        result.evidence
    )
    if leak:
        assert "exchange 0 (reset)" in result.evidence[0]


def test_complete_numeric_range_trace_reaches_isolation_through_runner(tmp_path):
    with served("numeric_range", leak=True) as base_url:
        provider = FakeRuntimeProvider()
        provider.subject.base_url = base_url
        report = run_validation(
            FIXTURE, max_level=Level.RUNTIME, provider=provider, artifacts_dir=tmp_path
        )
    results = {result.check_id: result for result in report.results}
    assert results["runtime.startup"].status is CheckStatus.PASS
    isolated = results["runtime.episode_isolation"]
    assert isolated.status is CheckStatus.FAIL, isolated.evidence
    assert "exchange 0 (reset)" in isolated.evidence[0]
    assert provider.subject.stopped
