"""Episode isolation over real WebSocket sessions whose resets leak nothing."""

import json
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.runtime.collector import collect_runtime_evidence
from openenv.validation.runtime.contracts import IsolationEvidence, RuntimePlan

PLAN = RuntimePlan.model_validate(
    {
        "plan_schema_version": "1",
        "reset": {"episode_id": "same", "seed": 42},
        "actions": [{"increment": 1}] * 5,
    }
)


@contextmanager
def served(mode):
    """A protocol server that resets all episode state; `mode` varies only the wire."""
    app = FastAPI()
    sessions = 0

    @app.get("/schema")
    def schema():
        return {"observation": {"type": "object"}}

    @app.websocket("/ws")
    async def ws(connection: WebSocket):
        nonlocal sessions
        sessions += 1
        session_value = sessions % 2 if mode == "per_session" else 0
        await connection.accept()
        resets = steps = 0
        try:
            while (request := await connection.receive_json())["type"] != "close":
                if request["type"] == "reset":
                    resets += 1
                    steps = 0
                elif request["type"] == "step":
                    steps += 1
                if request["type"] == "state":
                    data = {"episode_id": "same", "step_count": steps}
                    response = {"type": "state", "data": data}
                else:
                    observation = {"counter": steps, "session_value": session_value}
                    data = {"observation": observation, "reward": 0, "done": steps == 5}
                    response = {"type": "observation", "data": data}
                # The same values, encoded with another key order after a reset.
                reordered = mode == "reordered" and resets == 2
                await connection.send_text(json.dumps(response, sort_keys=reordered))
        except WebSocketDisconnect:
            pass

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
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
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


def measure(mode):
    with served(mode) as base_url:
        collect = lambda **k: collect_runtime_evidence(  # noqa: E731
            base_url, PLAN, episode_timeout_s=10, **k
        )
        return SimpleNamespace(
            runtime_evidence=collect(),
            isolation_evidence=IsolationEvidence(
                sequential=collect(episodes=2), fresh=collect()
            ),
        )


def test_reencoded_responses_after_reset_match_a_fresh_episode():
    subject = measure("reordered")
    sequential = subject.isolation_evidence.sequential.exchanges
    after_reset = sequential[len(sequential) // 2 :]
    fresh = subject.isolation_evidence.fresh.exchanges
    assert after_reset[0].response_json != fresh[0].response_json
    assert [json.loads(x.response_json) for x in after_reset] == [
        json.loads(x.response_json) for x in fresh
    ]
    result = EpisodeIsolationGrader().run(subject)
    assert result.status.value == "pass", result.evidence


def test_per_session_variation_is_unproven_not_a_leak():
    # Sessions 1 and 3 (reference, fresh) see 1; session 2 sees 0 in both episodes.
    subject = measure("per_session")
    sequential = subject.isolation_evidence.sequential.exchanges
    half = len(sequential) // 2
    assert sequential[:half] == sequential[half:]
    result = EpisodeIsolationGrader().run(subject)
    assert result.status.value == "skip", result.evidence
    assert "two fresh episodes differ at exchange 0 (reset)" in result.evidence[0]
