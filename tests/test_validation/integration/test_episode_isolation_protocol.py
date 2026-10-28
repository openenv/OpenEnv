"""Episode isolation over real WebSocket sessions whose resets leak nothing."""

import asyncio
import hashlib
import json
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.runner import run_validation
from openenv.validation.runtime.collector import (
    collect_runtime_evidence,
    MAX_TRACE_BYTES,
)
from openenv.validation.runtime.contracts import IsolationEvidence, RuntimePlan
from openenv.validation.types import Level
from support.runtime import FakeRuntimeProvider
from test_runtime_artifacts import rebuild
from test_runtime_execution import FIXTURE

PLAN = RuntimePlan.model_validate(
    {
        "plan_schema_version": "1",
        "reset": {"episode_id": "same", "seed": 42},
        "actions": [{"increment": 1}] * 5,
    }
)
EPISODE_TIMEOUT_S = 1.0
# Longer than one episode budget, but A and B together fit in two.
SLOW_STEP_S = 1.5


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
                    if mode == "slow_after_reset" and resets == 2 and steps == 1:
                        await asyncio.sleep(SLOW_STEP_S)
                if request["type"] == "state":
                    data = {"episode_id": "same", "step_count": steps}
                    response = {"type": "state", "data": data}
                else:
                    observation = {"counter": steps, "session_value": session_value}
                    if mode == "large":
                        # Under the message bound; each episode is about 5.4 MB.
                        observation["payload"] = "x" * 900_000
                    data = {"observation": observation, "reward": 0, "done": steps == 5}
                    response = {"type": "observation", "data": data}
                # The same values, encoded with another key order after a reset.
                reordered = mode == "reordered" and resets == 2
                raw = json.dumps(response, sort_keys=reordered)
                if request["type"] == "step" and mode == "precise_number":
                    number = "1.0" if resets == 2 else "1.0000000000000001"
                    raw = raw.replace(
                        '"counter": ' + str(steps), '"counter": ' + number
                    )
                if request["type"] == "step" and mode == "deep_json":
                    nested = "[" * 500 + "0" + "]" * 500
                    raw = raw.replace(
                        '"counter": ' + str(steps), '"counter": ' + nested
                    )
                    if resets == 2:
                        raw = raw.replace(":", ": ")
                await connection.send_text(raw)
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


def measure(mode, episode_timeout_s=10):
    with served(mode) as base_url:
        collect = lambda **k: collect_runtime_evidence(  # noqa: E731
            base_url, PLAN, episode_timeout_s=episode_timeout_s, **k
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


def test_deep_wire_json_ignores_only_formatting():
    subject = measure("deep_json")
    assert subject.isolation_evidence.sequential.failure_reason is None
    result = EpisodeIsolationGrader().run(subject)
    assert result.status.value == "pass", result.evidence


def test_precise_wire_divergence_survives_bundle_replay(tmp_path):
    with served("precise_number") as base_url:
        provider = FakeRuntimeProvider()
        provider.subject.base_url = base_url
        report = run_validation(
            FIXTURE, max_level=Level.RUNTIME, provider=provider, artifacts_dir=tmp_path
        )
    results = {result.check_id: result for result in report.results}
    assert results["runtime.startup"].status.value == "pass"
    assert provider.subject.stopped
    original = results["runtime.episode_isolation"]
    assert original.status.value == "fail", original.evidence
    replay = SimpleNamespace(
        runtime_evidence=rebuild(tmp_path),
        isolation_evidence=IsolationEvidence(
            sequential=rebuild(tmp_path, "isolation-sequential"),
            fresh=rebuild(tmp_path, "isolation-fresh"),
        ),
    )
    result = EpisodeIsolationGrader().run(replay)
    assert result.status == original.status
    assert result.evidence == original.evidence
    for session in ("collector", "isolation-sequential", "isolation-fresh"):
        metadata = json.loads((tmp_path / f"{session}-evidence.json").read_text())
        assert metadata["redacted"] is False
    for line in (tmp_path / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest


def test_per_session_variation_is_unproven_not_a_leak():
    # Sessions 1 and 3 (reference, fresh) see 1; session 2 sees 0 in both episodes.
    subject = measure("per_session")
    sequential = subject.isolation_evidence.sequential.exchanges
    half = len(sequential) // 2
    assert sequential[:half] == sequential[half:]
    result = EpisodeIsolationGrader().run(subject)
    assert result.status.value == "skip", result.evidence
    assert "two fresh episodes differ at exchange 0 (reset)" in result.evidence[0]


def test_each_episode_gets_the_full_trace_budget():
    subject = measure("large")
    episode = subject.runtime_evidence.exchanges
    size = sum(len(x.request_json) + len(x.response_json) for x in episode)
    assert MAX_TRACE_BYTES / 2 < size < MAX_TRACE_BYTES
    sequential = subject.isolation_evidence.sequential
    assert sequential.failure_reason is None
    assert len(sequential.exchanges) == 2 * len(episode)
    result = EpisodeIsolationGrader().run(subject)
    assert result.status.value == "pass", result.evidence


def test_each_episode_gets_its_own_deadline():
    # Time episode A saves must not let episode B outlast its own budget.
    subject = measure("slow_after_reset", episode_timeout_s=EPISODE_TIMEOUT_S)
    sequential = subject.isolation_evidence.sequential
    first = len(subject.runtime_evidence.exchanges)
    assert sequential.failure_reason == "step failed (TimeoutError)"
    assert [x.operation for x in sequential.exchanges[first:]] == ["reset", "state"]
    result = EpisodeIsolationGrader().run(subject)
    assert result.status.value == "skip", result.evidence
    assert result.evidence == [
        "the same-session episodes did not complete: step failed (TimeoutError)"
    ]
