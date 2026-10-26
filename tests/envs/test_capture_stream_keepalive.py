"""Delayed streaming must stay connected without manufacturing captured tokens."""

import asyncio
import json
import socket
import threading

import httpx
import pytest
from openenv.core.harness.capture import sse
from openenv.core.harness.capture.detection import APIType
from openenv.core.harness.capture.export import export_session
from openenv.core.harness.capture.runner import CaptureServer
from starlette.responses import JSONResponse


class DelayedEngine:
    served_model = "test-model"
    capture_level = "tokens"
    param_fixes = {}
    api_key = None

    def __init__(self):
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.calls = 0

    async def completion(self, request):
        self.calls += 1
        assert request["stream"] is False
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {
            "id": "delayed",
            "object": "chat.completion",
            "model": self.served_model,
            "prompt_token_ids": [1, 2],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                    "token_ids": [3, 4],
                    "logprobs": {
                        "content": [
                            {"token": "3", "logprob": -0.25},
                            {"token": "4", "logprob": -0.5},
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }


@pytest.fixture
def delayed_server(monkeypatch):
    monkeypatch.setattr(sse, "KEEPALIVE_INTERVAL_S", 0.02)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = CaptureServer(
        llm_url="http://127.0.0.1:9/v1",
        model="test-model",
        port=port,
        capture_level="tokens",
    )
    engine = DelayedEngine()
    server.app.state.inference = engine
    server.app.state.upstreams._default = (engine, "tokens")
    session = server.app.state.registry.create(max_model_calls=2)
    server.start()
    try:
        yield server, engine, session
    finally:
        engine.release.set()
        server.stop()


def test_heartbeat_arrives_before_generation_then_exact_capture(delayed_server):
    server, engine, session = delayed_server
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{server.port}/v1/chat/completions",
        headers={"Authorization": "Bearer " + session.session_id},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        timeout=3,
    ) as response:
        assert response.status_code == 200
        lines = response.iter_lines()
        assert next(lines) == ": openenv keepalive"
        assert session.graph.stats()["n_turns"] == 0
        assert engine.calls == session.model_calls == 1
        engine.release.set()
        text = "\n".join(lines)
        assert "hello" in text and "data: [DONE]" in text
    document = export_session(session, capture_level="tokens")
    assert len(document["turns"]) == 1
    row = document["sequences"][0]
    assert row["input_ids"] == [1, 2, 3, 4]
    assert row["logprobs"] == [0, 0, -0.25, -0.5]
    assert row["loss_mask"] == [0, 0, 1, 1]


def test_disconnect_cancels_pending_capture(delayed_server):
    server, engine, session = delayed_server
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{server.port}/v1/chat/completions",
        headers={"Authorization": "Bearer " + session.session_id},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        timeout=3,
    ) as response:
        assert next(response.iter_lines()) == ": openenv keepalive"
    assert engine.cancelled.wait(2)
    assert session.graph.stats()["n_turns"] == 0


@pytest.mark.parametrize("dialect", list(APIType))
def test_late_error_is_an_error_event_without_completion(monkeypatch, dialect):
    monkeypatch.setattr(sse, "KEEPALIVE_INTERVAL_S", 0.001)

    async def check():
        async def failed():
            await asyncio.sleep(0.01)
            return JSONResponse(
                {"error": {"message": "engine failed"}}, status_code=502
            )

        response = await sse.keepalive_response(failed(), dialect)
        body = "".join([part async for part in response.body_iterator])
        assert ": openenv keepalive" in body
        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert len(events) == 1
        assert events[0].get("type") == "error" or "error" in events[0]
        assert (
            "engine failed" in body and "[DONE]" not in body and "assistant" not in body
        )

    asyncio.run(check())


def test_fast_error_keeps_http_status():
    async def check():
        async def failed():
            return JSONResponse(
                {"error": {"message": "invalid request"}}, status_code=400
            )

        response = await sse.keepalive_response(failed(), APIType.OPENAI_CHAT)
        assert response.status_code == 400

    asyncio.run(check())
