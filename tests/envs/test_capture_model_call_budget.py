# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The per-session model-call budget.

Written because the thing it replaces was imaginary. `agent.build.steps` is a real key in opencode's
schema and is simply not honoured -- measured against a fake engine that always asks for one more
tool call, `steps=3`, `maxSteps=3` and no setting at all each produced 61 model calls. So the tests
that matter here are the two that distinguish a real cap from a decorative one: that the (n+1)th call
is never forwarded, and that it never enters the capture graph.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from openenv.core.harness.capture.server import create_app
from openenv.core.harness.capture.upstream import UpstreamHTTPError


class _CountingEngine:
    """Stands in for vLLM. Records what it was asked to do, so 'not forwarded' is observable."""

    def __init__(self) -> None:
        self.calls = 0
        self.served_model = "test-model"
        self.param_fixes: dict[str, Any] = {}
        self.capture_level = "text"

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "id": f"c{self.calls}",
            "object": "chat.completion",
            "model": self.served_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"turn {self.calls}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@pytest.fixture
def app_and_engine():
    app = create_app(
        llm_url="http://engine.invalid/v1", model="test-model", capture_level="text"
    )
    engine = _CountingEngine()
    app.state.inference = engine
    # The pool captured the real client at create_app time, so replacing `app.state.inference` alone
    # leaves every request going to `engine.invalid`. Sessions here name no upstream, so they take the
    # pool's default and this is the hook that matters.
    app.state.upstreams._default = (engine, "text")
    return app, engine


def _chat(client: TestClient, session_id: str) -> Any:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {session_id}"},
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )


@pytest.mark.parametrize("stream", [False, True])
def test_context_limit_stops_without_fabricating_a_captured_turn(
    app_and_engine, stream
):
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=17)
    with TestClient(app) as client:
        _chat(client, session.session_id)
        before = session.graph.stats()["n_turns"]

        async def too_long(request):
            raise UpstreamHTTPError(
                400,
                {
                    "error": {
                        "message": (
                            "This model's maximum context length is 131072 tokens. "
                            "However, you requested 4096 output tokens and your prompt contains "
                            "at least 126977 input tokens."
                        )
                    }
                },
            )

        engine.completion = too_long
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {session.session_id}"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "long"}],
                "stream": stream,
            },
        )
    assert response.status_code == 200
    assert session.budget_stop_count == 1
    assert session.upstream_errors == 0
    assert session.graph.stats()["n_turns"] == before
    assert any("context_budget_exhausted" in finding for finding in session.findings)
    from openenv.core.harness.capture.export import export_session

    document = export_session(session, capture_level="text")
    assert not any(
        "degenerate_rollout" in finding for finding in document["validation"]
    )
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert '"finish_reason":"stop"' in response.text.replace(" ", "")
    else:
        assert response.json()["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "upstream_status, expected_status", [(400, 400), (413, 413), (422, 422), (500, 502)]
)
def test_other_upstream_errors_are_not_converted_to_budget_stops(
    app_and_engine, upstream_status, expected_status
):
    app, engine = app_and_engine
    session = app.state.registry.create()

    async def invalid(request):
        raise UpstreamHTTPError(
            upstream_status, {"error": {"message": "invalid tools"}}
        )

    engine.completion = invalid
    with TestClient(app) as client:
        assert _chat(client, session.session_id).status_code == expected_status
    assert session.budget_stop_count == 0
    assert session.upstream_errors == 1
    assert session.graph.stats()["n_turns"] == 0


def test_single_turn_without_recorded_budget_stop_warns(app_and_engine):
    from openenv.core.harness.capture.export import export_session

    app, _ = app_and_engine
    session = app.state.registry.create()
    with TestClient(app) as client:
        _chat(client, session.session_id)
    document = export_session(session, capture_level="text")
    assert any(
        "[WARN] degenerate_rollout" in finding for finding in document["validation"]
    )


def test_budget_stop_does_not_make_an_empty_capture_valid(app_and_engine):
    from openenv.core.harness.capture.export import export_session

    app, _ = app_and_engine
    session = app.state.registry.create()
    session.budget_stop_count = 1
    document = export_session(session, capture_level="text")
    assert any("[FATAL] no_turns" in finding for finding in document["validation"])


def test_budget_stops_forwarding_at_the_cap(app_and_engine):
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=3)
    with TestClient(app) as client:
        for _ in range(5):
            assert _chat(client, session.session_id).status_code == 200
    # Five requests, three forwarded. Without the cap the engine would see all five.
    assert engine.calls == 3
    assert session.model_calls == 3


def test_the_capped_turn_is_terminal_and_never_captured(app_and_engine):
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=1)
    with TestClient(app) as client:
        _chat(client, session.session_id)
        over = _chat(client, session.session_id).json()

    # Terminal: this is what actually ends the agent's loop. opencode exits 0 on it.
    assert over["choices"][0]["finish_reason"] == "stop"
    assert not over["choices"][0]["message"].get("tool_calls")
    # Non-empty: an empty assistant message reads as a failed generation and is retried. See
    # `test_the_stop_message_is_not_empty`.
    assert over["choices"][0]["message"]["content"].strip()
    # And it is not in the graph. A synthetic turn in the training data is the failure this guards.
    assert session.graph.stats()["n_turns"] == 1

    # The harness may put the terminal response in ATIF. The independent cross-check needs
    # explicit evidence that this zero-token step came from the proxy, rather than the model.
    from openenv.core.harness.capture.export import export_session
    from openenv.harbor.atif import reconcile

    document = export_session(session, capture_level="text")
    trace = {
        "steps": [
            {
                "source": "agent",
                "message": "turn 1",
                "metrics": {"completion_tokens": 1},
            },
            {
                "source": "agent",
                "message": over["choices"][0]["message"]["content"],
                "metrics": over["usage"],
            },
        ]
    }
    assert document["budget_stop_count"] == 1
    assert not any(
        "degenerate_rollout" in finding for finding in document["validation"]
    )
    report = reconcile(document, trace)
    assert "proxy_budget_stops" in {f.code for f in report.findings}
    assert "atif_calls_missing" not in {f.code for f in report.findings}


def test_the_stop_is_streamed_when_the_caller_streams(app_and_engine):
    """A streaming caller must get SSE back, not a JSON body.

    This is the failure that made the cap useless in practice. opencode streams; answering it with a
    plain JSON body did not end its loop, so it retried, the proxy answered the stop again, and the
    rollout spun until its timeout -- "budget enforced" in the log, forever.
    """
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=1)
    with TestClient(app) as client:
        _chat(client, session.session_id)  # spends the budget
        over = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {session.session_id}"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert over.status_code == 200
    assert over.headers["content-type"].startswith("text/event-stream")
    body = over.text
    assert "data: " in body and "[DONE]" in body
    # The terminal signal has to be in the stream, or the loop never learns it should stop.
    assert '"finish_reason": "stop"' in body or '"finish_reason":"stop"' in body
    # And still nothing synthetic in the graph.
    assert session.graph.stats()["n_turns"] == 1


def test_the_stop_message_is_not_empty(app_and_engine):
    """An empty assistant message reads as a failed generation and gets retried."""
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=1)
    with TestClient(app) as client:
        _chat(client, session.session_id)
        over = _chat(client, session.session_id).json()
    assert over["choices"][0]["message"]["content"].strip()


def test_zero_means_unlimited(app_and_engine):
    app, engine = app_and_engine
    session = app.state.registry.create()
    assert session.max_model_calls == 0
    with TestClient(app) as client:
        for _ in range(6):
            _chat(client, session.session_id)
    assert engine.calls == 6
    assert not session.over_budget


def test_budget_is_per_session_not_per_server(app_and_engine):
    """One deployment serves a capped training run and an uncapped eval run at the same time."""
    app, engine = app_and_engine
    capped = app.state.registry.create(max_model_calls=2)
    uncapped = app.state.registry.create()
    with TestClient(app) as client:
        for _ in range(4):
            _chat(client, capped.session_id)
            _chat(client, uncapped.session_id)
    assert capped.model_calls == 2
    assert uncapped.model_calls == 4
