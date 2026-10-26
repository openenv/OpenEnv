"""One shared proxy applies independent rollout caps without relaxing its own limit."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from openenv.core.harness.capture.server import create_app


class Engine:
    served_model = "test-model"
    capture_level = "text"
    param_fixes = {}

    async def completion(self, request):
        return {
            "id": "cap",
            "model": self.served_model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": str(request["max_tokens"]),
                    },
                    "finish_reason": "stop",
                }
            ],
        }


def test_parallel_rollouts_keep_distinct_caps_and_cannot_raise_server_limit():
    app = create_app(
        llm_url="http://unused.invalid",
        model="test-model",
        capture_level="text",
        max_output_tokens=16384,
    )
    app.state.upstreams._default = (Engine(), "text")
    caps = [4096, 16384, 32768]
    sessions = [app.state.registry.create(max_output_tokens=cap) for cap in caps]
    with TestClient(app) as client:

        def call(index):
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer " + sessions[index].session_id},
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 32768,
                },
            )
            assert response.status_code == 200
            return int(response.json()["choices"][0]["message"]["content"])

        with ThreadPoolExecutor(max_workers=3) as pool:
            assert list(pool.map(call, [0, 1, 2] * 3)) == [4096, 16384, 16384] * 3


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "4096"])
def test_invalid_session_budget_never_reaches_inference(cap):
    app = create_app(
        llm_url="http://unused.invalid", model="test-model", capture_level="text"
    )
    session = app.state.registry.create(max_output_tokens=cap)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + session.session_id},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    assert session.model_calls == 0
