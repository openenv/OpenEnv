# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The engine is a per-rollout property, not a per-server one.

A dataset server is the expensive thing to keep alive: thousands of task files, prebuilt sandbox
templates. An inference engine is the cheap, changing part — it restarts every training run, and a
train-tier engine and an eval-tier one are usually both wanted against the same task suite. Pinning
the engine at boot made the durable thing hostage to the ephemeral one: no URL meant no capture proxy
at all, and every rollout answered "server not initialised".

So a caller names its engine when it mints a session, that engine is probed THEN (so the caller learns
its tier at submit time rather than when the token fields come back empty), and the measurement is
cached per engine so a whole GRPO group naming one vLLM pays for it once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

server = pytest.importorskip("openenv.core.harness.capture.server")
sessions = pytest.importorskip("openenv.core.harness.capture.sessions")


def app_with(monkeypatch, *, llm_url="", probe=None):
    """A capture app whose probe is stubbed, so no engine is contacted."""
    calls: list[str] = []

    def fake_probe(self, upstream):
        calls.append(upstream.llm_url)
        return (
            upstream.model or "stub-model",
            (probe or {}).get(upstream.llm_url, "tokens"),
        )

    monkeypatch.setattr(server.UpstreamPool, "_probe", fake_probe, raising=True)
    app = server.create_app(llm_url=llm_url, model="boot-model" if llm_url else None)
    app.state.admin_key = None  # these tests are about engines, not auth
    return app, calls


def test_a_server_boots_with_no_engine_at_all(monkeypatch):
    """The regression: this used to be unusable rather than merely engineless."""
    app, _ = app_with(monkeypatch)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        # Nothing to list, and an empty list is the honest answer rather than a 500.
        assert client.get("/v1/models").json() == {"object": "list", "data": []}


def test_naming_an_engine_probes_it_and_returns_the_tier(monkeypatch):
    app, calls = app_with(monkeypatch, probe={"http://train:8000": "tokens"})
    with TestClient(app) as client:
        body = client.post("/sessions", json={"llm_url": "http://train:8000"}).json()
    assert body["capture_level"] == "tokens"
    assert body["rollout_type"] == "train", "token-capable engine must be trainable"
    assert calls == ["http://train:8000"], "the engine should be probed exactly once"


def test_a_weaker_engine_comes_back_as_eval(monkeypatch):
    """Same server, same session route, different engine — the tier follows the engine."""
    app, _ = app_with(monkeypatch, probe={"http://evalonly:8000": "text"})
    with TestClient(app) as client:
        body = client.post("/sessions", json={"llm_url": "http://evalonly:8000"}).json()
    assert body["capture_level"] == "text"
    assert body["rollout_type"] == "eval"


def test_two_engines_coexist_on_one_server(monkeypatch):
    """The whole point: a trainer and an eval run share a server and get different tiers."""
    app, _ = app_with(
        monkeypatch, probe={"http://train:8000": "tokens", "http://eval:8000": "text"}
    )
    with TestClient(app) as client:
        train = client.post("/sessions", json={"llm_url": "http://train:8000"}).json()
        evaluate = client.post("/sessions", json={"llm_url": "http://eval:8000"}).json()
    assert (train["rollout_type"], evaluate["rollout_type"]) == ("train", "eval")
    assert train["session_id"] != evaluate["session_id"]


def test_the_probe_is_cached_per_engine(monkeypatch):
    """A GRPO group is N sessions on ONE engine; probing N times would add round trips per rollout."""
    app, calls = app_with(monkeypatch, probe={"http://train:8000": "tokens"})
    with TestClient(app) as client:
        for _ in range(5):
            client.post("/sessions", json={"llm_url": "http://train:8000"})
    assert calls == ["http://train:8000"], f"probed {len(calls)} times, expected 1"


def test_a_forgotten_engine_drops_its_client_and_is_probed_again(monkeypatch):
    """`forget` is for a caller whose key should not outlive their use of it."""
    app, calls = app_with(monkeypatch, probe={"http://train:8000": "tokens"})
    with TestClient(app) as client:
        client.post("/sessions", json={"llm_url": "http://train:8000", "api_key": "k"})
        pool = app.state.upstreams
        upstream = sessions.Upstream(llm_url="http://train:8000", api_key="k")
        pool.forget(upstream)
        assert pool.known() == []
        pool.forget(upstream)  # forgetting twice is harmless
        client.post("/sessions", json={"llm_url": "http://train:8000", "api_key": "k"})
    assert calls == ["http://train:8000", "http://train:8000"]


def test_a_session_with_no_engine_and_no_default_is_told_so(monkeypatch):
    """Better than forwarding to an empty base URL, which reads as a connection fault."""
    app, _ = app_with(monkeypatch)
    with TestClient(app) as client:
        sid = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {sid}"},
        )
    assert response.status_code == 503
    assert "no inference engine" in response.json()["error"]["message"]


def test_booting_with_an_engine_still_works(monkeypatch):
    """Backwards compatibility: the boot engine is the default for sessions that name none."""
    app, calls = app_with(monkeypatch, llm_url="http://default:8000")
    with TestClient(app) as client:
        body = client.post("/sessions", json={}).json()
    assert body["llm_url"] == "http://default:8000"
    assert body["capture_level"] == "tokens"  # create_app's default level
    assert calls == [], "naming no engine must not trigger a probe"


def test_health_lists_the_engines_it_has_measured(monkeypatch):
    app, _ = app_with(monkeypatch, probe={"http://train:8000": "tokens"})
    with TestClient(app) as client:
        client.post("/sessions", json={"llm_url": "http://train:8000"})
        upstreams = client.get("/health").json()["upstreams"]
    assert upstreams == [
        {
            "llm_url": "http://train:8000",
            "model": "stub-model",
            "capture_level": "tokens",
        }
    ]


def test_health_keeps_other_callers_engines_behind_the_admin_key(monkeypatch):
    """On a public Space, `/health` is public, and probed engines are other visitors' endpoints."""
    app, _ = app_with(monkeypatch, probe={"http://train:8000": "tokens"})
    app.state.admin_key = "admin"
    with TestClient(app) as client:
        client.post(
            "/sessions",
            json={"llm_url": "http://train:8000"},
            headers={"Authorization": "Bearer admin"},
        )
        assert client.get("/health").json()["upstreams"] == []
        listed = client.get(
            "/health", headers={"Authorization": "Bearer admin"}
        ).json()["upstreams"]
    assert [u["llm_url"] for u in listed] == ["http://train:8000"]


def test_an_unprobeable_engine_is_the_weakest_tier_not_a_crash():
    """`text` is the floor because claiming `tokens` without evidence is how an eval rollout gets
    stamped trainable — the one failure the capture level exists to prevent."""
    pool = server.UpstreamPool(default_client=None, default_level="tokens")
    model, level = pool._probe(
        sessions.Upstream(llm_url="http://nope.invalid:9/v1", model="m")
    )
    assert level == "text"
    assert model == "m"


def test_credentials_isolate_clients_without_exposing_secrets():
    """Credentials may select different tenants, quotas, and capabilities at the same URL."""
    a = sessions.Upstream(llm_url="http://x/v1", model="m", api_key="secret-a")
    b = sessions.Upstream(llm_url="http://x/v1", model="m", api_key="secret-b")
    assert a.cache_key != b.cache_key
    assert "secret-a" not in str(a.cache_key)
    assert "secret-a" not in repr(a)


def test_the_outgoing_model_comes_from_the_session_engine(monkeypatch):
    """The engine 404s on a mangled model name, and an engineless server used to send one.

    Harnesses rewrite the model: opencode is configured with `intercepted/<model>` and its provider
    layer forwards only the last path segment, so `Qwen/Qwen3.5-2B` arrives as `Qwen3.5-2B` and the
    engine answers `404 The model does not exist`. The proxy rewriting `model` is what makes the call
    work. Reading the SERVER's model to do it meant an engineless server skipped the rewrite entirely
    and every agent call 404'd — captured live before this test existed.
    """
    app, _ = app_with(monkeypatch)
    sent: dict = {}

    class FakeClient:
        served_model = "Qwen/Qwen3.5-2B"

        async def completion(self, request):
            sent.update(request)
            raise server.UpstreamError("stop here; the request is what matters")

    with TestClient(app) as client:
        body = client.post(
            "/sessions",
            json={"llm_url": "http://train:8000", "model": "Qwen/Qwen3.5-2B"},
        ).json()
        # Swap in a client that records what it was asked to send.
        key = sessions.Upstream(
            llm_url="http://train:8000", model="Qwen/Qwen3.5-2B"
        ).cache_key
        app.state.upstreams._by_engine[key] = (FakeClient(), "tokens")
        client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3.5-2B",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": f"Bearer {body['session_id']}"},
        )

    assert sent.get("model") == "Qwen/Qwen3.5-2B", (
        f"the mangled name reached the engine: {sent.get('model')!r}"
    )
