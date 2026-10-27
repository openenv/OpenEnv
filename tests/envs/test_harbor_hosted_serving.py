# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serving on a hosted platform, where there is one port and one URL.

Locally the capture proxy runs on its own port and is published to the sandbox. A Space exposes
exactly one port and already has a public URL, so the proxy is mounted onto the env server's app
instead and nothing is forwarded. These tests pin that split, because getting it wrong is not a
crash: it is a deployment that quietly opens a second listener it cannot publish.
"""

from __future__ import annotations

import pytest

serving = pytest.importorskip("openenv.harbor.serving")

CAPTURE_MOUNT = serving.CAPTURE_MOUNT
HarborService = serving.HarborService
space_public_url = serving.space_public_url

# Never contacted: nothing here reaches an engine.
UNUSED_LLM = "http://127.0.0.1:9/v1"


@pytest.fixture(autouse=True)
def _clear_space_env(monkeypatch):
    """Tests must not inherit a Space identity from the developer's shell."""
    monkeypatch.delenv("SPACE_HOST", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)


def test_no_space_means_no_public_url():
    assert space_public_url() == ""


def test_space_host_is_used_verbatim(monkeypatch):
    monkeypatch.setenv("SPACE_HOST", "owner-env.hf.space")
    assert space_public_url() == "https://owner-env.hf.space"


def test_space_host_tolerates_a_scheme_already_present(monkeypatch):
    monkeypatch.setenv("SPACE_HOST", "https://owner-env.hf.space/")
    assert space_public_url() == "https://owner-env.hf.space"


def test_space_id_is_slugged_when_host_is_absent(monkeypatch):
    """`SPACE_ID` is always set; the hostname lowercases and dash-separates it."""
    monkeypatch.setenv("SPACE_ID", "AdithyaSK/harbor_data.agent-env")
    assert space_public_url() == "https://adithyask-harbor-data-agent-env.hf.space"


def test_hosted_start_mounts_and_never_forwards(monkeypatch):
    """The regression that got a Space flagged: a hosted deployment must not forward."""
    monkeypatch.setenv("SPACE_ID", "owner/env")

    def explode(*_args, **_kwargs):
        raise AssertionError("a hosted deployment must not create a forwarder")

    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder", explode, raising=False
    )

    service = HarborService(llm_url=UNUSED_LLM, model="m", datasets=[])
    url = service.start()

    assert service.mounted is True
    assert url == f"https://owner-env.hf.space{CAPTURE_MOUNT}"
    # No port was bound, so stop() must be safe even though start() never launched a server.
    service.stop()


def test_mounted_capture_answers_under_the_prefix():
    """Mounting strips the prefix, so every dialect route keeps working unchanged."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openenv.core.harness.capture.server import create_app

    host = fastapi.FastAPI()
    host.mount(CAPTURE_MOUNT, create_app(llm_url=UNUSED_LLM, model="m"))
    client = TestClient(host)

    assert client.get(f"{CAPTURE_MOUNT}/health").json()["status"] == "ok"

    # The proxy's catch-all must match /v1/chat/completions, not /capture/v1/chat/completions.
    response = client.post(
        f"{CAPTURE_MOUNT}/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer not-a-session"},
    )
    # 401 rather than 404 proves it routed to the proxy and was rejected on identity, which is also
    # what stops a publicly mounted proxy being an open relay.
    assert response.status_code == 401
    assert "unknown API key" in response.json()["error"]["message"]


def test_harbor_app_serves_its_own_state_class(monkeypatch):
    """Harbor's package-backed app must retain custom state fields over HTTP."""
    from fastapi.testclient import TestClient
    from openenv.harbor.environment import HarborEnvironment
    from openenv.harbor.models import HarborState

    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    monkeypatch.setattr(HarborService, "_instance", None)
    # build_app configures these class attributes; restore the previous config afterward.
    for name in ("_datasets", "_llm_url", "_model", "_llm"):
        monkeypatch.setattr(HarborEnvironment, name, getattr(HarborEnvironment, name))

    app = serving.build_app(datasets=[], llm_url=UNUSED_LLM)
    with TestClient(app) as client:
        schema = client.get("/schema")
        state = client.get("/state")

    assert schema.status_code == state.status_code == 200
    expected_fields = set(HarborState.model_fields)
    assert expected_fields <= set(schema.json()["state"]["properties"])
    assert expected_fields <= set(state.json())
    assert state.json()["llm_url"] == UNUSED_LLM


def test_a_failed_forwarder_does_not_leave_the_capture_server_running(monkeypatch):
    """A half-started service poisons every later attempt.

    The capture server binds its port and starts a thread before the forwarder is built. If the
    forwarder then fails, leaving that server up means the next `start()` fails on a port conflict
    that says nothing about the real error.
    """
    stopped: list[bool] = []

    class FakeCapture:
        port = 8123

        def start(self):
            pass

        def stop(self):
            stopped.append(True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("cloudflared is not installed")

    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder", explode, raising=False
    )

    service = HarborService(llm_url=UNUSED_LLM, model="m", datasets=[])
    service.capture = FakeCapture()

    with pytest.raises(RuntimeError, match="cloudflared"):
        service.start()

    assert stopped == [True], "the capture server was left holding its port"
    assert service.public_url in (None, "")


def test_a_space_that_cannot_probe_does_not_claim_to_be_trainable(
    monkeypatch, tmp_path
):
    """The Space entry point defaulted `_CAPTURE_LEVEL` to "tokens" and only corrected it when a model
    resolved AND the probe succeeded. An ambiguous model list, an unset model or a raising probe left
    it at token level, so the proxy was built for capture and every rollout was stamped trainable —
    the exact mislabelling the capture level exists to prevent. Unknown must mean the weaker tier.
    """
    import runpy

    monkeypatch.setenv("OPENENV_LLM_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENENV_MODEL", raising=False)
    monkeypatch.delenv("SPACE_HOST", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    # Serves two models and names neither, so nothing resolves and the probe never runs.
    #
    # Patched through the module OBJECT, not the dotted string: the package re-exports a function
    # called `validate_llm` which shadows the same-named submodule, so the string form resolves to the
    # function and monkeypatch fails with "'function' object has no attribute 'list_models'".
    import importlib

    validate_llm_mod = importlib.import_module(
        "openenv.core.harness.capture.validate_llm"
    )
    monkeypatch.setattr(validate_llm_mod, "list_models", lambda *a, **k: ["one", "two"])
    # The service must not be started for real; capture the level it would have been built with.
    built: dict = {}

    class FakeService:
        def __init__(self, **kwargs):
            built.update(kwargs)

        def start(self):
            return ""

        @classmethod
        def set_current(cls, _service):
            pass

    monkeypatch.setattr(serving, "HarborService", FakeService)
    monkeypatch.setattr(serving, "build_app", lambda **kwargs: kwargs)

    runpy.run_module("harbor_env.server.app", run_name="not_main")
    assert built.get("capture_level") == "text", (
        "an unprobed endpoint must default to the weakest tier, never to tokens"
    )
