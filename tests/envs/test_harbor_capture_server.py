# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Port-ownership guarantees for the capture proxy.

A capture server that reports healthy while a *different* process owns its port is the worst
failure this layer has: sessions are minted in one registry and validated against another, so the
agent is rejected with 401, every rollout reports zero model calls, and the UI shows a live view
that can never advance. Nothing in that chain names the port, so these tests pin the invariant that
`start()` refuses rather than proceeds.

No credentials and no engine are needed: `start()` binds a socket and never contacts `llm_url`.

The last tests cover the standalone CLI (`python -m openenv.core.harness.capture.server`): it had no
way to set the admin key, so a port published from it served the session-management routes to
anyone who could reach it.
"""

from __future__ import annotations

import socket

import pytest

harbor_runner = pytest.importorskip("openenv.harbor.runner")
capture_server = pytest.importorskip("openenv.core.harness.capture.server")

CaptureServer = harbor_runner.CaptureServer

# Never contacted. Discard port, so a stray request would fail loudly rather than reach a real host.
UNUSED_ENGINE = "http://127.0.0.1:9/v1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def capture():
    """Yield a factory that tears every server it built back down."""
    built = []

    def make(port: int) -> CaptureServer:
        server = CaptureServer(llm_url=UNUSED_ENGINE, model="test-model", port=port)
        built.append(server)
        return server

    yield make
    for server in reversed(built):
        server.stop()


def test_health_reports_this_instance(capture):
    """`/health` must identify which app answered, so a probe can check identity not reachability."""
    import httpx

    server = capture(_free_port())
    server.start()

    payload = httpx.get(f"http://127.0.0.1:{server.port}/health", timeout=5.0).json()
    assert payload["instance"] == server.app.state.instance_id
    assert payload["status"] == "ok"


def test_instance_ids_are_distinct(capture):
    """Two apps must never share an id, or the identity probe cannot tell them apart."""
    first, second = capture(_free_port()), capture(_free_port())
    assert first.app.state.instance_id != second.app.state.instance_id


def test_start_refuses_a_port_another_server_holds(capture):
    """The regression: a second server on a held port used to report healthy.

    Its own uvicorn fails to bind on a background thread where nothing observes the error, while the
    liveness probe connects successfully to the *incumbent*. Reachability is not ownership.
    """
    port = _free_port()
    incumbent = capture(port)
    incumbent.start()

    intruder = capture(port)
    with pytest.raises(RuntimeError, match=f"{port}"):
        intruder.start()

    # The incumbent must be untouched: a failed start may not disturb a working server.
    import httpx

    payload = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5.0).json()
    assert payload["instance"] == incumbent.app.state.instance_id


def test_start_refuses_a_port_held_by_a_non_capture_listener(capture):
    """Any listener counts, not just another capture server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = int(squatter.getsockname()[1])

        with pytest.raises(RuntimeError, match="already in use"):
            capture(port).start()


def test_start_succeeds_on_a_free_port_after_a_refusal(capture):
    """A refusal must leave no state behind that breaks the next attempt."""
    port = _free_port()
    capture(port).start()

    with pytest.raises(RuntimeError):
        capture(port).start()

    recovered = capture(_free_port())
    recovered.start()
    assert recovered._thread is not None and recovered._thread.is_alive()


def test_stop_releases_the_port(capture):
    """Otherwise a restart in the same process hits the new guard and looks like a collision."""
    port = _free_port()
    server = capture(port)
    server.start()
    server.stop()

    successor = capture(port)
    successor.start()
    assert successor.app.state.instance_id != server.app.state.instance_id


def run_cli(monkeypatch, *argv: str) -> dict:
    """Run the CLI entry point with `argv`, returning the kwargs it built the app with.

    `--capture-level` is forced so the CLI never probes `UNUSED_ENGINE`; `uvicorn.run` is stubbed so
    nothing binds a port.
    """
    import uvicorn

    seen: dict = {}

    def fake_create_app(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(capture_server, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "sys.argv",
        ["server", "--llm-url", UNUSED_ENGINE, "--capture-level", "tokens", *argv],
    )
    capture_server.main()
    return seen


def test_cli_passes_admin_key_flag_to_the_app(monkeypatch):
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)

    seen = run_cli(monkeypatch, "--admin-key", "from-flag")

    assert seen["admin_key"] == "from-flag"


def test_cli_admin_key_defaults_to_the_env_var(monkeypatch):
    """Same variable `serve` and `rollout` read, so one setting covers every entry point."""
    monkeypatch.setenv("OPENENV_CAPTURE_ADMIN_KEY", "from-env")

    seen = run_cli(monkeypatch)

    assert seen["admin_key"] == "from-env"


def test_cli_default_loopback_leaves_admin_key_unset(monkeypatch, capsys):
    """Default `--host 127.0.0.1` may leave `/sessions*` ungated on a private local port."""
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)

    seen = run_cli(monkeypatch)

    assert seen["admin_key"] is None
    out = capsys.readouterr().out
    assert "admin key" not in out.lower()


def test_cli_refuses_non_loopback_without_admin_key(monkeypatch):
    """Non-loopback without a key must fail closed — never mint-and-print a credential."""
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)

    with pytest.raises(SystemExit, match="OPENENV_CAPTURE_ADMIN_KEY"):
        run_cli(monkeypatch, "--host", "0.0.0.0")


def test_cli_non_loopback_accepts_explicit_admin_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)

    seen = run_cli(monkeypatch, "--host", "0.0.0.0", "--admin-key", "explicit")

    assert seen["admin_key"] == "explicit"
    out = capsys.readouterr().out
    assert "explicit" not in out
    assert "admin key" not in out.lower()
