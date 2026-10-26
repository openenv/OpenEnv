# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`openenv harbor rollout` must not strand its capture proxy when setup fails.

`run_batch` starts the capture server — which binds a port in a background thread — and only then
builds the tunnel the sandbox reaches it through. Anything that throws in between used to escape with
that thread still running, so the port stayed held and the *next* invocation died on a port conflict
that named nothing about the actual cause. Teardown had the mirror of the same gap: `forwarder.stop()`
ran before `capture.stop()` without a guard, so a tunnel that failed to shut down took the port with it.

The same entry point also publishes the proxy through a tunnel, so its session-management routes are
reachable by anyone with the URL. `_admin_ok` admits every caller while no admin key is set, which
made `run_batch` an open control plane: the tests at the bottom pin that it always hands
`CaptureServer` a key.

These tests stub every external dependency; no engine, no sandbox and no network are involved.
"""

from __future__ import annotations

import asyncio

import pytest

runner = pytest.importorskip("openenv.harbor.runner")


class FakeCaps:
    llm = {"model": "m", "capture_level": "tokens"}
    available_sandboxes = ("e2b",)
    sandboxes: tuple = ()


def wire(monkeypatch, tmp_path, forwarder):
    """Replace startup, the capture server and the dataset with local stubs.

    Returns the teardown log and the list of kwargs each `CaptureServer` was built with.
    """
    stopped: list[str] = []
    built: list[dict] = []

    monkeypatch.setattr(runner, "resolve_task_dirs", lambda _d: [tmp_path])
    monkeypatch.setattr(
        "openenv.harbor.startup.prepare", lambda **_k: FakeCaps(), raising=False
    )

    class FakeCapture:
        registry = None
        inference = None

        def __init__(self, **kwargs):
            built.append(kwargs)
            self.port = kwargs.get("port", 8100)

        def start(self):
            pass

        def stop(self):
            stopped.append("capture")

    monkeypatch.setattr(runner, "CaptureServer", FakeCapture)
    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder",
        lambda _kind: forwarder(stopped),
        raising=False,
    )
    return stopped, built


class Quiet:
    """A forwarder that starts and stops without incident."""

    name = "gradio"

    def __init__(self, _stopped):
        pass

    def start(self, _port):
        return "https://tunnel.invalid"

    def stop(self):
        pass


def test_a_forwarder_that_cannot_start_releases_the_port(monkeypatch, tmp_path):
    class Exploding:
        name = "cloudflare"

        def __init__(self, _stopped):
            pass

        def start(self, _port):
            raise RuntimeError("cloudflared is not installed")

    stopped, _ = wire(monkeypatch, tmp_path, Exploding)

    with pytest.raises(RuntimeError, match="cloudflared"):
        asyncio.run(
            runner.run_batch(llm_url="http://x/v1", dataset="d", task_indices=[0])
        )

    assert stopped == ["capture"], (
        "the capture server kept its port after a failed forwarder"
    )


def test_a_forwarder_that_cannot_stop_still_lets_the_port_go(monkeypatch, tmp_path):
    class BadTeardown:
        name = "gradio"

        def __init__(self, stopped):
            self._stopped = stopped

        def start(self, _port):
            return "https://tunnel.invalid"

        def stop(self):
            self._stopped.append("forwarder")
            raise RuntimeError("tunnel already gone")

    stopped, _ = wire(monkeypatch, tmp_path, BadTeardown)
    monkeypatch.setattr(
        runner, "run_rollout", None
    )  # never reached: no indices are in range

    with pytest.raises(RuntimeError, match="tunnel already gone"):
        asyncio.run(
            runner.run_batch(llm_url="http://x/v1", dataset="d", task_indices=[99])
        )

    assert stopped == ["forwarder", "capture"], (
        "capture.stop() must run even when the forwarder's teardown raises"
    )


def run_empty_batch(**kwargs) -> None:
    """Drive `run_batch` through capture + forwarder setup with no rollout in range."""
    asyncio.run(
        runner.run_batch(
            llm_url="http://x/v1", dataset="d", task_indices=[99], **kwargs
        )
    )


def test_run_batch_always_gates_the_control_plane(monkeypatch, tmp_path):
    """The regression: the proxy went out on a public tunnel with `admin_key` unset.

    `_admin_ok` returns True for every caller while the key is unset, so `POST /sessions` minted keys
    for anyone (an open relay to the upstream) and `GET /sessions/{id}/rollout` served token-level
    training data to anyone who found the URL.
    """
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch()

    (kwargs,) = built
    assert kwargs.get("admin_key"), "CaptureServer was built without an admin key"
    assert len(kwargs["admin_key"]) >= 32


def test_run_batch_mints_a_fresh_key_per_batch(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENENV_CAPTURE_ADMIN_KEY", raising=False)
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch()
    run_empty_batch()

    first, second = (k["admin_key"] for k in built)
    assert first != second


def test_run_batch_forwards_an_explicit_admin_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENENV_CAPTURE_ADMIN_KEY", "from-env")
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch(admin_key="explicit")

    assert built[0]["admin_key"] == "explicit", "the argument must beat the env var"


def test_run_batch_honours_the_admin_key_env_var(monkeypatch, tmp_path):
    """Same variable `HarborService` reads, so one setting covers `serve` and `rollout`."""
    monkeypatch.setenv("OPENENV_CAPTURE_ADMIN_KEY", "from-env")
    _, built = wire(monkeypatch, tmp_path, Quiet)

    run_empty_batch()

    assert built[0]["admin_key"] == "from-env"
