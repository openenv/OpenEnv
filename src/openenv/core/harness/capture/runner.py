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

"""Run the capture proxy in-process, for any environment that needs one.

MOVED HERE FROM `openenv.harbor.runner` DELIBERATELY.
`CaptureServer` never touched a Harbor type -- it wraps `create_app`, binds a port and hands back the
live `SessionRegistry`. Living under `harbor/` meant a second environment wanting an in-process proxy
had to either import Harbor (dragging in its models, task resolution and rollout engine for a class
that uses none of them) or hand-roll HTTP calls to something already running in the same process.
Both are worse than moving it. `openenv.harbor.runner` re-exports it, so existing imports keep
working.

A THREAD, NOT A SUBPROCESS, and that is the point: the rollout path needs the live registry object so
it can mint a session and then read the graph straight back out of it. Going through HTTP for that
would add a serialisation round trip and a failure mode for no benefit.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from typing import Any

from .server import create_app


def _require_free_port(port: int) -> None:
    """Raise if anything is already listening on `port`, naming the holder when we can find it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(
                f"capture port :{port} is already in use ({exc.strerror}). {_port_holder(port)}"
                " Stop it or pass a different port: a second server on this port cannot bind, and "
                "the agent would silently talk to the older one."
            ) from exc


def _port_holder(port: int) -> str:
    """Best-effort description of the process holding `port`, for the error message only."""
    import shutil
    import subprocess

    if not shutil.which("ss"):
        return ""
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if f":{port} " in line and "users:" in line:
                return f"Held by {line.split('users:', 1)[1].strip()}."
    return ""


def _health_instance(port: int) -> str | None:
    """Instance id reported by whatever is serving `port`, or `None` if nothing answers yet."""
    import httpx

    with contextlib.suppress(Exception):
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        if resp.status_code == 200:
            return str(resp.json().get("instance") or "unknown")


class CaptureServer:
    """The capture proxy, running in a background thread for the life of a batch.

    A thread rather than a subprocess because the rollout path needs the live `SessionRegistry` — it
    mints a session, then reads the graph back out of it directly. Going through HTTP for that would
    add a serialisation round trip and a failure mode for no benefit.
    """

    def __init__(
        self,
        *,
        llm_url: str,
        model: str,
        port: int = 8100,
        max_output_tokens: int = 8192,
        api_key: str | None = None,
        auth_header: str = "Authorization",
        capture_level: str = "tokens",
        provider: str = "openai",
        admin_key: str | None = None,
        max_model_calls: int = 0,
    ) -> None:
        self.app = create_app(
            llm_url=llm_url,
            model=model,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            auth_header=auth_header,
            capture_level=capture_level,
            provider=provider,
            admin_key=admin_key,
            max_model_calls=max_model_calls,
        )
        self.capture_level = "text" if provider == "anthropic" else capture_level
        self.admin_key = admin_key
        self.port = port
        self._thread: threading.Thread | None = None
        self._server: Any = None

    @property
    def registry(self) -> Any:
        return self.app.state.registry

    @property
    def inference(self) -> Any:
        """The upstream client, for reading back what it had to work around. See `param_fixes`."""
        return self.app.state.inference

    def start(self, timeout_s: float = 30.0) -> None:
        """Bind the port and confirm that the process answering on it is *this* one.

        Raises:
            RuntimeError:
                If the port is already held, or if the server that comes up on it is not ours.
        """
        import uvicorn

        # Fail before uvicorn does. Its bind error surfaces on a background thread, where nothing
        # observes it, and the port stays served by whoever holds it.
        _require_free_port(self.port)

        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Reachability is not identity. A stale process on this port answers every probe, so the
        # check is that /health reports our own instance id.
        want = self.app.state.instance_id
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"capture server thread exited while starting on :{self.port} "
                    "(most likely the port was taken between the check and the bind)"
                )
            got = _health_instance(self.port)
            if got == want:
                return
            if got is not None:
                raise RuntimeError(
                    f"port :{self.port} is served by a different capture server (instance {got}, "
                    f"expected {want}). Stop the process holding it, or pass a different port; "
                    "sessions minted here would be rejected there and every rollout would see "
                    "no model calls."
                )
            time.sleep(0.1)
        raise RuntimeError(
            f"capture server did not come up on :{self.port} within {timeout_s:.0f}s"
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
