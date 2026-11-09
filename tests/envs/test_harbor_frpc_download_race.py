"""Concurrent first use of the gradio forwarder must not run a half-written frpc binary."""

import sys
import threading
from types import SimpleNamespace

from openenv.core.harness.capture.forwarding import GradioForwarder


def test_second_caller_waits_for_the_first_frpc_download(monkeypatch, tmp_path):
    binary = tmp_path / "frpc" / "frpc_test_v0.3"
    first_half_written = threading.Event()
    second_caller_done = threading.Event()
    downloads = []

    def download_binary():
        # Same shape as gradio's Tunnel.download_binary: check, then write in place.
        if binary.exists():
            return
        downloads.append(threading.current_thread().name)
        with open(binary, "wb") as file:
            file.write(b"first-half|")
            file.flush()
            first_half_written.set()
            # Without a lock the second caller finishes inside this window.
            second_caller_done.wait(timeout=1.0)
            file.write(b"second-half")

    monkeypatch.setitem(
        sys.modules,
        "gradio.tunneling",
        SimpleNamespace(
            BINARY_PATH=str(binary),
            Tunnel=SimpleNamespace(download_binary=staticmethod(download_binary)),
        ),
    )

    first = threading.Thread(target=GradioForwarder._ensure_frpc_binary, name="first")
    first.start()
    assert first_half_written.wait(timeout=5)

    seen_by_second = []

    def second_caller():
        GradioForwarder._ensure_frpc_binary()
        # The point where gradio would execute the binary.
        seen_by_second.append(binary.read_bytes())
        second_caller_done.set()

    second = threading.Thread(target=second_caller, name="second")
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert seen_by_second == [b"first-half|second-half"]
    assert downloads == ["first"]
