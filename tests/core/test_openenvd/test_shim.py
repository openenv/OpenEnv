# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest
from openenv.core.openenvd.shim import shim_spec

SRC = str(Path(__file__).resolve().parents[3] / "src")


@pytest.fixture
def sock_dir():
    path = tempfile.mkdtemp(prefix="oe-", dir="/tmp" if os.path.isdir("/tmp") else None)
    yield path
    shutil.rmtree(path, ignore_errors=True)


class _Echo(socketserver.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(1024)
        self.request.sendall(b"echo:" + data)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_shim(spec, tmp_path, timeout=20):
    path = tmp_path / "shim.json"
    path.write_text(json.dumps(spec))
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(
        [sys.executable, "-m", "openenv.core.openenvd.shim", "--spec", str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_principal_runs_with_forward_and_exit_code(tmp_path, sock_dir):
    unix_path = os.path.join(sock_dir, "agent.sock")
    server = socketserver.UnixStreamServer(unix_path, _Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = _free_port()
    principal = (
        "import socket\n"
        f"s = socket.create_connection(('127.0.0.1', {port}))\n"
        "s.sendall(b'ping')\n"
        "print(s.recv(1024).decode())\n"
        "print('hi')\n"
        "raise SystemExit(3)\n"
    )
    spec = shim_spec(
        argv=[sys.executable, "-c", principal],
        env={"PATH": os.environ.get("PATH", "")},
        forwards=[
            {
                "kind": "tcp_to_unix",
                "host": "127.0.0.1",
                "port": port,
                "path": unix_path,
            }
        ],
        workdir=str(tmp_path),
    )
    try:
        result = _run_shim(spec, tmp_path)
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 3, result.stderr
    assert result.stdout.splitlines() == ["echo:ping", "hi"]


def test_signaled_principal_exits_128_plus_signal(tmp_path):
    spec = shim_spec(
        argv=[sys.executable, "-c", "import os, signal; os.kill(os.getpid(), 9)"],
        env={},
        forwards=[],
        workdir=None,
    )
    assert _run_shim(spec, tmp_path).returncode == 128 + 9


def test_missing_command_exits_127(tmp_path):
    spec = shim_spec(["/nonexistent/bin"], {}, [], None)
    result = _run_shim(spec, tmp_path)
    assert result.returncode == 127
    assert "failed to start the principal" in result.stderr


def test_shim_spec_validates_forwards():
    with pytest.raises(ValueError):
        shim_spec(["x"], {}, [{"kind": "udp"}], None)
    with pytest.raises(ValueError):
        shim_spec(["x"], {}, [{"kind": "unix_to_tcp", "path": "/p"}], None)
    with pytest.raises(ValueError):
        shim_spec([], {}, [], None)
    spec = shim_spec(["x"], {"A": "1"}, [], "/w")
    assert json.loads(json.dumps(spec))["workdir"] == "/w"
