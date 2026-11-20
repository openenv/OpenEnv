# SPDX-License-Identifier: BSD-3-Clause

"""Live test of a whole unit, inside the integration image.

`run.sh` runs this file twice: in a privileged container, where openenvd
isolates, and in a plain one, where it must serve the environment unchanged.
"""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not Path("/opt/unit/openenv.yaml").exists(), reason="needs the integration image"
)

PRIVILEGED = os.environ.get("OPENENVD_IT_MODE", "privileged") == "privileged"
MANIFEST = "/opt/unit/openenv.yaml"
privileged_only = pytest.mark.skipif(not PRIVILEGED, reason="needs isolation")

BASE = "http://127.0.0.1:8000"
ORCH = "orchestrator-token-for-tests"
H = {"Authorization": f"Bearer {ORCH}"}
STATE = Path("/var/lib/openenvd")


MODEL_KEY = "real-model-key-never-in-a-zone"


class _StubModel(BaseHTTPRequestHandler):
    """An Anthropic-style upstream that insists on the real key."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("x-api-key") != MODEL_KEY:
            self.send_response(403)
            self.end_headers()
            return
        body = json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "secret plan"},
                    {"type": "text", "text": "hello"},
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


_MODEL = None


def _start(tmp: Path, *extra: str) -> tuple[subprocess.Popen, list[str]]:
    global _MODEL
    if _MODEL is None:
        _MODEL = ThreadingHTTPServer(("127.0.0.1", 9999), _StubModel)
        threading.Thread(target=_MODEL.serve_forever, daemon=True).start()
    model_key = tmp / "model-key"
    model_key.write_text(MODEL_KEY)
    extra = (
        "--model-url",
        "http://127.0.0.1:9999",
        "--model-key-file",
        str(model_key),
        *extra,
    )
    tokens = tmp / "tokens.json"
    tokens.write_text(json.dumps({"orchestrator": ORCH}))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openenv.core.openenvd",
            "--manifest",
            MANIFEST,
            "--assets",
            "/opt/unit",
            "--state",
            str(STATE),
            "--tokens",
            str(tokens),
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    threading.Thread(
        target=lambda: [lines.append(line) for line in proc.stdout], daemon=True
    ).start()
    return proc, lines


def _wait_listening(proc, lines):
    deadline = time.time() + 90
    while True:
        try:
            if httpx.get(f"{BASE}/health", timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        if proc.poll() is not None or time.time() > deadline:
            pytest.fail("the unit did not start:\n" + "".join(lines))
        time.sleep(0.2)


@pytest.fixture(scope="module")
def daemon(tmp_path_factory):
    proc, lines = _start(tmp_path_factory.mktemp("d"))
    if PRIVILEGED:
        deadline = time.time() + 90
        while not any("openenvd ready" in line for line in lines):
            if proc.poll() is not None or time.time() > deadline:
                pytest.fail("openenvd did not start:\n" + "".join(lines))
            time.sleep(0.2)
    else:
        _wait_listening(proc, lines)
    yield lines
    proc.terminate()
    proc.wait(30)
    sys.stdout.write("".join(lines[-80:]))


def _crun(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["crun", "--root", str(STATE / "run" / "oci"), *args],
        capture_output=True,
        text=True,
    )


def _container(suffix: str) -> str:
    listed = json.loads(_crun("list", "--format", "json").stdout or "[]")
    return next(c["id"] for c in listed if c["id"].endswith(suffix))


def _exec(suffix: str, code: str) -> str:
    result = _crun("exec", _container(suffix), "/usr/local/bin/python3", "-c", code)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@privileged_only
def test_enforcement_check(tmp_path):
    """Probe containers in every zone try what the zone must not be able to do."""
    proc, lines = _start(tmp_path, "--check", "--state", str(STATE / "check"))
    code = proc.wait(300)
    time.sleep(0.2)
    start = lines.index("{\n")
    end = len(lines) - lines[::-1].index("}\n")
    report = json.loads("".join(lines[start:end]))
    assert code == 0, "".join(lines)
    assert report["passed"] is True, report


def test_without_privileges_the_environment_runs_unchanged(daemon):
    if PRIVILEGED:
        pytest.skip("only meaningful without privileges")
    assert any("isolation disabled" in line for line in daemon), "".join(daemon)
    listed = httpx.post(
        f"{BASE}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    ).json()
    assert "echo_message" in {t["name"] for t in listed["result"]["tools"]}
    assert httpx.get(f"{BASE}/info").status_code == 404


@privileged_only
def test_full_episode(daemon):
    info = httpx.get(f"{BASE}/info", headers=H).json()
    assert info["isolation"] == "enabled"
    assert httpx.get(f"{BASE}/info").status_code == 401

    info = httpx.post(f"{BASE}/reset_unit", headers=H, timeout=120).json()
    assert info["phase"] == "ready", info

    # The scripted harness runs on its own and writes what it could reach into
    # its socket directory.
    probe_file = STATE / "run" / "sock" / "agent-harness" / "probe.json"
    deadline = time.time() + 60
    while not probe_file.exists():
        assert time.time() < deadline, "harness never finished"
        time.sleep(0.2)
    probe = json.loads(probe_file.read_text())
    assert probe["tools"] and probe["model_reply"] == "hello"
    assert probe["other_key"] == 401
    assert probe["egress"] != "open" and probe["key_in_env"] is False
    assert probe["asset_readable"] is False and probe["state_readable"] is False

    # The agent surface is the unmodified env server.
    listed = httpx.post(
        f"{BASE}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    ).json()
    names = {t["name"] for t in listed["result"]["tools"]}
    assert "echo_message" in names

    # Simulation controls are not reachable without the orchestrator token.
    assert httpx.post(f"{BASE}/reset", json={}).status_code == 401

    with connect(f"ws://127.0.0.1:8000/ws?token={ORCH}") as ws:
        ws.send(json.dumps({"type": "reset", "data": {}}))
        json.loads(ws.recv(timeout=30))
        for message in ("one", "two"):
            ws.send(
                json.dumps(
                    {
                        "type": "step",
                        "data": {
                            "type": "call_tool",
                            "tool_name": "echo_message",
                            "arguments": {"message": message},
                        },
                    }
                )
            )
            json.loads(ws.recv(timeout=30))
    assert httpx.get(f"{BASE}/info", headers=H).json()["phase"] == "running"

    # Inside the env container: control-plane state and assets are masked,
    # nothing outside is reachable, and the hidden service answers only via its relay.
    out = _exec(
        "agent-env",
        "import os, socket, json, urllib.request\n"
        "r = {}\n"
        "def ls(p):\n"
        " try:\n  return os.listdir(p)\n"
        " except (FileNotFoundError, PermissionError):\n  return []\n"
        "r['state'] = ls('/var/lib/openenvd')\n"
        "r['assets'] = ls('/opt/unit/assets')\n"
        "r['pids'] = len([p for p in os.listdir('/proc') if p.isdigit()])\n"
        "s = socket.socket(); s.settimeout(2)\n"
        "try:\n s.connect(('1.1.1.1', 53)); r['egress'] = 'open'\n"
        "except OSError as e:\n r['egress'] = type(e).__name__\n"
        "r['svc'] = urllib.request.urlopen('http://airbnb.sim/', timeout=5).status\n"
        "open('/workspace/agent-note.txt', 'w').write('left by the agent')\n"
        "print(json.dumps(r))",
    )
    seen = json.loads(out)
    assert seen["state"] == [] and seen["assets"] == []
    assert seen["egress"] != "open"
    assert seen["pids"] < 10
    assert seen["svc"] == 200

    # Freeze and thaw the agent zone.
    assert (
        httpx.post(f"{BASE}/inspect", headers=H, timeout=60).json()["phase"] == "frozen"
    )
    assert httpx.post(f"{BASE}/resume", headers=H).json()["phase"] == "running"

    info = httpx.post(f"{BASE}/end_episode", headers=H, timeout=300).json()
    assert info["phase"] == "closed", info
    assert info["verdict"] == "graded"
    verdict = httpx.get(f"{BASE}/verdict", headers=H).json()
    results = verdict["observers"]
    assert results["runner"]["assets_visible"] is False
    assert results["grader"]["score"] == 1.0
    assert results["grader"]["runner_saw_assets"] is False
    assert "agent-note.txt" in results["runner"]["workspace_files"]

    # The agent surface is closed once the episode is over.
    assert httpx.post(f"{BASE}/mcp", json={}).status_code == 503
