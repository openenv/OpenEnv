"""Trusted network-namespace helper for the Docker-local provider.

The helper joins only the subject's network namespace. It runs the validator's own
digest-pinned image, so probes and the no-network control relay never execute code
from the subject's image. Every script here is stdlib Python passed with `-c`.
"""

import socket
import subprocess
import threading

# Digest-pinned validator image for the helper, sink and control containers.
HELPER_IMAGE = (
    "python:3.12-slim@sha256:"
    "c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28"
)
SINK_TCP_PORT = 8080
SINK_UDP_PORT = 8081

# Hardening shared by the helper, sink and control containers.
HARDENING = [
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--user",
    "65532:65532",
    "--pids-limit",
    "64",
    "--memory",
    "64m",
    "--stop-timeout",
    "0",
]

# A validator-owned destination: TCP greeting and UDP echo.
SINK_SCRIPT = f"""
import socket, threading
def tcp():
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", {SINK_TCP_PORT})); s.listen(16)
    while True:
        c, _ = s.accept()
        try: c.sendall(b"openenv-sink\\n")
        finally: c.close()
def udp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("0.0.0.0", {SINK_UDP_PORT}))
    while True:
        data, peer = s.recvfrom(512); s.sendto(data, peer)
threading.Thread(target=udp, daemon=True).start()
tcp()
"""

# Probes one sink and reports the namespace. argv: sink_ip [wait_seconds].
# Only network-level refusals count as denial; a refusal from the sink host itself
# proves the host was reached.
PROBE_SCRIPT = f"""
import errno, json, os, socket, sys, time
sink = sys.argv[1]
wait = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
DENIED = {{errno.ENETUNREACH, errno.EHOSTUNREACH}}
def outcome(fn):
    try:
        fn()
        return {{"result": "reachable"}}
    except socket.timeout:
        return {{"result": "inconclusive", "errno": None}}
    except ConnectionRefusedError:
        return {{"result": "reachable"}}
    except OSError as exc:
        kind = "denied" if exc.errno in DENIED else "inconclusive"
        return {{"result": kind, "errno": exc.errno}}
def tcp():
    with socket.create_connection((sink, {SINK_TCP_PORT}), timeout=3) as s:
        s.recv(32)
def udp():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(3); s.connect((sink, {SINK_UDP_PORT})); s.send(b"probe"); s.recv(32)
def icmp():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP) as s:
        s.settimeout(3); s.connect((sink, 0)); s.send(b"\\x08\\x00\\xf7\\xff\\x00\\x00\\x00\\x00"); s.recv(64)
deadline = time.monotonic() + wait
while time.monotonic() < deadline and outcome(tcp)["result"] != "reachable":
    time.sleep(0.2)
def lines(path):
    try:
        with open(path) as f: return f.read().splitlines()
    except OSError: return []
routes = [line.split() for line in lines("/proc/net/route")[1:]]
v6 = [line.split() for line in lines("/proc/net/if_inet6")]
listening = sorted({{int(f[1].split(":")[1], 16)
    for path in ("/proc/net/tcp", "/proc/net/tcp6") for f in (l.split() for l in lines(path)[1:])
    if len(f) > 3 and f[3] == "0A"}})
print(json.dumps({{
    "namespace": {{
        "interfaces": sorted(os.listdir("/sys/class/net")),
        "default_route": any(len(r) > 1 and r[1] == "00000000" for r in routes),
        "ipv6_non_loopback": sum(1 for f in v6 if f and f[-1] != "lo"),
        "listening_ports": listening,
    }},
    "probes": {{"tcp": outcome(tcp), "udp": outcome(udp), "icmp": outcome(icmp)}},
}}))
"""

# Waits inside the shared namespace for the subject's readiness. argv: timeout.
WAIT_READY_SCRIPT = """
import sys, time, urllib.request
deadline = time.monotonic() + float(sys.argv[1])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while time.monotonic() < deadline:
    try:
        with opener.open("http://127.0.0.1:8000/health", timeout=1) as r:
            if r.status == 200:
                sys.exit(0)
    except OSError:
        pass
    time.sleep(0.1)
sys.exit(1)
"""

# Copies one connection between stdin/stdout and the subject's loopback port 8000.
RELAY_SCRIPT = """
import os, socket, threading
s = socket.create_connection(("127.0.0.1", 8000), timeout=10)
s.settimeout(None)
def upstream():
    try:
        while chunk := os.read(0, 65536):
            s.sendall(chunk)
    finally:
        try: s.shutdown(socket.SHUT_WR)
        except OSError: pass
threading.Thread(target=upstream, daemon=True).start()
while chunk := s.recv(65536):
    pending = memoryview(chunk)
    while pending:
        pending = pending[os.write(1, pending):]
"""


class ExecRelayBridge:
    """Host-loopback listener relaying each connection through `docker exec -i`.

    The listener binds host loopback only; the relay target inside the namespace is
    fixed, and the helper itself never listens, so the subject cannot use either end.
    """

    def __init__(self, helper: str):
        self.helper = helper
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(16)
        self.port = self._server.getsockname()[1]
        self._processes: list[subprocess.Popen] = []
        self._lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(connection,), daemon=True
            ).start()

    def _serve(self, connection):
        try:
            process = subprocess.Popen(
                ["docker", "exec", "-i", self.helper, "python3", "-c", RELAY_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            connection.close()
            return
        with self._lock:
            if self._closed:
                process.kill()
                connection.close()
                return
            self._processes.append(process)

        def downstream():
            try:
                while chunk := process.stdout.read1(65536):
                    connection.sendall(chunk)
            except OSError:
                pass
            finally:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        reader = threading.Thread(target=downstream, daemon=True)
        reader.start()
        try:
            while chunk := connection.recv(65536):
                process.stdin.write(chunk)
                process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            reader.join(timeout=5)
            connection.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    def close(self):
        with self._lock:
            self._closed = True
            processes = list(self._processes)
        try:
            self._server.close()
        except OSError:
            pass
        for process in processes:
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
