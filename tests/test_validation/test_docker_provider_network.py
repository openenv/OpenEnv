"""Docker-local no-network launch and network measurement, against a fake engine."""

import json
import subprocess
import sys
import types

import pytest
from openenv.validation.manifest import NetworkPolicy, ResourceDeclaration
from openenv.validation.providers import _netns, docker, ProviderError, StartupError
from openenv.validation.runtime.contracts import LaunchSpec, NetworkEvidence
from openenv.validation.types import ProviderCapability

IMAGE = "sha256:" + "a" * 64
SINK_IP = "10.88.0.5"


def test_relay_preserves_bytes_when_stdout_writes_are_short():
    setup = """
import os, socket, threading
write = os.write
os.write = lambda fd, data: write(fd, data[:3])
class Socket:
    chunks = iter([b"first response", b"second response", b""])
    def settimeout(self, value): pass
    def recv(self, size): return next(self.chunks)
socket.create_connection = lambda *a, **k: Socket()
class Thread:
    def __init__(self, **kwargs): pass
    def start(self): pass
threading.Thread = Thread
"""
    result = subprocess.run(
        [sys.executable, "-c", setup + _netns.RELAY_SCRIPT],
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == b"first responsesecond response"


def launch_spec(mode="no-network"):
    return LaunchSpec(
        image_ref=IMAGE,
        run_id="test-run",
        network=NetworkPolicy(mode=mode),
        resources=ResourceDeclaration(
            cpu=0.5, memory_mb=128, disk_mb=32, episode_timeout_s=5
        ),
    )


def probe_output(*, namespace=None, tcp="denied"):
    outcome = {"result": tcp, "errno": 101} if tcp == "denied" else {"result": tcp}
    return json.dumps(
        {
            "namespace": namespace
            or {
                "interfaces": ["lo"],
                "default_route": False,
                "ipv6_non_loopback": 0,
                "listening_ports": [8000],
            },
            "probes": {
                "tcp": outcome,
                "udp": outcome,
                "icmp": {"result": "inconclusive", "errno": 13},
            },
        }
    )


class FakeEngine:
    def __init__(self, *, ready=True, image_present=True, subject_probe=None):
        self.calls, self.containers = [], {}
        self.ready, self.image_present = ready, image_present
        self.subject_probe = subject_probe or probe_output()

    def __call__(self, argv, timeout_s, max_bytes=docker._MAX_OUTPUT):
        self.calls.append(argv)
        verb = argv[1]
        if verb == "image":
            return (0, "[]", "") if self.image_present else (1, "", "missing")
        if verb == "create":
            name = argv[argv.index("--name") + 1]
            label = (
                argv[argv.index("--label") + 1] if "--label" in argv else "x=test-run"
            )
            self.containers[name] = label.split("=", 1)[1]
            return 0, "id", ""
        if verb == "ps":
            query = argv[argv.index("--filter") + 1].removeprefix("name=")
            rows = [f"{n}\t{o}" for n, o in self.containers.items() if query in n]
            return 0, "\n".join(rows), ""
        if verb == "rm":
            self.containers.pop(argv[-1], None)
            return 0, "", ""
        if verb == "inspect":
            name = argv[-1]
            if name.endswith("-sink"):
                return (
                    0,
                    json.dumps(
                        [
                            {
                                "NetworkSettings": {
                                    "IPAddress": "",
                                    "Networks": {"podman": {"IPAddress": SINK_IP}},
                                }
                            }
                        ]
                    ),
                    "",
                )
            return (
                0,
                json.dumps(
                    [
                        {
                            "Config": {"Labels": {docker._LABEL: "test-run"}},
                            "HostConfig": {"NetworkMode": "none"},
                            "Mounts": [],
                        }
                    ]
                ),
                "",
            )
        if verb == "run":
            return 0, probe_output(tcp="reachable"), ""
        if verb == "exec":
            script = argv[argv.index("-c") + 1]
            if script == _netns.WAIT_READY_SCRIPT:
                return (0, "", "") if self.ready else (1, "", "")
            if script == _netns.PROBE_SCRIPT:
                return 0, self.subject_probe, ""
        return 0, "", ""

    def created(self, suffix=""):
        return next(
            c
            for c in self.calls
            if c[1] == "create" and c[c.index("--name") + 1].endswith(suffix)
        )


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(docker, "_command", fake)
    return fake


def test_provider_declares_network_enforcement():
    provider = docker.DockerValidationProvider()
    assert ProviderCapability.NETWORK_POLICY in provider.capabilities
    assert provider.supported_network_modes == frozenset({"public", "no-network"})
    with pytest.raises(ValueError, match="digest"):
        docker.DockerValidationProvider(helper_image="python:3.12-slim")


def test_no_network_subject_has_no_network_and_no_published_port(engine):
    subject = docker.DockerValidationProvider().start(launch_spec())
    create = engine.created(subject.name)
    assert create[create.index("--network") + 1] == "none"
    assert "--publish" not in create
    helper = engine.created("-netns")
    assert helper[helper.index("--network") + 1] == f"container:{subject.name}"
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--user"):
        assert flag in helper
    assert _netns.HELPER_IMAGE in helper and IMAGE not in helper
    assert subject.base_url.startswith("http://127.0.0.1:")
    subject.stop()


def test_readiness_runs_inside_the_helper(engine):
    subject = docker.DockerValidationProvider().start(launch_spec())
    waits = [
        c for c in engine.calls if c[1] == "exec" and _netns.WAIT_READY_SCRIPT in c
    ]
    assert len(waits) == 1 and waits[0][2] == f"{subject.name}-netns"
    subject.stop()


def test_failed_readiness_removes_helper_and_subject(monkeypatch):
    fake = FakeEngine(ready=False)
    monkeypatch.setattr(docker, "_command", fake)
    with pytest.raises(StartupError, match="/health"):
        docker.DockerValidationProvider().start(launch_spec())
    assert fake.containers == {}


def test_missing_helper_image_is_pulled_once(monkeypatch):
    fake = FakeEngine(image_present=False)
    monkeypatch.setattr(docker, "_command", fake)
    docker.DockerValidationProvider().start(launch_spec()).stop()
    assert sum(c[1] == "pull" for c in fake.calls) == 1


def test_helpers_are_removed_before_the_subject(engine):
    subject = docker.DockerValidationProvider().start(launch_spec())
    subject.stop()
    removed = [c[-1] for c in engine.calls if c[1] == "rm"]
    assert removed == [f"{subject.name}-netns", subject.name]
    assert engine.containers == {}


@pytest.mark.parametrize("subject_present", [True, False])
@pytest.mark.parametrize("failure", ["removal", "timeout", "ownership"])
def test_auxiliary_cleanup_failure_still_removes_subject_and_can_retry(
    engine, monkeypatch, subject_present, failure
):
    subject = docker.DockerValidationProvider().start(launch_spec())
    failed = f"{subject.name}-control"
    subject._auxiliary.append(failed)
    engine.containers[failed] = (
        "other-run" if failure == "ownership" else subject.run_id
    )
    if not subject_present:
        engine.containers.pop(subject.name)

    def fail_auxiliary_removal(argv, *args):
        if argv[1] == "rm" and argv[-1] == failed:
            engine.calls.append(argv)
            if failure == "timeout":
                raise ProviderError("Docker operation exceeded its deadline")
            return 1, "", "removal failed"
        return engine(argv, *args)

    monkeypatch.setattr(docker, "_command", fail_auxiliary_removal)
    with pytest.raises(ProviderError, match="network helper cleanup"):
        subject.stop()

    removed = [call[-1] for call in engine.calls if call[1] == "rm"]
    expected = [] if failure == "ownership" else [failed]
    expected.append(f"{subject.name}-netns")
    if subject_present:
        expected.append(subject.name)
    assert removed == expected
    assert set(engine.containers) == {failed}
    assert not subject._stopped

    # An incomplete teardown remains retryable, including after the subject is gone.
    engine.containers[failed] = subject.run_id
    monkeypatch.setattr(docker, "_command", engine)
    subject.stop()
    assert engine.containers == {}
    assert subject._stopped
    calls = len(engine.calls)
    subject.stop()
    assert len(engine.calls) == calls


def test_measure_network_assembles_validated_evidence(engine):
    subject = docker.DockerValidationProvider().start(launch_spec())
    measured = subject.measure_network()
    evidence = NetworkEvidence.model_validate(measured)
    assert evidence.requested_mode == "no-network"
    assert evidence.subject_network_mode == "none"
    tcp = next(p for p in evidence.probes if p.kind == "tcp")
    assert tcp.control.result == "reachable" and tcp.subject.result == "denied"
    control = next(c for c in engine.calls if c[1] == "run")
    assert SINK_IP in control and control[control.index("--network") + 1] == "bridge"
    assert f"{subject.name}-sink" not in engine.containers
    subject.stop()
    assert engine.containers == {}


def test_helper_image_pull_does_not_consume_the_measurement_budget(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(docker, "_command", fake)
    subject = docker.DockerValidationProvider().start(launch_spec())
    clock = [0.0]
    monkeypatch.setattr(
        docker, "time", types.SimpleNamespace(monotonic=lambda: clock[0])
    )
    fake.image_present = False
    budgets = []

    def slow_pull(argv, timeout_s, max_bytes=docker._MAX_OUTPUT):
        if argv[1] == "pull":
            clock[0] += 1000
        elif clock[0]:
            budgets.append(timeout_s)
        return fake(argv, timeout_s, max_bytes)

    monkeypatch.setattr(docker, "_command", slow_pull)
    subject.measure_network(timeout_s=120)
    assert budgets and min(budgets) > 0
    subject.stop()


@pytest.mark.parametrize(
    "subject_probe",
    ["not json", json.dumps({"namespace": {}}), json.dumps({"probes": {}})],
    ids=["unparseable", "missing-probes", "missing-namespace"],
)
def test_measure_network_removes_the_sink_when_a_probe_fails(
    monkeypatch, subject_probe
):
    fake = FakeEngine(subject_probe=subject_probe)
    monkeypatch.setattr(docker, "_command", fake)
    subject = docker.DockerValidationProvider().start(launch_spec())
    with pytest.raises(ProviderError):
        subject.measure_network()
    assert f"{subject.name}-sink" not in fake.containers
    subject.stop()


def test_public_subject_creates_the_helper_only_when_measured(engine, monkeypatch):
    def healthy(*args, **kwargs):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class Opener:
            def open(self, *a, **k):
                return Response()

        return Opener()

    monkeypatch.setattr(docker.urllib.request, "build_opener", healthy)
    original = engine.__call__

    def with_port(argv, timeout_s, max_bytes=docker._MAX_OUTPUT):
        if argv[1] == "inspect" and not argv[-1].endswith("-sink"):
            return (
                0,
                json.dumps(
                    [
                        {
                            "Config": {"Labels": {docker._LABEL: "test-run"}},
                            "HostConfig": {"NetworkMode": "bridge"},
                            "Mounts": [],
                            "NetworkSettings": {
                                "Ports": {
                                    "8000/tcp": [
                                        {"HostIp": "127.0.0.1", "HostPort": "49231"}
                                    ]
                                }
                            },
                        }
                    ]
                ),
                "",
            )
        return original(argv, timeout_s, max_bytes)

    monkeypatch.setattr(docker, "_command", with_port)
    subject = docker.DockerValidationProvider().start(launch_spec("public"))
    assert not any(
        c[1] == "create" and "-netns" in c[c.index("--name") + 1] for c in engine.calls
    )
    subject.measure_network()
    assert engine.created("-netns")
    subject.stop()
    assert engine.containers == {}
