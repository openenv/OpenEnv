"""Real HTTP/WebSocket evidence with installed OpenEnv and a test-only process.

This covers protocol and grading on CPU Jobs. It does not establish Docker,
resource-limit, network-isolation or fresh-container acceptance.
"""

import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import httpx
import pytest
from openenv.validation.providers import StartupError, UnsupportedCapability
from openenv.validation.runner import run_validation
from openenv.validation.types import CheckStatus, Level, ProviderCapability

FIXTURE = Path(__file__).parents[2] / "fixtures/validation/runtime/served_probe"
SERVER = """
import importlib.util, os, sys, uvicorn
spec = importlib.util.spec_from_file_location('process_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
uvicorn.run(module.make_app(os.environ['VALIDATION_FAULT']),
            fd=int(sys.argv[2]), log_level='warning')
"""


class ProcessSubject:
    def __init__(self, process, port, log_path, record_sha256):
        self.process = process
        self.port = port
        self.log_path = log_path
        self.record_sha256 = record_sha256
        self.base_url = f"http://127.0.0.1:{port}"

    def inspect(self):
        return {
            "test_only": True,
            "isolation": "process",
            "pid": self.process.pid,
            "installed_record_sha256": self.record_sha256,
            "container_build_exercised": False,
            "resource_limits_enforced": False,
            "network_policy_enforced": False,
        }

    def logs(self, max_bytes=65536):
        with self.log_path.open("rb") as stream:
            stream.seek(max(0, self.log_path.stat().st_size - max_bytes))
            return stream.read(max_bytes).decode("utf-8", "replace")

    def exec(self, argv, timeout_s):
        raise UnsupportedCapability("test process provider has no sandbox exec")

    def stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
        assert self.process.poll() is not None
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", self.port)) != 0


class ProcessProvider:
    """Substitute only process launch; all wire collection and grading are real.

    IMAGE_BUILD satisfies the runner's lifecycle seam in this test only. The
    reference identifies installed wheel metadata, never a purported Docker image.
    """

    name = "test-process"
    capabilities = frozenset({ProviderCapability.IMAGE_BUILD})
    supported_network_modes = frozenset({"public"})

    def __init__(self, work, mode="good"):
        self.work = work
        self.work.mkdir(parents=True)
        self.mode = mode
        self.subjects = []
        installed = next(
            item
            for item in importlib.metadata.distributions(
                path=[sysconfig.get_path("purelib")]
            )
            if item.metadata["Name"] == "openenv"
        )
        self.record_sha256 = hashlib.sha256(
            installed.read_text("RECORD").encode()
        ).hexdigest()

    def build(self, root, execution):
        self.package = self.work / "fixture"
        shutil.copytree(
            root, self.package, ignore=shutil.ignore_patterns("__pycache__")
        )
        return "sha256:" + self.record_sha256

    def start(self, spec):
        assert spec.network.mode == "public" and spec.resources.gpus == 0
        log_path = self.work / f"server-{len(self.subjects)}.log"
        with socket.socket() as listener, log_path.open("wb") as log:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    SERVER,
                    str(self.package / "app.py"),
                    str(listener.fileno()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    "PATH": os.defpath,
                    "GRADIO_ANALYTICS_ENABLED": "False",
                    "VALIDATION_FAULT": self.mode,
                    **spec.env_vars,
                },
                pass_fds=(listener.fileno(),),
                start_new_session=True,
            )
        subject = ProcessSubject(process, port, log_path, self.record_sha256)
        self.subjects.append(subject)
        deadline = time.monotonic() + min(spec.startup_timeout_s, 10)
        try:
            with httpx.Client(trust_env=False, timeout=0.2) as client:
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        if client.get(subject.base_url + "/health").status_code == 200:
                            return subject
                    except httpx.HTTPError:
                        pass  # The process may still be starting its HTTP listener.
                    time.sleep(0.02)
            raise StartupError("test process failed readiness")
        except BaseException:
            subject.stop()
            raise


@pytest.mark.parametrize(
    "mode,failed_check",
    [
        ("good", None),
        ("bad_reward", "runtime.reward_well_formed"),
        ("bad_observation", "runtime.observation_schema"),
        ("missing_done", "runtime.observation_schema"),
        ("bad_state", "runtime.state_contract"),
        ("startup_failure", "runtime.startup"),
    ],
)
def test_installed_server_collector_and_graders_over_loopback(
    tmp_path, mode, failed_check
):
    artifacts = (
        Path(os.environ.get("OPENENV_VALIDATION_ARTIFACTS", tmp_path))
        / "process"
        / mode
    )
    provider = ProcessProvider(artifacts / "subject", mode)
    try:
        report = run_validation(
            FIXTURE,
            max_level=Level.RUNTIME,
            provider=provider,
            artifacts_dir=artifacts / "report",
        )
    finally:
        for subject in provider.subjects:
            subject.stop()
    checks = {result.check_id: result for result in report.results}
    assert checks["static.manifest"].status is CheckStatus.PASS
    if failed_check:
        assert checks[failed_check].status is CheckStatus.FAIL
    else:
        for check in (
            "startup",
            "reward_well_formed",
            "observation_schema",
            "state_contract",
        ):
            assert checks[f"runtime.{check}"].status is CheckStatus.PASS
    if mode != "startup_failure":
        trace = json.loads((artifacts / "report/collector-trace.json").read_text())
        assert sum(row["operation"] == "step" for row in trace) == 2
        manifest = json.loads((artifacts / "report/run-manifest.json").read_text())
        assert manifest["provider"]["isolation"] == "process"
        assert manifest["provider"]["container_build_exercised"] is False
        telemetry_path = artifacts / "report/session-telemetry.json"
        telemetry = json.loads(telemetry_path.read_text())
        assert telemetry["seed"]["accepted"] is True
        assert len(telemetry["trajectory"]["records"]) == len(trace)
        assert len(telemetry["attribution"]) == 2
        assert telemetry["trajectory"]["complete"] is True
    for line in (artifacts / "report/SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        assert (
            hashlib.sha256((artifacts / "report" / name).read_bytes()).hexdigest()
            == checksum
        )
    assert provider.subjects and all(
        subject.process.poll() is not None for subject in provider.subjects
    )
