"""Installed-wheel CLI acceptance: real builds, wire faults, reports and cleanup."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest


pytestmark = pytest.mark.docker

IMPLEMENTED = {
    "runtime.startup",
    "runtime.reward_well_formed",
    "runtime.observation_schema",
    "runtime.state_contract",
    "runtime.network_policy",
}
PENDING = {
    "runtime.trajectory_record",
    "runtime.tool_declaration_accuracy",
    "runtime.seed_control",
    "runtime.episode_determinism",
    "runtime.host_containment",
    "runtime.resource_bounds",
    "runtime.episode_isolation",
    "runtime.oracle_containment",
}
NOT_APPLICABLE = {
    "runtime.rubric_introspectable",
    "runtime.reward_attribution",
    "runtime.task_declaration_accuracy",
}


def _copy_asset(source, destination):
    # Wheel bytes are immutable inputs; hardlinks avoid copying the locked
    # wheelhouse for every fault case. Fall back across filesystem boundaries.
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


@pytest.fixture
def cli_context(tmp_path):
    root = os.environ.get("OPENENV_VALIDATION_CONTEXT")
    if not root:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required CLI acceptance needs OPENENV_VALIDATION_CONTEXT")
        pytest.skip("Run the validation lab to provide its offline build context")
    context = tmp_path / "subject"
    shutil.copytree(root, context, copy_function=_copy_asset)
    # Fault injection must never modify the shared context's Dockerfile inode.
    dockerfile = context / "Dockerfile"
    content = dockerfile.read_bytes()
    dockerfile.unlink()
    dockerfile.write_bytes(content)
    return context


def _case_label(context):
    return hashlib.sha256(str(context).encode()).hexdigest()[:20]


def _mark_context(context):
    # An image label is inherited by every container, including failed starts.
    # This scopes external cleanup verification to this exact test invocation.
    dockerfile = context / "Dockerfile"
    content = dockerfile.read_bytes()
    dockerfile.unlink()
    dockerfile.write_bytes(
        content
        + f"\nLABEL org.openenv.validation.test={_case_label(context)}\n".encode()
    )


def _container_ids(context):
    checked = subprocess.run(
        [
            "docker",
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"label=org.openenv.validation.test={_case_label(context)}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return set(checked.stdout.split())


def _wait_for_blocked_step(process, context, work):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail("CLI exited before the deliberate blocked step")
        for container_id in _container_ids(context):
            logs = subprocess.run(
                ["docker", "logs", "--tail", "30", container_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "OPENENV_VALIDATION_STEP_BLOCKED" in logs.stdout:
                (work / "blocked-handshake.json").write_text(
                    json.dumps(
                        {
                            "container_id": container_id,
                            "marker": "OPENENV_VALIDATION_STEP_BLOCKED",
                            "milestone": "second_step_received",
                        }
                    )
                    + "\n"
                )
                return time.monotonic()
        # Poll an observable protocol milestone rather than guessing when the
        # CLI has reached its collector after an arbitrarily long image build.
        time.sleep(0.05)
    pytest.fail("Subject never reached its deliberately blocked second step")


def _verify_bundle(bundle, report):
    schema = json.loads(
        files("openenv.validation")
        .joinpath("schemas/report-v2.schema.json")
        .read_text()
    )
    jsonschema.validate(report, schema)
    assert report["report_schema_version"] == "2"
    assert report["manifest"]["manifest_schema_version"] == "2"
    assert report["policy_version"] == "v2"
    assert report["lane"] == "local"
    assert len(report["source_digest"]) == 64
    artifact_report = json.loads((bundle / "report.json").read_text())
    assert artifact_report == report
    recorded = set()
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in recorded
        recorded.add(name)
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    assert recorded == {
        path.name for path in bundle.iterdir() if path.name != "SHA256SUMS"
    }
    return {name: json.loads((bundle / name).read_text()) for name in recorded}


def _default_signal_handlers():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _invoke_cli(
    context,
    tmp_path,
    case,
    *,
    skip_build=False,
    wait_for_blocked=False,
    interrupt=None,
):
    artifact_root = Path(os.environ.get("OPENENV_VALIDATION_ARTIFACTS", tmp_path))
    work = artifact_root / "cli" / case
    work.mkdir(parents=True, exist_ok=True)
    _mark_context(context)
    report_path = work / "report.json"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    marker = tmp_path / "unexpected-docker-call"
    if skip_build:
        binary_dir = tmp_path / "bin"
        binary_dir.mkdir()
        docker = binary_dir / "docker"
        docker.write_text(
            f"#!{sys.executable}\nfrom pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('called')\nraise SystemExit(91)\n"
        )
        docker.chmod(0o755)
        environment["PATH"] = str(binary_dir) + os.pathsep + environment["PATH"]
    command = [
        sys.executable,
        "-m",
        "openenv.cli",
        "validate",
        str(context),
        "--level",
        "runtime",
        "--local",
        "--policy",
        "v2",
        "--json",
        "--output",
        str(report_path),
    ]
    if skip_build:
        command.append("--skip-build")
    process = None
    try:
        with (
            (work / "stdout.json").open("w") as stdout,
            (work / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=work,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                # Children inherit ignored signals; start the CLI with defaults.
                preexec_fn=_default_signal_handlers,
            )
            blocked_at = None
            if wait_for_blocked:
                blocked_at = _wait_for_blocked_step(process, context, work)
            if interrupt is not None:
                assert blocked_at is not None, "Signal requires a protocol handshake"
                process.send_signal(interrupt)
            process.wait(timeout=15 if blocked_at else 180)
            if blocked_at is not None:
                elapsed = time.monotonic() - blocked_at
                (work / "termination.json").write_text(
                    json.dumps(
                        {
                            "seconds_after_blocked_step": elapsed,
                            "signal": interrupt.name if interrupt else None,
                        }
                    )
                    + "\n"
                )
                assert elapsed < 15, "CLI did not terminate within its time budget"
        result = subprocess.CompletedProcess(
            command,
            process.returncode,
            (work / "stdout.json").read_text(),
            (work / "stderr.log").read_text(),
        )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        remaining = _container_ids(context)
        (work / "cleanup-external.json").write_text(
            json.dumps(
                {
                    "test_label": _case_label(context),
                    "remaining_container_ids": sorted(remaining),
                }
            )
            + "\n"
        )
        if remaining:
            # Retain the failed assertion and evidence, but do not leave a failed
            # acceptance test consuming resources or touching other test runs.
            subprocess.run(
                ["docker", "rm", "--force", "--volumes", *sorted(remaining)],
                capture_output=True,
                timeout=15,
                check=True,
            )
        assert not remaining, "CLI validation leaked a container"
    assert report_path.is_file(), result.stderr
    report = json.loads(report_path.read_text())
    assert json.loads(result.stdout) == report
    artifacts = _verify_bundle(work / "report.artifacts", report)
    assert not marker.exists(), "--skip-build invoked Docker"
    checks = {row["check_id"]: row for row in report["results"]}
    assert len(checks) == len(report["results"]), "Report contains duplicate check IDs"
    assert {
        key for key in checks if key.startswith("runtime.")
    } == IMPLEMENTED | PENDING
    assert not NOT_APPLICABLE & checks.keys()
    assert all(checks[key]["status"] == "skip" for key in PENDING)
    assert all(checks[key]["evidence"] for key in PENDING)
    assert (
        set(artifacts["coverage.json"]["requested_runtime_checks"])
        == IMPLEMENTED | PENDING
    )
    assert artifacts["cleanup.json"]["completed"] is True
    return result, report, checks, artifacts


@pytest.mark.parametrize(
    "mode,failed_check",
    [
        ("good", None),
        ("slow_step", None),
        ("bad_reward", "runtime.reward_well_formed"),
        ("bad_observation", "runtime.observation_schema"),
        ("missing_done", "runtime.observation_schema"),
        ("bad_state", "runtime.state_contract"),
    ],
)
def test_cli_runtime_contract_findings(cli_context, tmp_path, mode, failed_check):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write(f"\nENV VALIDATION_FAULT={mode}\n")
    result, report, checks, artifacts = _invoke_cli(cli_context, tmp_path, mode)
    assert report["levels_run"] == [1, 2]
    assert checks["static.manifest"]["status"] == "pass"
    assert checks["runtime.startup"]["status"] == "pass"
    assert checks["runtime.startup"]["measured"]["image_ref"].startswith("sha256:")
    assert artifacts["cleanup.json"]["required"] is True
    assert artifacts["run-manifest.json"]["provider"]["container_id"]
    assert artifacts["run-manifest.json"]["source_digest"] == report["source_digest"]
    trace = artifacts["collector-trace.json"]
    assert [row["operation"] for row in trace] == [
        "reset",
        "state",
        "step",
        "state",
        "step",
        "state",
    ]
    if failed_check:
        assert result.returncode == 1
        assert report["verdict"] == "fail"
        assert checks[failed_check]["status"] == "fail"
        assert checks[failed_check]["evidence"]
    else:
        assert result.returncode == 0
        assert report["verdict"] == "warn", (
            "Missing later graders must keep the result incomplete"
        )
        assert all(checks[key]["status"] == "pass" for key in IMPLEMENTED)
        assert set(artifacts["coverage.json"]["incomplete"]) == PENDING


def _assert_partial_episode(artifacts, expected_failure):
    trace = artifacts["collector-trace.json"]
    assert [row["operation"] for row in trace] == ["reset", "state", "step", "state"]
    assert trace[0]["response_json"]["data"]["observation"]["counter"] == 0
    assert trace[2]["response_json"]["data"]["observation"]["counter"] == 1
    assert trace[3]["response_json"]["data"]["step_count"] == 1
    evidence = artifacts["collector-evidence.json"]
    assert evidence["complete"] is False
    assert evidence["failure_phase"] == "step"
    assert expected_failure in evidence["failure_reason"]
    assert artifacts["cleanup.json"] == {"required": True, "completed": True}


def test_cli_hung_step_times_out_with_partial_evidence(cli_context, tmp_path):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write("\nENV VALIDATION_FAULT=hung_step\n")
    manifest = cli_context / "openenv.yaml"
    content = manifest.read_text()
    assert content.count("episode_timeout_s: 30.0") == 1
    content = content.replace("episode_timeout_s: 30.0", "episode_timeout_s: 3.0")
    # The context hardlinks immutable inputs; changing one must not alter the
    # shared source used by subsequent test cases.
    manifest.unlink()
    manifest.write_text(content)
    result, report, checks, artifacts = _invoke_cli(
        cli_context, tmp_path, "hung_step", wait_for_blocked=True
    )
    assert result.returncode == 1
    assert report["verdict"] == "fail"
    assert report["manifest"]["resources"]["episode_timeout_s"] == 3.0
    assert checks["runtime.startup"]["status"] == "fail"
    assert all(
        checks[key]["status"] == "skip" for key in IMPLEMENTED - {"runtime.startup"}
    )
    _assert_partial_episode(artifacts, "TimeoutError")


def test_cli_sigint_retains_partial_evidence_and_cleans_up(cli_context, tmp_path):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write("\nENV VALIDATION_FAULT=hung_step\n")
    result, report, checks, artifacts = _invoke_cli(
        cli_context, tmp_path, "sigint", wait_for_blocked=True, interrupt=signal.SIGINT
    )
    # Recorded check errors fail closed through the existing policy (exit 1).
    # Exit 3 is reserved for an internal error that cannot produce this report.
    assert result.returncode == 1
    assert report["verdict"] == "fail"
    assert checks["runtime.startup"]["status"] == "error"
    assert "interrupt" in " ".join(checks["runtime.startup"]["evidence"]).lower()
    assert all(
        checks[key]["status"] == "skip" for key in IMPLEMENTED - {"runtime.startup"}
    )
    _assert_partial_episode(artifacts, "KeyboardInterrupt")


def test_cli_sigterm_retains_partial_evidence_and_cleans_up(cli_context, tmp_path):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write("\nENV VALIDATION_FAULT=hung_step\n")
    result, report, checks, artifacts = _invoke_cli(
        cli_context,
        tmp_path,
        "sigterm",
        wait_for_blocked=True,
        interrupt=signal.SIGTERM,
    )
    assert result.returncode == 1
    assert report["verdict"] == "fail"
    assert checks["runtime.startup"]["status"] == "error"
    assert "interrupt" in " ".join(checks["runtime.startup"]["evidence"]).lower()
    assert all(
        checks[key]["status"] == "skip" for key in IMPLEMENTED - {"runtime.startup"}
    )
    _assert_partial_episode(artifacts, "KeyboardInterrupt")


def test_cli_startup_failure_is_a_finding_and_leaves_no_container(
    cli_context, tmp_path
):
    with (cli_context / "Dockerfile").open("a") as stream:
        stream.write("\nENV VALIDATION_FAULT=startup_failure\n")
    result, report, checks, artifacts = _invoke_cli(
        cli_context, tmp_path, "startup_failure"
    )
    assert result.returncode == 1
    assert report["verdict"] == "fail"
    assert report["levels_run"] == [1, 2]
    assert checks["runtime.startup"]["status"] == "fail"
    assert all(
        checks[key]["status"] == "skip" for key in IMPLEMENTED - {"runtime.startup"}
    )
    assert artifacts["collector-trace.json"] == []


def test_cli_skip_build_does_not_invoke_docker(cli_context, tmp_path):
    result, report, checks, artifacts = _invoke_cli(
        cli_context, tmp_path, "skip_build", skip_build=True
    )
    assert result.returncode == 0
    assert report["verdict"] == "warn"
    assert report["levels_run"] == [1]
    assert all(checks[key]["status"] == "skip" for key in IMPLEMENTED | PENDING)
    assert "--skip-build" in " ".join(checks["runtime.startup"]["evidence"])
    assert artifacts["cleanup.json"]["required"] is False
    assert artifacts["run-manifest.json"]["provider"] == {}
    assert artifacts["collector-trace.json"] == []
