"""Exercise Docker lab orchestration with inert workers, without an engine."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from tests.test_validation.test_reproduction import ACCEPTANCE, evidence, module, ROOT


@pytest.fixture
def lab(tmp_path, monkeypatch):
    reproduction = module("reproduce")
    monkeypatch.setattr(reproduction, "DOCKER_CLEANUP_GRACE_SECONDS", 0.1)
    executable = tmp_path / "pytest-double"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, signal, subprocess, sys, time
from pathlib import Path
from xml.etree import ElementTree as ET
args = sys.argv[1:]
selection = args[args.index('-k') + 1]
name = 'remaining' if selection.startswith('not ') else 'contracts'
root = Path(os.environ['OPENENV_VALIDATION_ARTIFACTS'])
mode = os.environ.get('DOUBLE_MODE', 'success')
(root / (name + '-argv.json')).write_text(json.dumps(args))
assert Path(os.environ['OPENENV_VALIDATION_CONTEXT']).is_dir()
assert os.environ['OPENENV_VALIDATION_IMAGE'] == 'sha256:same-staged-wheel'
print('progress ' + name, flush=True)
if mode == 'hang':
    child = subprocess.Popen([sys.executable, '-c',
        'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)'],
        start_new_session=True)
    (root / (name + '-pids.json')).write_text(json.dumps([os.getpid(), child.pid]))
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    time.sleep(60)
if name == 'contracts' and mode == 'no-junit':
    sys.exit(0)
path = Path(args[args.index('--junitxml') + 1])
if name == 'contracts' and mode == 'invalid-junit':
    path.write_text('<testsuites>')
    sys.exit(0)
cases = json.loads(Path(os.environ['DOUBLE_ACCEPTANCE']).read_text())['suites']['docker']
cases = [c for c in cases if ('test_cli_runtime_contract_findings' in c) == (name == 'contracts')]
if name == 'contracts' and mode == 'incomplete':
    cases.pop()
if name == 'remaining' and mode == 'duplicate':
    cases.append(next(c for c in json.loads(Path(os.environ['DOUBLE_ACCEPTANCE']).read_text())['suites']['docker']
                      if 'test_cli_runtime_contract_findings' in c))
doc = ET.Element('testsuites')
suite = ET.SubElement(doc, 'testsuite', tests=str(len(cases)),
    failures=str(int(name == 'contracts' and mode == 'failed-case')),
    errors=str(int(name == 'contracts' and mode == 'error')),
    skipped=str(int(name == 'contracts' and mode == 'skip')))
for identity in cases:
    classname, case = identity.split('::', 1)
    ET.SubElement(suite, 'testcase', classname=classname, name=case)
ET.ElementTree(doc).write(path)
sys.exit(7 if name == 'contracts' and mode == 'failure' else 0)
"""
    )
    executable.chmod(0o755)
    monkeypatch.setattr(reproduction, "sys", SimpleNamespace(executable=executable))
    context = tmp_path / "staged-context"
    context.mkdir()
    environment = {
        **os.environ,
        "OPENENV_VALIDATION_ARTIFACTS": str(tmp_path),
        "OPENENV_VALIDATION_CONTEXT": str(context),
        "OPENENV_VALIDATION_IMAGE": "sha256:same-staged-wheel",
        "DOUBLE_ACCEPTANCE": str(ACCEPTANCE),
    }
    return reproduction, environment


def execute(lab, tmp_path, mode="success", timeout=5):
    reproduction, environment = lab
    environment["DOUBLE_MODE"] = mode
    return reproduction.run_docker_tests(
        [str(ROOT / "tests/test_validation/integration"), "-m", "docker"],
        output=tmp_path,
        env=environment,
        basetemp=tmp_path / "pytest",
        processes=[],
        timeout=timeout,
    )


def seal(tmp_path):
    reproduction = module("reproduce")
    sums = tmp_path / "SHA256SUMS"
    sums.unlink(missing_ok=True)
    sums.write_text(
        "".join(
            f"{value}  {name}\n"
            for name, value in reproduction.hashes(tmp_path).items()
        )
    )


def acceptance_metadata(tmp_path):
    inventory = ACCEPTANCE.read_bytes()
    (tmp_path / "acceptance.json").write_bytes(inventory)
    module("reproduce").write_json(
        tmp_path / "run-manifest.json",
        {
            "success": True,
            "suite": "docker",
            "acceptance_inventory_sha256": module("reproduce").digest(ACCEPTANCE),
        },
    )


def test_partition_collects_exact_required_union_without_docker(tmp_path):
    reproduction = module("reproduce")
    collected = []
    for name, selector in reproduction.DOCKER_WORKERS:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_validation/integration",
                "-m",
                "docker",
                "-k",
                selector,
                "--collect-only",
                "-q",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": "src:envs"},
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        cases = {
            line.replace("/", ".").replace(".py::", "::")
            for line in result.stdout.splitlines()
            if line.startswith("tests/")
        }
        assert len(cases) == (20 if name == "contracts" else 38)
        collected.append(cases)
    assert not collected[0] & collected[1]
    assert collected[0] | collected[1] == set(
        json.loads(ACCEPTANCE.read_text())["suites"]["docker"]
    )


def test_workers_share_inputs_keep_logs_and_establish_complete_evidence(lab, tmp_path):
    execute(lab, tmp_path)
    acceptance_metadata(tmp_path)
    seal(tmp_path)
    assert module("verify_artifacts").verify(tmp_path) == 58
    records = json.loads((tmp_path / "docker-workers.json").read_text())
    assert records["timeout_seconds"] == 5
    basetemps = []
    for worker in records["workers"]:
        assert worker["returncode"] == 0
        assert worker["elapsed_seconds"] >= 0
        name = worker["name"]
        args = json.loads((tmp_path / f"{name}-argv.json").read_text())
        assert args[:2] == ["-m", "pytest"]
        assert "-vv" in args and "--durations=0" in args
        assert "faulthandler_timeout=180" in args
        basetemps.append(args[args.index("--basetemp") + 1])
        assert (tmp_path / f"junit-{name}.xml").is_file()
        assert f"progress {name}" in (tmp_path / f"logs/tests-{name}.log").read_text()
    assert len(set(basetemps)) == 2


@pytest.mark.parametrize("mode", ["failure", "no-junit", "invalid-junit"])
def test_failure_retains_both_worker_diagnostics_and_fails(lab, tmp_path, mode):
    with pytest.raises((RuntimeError, ValueError), match="worker|JUnit"):
        execute(lab, tmp_path, mode)
    records = json.loads((tmp_path / "docker-workers.json").read_text())
    assert len(records["workers"]) == 2
    assert all(row["returncode"] is not None for row in records["workers"])
    assert (tmp_path / "junit-remaining.xml").is_file()
    assert (tmp_path / "logs/tests-contracts.log").is_file()
    assert (tmp_path / "logs/tests-remaining.log").is_file()
    assert (tmp_path / "junit.xml").exists() == (mode == "failure")


@pytest.mark.parametrize(
    "mode", ["duplicate", "incomplete", "skip", "error", "failed-case"]
)
def test_real_verifier_rejects_invalid_union(lab, tmp_path, mode):
    execute(lab, tmp_path, mode)
    acceptance_metadata(tmp_path)
    seal(tmp_path)
    match = (
        "Duplicate"
        if mode == "duplicate"
        else "Missing required|failures, errors or skips"
    )
    with pytest.raises(ValueError, match=match):
        module("verify_artifacts").verify(tmp_path)


def test_verifier_rejects_duplicate_even_when_all_required_cases_pass(tmp_path):
    cases = json.loads(ACCEPTANCE.read_text())["suites"]["docker"]
    evidence(tmp_path, suite="docker", cases=[*cases, cases[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        module("verify_artifacts").verify(tmp_path)


def assert_stopped(pid):
    # An orphan killed after its parent may briefly be a zombie awaiting init.
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 or result.stdout.strip().startswith("Z")


def test_timeout_is_shared_and_kills_separate_child_sessions(lab, tmp_path):
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        execute(lab, tmp_path, "hang", timeout=0.8)
    assert time.monotonic() - started < 15
    records = json.loads((tmp_path / "docker-workers.json").read_text())
    assert records["timed_out"] is True
    for name in ("contracts", "remaining"):
        for pid in json.loads((tmp_path / f"{name}-pids.json").read_text()):
            assert_stopped(pid)
    assert not (tmp_path / "junit.xml").exists()


def test_later_worker_does_not_receive_another_full_deadline(tmp_path, monkeypatch):
    reproduction = module("reproduce")
    clock = [0.0]
    stopped = []

    class Worker:
        def __init__(self, finish):
            self.finish = finish
            self.stdout = io.BytesIO(b"progress\n")
            self.returncode = None

        def poll(self):
            if clock[0] >= self.finish:
                self.returncode = 0
            return self.returncode

    workers = iter([Worker(0.75), Worker(1.5)])

    def stop(processes):
        stopped.append(clock[0])
        for process in processes:
            process.returncode = -9

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(
        reproduction, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    )
    monkeypatch.setattr(
        reproduction,
        "subprocess",
        SimpleNamespace(
            Popen=lambda *args, **kwargs: next(workers),
            PIPE=subprocess.PIPE,
            STDOUT=subprocess.STDOUT,
            DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    monkeypatch.setattr(reproduction, "stop_docker_workers", stop)
    with pytest.raises(subprocess.TimeoutExpired):
        reproduction.run_docker_tests(
            [],
            output=tmp_path,
            env={},
            basetemp=tmp_path / "pytest",
            processes=[],
            timeout=1,
        )
    assert stopped == [1.0]


def test_staging_is_retained_if_a_worker_cannot_be_reaped():
    reproduction = module("reproduce")
    work = None
    try:
        with pytest.raises(RuntimeError, match="staged context retained"):
            with reproduction.staging_directory(
                [SimpleNamespace(poll=lambda: None)]
            ) as work:
                assert work.is_dir()
        assert work.is_dir()
    finally:
        if work is not None:
            reproduction.shutil.rmtree(work)


def test_main_fails_incomplete_docker_union_and_seals_failure(
    lab, tmp_path, monkeypatch
):
    reproduction, environment = lab
    monkeypatch.setenv("DOUBLE_ACCEPTANCE", environment["DOUBLE_ACCEPTANCE"])
    monkeypatch.setenv("DOUBLE_MODE", "incomplete")
    reproduction.sys.argv = [
        "reproduce.py",
        "--suite",
        "docker",
        "--require-complete",
        "--output",
        str(tmp_path / "runs"),
    ]
    reproduction.sys.stderr = sys.stderr
    monkeypatch.setattr(sys, "argv", reproduction.sys.argv)
    pins = json.loads((ROOT / "tests/validation_runtime/toolchain.json").read_text())

    def run(argv, **kwargs):
        return {
            ("uv", "--version"): f"uv {pins['uv']}",
            ("git", "rev-parse", "HEAD"): "28486f89469d3b791c587a6a4c2d885ba5ed5669",
            ("git", "status", "--porcelain"): "",
            ("docker", "info", "--format", "{{json .}}"): "{}",
        }[tuple(argv)]

    def stage(work, output, pins, manifest):
        context = work / "subject"
        context.mkdir()
        return context, environment["OPENENV_VALIDATION_IMAGE"]

    monkeypatch.setattr(reproduction, "run", run)
    monkeypatch.setattr(reproduction, "command_digest", lambda argv: "no-docker-double")
    monkeypatch.setattr(reproduction, "stage_image", stage)
    monkeypatch.setattr(
        reproduction,
        "stage_echo_canary",
        lambda work, context, output, manifest: context,
    )
    monkeypatch.syspath_prepend(str(ROOT / "scripts/validation"))
    assert reproduction.main() == 1
    (output,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["success"] is False
    assert "Missing required acceptance cases" in manifest["error"]
    assert not list(output.glob("tmp-*"))
    for line in (output / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        assert reproduction.digest(output / name) == checksum
    with pytest.raises(ValueError, match="did not complete successfully"):
        module("verify_artifacts").verify(output)


def test_sigterm_cancellation_reaps_workers_and_detached_children(lab, tmp_path):
    reproduction, environment = lab
    environment["DOUBLE_MODE"] = "hang"
    script = tmp_path / "cancel.py"
    script.write_text(
        f"import importlib.util\nfrom pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('lab', {str(ROOT / 'scripts/validation/reproduce.py')!r})\n"
        "lab = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(lab)\n"
        "lab.DOCKER_CLEANUP_GRACE_SECONDS = 0.1\n"
        f"lab.sys = type('Python', (), {{'executable': {str(reproduction.sys.executable)!r}}})\n"
        f"lab.run_docker_tests([], output=Path({str(tmp_path)!r}), env=dict(__import__('os').environ), "
        f"basetemp=Path({str(tmp_path)!r}) / 'pytest', processes=[], timeout=30)\n"
    )
    # A process outside either worker tree must survive the owned cleanup.
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    process = subprocess.Popen(
        [sys.executable, str(script)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not all(
            (tmp_path / f"{name}-pids.json").is_file()
            for name in ("contracts", "remaining")
        ):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=15)
        assert process.returncode != 0, stderr
        assert unrelated.poll() is None
        records = json.loads((tmp_path / "docker-workers.json").read_text())
        assert records["cancelled"] is True
        for name in ("contracts", "remaining"):
            for pid in json.loads((tmp_path / f"{name}-pids.json").read_text()):
                assert_stopped(pid)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)
