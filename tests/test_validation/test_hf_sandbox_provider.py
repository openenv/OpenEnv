"""The HF Sandbox provider: Space revision pinning, server start and verified cleanup."""

import asyncio
import hashlib
import importlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest
from huggingface_hub import _local_folder
from openenv.cli.__main__ import app
from openenv.validation.manifest import (
    ExecutionDeclaration,
    NetworkPolicy,
    ResourceDeclaration,
)
from openenv.validation.providers import (
    hf_sandbox,
    ProviderError,
    StartupError,
    UnsupportedCapability,
)
from openenv.validation.providers.hf_sandbox import (
    _app_port,
    _server_command,
    HFSandboxValidationProvider,
)
from openenv.validation.runner import source_digest
from openenv.validation.runtime.contracts import LaunchSpec
from openenv.validation.types import ProviderCapability
from pydantic import ValidationError
from typer.testing import CliRunner

GPU = SimpleNamespace(quantity="1")
PROBE = Path(__file__).parents[1] / "fixtures/validation/runtime/served_probe"


class FakeApi:
    def __init__(self, token=None):
        self.token = token
        self.sha = "a" * 40
        self.space_stage = "RUNNING"
        self.built = "a" * 40
        self.stages = ["RUNNING", "CANCELED"]

    def list_jobs_hardware(self):
        return [
            SimpleNamespace(name="cpu-basic", accelerator=None),
            SimpleNamespace(name="t4-small", accelerator=GPU),
        ]

    def space_info(self, space_id, revision=None):
        runtime = SimpleNamespace(raw={"stage": self.space_stage, "sha": self.built})
        return SimpleNamespace(sha=self.sha, runtime=runtime)

    def inspect_job(self, job_id):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return SimpleNamespace(status=SimpleNamespace(stage=stage))


class FakeSandbox:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = "job-1"
        self.commands = []
        self.killed = False
        self.proxy_headers = {"Authorization": "Bearer sandbox-only"}

    @classmethod
    def create(cls, **kwargs):
        sandbox = cls(**kwargs)
        cls.created.append(sandbox)
        return sandbox

    def run(self, cmd, **kwargs):
        self.commands.append((cmd, kwargs))
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    def proxy_url_for(self, port, path="/"):
        return f"https://job-1--1.hf.jobs/proxy/{port}/"

    def kill(self):
        self.killed = True


class FakeProxy:
    def __init__(self, *, target_url, headers):
        self.target_url = target_url

    def start(self):
        return "http://127.0.0.1:9"

    def stop(self):
        pass


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "Sandbox", FakeSandbox)
    monkeypatch.setattr(hf_sandbox, "_LocalAuthProxy", FakeProxy)
    monkeypatch.setattr(hf_sandbox.time, "sleep", lambda _: None)
    FakeSandbox.created = []
    return lambda flavor="cpu-basic": HFSandboxValidationProvider(
        "openenv-testing/probe", "a" * 40, flavor=flavor
    )


def space(tmp_path, cmd='CMD ["python", "-m", "served_probe.app"]'):
    (tmp_path / "Dockerfile").write_text(f"FROM python:3.12-slim\n{cmd}\n")
    (tmp_path / "README.md").write_text("---\nsdk: docker\napp_port: 8000\n---\n")
    (tmp_path / "openenv.yaml").write_text((PROBE / "openenv.yaml").read_text())
    shutil.copytree(PROBE / "validation", tmp_path / "validation", dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
def downloads(provider, monkeypatch, tmp_path_factory):
    """Repository snapshots with real SDK bookkeeping kept in a separate cache."""
    revisions = []
    cache_root = tmp_path_factory.mktemp("hf-cache")

    def snapshot_download(repo_id, *, repo_type, revision, local_dir=None):
        revisions.append(revision)
        cache = cache_root / f"download-{len(revisions)}"
        root = Path(local_dir) if local_dir else cache / "snapshot"
        root.mkdir(parents=True, exist_ok=True)
        metadata_root = Path(local_dir) if local_dir else cache
        for path in list(space(root).rglob("*")):
            if not path.is_file():
                continue
            # SDK bookkeeping contains download timestamps.
            _local_folder.write_download_metadata(
                metadata_root,
                path.relative_to(root).as_posix(),
                revision,
                "etag",
            )
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return revisions


def validate_space(*args):
    result = CliRunner().invoke(
        app,
        ["validate", "--space", "openenv-testing/probe", "--level", "static"]
        + ["--json", *args],
    )
    return json.loads(result.output)


def spec(image="sha256:" + "0" * 64, gpus=0):
    return LaunchSpec(
        image_ref=image,
        resources=ResourceDeclaration(
            cpu=1, memory_mb=512, disk_mb=64, episode_timeout_s=10, gpus=gpus
        ),
        network=NetworkPolicy(),
        run_id="validation-test",
    )


def healthy(monkeypatch, status=200):
    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, headers, timeout):
            assert headers == {"Authorization": "Bearer sandbox-only"}
            return SimpleNamespace(status_code=status)

    monkeypatch.setattr(hf_sandbox.httpx, "Client", Client)


def test_flavor_hardware_sets_gpu_capability(provider):
    assert ProviderCapability.GPU not in provider().capabilities
    gpu = provider("t4-small")
    assert ProviderCapability.GPU in gpu.capabilities
    assert gpu.gpus == 1
    with pytest.raises(ValueError, match="unknown HF Jobs flavor"):
        provider("not-a-flavor")


def test_revision_must_be_a_commit_hash(provider):
    with pytest.raises(ValueError, match="full commit hash"):
        HFSandboxValidationProvider("openenv-testing/probe", "main")


@pytest.mark.parametrize(
    "args",
    [
        ["--level", "static"],
        ["--level", "runtime", "--skip-build"],
        ["--skip-build"],
    ],
    ids=["static", "runtime-skip-build", "semantic-skip-build"],
)
def test_space_without_runtime_does_not_construct_jobs_provider(
    downloads, monkeypatch, args
):
    def unexpected_provider(*args, **kwargs):
        raise AssertionError("HF Jobs provider preflight must not run")

    monkeypatch.setattr(hf_sandbox, "HFSandboxValidationProvider", unexpected_provider)
    result = CliRunner().invoke(
        app,
        ["validate", "--space", "openenv-testing/probe", "--json", *args],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40
    assert report["levels_run"] == [1]
    if "--skip-build" in args:
        startup = next(
            r for r in report["results"] if r["check_id"] == "runtime.startup"
        )
        assert startup["status"] == "skip"
        assert "--skip-build" in " ".join(startup["evidence"])
    assert not FakeSandbox.created


@pytest.mark.parametrize(
    "manifest, exit_code, evidence",
    [
        (None, 2, "unrecognized package"),
        ("name: incomplete\n", 1, "has no `validation:` block"),
        ("validation: [\n", 1, "not valid YAML"),
    ],
    ids=["missing-manifest", "missing-validation-block", "invalid-yaml"],
)
def test_invalid_space_source_precedes_jobs_hardware_preflight(
    downloads, monkeypatch, manifest, exit_code, evidence
):
    download = huggingface_hub.snapshot_download
    preflights = []

    def invalid_snapshot(*args, **kwargs):
        root = Path(download(*args, **kwargs))
        path = root / "openenv.yaml"
        if manifest is None:
            path.unlink()
        else:
            path.write_text(manifest)
        return str(root)

    def unavailable_hardware(self):
        preflights.append(True)
        raise RuntimeError("HF Jobs hardware is unavailable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", invalid_snapshot)
    monkeypatch.setattr(FakeApi, "list_jobs_hardware", unavailable_hardware)
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--space",
            "openenv-testing/probe",
            "--level",
            "runtime",
            "--json",
        ],
    )
    assert result.exit_code == exit_code, result.output
    assert evidence in result.output
    assert preflights == []
    if manifest is not None:
        report = json.loads(result.output)
        assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40
        assert report["verdict"] == "fail"
        assert (
            next(r for r in report["results"] if r["check_id"] == "static.manifest")[
                "status"
            ]
            == "fail"
        )
    assert not FakeSandbox.created


@pytest.mark.parametrize(
    "level, manifest, exit_code",
    [
        ("static", None, 0),
        ("runtime", "name: incomplete\n", 1),
    ],
    ids=["static", "invalid-manifest"],
)
def test_space_report_and_artifacts_metadata_do_not_preflight_jobs_hardware(
    downloads, monkeypatch, tmp_path, level, manifest, exit_code
):
    command = importlib.import_module("openenv.cli.commands.validate")
    run = command.run_validation
    download = huggingface_hub.snapshot_download
    names = []

    def snapshot(*args, **kwargs):
        root = Path(download(*args, **kwargs))
        if manifest is not None:
            (root / "openenv.yaml").write_text(manifest)
        return str(root)

    def unavailable_hardware(self):
        raise AssertionError("forbidden HF Jobs hardware preflight")

    def record_report(package_root, **kwargs):
        names.append(kwargs["provider"].name)
        return run(package_root, **kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(FakeApi, "list_jobs_hardware", unavailable_hardware)
    monkeypatch.setattr(command, "run_validation", record_report)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--space",
            "openenv-testing/probe",
            "--level",
            level,
            "--json",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == exit_code, result.output
    report = json.loads(result.output)
    bundled_path = tmp_path / "report.artifacts/report.json"
    assert (
        report == json.loads(output.read_text()) == json.loads(bundled_path.read_text())
    )
    assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40
    assert names == ["hf-sandbox"]
    assert not FakeSandbox.created


@pytest.mark.parametrize(
    "flavor, exit_code, status, evidence",
    [
        ("cpu-basic", 0, "skip", "immutable_image"),
        ("t4-small", 0, "skip", "immutable_image"),
        ("not-a-flavor", 1, "error", "ValueError"),
    ],
)
def test_space_runtime_still_checks_jobs_flavor_once(
    downloads, monkeypatch, flavor, exit_code, status, evidence
):
    hardware = FakeApi.list_jobs_hardware
    preflights = []

    def record_hardware(self):
        preflights.append(True)
        return hardware(self)

    monkeypatch.setattr(FakeApi, "list_jobs_hardware", record_hardware)
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--space",
            "openenv-testing/probe",
            "--level",
            "runtime",
            "--flavor",
            flavor,
            "--json",
        ],
    )
    assert result.exit_code == exit_code, result.output
    startup = next(
        r
        for r in json.loads(result.output)["results"]
        if r["check_id"] == "runtime.startup"
    )
    assert startup["status"] == status
    assert evidence in " ".join(startup["evidence"])
    assert preflights == [True]
    assert not FakeSandbox.created


@pytest.mark.parametrize("revision", [None, "main", "release/v1", "a" * 40, ""])
def test_source_resolution_defaults_only_an_omitted_revision_to_main(
    downloads, monkeypatch, revision
):
    requested = []
    original = FakeApi.space_info

    def info(self, space_id, revision=None):
        requested.append(revision)
        return original(self, space_id, revision=revision)

    monkeypatch.setattr(FakeApi, "space_info", info)
    args = [] if revision is None else ["--revision", revision]
    report = validate_space(*args, "--level", "static")
    assert requested == ["main" if revision is None else revision]
    assert downloads == ["a" * 40]
    assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40


@pytest.mark.parametrize("revision", ["main", "release/v1", "a" * 40])
def test_named_revision_is_validated_as_its_commit(downloads, monkeypatch, revision):
    revisions = []
    real = hf_sandbox.HFSandboxValidationProvider

    def recording(space_id, revision, **kwargs):
        revisions.append(revision)
        return real(space_id, revision, **kwargs)

    monkeypatch.setattr(hf_sandbox, "HFSandboxValidationProvider", recording)
    report = validate_space("--revision", revision, "--level", "runtime")
    assert downloads == revisions == ["a" * 40]
    assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40


def test_repeat_downloads_hash_only_the_space_files(downloads, monkeypatch, tmp_path):
    clock = iter(range(1, 100))
    monkeypatch.setattr(
        _local_folder, "time", SimpleNamespace(time=lambda: next(clock))
    )
    digests = [validate_space()["source_digest"] for _ in range(2)]
    assert digests == [source_digest(space(tmp_path))] * 2


@pytest.mark.parametrize("level", ["static", "runtime"])
def test_space_reports_share_durable_source_identity(
    downloads, monkeypatch, tmp_path, level
):
    command = importlib.import_module("openenv.cli.commands.validate")
    run = command.run_validation
    returned = []
    source_paths = []

    def record_report(package_root, **kwargs):
        source_paths.append(package_root)
        report = run(package_root, **kwargs)
        returned.append(json.loads(report.model_dump_json()))
        return report

    monkeypatch.setattr(command, "run_validation", record_report)
    output = tmp_path / "report.json"
    printed = validate_space("--level", level, "--output", str(output))
    bundled_path = tmp_path / "report.artifacts/report.json"
    bundled = json.loads(bundled_path.read_text())
    assert printed == returned[0] == bundled == json.loads(output.read_text())
    assert printed["target"] == "spaces/openenv-testing/probe@" + "a" * 40
    expected = tmp_path / "expected"
    expected.mkdir()
    assert printed["source_digest"] == source_digest(space(expected))
    assert not source_paths[0].exists()
    assert not FakeSandbox.created
    for line in (bundled_path.parent / "SHA256SUMS").read_text().splitlines():
        expected_hash, name = line.split(maxsplit=1)
        assert (
            hashlib.sha256((bundled_path.parent / name).read_bytes()).hexdigest()
            == expected_hash
        )


def test_build_reads_the_space_command_and_port(provider, tmp_path):
    hf = provider()
    with pytest.raises(UnsupportedCapability, match="immutable_image"):
        hf.build(space(tmp_path), ExecutionDeclaration())
    assert not FakeSandbox.created
    assert hf._command == ["python", "-m", "served_probe.app"]
    assert hf._port == 8000


@pytest.mark.parametrize(
    "hub, error",
    [
        ({"sha": "b" * 40, "built": "b" * 40}, "is at revision b"),
        ({"built": "c" * 40}, "runtime revision c"),
        ({"space_stage": "BUILDING", "built": None}, "stage BUILDING"),
        ({"space_stage": "SLEEPING", "built": None}, "stage SLEEPING"),
    ],
    ids=["moved", "older-build", "building", "sleeping"],
)
def test_start_needs_a_running_build_of_the_validated_commit(
    provider, tmp_path, monkeypatch, hub, error
):
    healthy(monkeypatch)
    hf = provider()
    with pytest.raises(UnsupportedCapability, match="immutable_image"):
        hf.build(space(tmp_path), ExecutionDeclaration())
    vars(hf._api).update(hub)
    with pytest.raises(ProviderError, match=error) as refused:
        hf.start(spec())
    assert type(refused.value) is ProviderError
    assert FakeSandbox.created == []


def test_start_refuses_even_a_digest_without_source_provenance(provider):
    hf = provider()
    with pytest.raises(UnsupportedCapability, match="immutable_image"):
        hf.start(spec())
    assert not FakeSandbox.created


@pytest.mark.parametrize("operation", ["build", "start"])
@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"stage": "RUNNING"},
        {"sha": "a" * 40},
        {"stage": None, "sha": "a" * 40},
        {"stage": "RUNNING", "sha": None},
    ],
    ids=["no-runtime", "empty", "no-sha", "no-stage", "null-stage", "null-sha"],
)
def test_unavailable_runtime_metadata_is_unsupported(
    provider, tmp_path, monkeypatch, operation, raw
):
    hf = provider()
    info = SimpleNamespace(
        sha=hf.revision, runtime=None if raw is None else SimpleNamespace(raw=raw)
    )
    monkeypatch.setattr(hf._api, "space_info", lambda _: info)
    with pytest.raises(UnsupportedCapability, match="space_runtime_metadata"):
        if operation == "build":
            hf.build(space(tmp_path), ExecutionDeclaration())
        else:
            hf.start(spec())
    assert not FakeSandbox.created


def test_explicit_runtime_revision_mismatch_precedes_missing_stage(
    provider, monkeypatch
):
    hf = provider()
    info = SimpleNamespace(
        sha=hf.revision, runtime=SimpleNamespace(raw={"sha": "b" * 40})
    )
    monkeypatch.setattr(hf._api, "space_info", lambda _: info)
    with pytest.raises(ProviderError, match="runtime revision b") as refused:
        hf.start(spec())
    assert type(refused.value) is ProviderError
    assert not FakeSandbox.created


def test_missing_runtime_sha_cli_skips_without_collection(downloads, monkeypatch):
    from openenv.validation import runner

    def no_runtime_sha(self, space_id, revision=None):
        return SimpleNamespace(
            sha="a" * 40, runtime=SimpleNamespace(raw={"stage": "RUNNING"})
        )

    def unexpected_collection(*args, **kwargs):
        raise AssertionError("unsupported runtime must never collect evidence")

    monkeypatch.setattr(FakeApi, "space_info", no_runtime_sha)
    monkeypatch.setattr(runner, "collect_runtime_evidence", unexpected_collection)
    report = validate_space("--level", "runtime")
    startup = next(r for r in report["results"] if r["check_id"] == "runtime.startup")
    assert startup["status"] == "skip"
    assert "space_runtime_metadata" in " ".join(startup["evidence"])
    assert report["verdict"] == "warn"
    assert not FakeSandbox.created


def test_mutable_space_image_is_not_a_launch_spec():
    with pytest.raises(ValidationError, match="image_ref"):
        spec("hf.co/spaces/openenv-testing/probe")


def test_subject_serves_the_space_command_and_stop_verifies_the_job(
    provider, tmp_path, monkeypatch
):
    healthy(monkeypatch)
    hf = provider("t4-small")
    with pytest.raises(UnsupportedCapability, match="immutable_image"):
        hf.build(space(tmp_path), ExecutionDeclaration())
    sandbox = FakeSandbox()
    running = hf_sandbox.HFSandboxSubject(sandbox, hf)
    running._serve(hf._command, hf._port, 1)
    command, options = sandbox.commands[0]
    assert command.startswith("python -m served_probe.app")
    assert options == {"shell": True, "background": True}
    assert running.base_url == "http://127.0.0.1:9"
    assert running.inspect() == {
        "job_id": "job-1",
        "image": "hf.co/spaces/openenv-testing/probe",
        "image_verified": False,
        "space_revision": "a" * 40,
        "built_revision": None,
        "flavor": "t4-small",
    }
    running.stop()
    assert sandbox.killed


def test_subject_rejects_an_unhealthy_server(provider, monkeypatch):
    healthy(monkeypatch, status=503)
    running = hf_sandbox.HFSandboxSubject(FakeSandbox(), provider())
    with pytest.raises(StartupError, match="/health"):
        running._serve(["serve"], 8000, 0.01)


def test_gpu_declaration_needs_a_gpu_flavor(provider):
    with pytest.raises(UnsupportedCapability, match="0 GPUs"):
        provider().start(spec(gpus=1))
    assert FakeSandbox.created == []


def test_unverified_cleanup_is_an_error(provider, monkeypatch):
    hf = provider()
    running = hf_sandbox.HFSandboxSubject(FakeSandbox(), hf)
    hf._api.stages = ["RUNNING"]
    clock = iter(range(0, 1000, 30))
    monkeypatch.setattr(hf_sandbox.time, "monotonic", lambda: next(clock))
    with pytest.raises(ProviderError, match="did not stop"):
        running.stop()


@pytest.fixture
def stopping_subject(provider, monkeypatch):
    hf = provider()
    hf._api.inspect_job = Mock(wraps=hf._api.inspect_job)
    sandbox = FakeSandbox()
    sandbox.kill = Mock(wraps=sandbox.kill)
    running = hf_sandbox.HFSandboxSubject(sandbox, hf)
    running._proxy = SimpleNamespace(stop=Mock())
    clock = SimpleNamespace(now=0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(
        hf_sandbox,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, sleep=Mock(side_effect=sleep)),
    )
    return running


@pytest.mark.parametrize("stage", ["COMPLETED", "CANCELED", "ERROR", "DELETED"])
@pytest.mark.parametrize("has_proxy", [True, False])
def test_stop_confirms_terminal_job_and_is_idempotent(
    stopping_subject, stage, has_proxy
):
    running = stopping_subject
    proxy = running._proxy
    if not has_proxy:
        running._proxy = None
    api = running._provider._api
    api.stages = ["RUNNING", stage]

    assert running.stop() is None
    assert running._stopped
    assert running._proxy is None
    assert proxy.stop.call_count == int(has_proxy)
    running._sandbox.kill.assert_called_once_with()
    assert api.inspect_job.call_count == 2
    api.inspect_job.assert_called_with(job_id="job-1")
    hf_sandbox.time.sleep.assert_called_once_with(2)

    running.stop()
    assert proxy.stop.call_count == int(has_proxy)
    running._sandbox.kill.assert_called_once_with()
    assert api.inspect_job.call_count == 2
    with pytest.raises(ProviderError, match="Subject has stopped"):
        running.exec(["echo", "stopped"], 1)


@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_stop_confirms_job_despite_proxy_failure_and_retries(
    stopping_subject, error_type
):
    running = stopping_subject
    proxy = running._proxy
    proxy_error = error_type("proxy shutdown failed")
    proxy.stop.side_effect = [proxy_error, None]
    api = running._provider._api

    with pytest.raises(error_type) as failure:
        running.stop()
    assert failure.value is proxy_error
    running._sandbox.kill.assert_called_once_with()
    assert api.inspect_job.call_count == 2
    hf_sandbox.time.sleep.assert_called_once_with(2)
    assert running._proxy is proxy
    assert not running._stopped

    running.stop()
    assert running._stopped
    assert running._proxy is None
    assert proxy.stop.call_count == 2
    assert running._sandbox.kill.call_count == 2
    assert api.inspect_job.call_count == 3
    running.stop()
    assert running._sandbox.kill.call_count == 2


@pytest.mark.parametrize("proxy_fails", [False, True])
@pytest.mark.parametrize("remote_failure", ["kill", "inspection", "timeout"])
def test_stop_retains_incomplete_cleanup_and_chains_dual_failures(
    stopping_subject, proxy_fails, remote_failure
):
    running = stopping_subject
    proxy = running._proxy
    proxy_error = RuntimeError("proxy shutdown failed")
    if proxy_fails:
        proxy.stop.side_effect = [proxy_error, None]
    api = running._provider._api
    remote_error = RuntimeError("remote teardown failed")
    if remote_failure == "kill":
        running._sandbox.kill.side_effect = remote_error
    elif remote_failure == "inspection":
        api.inspect_job.side_effect = remote_error
    else:
        # kill() can return normally even when cancellation failed.
        api.stages = ["RUNNING"]

    with pytest.raises(
        ProviderError if remote_failure == "timeout" else RuntimeError
    ) as failure:
        running.stop()
    if remote_failure == "timeout":
        assert "job-1 did not stop" in str(failure.value)
        assert api.inspect_job.call_count == 30
        assert hf_sandbox.time.monotonic() == 60
        assert [call.args for call in hf_sandbox.time.sleep.call_args_list] == [
            (2,)
        ] * 30
    else:
        assert failure.value is remote_error
        assert api.inspect_job.call_count == int(remote_failure == "inspection")
    assert failure.value.__context__ is (proxy_error if proxy_fails else None)
    running._sandbox.kill.assert_called_once_with()
    assert not running._stopped
    assert running._sandbox.id == "job-1"
    assert running._proxy is (proxy if proxy_fails else None)

    running._sandbox.kill.side_effect = None
    api.inspect_job.side_effect = None
    api.stages = ["CANCELED"]
    running.stop()
    assert running._stopped
    assert running._proxy is None
    assert proxy.stop.call_count == (2 if proxy_fails else 1)
    assert running._sandbox.kill.call_count == 2


@pytest.mark.parametrize(
    "dockerfile, expected",
    [
        ('CMD ["sh", "-c", "uvicorn app:app"]', ["sh", "-c", "uvicorn app:app"]),
        (
            "CMD uvicorn app:app --port 8000",
            ["/bin/sh", "-c", "uvicorn app:app --port 8000"],
        ),
        ('CMD ["a"]\nFROM x\nCMD ["b"]', ["b"]),
    ],
)
def test_server_command(dockerfile, expected):
    assert _server_command(dockerfile) == expected


def test_server_command_rejects_entrypoint_and_missing_cmd():
    with pytest.raises(UnsupportedCapability):
        _server_command('ENTRYPOINT ["x"]\nCMD ["y"]')
    with pytest.raises(StartupError):
        _server_command("FROM python:3.12")


@pytest.mark.parametrize(
    "heredoc",
    [
        "COPY <<EOF /tmp/example.txt\nFROM scratch\nEOF",
        "RUN <<EOF\nFROM scratch\nEOF",
        "COPY <<-'EOF' /tmp/example.txt\n\tFROM scratch\n\tEOF",
        'RUN <<-"EOF"\n\tENTRYPOINT ["wrong"]\n\tEOF',
        'COPY <<FIRST <<SECOND /tmp/\nCMD ["wrong"]\nFIRST\nFROM scratch\nSECOND',
        "RUN echo ready \\\n+    && cat <<EOF\nFROM scratch\nEOF",
    ],
    ids=["copy", "run", "quoted-copy", "quoted-run", "multiple", "continued"],
)
def test_server_command_refuses_heredocs_before_reading_payload(heredoc):
    dockerfile = 'FROM python:3.12-slim\nCMD ["python", "-m", "served_probe.app"]\n'
    with pytest.raises(UnsupportedCapability, match="dockerfile_heredoc"):
        _server_command(dockerfile + heredoc + "\n")


def test_copy_heredoc_runtime_is_capability_skip(downloads, monkeypatch, tmp_path):
    original = huggingface_hub.snapshot_download

    def heredoc_snapshot(*args, **kwargs):
        root = Path(original(*args, **kwargs))
        with (root / "Dockerfile").open("a") as dockerfile:
            dockerfile.write("COPY <<EOF /tmp/example.txt\nFROM scratch\nEOF\n")
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", heredoc_snapshot)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--space",
            "openenv-testing/probe",
            "--level",
            "runtime",
            "--json",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    startup = next(r for r in report["results"] if r["check_id"] == "runtime.startup")
    assert startup["status"] == "skip"
    assert "dockerfile_heredoc" in " ".join(startup["evidence"])
    assert report["verdict"] == "warn"
    assert report == json.loads((tmp_path / "report.artifacts/report.json").read_text())
    assert not FakeSandbox.created


def test_app_port_defaults_to_the_hub_default():
    assert _app_port("---\napp_port: 8000\n---\n") == 8000
    assert _app_port("no front matter") == 7860


@pytest.mark.parametrize(
    "dockerfile, expected",
    [
        (
            'FROM python:3.12\nCMD ["python", \\\n    "-m", \\\n    "app"]',
            ["python", "-m", "app"],
        ),
        (
            'FROM python AS build\nENTRYPOINT ["build"]\nCMD ["old"]\nFROM python\nCMD ["serve"]',
            ["serve"],
        ),
        ('FROM python AS build\nCMD ["serve"]\nFROM build', ["serve"]),
        ("FROM python\nCMD echo \\\n    hello", ["/bin/sh", "-c", "echo     hello"]),
        (
            'FROM python\nCMD ["serve"]\nHEALTHCHECK CMD true \\ \t\n    CMD ["probe"]',
            ["serve"],
        ),
    ],
)
def test_server_command_continuations_and_stages(dockerfile, expected):
    assert _server_command(dockerfile) == expected


@pytest.mark.parametrize("escape_count", [2, 3])
@pytest.mark.parametrize("trailing", ["", " ", "\t", " \t"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "tail,error,expected",
    [
        ('CMD ["serve"]', None, ["serve"]),
        ('ENTRYPOINT ["entry"]', UnsupportedCapability, "ENTRYPOINT"),
        ("FROM other", StartupError, "no CMD"),
        ('SHELL ["bash", "-c"]\nCMD echo hello', UnsupportedCapability, "SHELL"),
    ],
)
def test_escaped_backslashes_cannot_hide_following_instructions(
    escape_count, trailing, newline, tail, error, expected
):
    dockerfile = newline.join(
        [
            "FROM python",
            'CMD ["old"]',
            "HEALTHCHECK CMD true " + chr(92) * escape_count + trailing,
            *tail.splitlines(),
        ]
    )
    if error is None:
        assert _server_command(dockerfile) == expected
    else:
        with pytest.raises(error, match=expected):
            _server_command(dockerfile)


def test_final_stage_does_not_inherit_an_unrelated_stage_cmd():
    with pytest.raises(StartupError, match="no CMD"):
        _server_command('FROM python AS build\nCMD ["old"]\nFROM python')


def test_final_stage_inherits_unsupported_entrypoint():
    with pytest.raises(UnsupportedCapability, match="ENTRYPOINT"):
        _server_command('FROM python AS build\nENTRYPOINT ["x"]\nFROM build\nCMD ["y"]')


def test_server_log_file_is_bounded_while_output_is_drained(
    provider, tmp_path, monkeypatch
):
    healthy(monkeypatch)
    log = tmp_path / "server.log"
    completed = tmp_path / "completed"
    producer = (
        "import os; from pathlib import Path; "
        "[os.write(fd, b'x' * 8192) for fd in (1, 2) for _ in range(128)]; "
        f"Path({str(completed)!r}).write_text('done')"
    )
    hf = provider()
    sandbox = FakeSandbox()
    subject = hf_sandbox.HFSandboxSubject(sandbox, hf)
    subject._serve([sys.executable, "-c", producer], 8000, 1)
    command, options = sandbox.commands[0]
    assert options == {"shell": True, "background": True}
    result = subprocess.run(
        command.replace("/tmp/openenv-server.log", shlex.quote(str(log))),
        shell=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert completed.read_text() == "done"
    assert log.stat().st_size == hf_sandbox._MAX_OUTPUT
    assert result.stdout == result.stderr == b""


def test_server_command_rejects_custom_shell_and_escape_directive():
    with pytest.raises(UnsupportedCapability, match="SHELL"):
        _server_command('FROM python\nSHELL ["bash", "-c"]\nCMD echo hello')
    with pytest.raises(UnsupportedCapability, match="escapes"):
        _server_command("# escape=`\nFROM python\nCMD echo hello")


def test_runtime_skips_unverified_image_before_launch(downloads, monkeypatch, tmp_path):
    from openenv.validation import runner

    def unexpected_collection(*args, **kwargs):
        raise AssertionError("unverified image must never collect runtime evidence")

    monkeypatch.setattr(runner, "collect_runtime_evidence", unexpected_collection)
    healthy(monkeypatch)
    output = tmp_path / "report.json"
    report = validate_space("--level", "runtime", "--output", str(output))
    startup = next(r for r in report["results"] if r["check_id"] == "runtime.startup")
    assert startup["status"] == "skip"
    assert "immutable_image" in " ".join(startup["evidence"])
    assert "a" * 40 in " ".join(startup["evidence"])
    assert not FakeSandbox.created
    assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40
    assert report["source_digest"]
    assert report["verdict"] != "pass"
    bundle = json.loads((tmp_path / "report.artifacts/run-manifest.json").read_text())
    assert not bundle["provider"].get("image_verified")


def test_download_preserves_repository_files_in_the_sdk_metadata_path(
    downloads, monkeypatch, tmp_path
):
    original = huggingface_hub.snapshot_download

    def snapshot_with_repository_cache_file(*args, **kwargs):
        root = Path(original(*args, **kwargs))
        path = root / ".cache/huggingface/repository.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked repository content")
        return str(root)

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", snapshot_with_repository_cache_file
    )
    expected = space(tmp_path)
    tracked = expected / ".cache/huggingface/repository.txt"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("tracked repository content")
    assert validate_space()["source_digest"] == source_digest(expected)
