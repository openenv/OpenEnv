"""The HF Sandbox provider: Space revision pinning, server start and verified cleanup."""

import json
from pathlib import Path
from types import SimpleNamespace

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
from typer.testing import CliRunner

GPU = SimpleNamespace(quantity="1")
PROBE = Path(__file__).parents[1] / "fixtures/validation/runtime/served_probe"


class FakeApi:
    def __init__(self, token=None):
        self.token = token
        self.sha = "a" * 40
        self.stages = ["RUNNING", "CANCELED"]

    def list_jobs_hardware(self):
        return [
            SimpleNamespace(name="cpu-basic", accelerator=None),
            SimpleNamespace(name="t4-small", accelerator=GPU),
        ]

    def space_info(self, space_id, revision=None):
        return SimpleNamespace(sha=self.sha)

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
    return tmp_path


@pytest.fixture
def downloads(provider, monkeypatch):
    """`snapshot_download` into `local_dir`, recording each requested revision."""
    revisions = []

    def snapshot_download(repo_id, *, repo_type, revision, local_dir):
        revisions.append(revision)
        for path in list(space(Path(local_dir)).iterdir()):
            # The Hub client records each download, with its time, in the folder.
            _local_folder.write_download_metadata(
                Path(local_dir), path.name, revision, "etag"
            )
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return revisions


def validate_space(*args):
    result = CliRunner().invoke(
        app,
        ["validate", "--space", "openenv-testing/probe", "--level", "static"]
        + ["--json", *args],
    )
    return json.loads(result.output)


def spec(image, gpus=0):
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


def test_named_revision_is_validated_as_its_commit(downloads, monkeypatch):
    revisions = []
    real = hf_sandbox.HFSandboxValidationProvider

    def recording(space_id, revision, **kwargs):
        revisions.append(revision)
        return real(space_id, revision, **kwargs)

    monkeypatch.setattr(hf_sandbox, "HFSandboxValidationProvider", recording)
    report = validate_space("--revision", "main")
    assert downloads == revisions == ["a" * 40]
    assert report["target"] == "spaces/openenv-testing/probe@" + "a" * 40


def test_repeat_downloads_hash_only_the_space_files(downloads, monkeypatch, tmp_path):
    clock = iter(range(1, 100))
    monkeypatch.setattr(
        _local_folder, "time", SimpleNamespace(time=lambda: next(clock))
    )
    digests = [validate_space()["source_digest"] for _ in range(2)]
    assert digests == [source_digest(space(tmp_path))] * 2


def test_build_pins_the_validated_revision(provider, tmp_path):
    hf = provider()
    image = hf.build(space(tmp_path), ExecutionDeclaration())
    assert image == "hf.co/spaces/openenv-testing/probe"
    assert hf._command == ["python", "-m", "served_probe.app"]
    assert hf._port == 8000
    hf._api.sha = "b" * 40
    with pytest.raises(StartupError, match="moved"):
        hf.build(tmp_path, ExecutionDeclaration())


def test_start_serves_the_space_command_and_stop_verifies_the_job(
    provider, tmp_path, monkeypatch
):
    healthy(monkeypatch)
    hf = provider("t4-small")
    image = hf.build(space(tmp_path), ExecutionDeclaration())
    running = hf.start(spec(image, gpus=1))
    sandbox = FakeSandbox.created[0]
    assert sandbox.kwargs["flavor"] == "t4-small"
    assert sandbox.kwargs["image"] == "hf.co/spaces/openenv-testing/probe"
    assert "forward_hf_token" not in sandbox.kwargs
    command, options = sandbox.commands[0]
    assert command.startswith("python -m served_probe.app")
    assert options == {"shell": True, "background": True}
    assert running.base_url == "http://127.0.0.1:9"
    assert running.inspect()["space_revision"] == "a" * 40
    running.stop()
    assert sandbox.killed


def test_unhealthy_start_kills_the_job(provider, tmp_path, monkeypatch):
    healthy(monkeypatch, status=503)
    hf = provider()
    image = hf.build(space(tmp_path), ExecutionDeclaration())
    with pytest.raises(StartupError, match="/health"):
        hf.start(spec(image).model_copy(update={"startup_timeout_s": 0.01}))
    assert FakeSandbox.created[0].killed


def test_gpu_declaration_needs_a_gpu_flavor(provider, tmp_path):
    hf = provider()
    image = hf.build(space(tmp_path), ExecutionDeclaration())
    with pytest.raises(UnsupportedCapability, match="0 GPUs"):
        hf.start(spec(image, gpus=1))
    assert FakeSandbox.created == []


def test_unverified_cleanup_is_an_error(provider, tmp_path, monkeypatch):
    healthy(monkeypatch)
    hf = provider()
    running = hf.start(spec(hf.build(space(tmp_path), ExecutionDeclaration())))
    hf._api.stages = ["RUNNING"]
    clock = iter(range(0, 1000, 30))
    monkeypatch.setattr(hf_sandbox.time, "monotonic", lambda: next(clock))
    with pytest.raises(ProviderError, match="did not stop"):
        running.stop()


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


def test_app_port_defaults_to_the_hub_default():
    assert _app_port("---\napp_port: 8000\n---\n") == 8000
    assert _app_port("no front matter") == 7860
