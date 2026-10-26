import json
import sys
import time
import urllib.error
from pathlib import Path

import pytest
from openenv.validation.manifest import (
    ExecutionDeclaration,
    NetworkPolicy,
    ResourceDeclaration,
)
from openenv.validation.providers import (
    docker,
    ProviderError,
    StartupError,
    UnsupportedCapability,
)
from openenv.validation.runtime.contracts import LaunchSpec


IMAGE = "sha256:" + "a" * 64


def launch_spec(**kwargs):
    return LaunchSpec(
        image_ref=IMAGE,
        run_id="test-run",
        network=kwargs.pop("network", NetworkPolicy()),
        resources=kwargs.pop(
            "resources",
            ResourceDeclaration(
                cpu=0.5,
                memory_mb=128,
                disk_mb=32,
                episode_timeout_s=5,
            ),
        ),
        **kwargs,
    )


def details(name="test", run_id="test-run", mounts=None):
    return {
        "Id": name,
        "Image": IMAGE,
        "Config": {
            "Labels": {docker._LABEL: run_id},
            "User": "65532:65532",
            "Env": ["TOKEN=private"],
        },
        "State": {"Running": True, "OOMKilled": False},
        "HostConfig": {"Memory": 128 * 1024**2, "ReadonlyRootfs": True},
        "Mounts": mounts or [],
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "49231"}]}
        },
    }


class Healthy:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def open(self, *args, **kwargs):
        return self


def listing(argv, containers):
    """Answer a `docker ps` name query the way the Docker CLI formats it."""
    query = argv[argv.index("--filter") + 1].removeprefix("name=")
    return "".join(
        f"{name}\t{owner}\n" for name, owner in containers.items() if query in name
    )


@pytest.fixture
def commands(monkeypatch):
    calls = []
    containers = {}

    def run(argv, timeout_s, max_bytes=docker._MAX_OUTPUT):
        calls.append(argv)
        if argv[1] == "create":
            containers[argv[argv.index("--name") + 1]] = "test-run"
        if argv[1] == "ps":
            return 0, listing(argv, containers), ""
        if argv[1] == "inspect":
            return 0, json.dumps([details(argv[-1])]), ""
        if argv[1] == "rm":
            containers.pop(argv[-1], None)
        return 0, "ok", ""

    monkeypatch.setattr(docker, "_command", run)
    monkeypatch.setattr(docker.urllib.request, "build_opener", lambda *a: Healthy())
    return calls


def test_start_is_hardened_and_does_not_forward_host_tokens(commands, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_not_forwarded_to_subject")
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"MODE": "probe"})
    )
    create = commands[0]
    assert subject.base_url == "http://127.0.0.1:49231"
    for flag, value in {
        "--user": "65532:65532",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--memory": "128m",
        "--memory-swap": "128m",
        "--cpus": "0.5",
        "--pids-limit": "256",
        "--publish": "127.0.0.1::8000",
        "--network": "bridge",
        "--stop-timeout": "0",
    }.items():
        assert create[create.index(flag) + 1] == value
    assert "--read-only" in create and "--init" in create
    assert "MODE=probe" in create
    assert not any("HF_TOKEN" in arg or "not_forwarded" in arg for arg in create)
    assert create[-1] == IMAGE
    tmpfs = create[create.index("--tmpfs") + 1]
    assert "size=16777216" in tmpfs
    assert int(create[create.index("--shm-size") + 1]) == 16777216
    subject.stop()
    subject.stop()
    assert sum(call[1] == "rm" for call in commands) == 1
    assert all("prune" not in call for call in commands)


def test_unsupported_network_is_rejected_without_docker(commands):
    network = NetworkPolicy(mode="allowlist", allowed_hosts=["example.org"])
    with pytest.raises(UnsupportedCapability):
        docker.DockerValidationProvider().start(launch_spec(network=network))
    assert not commands


def test_gpu_is_rejected_without_docker(commands):
    spec = launch_spec(
        resources=ResourceDeclaration(
            cpu=1, memory_mb=128, disk_mb=32, episode_timeout_s=5, gpus=1
        )
    )
    with pytest.raises(UnsupportedCapability, match="GPU"):
        docker.DockerValidationProvider().start(spec)
    assert not commands


def test_startup_timeout_removes_container(commands, monkeypatch):
    class Unhealthy:
        def open(self, *args, **kwargs):
            raise urllib.error.URLError("refused")

    monkeypatch.setattr(docker.urllib.request, "build_opener", lambda *a: Unhealthy())
    with pytest.raises(StartupError, match="deadline"):
        docker.DockerValidationProvider().start(launch_spec(startup_timeout_s=0.02))
    assert commands[-2][1:4] == ["rm", "--force", "--volumes"]


def test_image_volumes_are_rejected_before_execution(commands, monkeypatch):
    containers = {}

    def run(argv, *args):
        commands.append(argv)
        if argv[1] == "create":
            containers[argv[argv.index("--name") + 1]] = "test-run"
        if argv[1] == "ps":
            return 0, listing(argv, containers), ""
        if argv[1] == "inspect":
            return 0, json.dumps([details(mounts=[{"Type": "volume"}])]), ""
        if argv[1] == "rm":
            containers.pop(argv[-1], None)
        return 0, "ok", ""

    monkeypatch.setattr(docker, "_command", run)
    with pytest.raises(StartupError, match="mounts"):
        docker.DockerValidationProvider().start(launch_spec())
    assert not any(call[1] == "start" for call in commands)
    assert commands[-2][1] == "rm"


def test_create_failure_still_attempts_owned_cleanup(commands, monkeypatch):
    containers = {}

    def run(argv, *args):
        commands.append(argv)
        if argv[1] == "create":
            # The client timed out, but the engine had already created it.
            containers[argv[argv.index("--name") + 1]] = "test-run"
            raise ProviderError("deadline")
        if argv[1] == "ps":
            return 0, listing(argv, containers), ""
        if argv[1] == "inspect":
            return 0, json.dumps([details(argv[-1])]), ""
        if argv[1] == "rm":
            containers.pop(argv[-1], None)
        return 0, "", ""

    monkeypatch.setattr(docker, "_command", run)
    with pytest.raises(StartupError):
        docker.DockerValidationProvider().start(launch_spec())
    assert commands[-2][1] == "rm"
    name = commands[0][commands[0].index("--name") + 1]
    assert commands[-2][-1] == name
    assert commands[-1][commands[-1].index("--filter") + 1] == f"name={name}"


def test_inspection_reports_effective_settings_not_environment(commands):
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"OPTION": "1"})
    )
    observed = subject.inspect()
    assert observed["limits"]["Memory"] == 128 * 1024**2
    assert observed["limits"]["ReadonlyRootfs"] is True
    assert "TOKEN" not in json.dumps(observed)
    subject.stop()


def test_exec_timeout_destroys_subject_to_stop_descendants(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command

    def timeout_exec(argv, *args):
        if argv[1] == "exec":
            raise ProviderError("deadline")
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", timeout_exec)
    with pytest.raises(ProviderError, match="deadline"):
        subject.exec(["python", "-c", "while True: pass"], 0.1)
    assert commands[-2][1] == "rm"
    with pytest.raises(ProviderError, match="stopped"):
        subject.exec(["true"], 1)


def test_exec_timeout_is_preserved_when_cleanup_verification_fails(
    commands, monkeypatch
):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command
    timeout = ProviderError("Docker operation exceeded its deadline")
    removed = False

    def fail_exec_and_verification(argv, *args):
        nonlocal removed
        if argv[1] == "exec":
            raise timeout
        if argv[1] == "ps" and removed:
            return 1, "", "Docker daemon unavailable"
        removed = removed or argv[1] == "rm"
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", fail_exec_and_verification)
    with pytest.raises(ProviderError, match="independently verified") as error:
        subject.exec(["sleep", "20"], 0.1)
    assert error.value.__cause__ is timeout
    assert commands[-1][1:4] == ["rm", "--force", "--volumes"]
    assert not subject._stopped


# Engines word a missing container differently; cleanup must not depend on it.
MISSING_CONTAINER_ERRORS = [
    "Error: No such container: openenv-validation-x",
    'Error response from daemon: no container with name or ID "x" found: '
    "no such container",
]


@pytest.mark.parametrize("missing_error", MISSING_CONTAINER_ERRORS)
def test_cleanup_verifies_removal_by_listing_not_error_text(
    commands, monkeypatch, missing_error
):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command

    def vanished_during_rm(argv, *args):
        if argv[1] == "rm":
            original(argv, *args)
            return 1, "", missing_error
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", vanished_during_rm)
    subject.stop()
    assert subject._stopped
    assert [call[1] for call in commands[-3:]] == ["ps", "rm", "ps"]
    assert not any(call[1] == "inspect" for call in commands[-3:])


@pytest.mark.parametrize("missing_error", MISSING_CONTAINER_ERRORS)
def test_failed_removal_is_reported_whatever_the_engine_says(
    commands, monkeypatch, missing_error
):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command

    def rm_fails_but_container_remains(argv, *args):
        if argv[1] == "rm":
            commands.append(argv)
            return 1, "", missing_error
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", rm_fails_but_container_remains)
    with pytest.raises(ProviderError, match="Could not remove validation container"):
        subject.stop()
    assert not subject._stopped


def test_cleanup_skips_a_container_that_is_already_gone(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(launch_spec())
    original = docker._command

    def already_gone(argv, *args):
        if argv[1] == "ps":
            commands.append(argv)
            return 0, "", ""
        return original(argv, *args)

    monkeypatch.setattr(docker, "_command", already_gone)
    subject.stop()
    assert subject._stopped
    assert not any(call[1] == "rm" for call in commands)


def test_unreachable_engine_is_never_treated_as_removed(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(launch_spec())

    def unreachable(argv, *args):
        commands.append(argv)
        return 1, "", "Cannot connect to the Docker daemon"

    monkeypatch.setattr(docker, "_command", unreachable)
    with pytest.raises(ProviderError, match="ownership"):
        subject.stop()
    assert not subject._stopped
    assert not any(call[1] == "rm" for call in commands)


def test_exec_preserves_argv_and_redacts_credentials(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"API_KEY": "private-value"})
    )

    def run(argv, *args):
        commands.append(argv)
        return 7, "private-value hf_abcdefgh12345", "token=bad-secret"

    monkeypatch.setattr(docker, "_command", run)
    result = subject.exec(["printf", "$(cat /host/key); anything"], 1)
    assert result.exit_code == 7
    assert commands[-1][-1] == "$(cat /host/key); anything"
    assert "private-value" not in result.stdout and "hf_" not in result.stdout
    assert "bad-secret" not in result.stderr


@pytest.mark.parametrize("uppercase_secret_names", [False, True])
def test_build_uses_filtered_snapshot_and_returns_immutable_image(
    tmp_path, monkeypatch, uppercase_secret_names
):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "data.txt").write_text("source")
    for name in (".git", ".venv", "outputs"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "private").write_text("not copied")
    for name in (
        ".env",
        ".env.local",
        ".env.example",
        ".netrc",
        "secret",
        "secrets",
        "credentials",
        "credentials.json",
        "secrets.json",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
        "api_secret.json",
        "service_secrets.yaml",
        "_secret.json",
        "_secrets.yaml",
        "id_rsa",
        "id_rsa.pub",
        "id_rsa_backup",
        "id_ed25519",
        "id_ed25519.pub",
        "id_ed25519_backup",
        "id_ecdsa",
        "id_ecdsa.pub",
        "key.pem",
        "key.key",
        "certificate.p12",
        "certificate.pfx",
    ):
        if uppercase_secret_names:
            name = name.upper()
        (tmp_path / name).write_text("private")
    for name in (
        "secret.py",
        "secrets.py",
        "credentials.py",
        "secret_utils.py",
        "credentials_helper.py",
        "api_secretary.py",
    ):
        (tmp_path / name).write_text("VALUE = 'source'\n")

    def build(argv, timeout_s):
        assert timeout_s <= 600
        context = Path(argv[-1])
        assert set(path.name for path in context.iterdir()) == {
            "Dockerfile",
            "api_secretary.py",
            "credentials.py",
            "credentials_helper.py",
            "data.txt",
            "secret.py",
            "secret_utils.py",
            "secrets.py",
        }
        assert context != tmp_path
        Path(argv[argv.index("--iidfile") + 1]).write_text(IMAGE)
        return 0, "", ""

    monkeypatch.setattr(docker, "_command", build)
    assert (
        docker.DockerValidationProvider().build(tmp_path, ExecutionDeclaration())
        == IMAGE
    )


def test_build_rejects_symlink_escape_without_invoking_docker(tmp_path, commands):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ProviderError, match="symbolic"):
        docker.DockerValidationProvider().build(tmp_path, ExecutionDeclaration())
    assert not commands


def test_build_rejects_oversized_snapshot(tmp_path, commands):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    with pytest.raises(ProviderError, match="budget"):
        docker.DockerValidationProvider(max_context_bytes=1).build(
            tmp_path, ExecutionDeclaration()
        )
    assert not commands


def test_build_failure_is_bounded_and_sanitized(tmp_path, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    monkeypatch.setattr(
        docker, "_command", lambda *a: (1, "", "x" * 8000 + " hf_1234567890123456")
    )
    with pytest.raises(ProviderError) as error:
        docker.DockerValidationProvider().build(tmp_path, ExecutionDeclaration())
    assert len(str(error.value)) < 4200
    assert "hf_123456" not in str(error.value)


def test_command_drains_large_stdout_and_stderr_with_bounded_memory():
    code, stdout, stderr = docker._command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('a'*200000); sys.stderr.write('b'*200000)",
        ],
        5,
        max_bytes=1024,
    )
    assert code == 0 and stdout == "a" * 1024 and stderr == "b" * 1024


def test_command_kills_timed_out_process_group():
    started = time.monotonic()
    with pytest.raises(ProviderError, match="deadline"):
        docker._command([sys.executable, "-c", "import time; time.sleep(20)"], 0.05)
    assert time.monotonic() - started < 3


def test_readiness_never_follows_redirects():
    handler = docker._NoRedirects()
    assert handler.redirect_request(None, None, 302, "", {}, "http://private/") is None


def test_cleanup_refuses_a_container_with_another_owner(commands, monkeypatch):
    subject = docker.DockerValidationProvider().start(launch_spec())

    def foreign(argv, *args):
        commands.append(argv)
        # The name filter also matches a sibling whose name has ours as a prefix.
        return 0, f"{subject.name}-2\ttest-run\n{subject.name}\tother-run\n", ""

    monkeypatch.setattr(docker, "_command", foreign)
    with pytest.raises(ProviderError, match="another owner"):
        subject.stop()
    assert not any(call[1] == "rm" for call in commands)


def test_logs_remain_byte_bounded_after_redaction_expands_short_values(
    commands, monkeypatch
):
    subject = docker.DockerValidationProvider().start(
        launch_spec(env_vars={"KEY": "x"})
    )
    monkeypatch.setattr(docker, "_command", lambda *a: (0, "x" * 32, ""))
    output = subject.logs(max_bytes=32)
    assert len(output.encode("utf-8")) <= 32
    assert "x" not in output


def test_large_image_metadata_cannot_prevent_cleanup_after_inspection_failure(
    monkeypatch,
):
    calls = []
    containers = {}

    def run(argv, *args):
        calls.append(argv)
        if argv[1] == "create":
            containers[argv[argv.index("--name") + 1]] = "test-run"
        if argv[1] == "ps":
            return 0, listing(argv, containers), ""
        if argv[1] == "inspect":
            # Simulate the retained tail after image metadata exceeds 64 KiB.
            return 0, "x" * docker._MAX_OUTPUT, ""
        if argv[1] == "rm":
            containers.pop(argv[-1], None)
        return 0, "created", ""

    monkeypatch.setattr(docker, "_command", run)
    with pytest.raises(StartupError, match="invalid inspection evidence"):
        docker.DockerValidationProvider().start(launch_spec())
    assert not containers
    assert [call[1] for call in calls[-3:]] == ["ps", "rm", "ps"]
