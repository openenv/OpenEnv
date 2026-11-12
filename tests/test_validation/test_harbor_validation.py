"""Native Harbor checks mock only the Trial boundary and trusted host inspector."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from openenv.validation import harbor
from openenv.validation.harbor import DockerInspector
from openenv.validation.manifest import ManifestError
from openenv.validation.parsers.harbor_toml import (
    HarborTaskParser,
    read_harbor_contract,
    read_harbor_task,
)
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.types import CheckStatus, Level, SignatureKind, Verdict

pytest.importorskip("harbor.models.task.task")

IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    for directory in ("environment", "solution", "tests", "validation"):
        (root / directory).mkdir(parents=True)
    (root / "task.toml").write_text(
        'schema_version = "1.4"\n[verifier]\nenvironment_mode = "shared"\n'
    )
    (root / "instruction.md").write_text("Write the correct answer.\n")
    (root / "environment/Dockerfile").write_text("FROM debian:bookworm-slim\n")
    (root / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "solution/solve.sh").write_text("#!/bin/sh\nexit 0\n")
    contract = {
        "profile": "harbor-semantic-v1",
        "reward_key": "reward",
        "reward": {"range": [0, 1], "oracle_tolerance": 0, "floor_margin": 0.1},
        "resources": {
            "cpu": 1,
            "memory_mb": 512,
            "disk_mb": 1024,
            "episode_timeout_s": 10,
        },
        "types": {"tags": ["swe"]},
        "original_source_digest": "b" * 64,
        "requires_credentials": False,
    }
    (root / "validation/openenv.json").write_text(json.dumps(contract))
    return root


@pytest.fixture
def boundary(monkeypatch, task):
    state = SimpleNamespace(
        rewards={"nop": b'{"reward":0}', "oracle": b'{"reward":1}'},
        trials=[],
        cleanup_ok=True,
        cleanup_fail_role=None,
        teardown_defect=None,
        mutate=False,
        private_mutate=False,
        image_drift=False,
        config_drift=False,
        oracle_exit=None,
        exception=False,
        exception_type=None,
        exception_phase="environment build/start",
        no_identity=False,
        timeout=False,
        retained=True,
        retained_error=False,
        overwrite_exit=False,
        excluded_copied=[],
        start_delay=0,
        agent_delay=0,
        pending_compose=False,
        inventory_problem=None,
        cleanup_calls=[],
        preflight_error=None,
        interrupt_role=None,
        on_build=None,
    )

    class Inspector:
        def preflight(self):
            if state.preflight_error is not None:
                raise state.preflight_error
            return {"os": "linux", "daemon_id": "mock-host"}

        def inventory(self, project):
            if state.inventory_problem and "canary-1__env" in project:
                if state.inventory_problem == "error":
                    raise OSError("Docker inventory failed")
                return {"containers": ["preexisting"], "networks": [], "volumes": []}
            return {"containers": [], "networks": [], "volumes": []}

        def capture(self, project, retained_tag, *, retain=True):
            role = _role(project.removesuffix("__env"))
            return {
                "container_id": role + "-container",
                "image_ref": "sha256:" + "c" * 64
                if state.image_drift and role == "oracle"
                else IMAGE,
                "configuration_digest": "drift"
                if state.config_drift and role == "oracle"
                else "stable",
                "retained_image_tag": retained_tag,
            }

        def cleanup(self, project):
            state.cleanup_calls.append(project)
            verified = (
                state.cleanup_ok
                and _role(project.removesuffix("__env")) != state.cleanup_fail_role
            )
            return {
                "verified": verified,
                "after": {
                    "containers": [] if verified else ["leftover"],
                    "networks": [],
                    "volumes": [],
                },
            }

        def remove_image(self):
            pass

        def verify_image(self, tag, image_ref):
            if state.retained_error:
                raise OSError("inspect failed")
            return state.retained

    class Trial:
        def __init__(self, config):
            self.config = config
            self.role = _role(config.trial_name)
            self.agent = config.agent.name
            self.solve = (config.task.path / "solution/solve.sh").read_bytes()
            directory = config.trials_dir / config.trial_name
            (directory / "verifier").mkdir(parents=True)
            (directory / "agent").mkdir()
            self.paths = SimpleNamespace(
                trial_dir=directory,
                reward_json_path=directory / "verifier/reward.json",
                reward_text_path=directory / "verifier/reward.txt",
                agent_dir=directory / "agent",
            )
            self.agent_environment = SimpleNamespace(
                session_id=config.trial_name + "__env", start=self.start, stop=self.stop
            )
            self.hooks = {}
            self._openenv_cancelled_compose = []
            environment = read_harbor_task(config.task.path).config.environment
            self._environment_build_timeout_sec = environment.build_timeout_sec
            self.docker_image = environment.docker_image
            state.excluded_copied.append(
                (config.task.path / "environment/.worktrees").exists()
            )
            state.trials.append(self)

        async def stop(self, *, delete):
            assert not delete
            if state.teardown_defect == "error":
                raise OSError("Harbor teardown failed")
            if state.teardown_defect == "drift":
                self.paths.reward_json_path.write_bytes(b'{"reward":0}')
            if state.teardown_defect == "unsafe":
                self.paths.reward_json_path.unlink()
                self.paths.reward_json_path.symlink_to("outside-reward.json")

        async def start(self, *, force_build):
            if self.role == "nop" and state.on_build is not None:
                state.on_build(self.agent_environment.session_id)
            if state.pending_compose:
                self._openenv_cancelled_compose.append(
                    {
                        "command": "build",
                        "project": self.agent_environment.session_id,
                        "client_group_killed": True,
                        "client_reaped": True,
                        "daemon_completion_verified": False,
                    }
                )
                raise TimeoutError()
            if self.role == "nop":
                await asyncio.sleep(state.start_delay)

        def add_hook(self, event, callback):
            self.hooks.setdefault(event.value, []).append(callback)

        async def run(self):
            for hook in self.hooks.get("environment-start", []):
                await hook(None)
            if (
                state.exception_type in ("RuntimeError", "EnvironmentStartTimeoutError")
                and state.exception_phase == "environment build/start"
            ):
                return SimpleNamespace(
                    exception_info=SimpleNamespace(exception_type=state.exception_type),
                    verifier_result=None,
                )
            from harbor.trial.trial import (
                EnvironmentStartTimeoutError,
                Trial as HarborTrial,
            )

            try:
                await HarborTrial._start_agent_environment(self)
            except EnvironmentStartTimeoutError:
                return SimpleNamespace(
                    exception_info=SimpleNamespace(
                        exception_type="EnvironmentStartTimeoutError"
                    ),
                    verifier_result=None,
                )
            if not state.no_identity:
                for hook in self.hooks.get("agent-start", []):
                    await hook(None)
            if self.role == state.interrupt_role:
                raise KeyboardInterrupt
            try:
                await asyncio.wait_for(
                    asyncio.sleep(state.agent_delay),
                    timeout=self.config.agent.override_timeout_sec,
                )
            except TimeoutError:
                return SimpleNamespace(
                    exception_info=SimpleNamespace(exception_type="AgentTimeoutError"),
                    verifier_result=None,
                )
            if state.timeout:
                raise TimeoutError()
            raw = state.rewards.get(self.role, state.rewards[self.agent])
            if raw is not None:
                self.paths.reward_json_path.write_bytes(raw)
            if self.agent == "oracle" and state.oracle_exit is not None:
                (self.paths.agent_dir / "exit-code.txt").write_text(
                    str(state.oracle_exit)
                )
            try:
                for hook in self.hooks.get("agent-end", []):
                    await hook(None)
            finally:
                if self.agent == "oracle" and state.overwrite_exit:
                    (self.paths.agent_dir / "exit-code.txt").write_text("0")
            if state.mutate:
                (task / "instruction.md").write_text("changed source")
            if state.private_mutate:
                (self.config.task.path / "instruction.md").write_text(
                    "changed execution source"
                )
            for hook in self.hooks.get("verification-start", []):
                await hook(None)
            try:
                rewards = json.loads(raw) if raw else None
            except ValueError:
                rewards = None
            exception_type = state.exception_type
            if raw is None:
                exception_type = "RewardFileNotFoundError"
            elif raw == b"":
                exception_type = "RewardFileEmptyError"
            elif raw in (b'{"reward":NaN}', b'{"reward":Infinity}', b'{"reward":"0"}'):
                exception_type = "VerifierOutputParseError"
            if state.exception:
                exception_type = "TestError"
            return SimpleNamespace(
                exception_info=SimpleNamespace(exception_type=exception_type)
                if exception_type
                else None,
                verifier_result=SimpleNamespace(rewards=rewards),
            )

    async def create(config):
        return Trial(config)

    monkeypatch.setattr(harbor, "DockerInspector", Inspector)
    monkeypatch.setattr(harbor, "_create_trial", create)
    return state


@pytest.fixture
def image_docker(boundary, monkeypatch):
    state = SimpleNamespace(
        images={"unrelated:tag": IMAGE},
        removed=set(),
        commands=[],
        image_os="linux",
        remove_error=False,
        retained_remove_error=False,
        image_drift=False,
        foreign_image=False,
    )

    def built(project):
        state.images[project + "-main:latest"] = (
            "sha256:" + "e" * 64 if state.foreign_image else IMAGE
        )

    boundary.on_build = built

    def docker(self, *args):
        state.commands.append(args)
        if args == ("info", "--format", "{{json .}}"):
            return json.dumps({"OSType": "linux", "ID": "fake", "ServerVersion": "29"})
        if args == ("compose", "version"):
            return "Docker Compose"
        if args[0] == "ps":
            project = args[-1].removeprefix("label=com.docker.compose.project=")
            return "\n".join(
                trial.role + "-container"
                for trial in boundary.trials
                if trial.agent_environment.session_id == project
                and trial.role not in state.removed
            )
        if args[:2] in (("network", "ls"), ("volume", "ls")):
            return ""
        if args[0] == "inspect":
            trial = next(t for t in boundary.trials if t.role + "-container" == args[1])
            project = trial.agent_environment.session_id
            return json.dumps(
                [
                    {
                        "Id": args[1],
                        "Image": IMAGE,
                        "State": {"Running": True},
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": project,
                                "com.docker.compose.service": "main",
                            },
                            "User": "1000",
                            "WorkingDir": "/app",
                        },
                        "HostConfig": {"NetworkMode": project + "_default"},
                        "Mounts": [],
                    }
                ]
            )
        if args[:2] == ("image", "tag"):
            state.images[args[3]] = args[2]
            return ""
        if args[:2] == ("image", "ls"):
            assert args[2:4] == ("-q", "--no-trunc")
            return state.images.get(args[4], "")
        if args[:2] == ("image", "inspect"):
            image = state.images[args[2]]
            if state.image_drift and args[2].endswith("-main:latest"):
                image = "sha256:" + "e" * 64
            project = args[2].removesuffix("-main:latest")
            return json.dumps(
                [
                    {
                        "Id": image,
                        "Os": state.image_os,
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": "unrelated"
                                if state.foreign_image
                                else project,
                                "com.docker.compose.service": "main",
                            }
                        },
                    }
                ]
            )
        if args[:3] == ("image", "rm", "--no-prune"):
            if (state.remove_error and args[3].endswith("-main:latest")) or (
                state.retained_remove_error
                and args[3].startswith("openenv-validation:")
            ):
                raise OSError("image removal failed")
            del state.images[args[3]]
            return ""
        if args[:3] == ("rm", "-f", "-v"):
            state.removed.add(args[3].removesuffix("-container"))
            return ""
        pytest.fail(f"unexpected Docker command: {args}")

    monkeypatch.setattr(DockerInspector, "_docker", docker)
    monkeypatch.setattr(harbor, "DockerInspector", DockerInspector)
    return state


def _role(trial_name):
    # openenv-validation-<run id>-<role>
    return trial_name.split("-", 3)[3]


def result(report, check_id):
    return next(check for check in report.results if check.check_id == check_id)


def statuses(report):
    return {check.check_id: check.status for check in report.results}


def test_parser_is_pure_and_describes_process_binding(task, monkeypatch):
    monkeypatch.setattr(
        harbor, "_create_trial", lambda config: pytest.fail("Trial created by parser")
    )
    manifest = HarborTaskParser().parse(task)
    assert manifest.signature is SignatureKind.HARBOR_TASK
    assert manifest.execution.kind == "harbor_task"
    assert manifest.execution.agent_boundary == "process"
    assert manifest.reward.floor_margin == 0.1


def test_success_grades_semantic_checks_with_native_raw_evidence(
    task, boundary, tmp_path
):
    original = source_digest(task)
    artifacts = tmp_path / "artifacts"
    report = run_validation(task, artifacts_dir=artifacts)
    # Checks this binding does not implement remain named SKIPs.
    assert report.verdict is Verdict.WARN
    assert report.levels_run == [Level.STATIC, Level.RUNTIME, Level.SEMANTIC]
    assert source_digest(task) == original
    passed = {
        "static.manifest",
        "runtime.startup",
        "runtime.reward_well_formed",
        "runtime.episode_determinism",
        "semantic.oracle_max",
        "semantic.floor_gap",
    }
    found = statuses(report)
    assert {check for check, status in found.items() if status is CheckStatus.PASS} == (
        passed
    )
    assert all(
        status is CheckStatus.SKIP
        for check, status in found.items()
        if check not in passed
    )
    assert result(report, "semantic.canary_floor").evidence == [
        "no canaries declared in validation/openenv.json"
    ]
    assert result(report, "static.sbom").evidence == [
        "grader not implemented in this build"
    ]
    assert "harbor_task" in result(report, "semantic.replayability").evidence[0]
    assert result(report, "semantic.floor_gap").measured["measured_floor"] == 0
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert native["profile"] == "harbor-semantic-v1"
    assert native["task_effective_digest"] == report.source_digest == original
    assert native["original_source_digest"] == "b" * 64
    assert native["image_ref"] == IMAGE
    assert native["floor_reward"] == 0 and native["oracle_reward"] == 1
    assert native["cleanup"]["verified"]
    assert [trial["role"] for trial in native["trials"]] == [
        "nop",
        "oracle",
        "oracle-replay",
    ]
    assert [trial.agent for trial in boundary.trials] == ["nop", "oracle", "oracle"]
    assert all(not trial.config.environment.delete for trial in boundary.trials)
    assert all(
        trial.config.environment.kwargs["keep_containers"] for trial in boundary.trials
    )
    assert len({trial.config.task.path for trial in boundary.trials}) == 3
    assert json.loads((artifacts / "report.json").read_text()) == json.loads(
        report.model_dump_json()
    )
    inventory = {}
    for line in (artifacts / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == expected
        inventory[name] = expected
    assert {
        "harbor-evidence.json",
        "harbor-trials/nop/reward.json",
        "harbor-trials/oracle/reward.json",
        "harbor-trials/oracle-replay/reward.json",
    } <= inventory.keys()
    assert (artifacts / "harbor-trials/nop/reward.json").read_bytes() == b'{"reward":0}'


def test_trial_does_not_share_phase_timeout(task, boundary, monkeypatch):
    timeouts = []
    timeout = harbor.asyncio.timeout

    def record_timeout(delay):
        timeouts.append(delay)
        return timeout(delay)

    monkeypatch.setattr(harbor.asyncio, "timeout", record_timeout)
    assert run_validation(task).verdict is Verdict.WARN
    assert timeouts == [600, 30, 30, 30]
    assert all(
        trial.config.agent.override_timeout_sec == 10
        and trial.config.agent.override_setup_timeout_sec == 10
        and trial.config.verifier.override_timeout_sec == 10
        for trial in boundary.trials
    )


@pytest.mark.parametrize(
    "outcome", ["kept", "fail", "error", "interrupted", "capture_error"]
)
def test_retained_image_policy(task, boundary, image_docker, tmp_path, outcome):
    if outcome == "fail":
        boundary.rewards["nop"] = b'{"reward":1}'
    elif outcome == "error":
        boundary.exception = True
    elif outcome == "interrupted":
        boundary.interrupt_role = "oracle"
    elif outcome == "capture_error":
        image_docker.image_os = "windows"
    artifacts = tmp_path / "image-policy"
    report = run_validation(task, artifacts_dir=artifacts)
    tag = "openenv-validation:" + boundary.trials[0].config.trial_name.split("-")[2]
    measured = result(report, "runtime.startup").measured
    compose_tag = boundary.trials[0].agent_environment.session_id + "-main:latest"
    removals = [
        cmd
        for cmd in image_docker.commands
        if cmd[:2] == ("image", "rm") and cmd[3] != compose_tag
    ]
    if outcome == "kept":
        assert report.verdict is Verdict.WARN
        assert measured["retained_image_tag"] == tag
        assert image_docker.images == {"unrelated:tag": IMAGE, tag: IMAGE}
        assert not removals
    else:
        assert report.verdict is Verdict.FAIL
        if outcome == "fail":
            assert result(report, "semantic.floor_gap").status is CheckStatus.FAIL
        else:
            assert result(report, "runtime.startup").status is CheckStatus.ERROR
        assert "retained_image_tag" not in measured
        assert image_docker.images == {"unrelated:tag": IMAGE}
        assert removals == [("image", "rm", "--no-prune", tag)]
    assert json.loads((artifacts / "report.json").read_text()) == json.loads(
        report.model_dump_json()
    )


@pytest.mark.parametrize("keep", [True, False], ids=["kept", "removed"])
def test_compose_build_tag_is_removed_on_every_run(
    task, boundary, image_docker, tmp_path, keep
):
    if not keep:
        boundary.rewards["nop"] = b'{"reward":1}'
    report = run_validation(task, artifacts_dir=tmp_path / "compose-tags")
    project = boundary.trials[0].agent_environment.session_id
    compose_tag = project + "-main:latest"
    retained_tag = "openenv-validation:" + project.split("-")[2]
    expected = {"unrelated:tag": IMAGE}
    if keep:
        expected[retained_tag] = IMAGE
        assert (
            result(report, "runtime.startup").measured["retained_image_tag"]
            == retained_tag
        )
    assert image_docker.images == expected
    removal = ("image", "rm", "--no-prune", compose_tag)
    index = image_docker.commands.index(removal)
    assert image_docker.commands[index - 1] == ("image", "inspect", compose_tag)
    assert image_docker.removed == {trial.role for trial in boundary.trials}


@pytest.mark.parametrize("defect", ["remove_error", "image_drift", "foreign_image"])
def test_compose_tag_cleanup_failure_is_a_startup_error(
    task, boundary, image_docker, tmp_path, defect
):
    setattr(image_docker, defect, True)
    artifacts = tmp_path / "compose-cleanup"
    report = run_validation(task, artifacts_dir=artifacts)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert "retained_image_tag" not in startup.measured
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert not native["cleanup"]["verified"]
    assert native["trials"][0]["failure"] == "CleanupNotVerified"
    assert "unrelated:tag" in image_docker.images
    if defect in {"image_drift", "foreign_image"}:
        assert not any(
            cmd[:2] == ("image", "rm") and cmd[3].endswith("-main:latest")
            for cmd in image_docker.commands
        )


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_retained_image_removed_when_reporting_does_not_complete(
    task, boundary, image_docker, tmp_path, monkeypatch, error
):
    def interrupted(*args, **kwargs):
        raise error

    monkeypatch.setattr(harbor, "write_runtime_bundle", interrupted)
    with pytest.raises(error):
        run_validation(task, artifacts_dir=tmp_path / "interrupted-report")
    assert image_docker.images == {"unrelated:tag": IMAGE}
    assert len(boundary.trials) == 3


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_image_removal_failure_preserves_the_reporting_exception(
    task, boundary, image_docker, tmp_path, monkeypatch, error
):
    original = error()
    cleanup = {}

    def interrupted(*args, **kwargs):
        nonlocal cleanup
        cleanup = kwargs["cleanup"]
        raise original

    image_docker.retained_remove_error = True
    monkeypatch.setattr(harbor, "write_runtime_bundle", interrupted)
    with pytest.raises(error) as caught:
        run_validation(task, artifacts_dir=tmp_path / "interrupted-report")
    assert caught.value is original
    assert not cleanup["verified"] and not cleanup["completed"]
    assert cleanup["failure"] == "OSError"
    assert (
        "retained image removal failed (OSError); cleanup incomplete"
        in original.__notes__
    )


def test_build_failure_has_no_retained_tag_to_remove(task, boundary, image_docker):
    boundary.exception_type = "RuntimeError"
    report = run_validation(task)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.FAIL
    assert "retained_image_tag" not in startup.measured
    assert image_docker.images == {"unrelated:tag": IMAGE}
    assert not any(cmd[:2] == ("image", "rm") for cmd in image_docker.commands)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        b"",
        b"{}",
        b'{"other":0}',
        b'{"reward":true}',
        b'{"reward":NaN}',
        b'{"reward":Infinity}',
        b'{"reward":1.01}',
        b'{"reward":-0.01}',
        b'{"reward":"0"}',
        b'{"reward":0,"aux":true}',
        b'{"reward":0,"reward":1}',
        b'{"reward":1.0000000000000000001}',
    ],
)
def test_bad_rewards_block_and_preserve_raw(task, boundary, tmp_path, raw):
    boundary.rewards["nop"] = raw
    artifacts = tmp_path / "bad"
    report = run_validation(task, artifacts_dir=artifacts)
    assert report.verdict is Verdict.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.PASS
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.FAIL
    assert result(report, "semantic.oracle_max").status is CheckStatus.SKIP
    assert len(boundary.trials) == 1
    if raw is not None:
        assert (artifacts / "harbor-trials/nop/reward.json").read_bytes() == raw


@pytest.mark.parametrize(
    "floor,oracle,oracle_max,floor_gap",
    [(1, 1, True, False), (0, 0.8, False, True), (0.95, 1, True, False)],
)
def test_actual_oracle_floor_arithmetic(
    task, boundary, tmp_path, floor, oracle, oracle_max, floor_gap
):
    boundary.rewards = {
        "nop": json.dumps({"reward": floor}).encode(),
        "oracle": json.dumps({"reward": oracle}).encode(),
    }
    report = run_validation(task)
    expected = {True: CheckStatus.PASS, False: CheckStatus.FAIL}
    assert report.verdict is Verdict.FAIL
    assert result(report, "semantic.oracle_max").status is expected[oracle_max]
    assert result(report, "semantic.floor_gap").status is expected[floor_gap]
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.PASS


def test_tiny_positive_floor_does_not_round_up_to_margin(task, boundary, tmp_path):
    # Decimal's default precision and float subtraction both round this gap up.
    raw_contract = json.loads((task / "validation/openenv.json").read_text())
    raw_contract["reward"]["oracle_tolerance"] = 0.1
    raw_contract["reward"]["floor_margin"] = 0.9
    (task / "validation/openenv.json").write_text(json.dumps(raw_contract))
    boundary.rewards = {"nop": b'{"reward":1e-300}', "oracle": b'{"reward":0.9}'}
    report = run_validation(task)
    assert result(report, "semantic.oracle_max").status is CheckStatus.PASS
    assert result(report, "semantic.floor_gap").status is CheckStatus.FAIL
    assert report.verdict is Verdict.FAIL


@pytest.mark.parametrize(
    "flag",
    [
        "mutate",
        "private_mutate",
        "image_drift",
        "config_drift",
        "exception",
        "no_identity",
        "timeout",
    ],
)
def test_source_identity_exception_timeout_fail_closed(task, boundary, flag):
    setattr(boundary, flag, True)
    report = run_validation(task)
    assert report.verdict is Verdict.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.ERROR


@pytest.mark.parametrize(
    "exception_type,phase",
    [
        ("RuntimeError", "environment"),
        ("AgentTimeoutError", "agent"),
        ("AgentSetupTimeoutError", "agent setup"),
        ("VerifierTimeoutError", "verifier"),
    ],
)
def test_subject_failures_name_the_phase(task, boundary, exception_type, phase):
    boundary.exception_type = exception_type
    report = run_validation(task)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.FAIL
    assert phase in startup.evidence[0]
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.SKIP


def test_runtime_error_outside_build_start_is_a_validator_error(task, boundary):
    boundary.exception_type = "RuntimeError"
    boundary.exception_phase = "verifier"
    report = run_validation(task)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert startup.measured["failures"] == ["Harbor Trial returned exception_info"]
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.SKIP


def test_cleanup_failure_blocks(task, boundary, tmp_path):
    boundary.cleanup_ok = False
    artifacts = tmp_path / "cleanup"
    report = run_validation(task, artifacts_dir=artifacts)
    assert report.verdict is Verdict.FAIL
    assert not json.loads((artifacts / "harbor-evidence.json").read_text())["cleanup"][
        "verified"
    ]


@pytest.mark.parametrize(
    "defect,failure",
    [
        ("error", "HarborTeardownError"),
        ("cleanup", "CleanupNotVerified"),
        ("drift", "RewardArtifactDrift"),
        ("unsafe", "UnsafeRewardArtifact"),
    ],
)
def test_host_failure_overrides_reward_error(task, boundary, tmp_path, defect, failure):
    boundary.rewards["nop"] = b'{"reward":true}'
    boundary.teardown_defect = defect
    boundary.cleanup_ok = defect != "cleanup"
    artifacts = tmp_path / "host-over-reward"
    report = run_validation(task, artifacts_dir=artifacts)
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.SKIP
    trial = json.loads((artifacts / "harbor-evidence.json").read_text())["trials"][0]
    assert trial["failure"] == failure
    assert trial["reward_error"]


def test_oracle_nonzero_exit_fails_oracle_max_and_retains_marker(
    task, boundary, tmp_path
):
    boundary.oracle_exit = 7
    artifacts = tmp_path / "exit"
    report = run_validation(task, artifacts_dir=artifacts)
    assert report.verdict is Verdict.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.PASS
    oracle_max = result(report, "semantic.oracle_max")
    assert oracle_max.status is CheckStatus.FAIL
    assert "oracle run 1 exited with code 7" in oracle_max.evidence
    assert (
        artifacts / "harbor-trials/oracle/oracle-exit-code.txt"
    ).read_bytes() == b"7"


def test_sandbox_log_sync_cannot_erase_trusted_oracle_exit(task, boundary, tmp_path):
    boundary.oracle_exit = 7
    boundary.overwrite_exit = True
    artifacts = tmp_path / "overwritten-exit"
    report = run_validation(task, artifacts_dir=artifacts)
    assert report.verdict is Verdict.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert native["trials"][1]["oracle_exit_code"] == 7
    assert (
        artifacts / "harbor-trials/oracle/oracle-exit-code.txt"
    ).read_bytes() == b"7"


def test_excluded_symlink_tree_is_never_dereferenced_or_copied(
    task, boundary, tmp_path
):
    sentinel = tmp_path / "outside-secret"
    sentinel.write_text("must not enter a build context")
    excluded = task / "environment/.worktrees"
    excluded.mkdir()
    (excluded / "payload").symlink_to(sentinel)
    assert run_validation(task).verdict is Verdict.WARN
    assert boundary.excluded_copied == [False, False, False]


def test_copy_change_and_restore_cannot_escape_source_binding(
    task, boundary, tmp_path, monkeypatch
):
    solution = task / "solution/solve.sh"
    solution.write_text("exit 7\n")
    digest = source_digest(task)
    copytree = harbor.shutil.copytree
    copies = 0

    def copying(*args, **kwargs):
        nonlocal copies
        if args[0] == task:
            copies += 1
            if copies == 2:
                solution.write_text("exit 0\n")
                try:
                    return copytree(*args, **kwargs)
                finally:
                    solution.write_text("exit 7\n")
        return copytree(*args, **kwargs)

    monkeypatch.setattr(harbor.shutil, "copytree", copying)
    artifacts = tmp_path / "copy-binding"
    report = run_validation(task, artifacts_dir=artifacts)
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert source_digest(task) == report.source_digest == digest
    assert report.verdict is Verdict.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    assert len(boundary.trials) == 1
    assert native["trials"][1]["unmodified_task_digest"] != digest
    assert "validated source digest" in native["trials"][1]["reason"]


@pytest.mark.parametrize("blocked", [None, "skip-build", "unsupported", "provider"])
def test_missing_oracle_fails_oracle_max_without_starting_trial(
    task, boundary, blocked
):
    (task / "solution/solve.sh").unlink()
    if blocked == "unsupported":
        (task / "task.toml").write_text('[environment]\ndocker_image="prebuilt"\n')
    report = run_validation(
        task,
        skip_build=blocked == "skip-build",
        provider=object() if blocked == "provider" else None,
    )
    assert report.verdict is Verdict.FAIL
    assert report.levels_run == [Level.STATIC, Level.SEMANTIC]
    # RFC 008: no oracle is a graded FAIL, not a SKIP.
    assert result(report, "semantic.oracle_max").status is CheckStatus.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.SKIP
    assert not boundary.trials


def test_retained_image_drift_blocks(task, boundary):
    boundary.retained = False
    assert run_validation(task).verdict is Verdict.FAIL


@pytest.mark.parametrize("inspect_error", [False, True])
def test_retained_image_failure_is_not_hidden_by_bad_reward(
    task, boundary, inspect_error
):
    boundary.rewards["nop"] = b'{"reward":true}'
    boundary.retained = False
    boundary.retained_error = inspect_error
    report = run_validation(task)
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.SKIP


def test_host_inspector_scopes_inventory_and_cleanup_to_owned_project(monkeypatch):
    inspector = harbor.DockerInspector()
    project = "openenv-validation-" + "d" * 32 + "-nop__env"
    commands = []
    resources = {
        "containers": ["c" * 64],
        "networks": ["e" * 64],
        "volumes": ["owned-volume"],
    }

    def docker(*args):
        commands.append(args)
        if args[:2] == ("image", "ls"):
            assert args == ("image", "ls", "-q", "--no-trunc", project + "-main:latest")
            return ""
        if args == ("volume", "ls", "-q"):
            return "\n".join(resources["volumes"])
        if args[0] == "inspect":
            return json.dumps(
                [
                    {
                        "Config": {"Labels": {"com.docker.compose.project": project}},
                        "Mounts": [],
                    }
                ]
            )
        if args[0] == "ps":
            assert args[-2:] == (
                "--filter",
                f"label=com.docker.compose.project={project}",
            )
            return "\n".join(resources["containers"])
        if args[:2] in (("network", "ls"), ("volume", "ls")):
            assert args[-2:] == (
                "--filter",
                f"label=com.docker.compose.project={project}",
            )
            return "\n".join(
                resources["networks" if args[0] == "network" else "volumes"]
            )
        if args[0] == "rm":
            assert args == ("rm", "-f", "-v", "c" * 64)
            resources["containers"] = []
        elif args[:2] == ("network", "rm"):
            assert args == ("network", "rm", "e" * 64)
            resources["networks"] = []
        elif args[:2] == ("volume", "rm"):
            assert args == ("volume", "rm", "owned-volume")
            resources["volumes"] = []
        else:
            pytest.fail(f"unexpected Docker command: {args}")
        return ""

    monkeypatch.setattr(inspector, "_docker", docker)
    assert inspector.cleanup(project)["verified"]
    assert commands[-4:] == [
        (
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ),
        (
            "network",
            "ls",
            "-q",
            "--no-trunc",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ),
        (
            "volume",
            "ls",
            "-q",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ),
        ("image", "ls", "-q", "--no-trunc", project + "-main:latest"),
    ]
    with pytest.raises(ValueError, match="run-owned"):
        inspector.cleanup("somebody-elses-project")


@pytest.mark.parametrize("volume_survives_removal", [False, True])
@pytest.mark.parametrize("automatic_volume_removal", [False, True])
def test_anonymous_volumes_are_registered_removed_and_rechecked(
    monkeypatch, volume_survives_removal, automatic_volume_removal
):
    inspector = harbor.DockerInspector()
    project = "openenv-validation-" + "d" * 32 + "-nop__env"
    containers = ["c" * 64]
    volumes = {"anonymous-owned", "unrelated-volume"}

    def inventory(value):
        assert value == project
        # The anonymous volume deliberately has no Compose project label.
        return {
            "containers": list(containers),
            "networks": [],
            "volumes": [],
            "images": [],
        }

    def docker(*args):
        if args == ("inspect", "c" * 64):
            return json.dumps(
                [
                    {
                        "Config": {"Labels": {"com.docker.compose.project": project}},
                        "Mounts": [{"Type": "volume", "Name": "anonymous-owned"}],
                    }
                ]
            )
        if args == ("volume", "ls", "-q"):
            return "\n".join(volumes)
        if args == ("rm", "-f", "-v", "c" * 64):
            containers.clear()  # Docker may log volume-removal errors and return 0.
            if automatic_volume_removal:
                volumes.remove("anonymous-owned")
        elif args == ("volume", "rm", "anonymous-owned"):
            if not volume_survives_removal:
                volumes.remove("anonymous-owned")
        else:
            pytest.fail(f"unexpected command: {args}")
        return ""

    monkeypatch.setattr(inspector, "inventory", inventory)
    monkeypatch.setattr(inspector, "_docker", docker)
    cleanup = inspector.cleanup(project)
    assert cleanup["verified"] is (
        automatic_volume_removal or not volume_survives_removal
    )
    assert "unrelated-volume" in volumes
    assert "anonymous-owned" in cleanup["before"]["volumes"]


def test_static_and_skip_build_do_not_start_trials(task, boundary):
    static = run_validation(task, max_level=Level.STATIC)
    assert result(static, "static.manifest").status is CheckStatus.PASS
    report = run_validation(task, skip_build=True)
    assert result(report, "runtime.startup").status is CheckStatus.SKIP
    assert not boundary.trials


@pytest.mark.parametrize("timeout_phase", [None, "build", "agent"])
def test_nop_build_has_a_separate_supervisor_budget(
    task, boundary, monkeypatch, tmp_path, timeout_phase
):
    from openenv.validation.providers import docker

    path = task / "validation/openenv.json"
    contract = json.loads(path.read_text())
    contract["resources"]["episode_timeout_s"] = 0.1
    path.write_text(json.dumps(contract))
    build_budget = 0.02 if timeout_phase == "build" else 0.5
    monkeypatch.setattr(
        docker,
        "DockerValidationProvider",
        lambda: SimpleNamespace(build_timeout_s=build_budget),
    )
    boundary.start_delay = 0.2
    boundary.agent_delay = 0.2 if timeout_phase == "agent" else 0
    artifacts = tmp_path / "build-budget"
    report = run_validation(task, artifacts_dir=artifacts)
    startup = result(report, "runtime.startup")
    assert startup.status is (
        CheckStatus.ERROR
        if timeout_phase == "build"
        else CheckStatus.FAIL
        if timeout_phase == "agent"
        else CheckStatus.PASS
    )
    if timeout_phase:
        assert len(boundary.trials) == 1
        phase = "build/start" if timeout_phase == "build" else "agent"
        assert any(phase in reason for reason in startup.evidence)
        assert (
            result(report, "runtime.episode_determinism").status is not CheckStatus.PASS
        )
    else:
        assert result(report, "semantic.oracle_max").status is CheckStatus.PASS
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert native["cleanup"]["verified"]
    assert native["trials"][0]["build_timeout_s"] == build_budget


@pytest.mark.parametrize("timeout_source", ["supervisor", "harbor", "unexpected"])
def test_validator_build_start_timeout_is_an_error(
    task, boundary, monkeypatch, timeout_source
):
    from openenv.validation.providers import docker

    monkeypatch.setattr(
        docker,
        "DockerValidationProvider",
        lambda: SimpleNamespace(build_timeout_s=0.01),
    )
    if timeout_source == "supervisor":
        boundary.start_delay = 0.1
    elif timeout_source == "harbor":
        boundary.exception_type = "EnvironmentStartTimeoutError"
    else:
        boundary.timeout = True
    report = run_validation(task)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    failures = startup.measured["failures"]
    if timeout_source == "unexpected":
        assert failures == ["unexpected Harbor timeout during agent"]
    else:
        assert failures == ["validator build/start exceeded its 0.01-second budget"]
        assert failures[0] in startup.evidence
    assert len(boundary.trials) == 1


def test_real_harbor_start_uses_build_budget_then_episode_budgets(
    task, boundary, monkeypatch
):
    timeouts = []
    wait_for = asyncio.wait_for

    async def record_timeout(awaitable, timeout):
        timeouts.append(timeout)
        return await wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", record_timeout)
    report = run_validation(task)
    assert result(report, "runtime.startup").status is CheckStatus.PASS
    # Harbor starts each environment, then runs its agent under a separate budget.
    assert timeouts == [600, 10, 10, 10, 10, 10]
    assert [trial.role for trial in boundary.trials] == [
        "nop",
        "oracle",
        "oracle-replay",
    ]


def test_missing_harbor_build_timeout_binding_is_a_startup_error(
    task, boundary, monkeypatch
):
    create = harbor._create_trial

    async def missing_binding(config):
        trial = await create(config)
        del trial._environment_build_timeout_sec
        return trial

    monkeypatch.setattr(harbor, "_create_trial", missing_binding)
    report = run_validation(task)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert startup.measured["failures"] == ["AttributeError"]
    assert "retained_image_tag" not in startup.measured
    assert len(boundary.trials) == 1


def test_harbor_report_preserves_resolved_space_target(task, boundary, tmp_path):
    target = "spaces/openenv/example@" + "d" * 40
    artifacts = tmp_path / "space-report"
    report = run_validation(task, report_target=target, artifacts_dir=artifacts)
    assert report.target == target
    assert json.loads((artifacts / "report.json").read_text())["target"] == target


def test_cancelled_compose_keeps_daemon_cleanup_unverified(task, boundary, tmp_path):
    boundary.pending_compose = True
    artifacts = tmp_path / "pending-compose"
    report = run_validation(task, artifacts_dir=artifacts)
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    cleanup = native["trials"][0]["cleanup"]
    assert not cleanup["verified"]
    assert not any(cleanup["after"].values())
    (pending,) = cleanup["pending_operations"]
    assert pending["command"] == "build"
    assert pending["project"] == boundary.trials[0].agent_environment.session_id
    assert pending["client_reaped"] and not pending["daemon_completion_verified"]


def test_separate_verifier_is_named_unsupported(task, boundary):
    (task / "task.toml").write_text('[verifier]\nenvironment_mode="separate"\n')
    report = run_validation(task)
    assert result(report, "runtime.startup").status is CheckStatus.SKIP
    assert "separate" in result(report, "runtime.startup").evidence[0]
    assert not boundary.trials


def test_unpinned_task_image_is_refused_before_trials(task, boundary):
    with (task / "task.toml").open("a") as stream:
        stream.write('[environment]\ndocker_image="registry.example/image:latest"\n')
    report = run_validation(task)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.SKIP
    assert "digest pin" in startup.evidence[0]
    assert not boundary.trials


def test_digest_pinned_task_image_runs_without_a_dockerfile(task, boundary, tmp_path):
    pinned = "registry.example/image@sha256:" + "d" * 64
    (task / "environment/Dockerfile").unlink()
    with (task / "task.toml").open("a") as stream:
        stream.write(f'[environment]\ndocker_image="{pinned}"\n')
    artifacts = tmp_path / "pinned"
    report = run_validation(task, artifacts_dir=artifacts)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.PASS
    assert startup.measured["source_image"] == pinned
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert native["source_image"] == pinned
    # The do-nothing trial pulls the pin; later trials run the image it retained.
    images = [trial.docker_image for trial in boundary.trials]
    assert images == [pinned, *[startup.measured["retained_image_tag"]] * 2]


def test_explicit_contract_and_valid_toml_required(task, boundary):
    (task / "validation/openenv.json").unlink()
    with pytest.raises(ManifestError):
        read_harbor_contract(task)
    assert run_validation(task).verdict is Verdict.FAIL
    assert not boundary.trials


@pytest.mark.parametrize(
    "field,token",
    [
        # As a float this margin is 0.1, so a measured gap of 0.1 would pass.
        ("floor_margin", "0.100000000000000001"),
        ("oracle_tolerance", "1e-999"),
        ("range", "[0, 1.0000000000000001]"),
    ],
)
def test_contract_numbers_must_keep_their_exact_value(task, boundary, field, token):
    path = task / "validation/openenv.json"
    contract = json.loads(path.read_text())
    contract["reward"][field] = "TOKEN"
    path.write_text(json.dumps(contract).replace('"TOKEN"', token))
    boundary.rewards["nop"] = b'{"reward":0.9}'
    with pytest.raises(ManifestError):
        read_harbor_contract(task)
    report = run_validation(task)
    assert report.verdict is Verdict.FAIL
    assert result(report, "static.manifest").status is CheckStatus.FAIL
    assert not boundary.trials


def test_malformed_toml_is_manifest_failure(task, boundary):
    (task / "task.toml").write_text("invalid = [")
    report = run_validation(task)
    assert result(report, "static.manifest").status is CheckStatus.FAIL
    assert not boundary.trials


def test_replay_divergence_fails_determinism_and_skips_semantics(task, boundary):
    boundary.rewards["oracle-replay"] = b'{"reward":0.5}'
    report = run_validation(task)
    assert report.verdict is Verdict.FAIL
    assert result(report, "runtime.episode_determinism").status is CheckStatus.FAIL
    for check_id in ("semantic.oracle_max", "semantic.floor_gap"):
        skipped = result(report, check_id)
        assert skipped.status is CheckStatus.SKIP
        assert skipped.evidence == ["unmet dependencies: runtime.episode_determinism"]


@pytest.fixture
def canaries(task):
    directory = task / "validation/canaries"
    directory.mkdir()
    (directory / "a-empty.sh").write_bytes(b"#!/bin/sh\ntrue\n")
    (directory / "b-hardcoded.sh").write_bytes(b"#!/bin/sh\necho 42 > /app/out\n")
    contract = json.loads((task / "validation/openenv.json").read_text())
    contract["canaries"] = "validation/canaries"
    (task / "validation/openenv.json").write_text(json.dumps(contract))
    return directory


def test_canaries_replace_the_solution_and_must_stay_at_the_floor(
    task, boundary, canaries, tmp_path
):
    boundary.rewards.update({"canary-1": b'{"reward":0}', "canary-2": b'{"reward":1}'})
    artifacts = tmp_path / "canaries"
    report = run_validation(task, artifacts_dir=artifacts)
    assert report.verdict is Verdict.FAIL
    canary = result(report, "semantic.canary_floor")
    assert canary.status is CheckStatus.FAIL
    assert canary.evidence == [
        "canary validation/canaries/b-hardcoded.sh earned 1.0, "
        "above the do-nothing reward 0.0"
    ]
    by_role = {trial.role: trial for trial in boundary.trials}
    assert list(by_role) == ["nop", "oracle", "oracle-replay", "canary-1", "canary-2"]
    assert by_role["oracle"].solve == (task / "solution/solve.sh").read_bytes()
    assert by_role["canary-1"].solve == (canaries / "a-empty.sh").read_bytes()
    assert by_role["canary-2"].solve == (canaries / "b-hardcoded.sh").read_bytes()
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    overlay = native["trials"][4]["task_overlay"]["solution/solve.sh"]
    assert overlay["canary"] == "validation/canaries/b-hardcoded.sh"


def test_canaries_at_the_floor_pass(task, boundary, canaries):
    boundary.rewards.update({"canary-1": b'{"reward":0}', "canary-2": b'{"reward":0}'})
    report = run_validation(task)
    assert result(report, "semantic.canary_floor").status is CheckStatus.PASS


@pytest.mark.parametrize("floor,oracle", [(1, 1), (0, 0.8)])
def test_canary_cleanup_failure_preserves_completed_semantics(
    task, boundary, canaries, tmp_path, floor, oracle
):
    boundary.rewards["nop"] = json.dumps({"reward": floor}).encode()
    boundary.rewards["oracle"] = json.dumps({"reward": oracle}).encode()
    boundary.cleanup_fail_role = "canary-2"
    artifacts = tmp_path / "partial-semantics"
    report = run_validation(task, artifacts_dir=artifacts)
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    assert result(report, "runtime.reward_well_formed").status is CheckStatus.SKIP
    assert result(report, "runtime.episode_determinism").status is CheckStatus.PASS
    assert result(report, "semantic.floor_gap").status is (
        CheckStatus.FAIL if floor else CheckStatus.PASS
    )
    assert result(report, "semantic.oracle_max").status is (
        CheckStatus.PASS if oracle == 1 else CheckStatus.FAIL
    )
    assert result(report, "semantic.canary_floor").status is CheckStatus.SKIP
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert len(native["trials"]) == 5
    assert not native["cleanup"]["verified"]
    assert native["floor_reward"] == floor


@pytest.mark.parametrize("problem", ["error", "collision"])
def test_preallocation_inventory_failure_preserves_completed_trials(
    task, boundary, canaries, tmp_path, problem
):
    boundary.inventory_problem = problem
    boundary.rewards["nop"] = b'{"reward":1}'
    artifacts = tmp_path / "inventory-error"
    report = run_validation(task, artifacts_dir=artifacts)
    assert result(report, "runtime.startup").status is CheckStatus.ERROR
    assert result(report, "runtime.episode_determinism").status is CheckStatus.PASS
    assert result(report, "semantic.floor_gap").status is CheckStatus.FAIL
    native = json.loads((artifacts / "harbor-evidence.json").read_text())
    assert len(native["trials"]) == 4
    assert len(boundary.trials) == len(boundary.cleanup_calls) == 3
    assert all(trial["cleanup"]["verified"] for trial in native["trials"][:3])
    assert not native["trials"][-1]["cleanup"]["verified"]
    assert (
        artifacts / "harbor-trials/oracle/reward.json"
    ).read_bytes() == b'{"reward":1}'


@pytest.mark.parametrize("safe", [False, True])
def test_collection_failure_keeps_only_safe_native_reasons(
    task, boundary, tmp_path, safe
):
    reason = "Harbor requires a Linux Docker daemon" if safe else "token=do-not-persist"
    boundary.preflight_error = (
        harbor.NativeEvidenceError(reason) if safe else RuntimeError(reason)
    )
    artifacts = tmp_path / "preflight"
    report = run_validation(task, artifacts_dir=artifacts)
    startup = result(report, "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert (reason in startup.evidence[0]) is safe
    native = (artifacts / "harbor-evidence.json").read_text()
    assert (reason in native) is safe


def test_canaries_run_only_at_the_semantic_level(task, boundary, canaries):
    report = run_validation(task, max_level=Level.RUNTIME)
    assert [trial.role for trial in boundary.trials] == [
        "nop",
        "oracle",
        "oracle-replay",
    ]
    assert result(report, "runtime.episode_determinism").status is CheckStatus.PASS
    assert not any(check.check_id.startswith("semantic.") for check in report.results)


@pytest.mark.parametrize("declared", ["../outside", "/abs", "validation/missing"])
def test_invalid_canaries_declaration_is_manifest_failure(task, boundary, declared):
    contract = json.loads((task / "validation/openenv.json").read_text())
    contract["canaries"] = declared
    (task / "validation/openenv.json").write_text(json.dumps(contract))
    report = run_validation(task)
    assert result(report, "static.manifest").status is CheckStatus.FAIL
    assert not boundary.trials


def test_canaries_directory_holds_only_scripts(task, boundary, canaries):
    (canaries / "notes.txt").write_text("not a canary")
    report = run_validation(task)
    assert result(report, "static.manifest").status is CheckStatus.FAIL
    assert not boundary.trials
