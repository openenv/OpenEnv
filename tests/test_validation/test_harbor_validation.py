"""Native Harbor checks mock only the Trial boundary and trusted host inspector."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from openenv.validation import harbor
from openenv.validation.manifest import ManifestError
from openenv.validation.parsers.harbor_toml import (
    HarborTaskParser,
    read_harbor_contract,
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
        mutate=False,
        private_mutate=False,
        image_drift=False,
        config_drift=False,
        oracle_exit=None,
        exception=False,
        no_identity=False,
        timeout=False,
        retained=True,
        overwrite_exit=False,
        excluded_copied=[],
    )

    class Inspector:
        def preflight(self):
            return {"os": "linux", "daemon_id": "mock-host"}

        def inventory(self, project):
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
            return {
                "verified": state.cleanup_ok,
                "after": {
                    "containers": [] if state.cleanup_ok else ["leftover"],
                    "networks": [],
                    "volumes": [],
                },
            }

        def verify_image(self, tag, image_ref):
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
                session_id=config.trial_name + "__env", stop=self.stop
            )
            self.hooks = {}
            state.excluded_copied.append(
                (config.task.path / "environment/.worktrees").exists()
            )
            state.trials.append(self)

        async def stop(self, *, delete):
            assert not delete

        def add_hook(self, event, callback):
            self.hooks.setdefault(event.value, []).append(callback)

        async def run(self):
            if not state.no_identity:
                for hook in self.hooks.get("agent-start", []):
                    await hook(None)
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
            try:
                rewards = json.loads(raw) if raw else None
            except ValueError:
                rewards = None
            return SimpleNamespace(
                exception_info={"exception_type": "TestError"}
                if state.exception
                else None,
                verifier_result=SimpleNamespace(rewards=rewards),
            )

    async def create(config):
        return Trial(config)

    monkeypatch.setattr(harbor, "DockerInspector", Inspector)
    monkeypatch.setattr(harbor, "_create_trial", create)
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


@pytest.mark.parametrize(
    "raw",
    [
        None,
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


def test_cleanup_failure_blocks(task, boundary, tmp_path):
    boundary.cleanup_ok = False
    artifacts = tmp_path / "cleanup"
    report = run_validation(task, artifacts_dir=artifacts)
    assert report.verdict is Verdict.FAIL
    assert not json.loads((artifacts / "harbor-evidence.json").read_text())["cleanup"][
        "verified"
    ]


def test_temp_cleanup_failure_does_not_discard_successful_trials(
    task, boundary, monkeypatch
):
    temporary_directory = harbor.tempfile.TemporaryDirectory

    class CleanupSensitiveTemporaryDirectory:
        def __init__(self, *args, ignore_cleanup_errors=False, **kwargs):
            self.directory = temporary_directory(*args, **kwargs)
            self.ignore_cleanup_errors = ignore_cleanup_errors

        def __enter__(self):
            return self.directory.__enter__()

        def __exit__(self, *args):
            self.directory.__exit__(*args)
            if not self.ignore_cleanup_errors:
                raise PermissionError("container-owned trial file")

    monkeypatch.setattr(
        harbor.tempfile, "TemporaryDirectory", CleanupSensitiveTemporaryDirectory
    )
    report = run_validation(task)
    assert result(report, "runtime.startup").status is CheckStatus.PASS
    assert result(report, "semantic.oracle_max").status is CheckStatus.PASS


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


def test_missing_oracle_fails_oracle_max_without_starting_trial(task, boundary):
    (task / "solution/solve.sh").unlink()
    report = run_validation(task)
    assert report.verdict is Verdict.FAIL
    assert report.levels_run == [Level.STATIC, Level.SEMANTIC]
    # RFC 008: no oracle is a graded FAIL, not a SKIP.
    assert result(report, "semantic.oracle_max").status is CheckStatus.FAIL
    assert result(report, "runtime.startup").status is CheckStatus.SKIP
    assert not boundary.trials


def test_retained_image_drift_blocks(task, boundary):
    boundary.retained = False
    assert run_validation(task).verdict is Verdict.FAIL


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
    assert commands[-3:] == [
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
        return {"containers": list(containers), "networks": [], "volumes": []}

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


def test_separate_verifier_is_named_unsupported(task, boundary):
    (task / "task.toml").write_text('[verifier]\nenvironment_mode="separate"\n')
    report = run_validation(task)
    assert result(report, "runtime.startup").status is CheckStatus.SKIP
    assert "separate" in result(report, "runtime.startup").evidence[0]
    assert not boundary.trials


def test_explicit_contract_and_valid_toml_required(task, boundary):
    (task / "validation/openenv.json").unlink()
    with pytest.raises(ManifestError):
        read_harbor_contract(task)
    assert run_validation(task).verdict is Verdict.FAIL
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
