# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Harbor environment.

Everything here is hermetic: no Docker, no network, no API keys. The bundled
`fix-sum-bug` example task doubles as the end-to-end fixture, so these tests also
guard the claim that a real Harbor task directory runs unmodified.
"""

from __future__ import annotations

import io
import json
import tarfile
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
from harbor_env import (
    AGENT_ACTIONS,
    CONTROL_ACTIONS,
    HarborAction,
    HarborEnv,
    HarborObservation,
    HarborState,
)
from harbor_env.server import (
    DockerSandbox,
    ExecResult,
    HarborEnvironment,
    HarborTask,
    LocalSandbox,
    Sandbox,
    SandboxError,
    TaskCatalog,
    TaskFormatError,
)
from harbor_env.server.reward import read_reward
from harbor_env.server.sandbox import (
    _container_limits,
    expand_env_refs,
    resolve_within,
    SandboxPaths,
)
from harbor_env.server.task import (
    BUNDLED_TASKS_DIR,
    NetworkPolicy,
    resolve_task_source,
    ResourceLimits,
)
from openenv.core.env_server.serialization import serialize_observation


EXAMPLE_TASK_ID = "fix-sum-bug"

#: The bundled task's grader runs five checks; the shipped buggy `stats.py`
#: passes three of them.
BUGGY_REWARD = 3 / 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_task(
    root: Path,
    task_id: str,
    *,
    config: str | None = None,
    instruction: str = "do the thing",
    test_script: str | None = None,
    solve_script: str | None = None,
    seed_files: dict[str, str] | None = None,
    dockerfile: str | None = None,
) -> Path:
    """Materialize a minimal Harbor task directory for a test."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("task.toml").write_text(
        config
        if config is not None
        else textwrap.dedent(
            f"""
            schema_version = "1.4"

            [task]
            name = "tests/{task_id}"
            description = "fixture"
            """
        ).strip(),
        encoding="utf-8",
    )
    task_dir.joinpath("instruction.md").write_text(instruction, encoding="utf-8")

    if test_script is not None:
        tests = task_dir / "tests"
        tests.mkdir(exist_ok=True)
        tests.joinpath("test.sh").write_text(test_script, encoding="utf-8")
    if solve_script is not None:
        solution = task_dir / "solution"
        solution.mkdir(exist_ok=True)
        solution.joinpath("solve.sh").write_text(solve_script, encoding="utf-8")
    for name, content in (seed_files or {}).items():
        target = task_dir / "environment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if dockerfile is not None:
        environment = task_dir / "environment"
        environment.mkdir(exist_ok=True)
        environment.joinpath("Dockerfile").write_text(dockerfile, encoding="utf-8")
    return task_dir


def reader(files: dict[str, str]):
    """Build the `read_text` callable that `read_reward` expects."""
    return lambda name: files.get(name)


# ---------------------------------------------------------------------------
# task.toml parsing
# ---------------------------------------------------------------------------


def test_load_reads_harbor_schema(tmp_path: Path) -> None:
    task_dir = write_task(
        tmp_path,
        "demo",
        config=textwrap.dedent(
            """
            schema_version = "1.4"

            [task]
            name = "acme/demo"
            description = "a demo"

            [metadata]
            difficulty = "easy"

            [environment]
            docker_image = "python:3.12-slim"

            [agent]
            timeout_sec = 60.0

            [verifier]
            timeout_sec = 30.0
            """
        ).strip(),
        instruction="# Demo\n",
    )

    task = HarborTask.load(task_dir)

    assert task.name == "acme/demo"
    assert task.description == "a demo"
    assert task.schema_version == "1.4"
    assert task.instruction == "# Demo\n"
    assert task.metadata["difficulty"] == "easy"
    assert task.docker_image == "python:3.12-slim"
    assert task.agent_timeout_s == 60.0
    assert task.verifier_timeout_s == 30.0


def test_load_reads_repo2rlenv_version_key(tmp_path: Path) -> None:
    """Repo2RLEnv writes `version` where Harbor writes `schema_version`."""
    task_dir = write_task(
        tmp_path,
        "pallets__click-2951",
        config=textwrap.dedent(
            """
            version = "1.0"

            [task]
            name = "acme/pallets__click-2951"
            description = "Fix the thing"

            [metadata.repo2env]
            pipeline = "pr_diff"
            reward_kinds = ["diff_similarity"]
            """
        ).strip(),
    )

    task = HarborTask.load(task_dir)

    assert task.schema_version == "1.0"
    assert task.metadata["repo2env"]["pipeline"] == "pr_diff"
    # Absent timeout tables fall back to Harbor's documented defaults.
    assert task.verifier_timeout_s == 300.0
    assert task.agent_timeout_s == 1800.0


def test_load_rejects_non_task_directory(tmp_path: Path) -> None:
    with pytest.raises(TaskFormatError, match="no task.toml"):
        HarborTask.load(tmp_path)


def test_load_rejects_windows_tasks(tmp_path: Path) -> None:
    task_dir = write_task(
        tmp_path,
        "win",
        config='schema_version = "1.4"\n\n[environment]\nos = "windows"\n',
    )
    with pytest.raises(TaskFormatError, match="only supports Linux"):
        HarborTask.load(task_dir)


def test_instruction_falls_back_to_description(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "terse")
    (task_dir / "instruction.md").unlink()
    assert HarborTask.load(task_dir).instruction == "fixture"


# ---------------------------------------------------------------------------
# self-contained vs image-backed tasks
# ---------------------------------------------------------------------------


def test_seed_files_make_a_task_self_contained(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "seeded", seed_files={"app.py": "x = 1\n"})
    task = HarborTask.load(task_dir)

    assert [p.name for p in task.seed_files] == ["app.py"]
    assert task.needs_image is False


def test_dockerfile_suppresses_seed_upload(tmp_path: Path) -> None:
    """Harbor only uploads environment/ when it holds no build recipe."""
    task_dir = write_task(
        tmp_path,
        "built",
        seed_files={"app.py": "x = 1\n"},
        dockerfile="FROM python:3.12-slim\n",
    )
    task = HarborTask.load(task_dir)

    assert task.seed_files == ()
    assert task.needs_image is True


def test_docker_image_alone_needs_an_image(tmp_path: Path) -> None:
    task_dir = write_task(
        tmp_path,
        "prebuilt",
        config='schema_version = "1.4"\n\n[environment]\ndocker_image = "ghcr.io/acme/x:1"\n',
    )
    assert HarborTask.load(task_dir).needs_image is True


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_catalog_finds_flat_and_nested_tasks(tmp_path: Path) -> None:
    write_task(tmp_path, "alpha")
    write_task(tmp_path / "tasks", "beta")  # the layout `repo2rlenv push` writes

    catalog = TaskCatalog(tmp_path)

    assert catalog.task_ids() == ["alpha", "tasks/beta"]
    assert catalog.get("tasks/beta").task_id == "tasks/beta"
    # A bare directory name resolves when it is unambiguous.
    assert catalog.get("beta").task_id == "tasks/beta"


def test_catalog_accepts_a_single_task_directory(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "solo")
    assert TaskCatalog(task_dir).task_ids() == ["solo"]


def test_catalog_reports_ambiguous_and_unknown_ids(tmp_path: Path) -> None:
    write_task(tmp_path / "a", "dup")
    write_task(tmp_path / "b", "dup")
    catalog = TaskCatalog(tmp_path)

    with pytest.raises(KeyError, match="ambiguous"):
        catalog.get("dup")
    with pytest.raises(KeyError, match="unknown task"):
        catalog.get("nope")


def test_catalog_caches_its_scan_until_refreshed(tmp_path: Path) -> None:
    write_task(tmp_path, "first")
    catalog = TaskCatalog(tmp_path)
    assert catalog.task_ids() == ["first"]

    write_task(tmp_path, "second")
    assert catalog.task_ids() == ["first"]  # served task sets are fixed

    catalog.refresh()
    assert catalog.task_ids() == ["first", "second"]


def test_resolve_task_source_defaults_to_bundled_tasks() -> None:
    assert resolve_task_source(None) == BUNDLED_TASKS_DIR
    assert resolve_task_source("  ") == BUNDLED_TASKS_DIR


def test_resolve_task_source_rejects_nonsense() -> None:
    with pytest.raises(FileNotFoundError, match="neither an existing directory"):
        resolve_task_source("./definitely/not/here")


# ---------------------------------------------------------------------------
# reward contract
# ---------------------------------------------------------------------------


def test_reward_json_takes_precedence_over_reward_txt() -> None:
    report = read_reward(
        reader(
            {"reward.json": '{"reward": 0.75, "f2p_rate": 0.5}', "reward.txt": "0.1"}
        )
    )

    assert report.value == 0.75
    assert report.source == "reward.json"
    assert report.metrics == {"reward": 0.75, "f2p_rate": 0.5}
    assert report.graded is True


def test_reward_json_with_a_single_metric_is_unambiguous() -> None:
    report = read_reward(reader({"reward.json": '{"accuracy": 0.5}'}))
    assert report.value == 0.5
    assert report.source == "reward.json"


def test_reward_json_without_a_primary_metric_falls_back_to_txt() -> None:
    report = read_reward(
        reader(
            {
                "reward.json": '{"accuracy": 0.5, "runtime_sec": 2.0}',
                "reward.txt": "0.9",
            }
        )
    )

    assert report.value == 0.9
    assert report.source == "reward.txt"
    assert report.metrics == {"accuracy": 0.5, "runtime_sec": 2.0}
    assert any("no 'reward' key" in e for e in report.errors)


def test_reward_txt_alone_is_honoured() -> None:
    report = read_reward(reader({"reward.txt": "1\n"}))
    assert report.value == 1.0
    assert report.source == "reward.txt"


def test_missing_reward_files_are_not_synthesized_into_zero() -> None:
    report = read_reward(reader({}))

    assert report.value is None
    assert report.graded is False
    assert report.source == "missing"


def test_malformed_reward_files_are_reported_not_guessed() -> None:
    report = read_reward(reader({"reward.json": "{oops", "reward.txt": "not-a-number"}))

    assert report.value is None
    assert len(report.errors) == 2


def test_repo2rlenv_details_sidecar_is_surfaced_but_never_scored() -> None:
    details = {"reward": 0.83, "f2p_passed": 2, "components": {"similarity": 0.4}}
    report = read_reward(
        reader({"reward-details.json": json.dumps(details), "reward.txt": "0.83"})
    )

    assert report.value == 0.83
    assert report.source == "reward.txt"
    assert report.details["components"]["similarity"] == 0.4


# ---------------------------------------------------------------------------
# sandbox primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative", ["a.py", "pkg/mod.py", "./pkg/../a.py"])
def test_resolve_within_accepts_paths_inside_the_workdir(relative: str) -> None:
    assert resolve_within("/work", relative).startswith("/work")


@pytest.mark.parametrize(
    "relative", ["../tests/grade.py", "../../etc/passwd", "/etc/passwd", "a/../../b"]
)
def test_resolve_within_rejects_escapes(relative: str) -> None:
    with pytest.raises(ValueError):
        resolve_within("/work", relative)


@pytest.mark.parametrize("relative", ["", "   ", ".", "a/.."])
def test_resolve_within_rejects_paths_naming_the_workdir(relative: str) -> None:
    """An empty path must not resolve to the working directory itself.

    `read` would try to cat a directory and report a confusing "no such file",
    but `write` is worse: the target splits into ("/", "workspace"), so the
    upload would drop a regular file over the whole checkout.
    """
    with pytest.raises(ValueError):
        resolve_within("/work", relative)


def test_empty_paths_are_rejected_as_actions(env: HarborEnvironment) -> None:
    """The action layer surfaces the rejection instead of a backend error."""
    env.reset(task_id=EXAMPLE_TASK_ID)

    for action in (
        HarborAction(action_type="read", path=""),
        HarborAction(action_type="write", path="", content="x"),
    ):
        observation = env.step(action)
        assert not observation.success
        assert "path" in observation.error

    # The working directory survived the write attempt.
    listing = env.step(HarborAction(action_type="exec", command="ls"))
    assert listing.output.split() == ["stats.py"]


def test_docker_upload_dir_archives_each_entry_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rglob` yields directories too, and `tarfile.add` recurses by default.

    Left alone, every nested file is archived once through its parent directory
    and again on its own iteration.
    """
    source = tmp_path / "tests"
    (source / "nested").mkdir(parents=True)
    (source / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (source / "nested" / "grade.py").write_text("x = 1\n", encoding="utf-8")

    captured: dict[str, bytes] = {}

    class _FakeContainer:
        def put_archive(self, path: str, data: bytes) -> bool:
            captured["data"] = data
            return True

    sandbox = DockerSandbox()
    monkeypatch.setattr(sandbox, "_require_container", lambda: _FakeContainer())
    monkeypatch.setattr(sandbox, "mkdirs", lambda *args, **kwargs: None)

    sandbox.upload_dir(source, "/tests")

    with tarfile.open(fileobj=io.BytesIO(captured["data"])) as archive:
        names = archive.getnames()

    assert sorted(names) == ["nested", "nested/grade.py", "test.sh"]
    assert len(names) == len(set(names))


def test_action_types_partition_into_agent_and_control() -> None:
    """The two sets must stay a partition of what the server actually handles.

    They are what documents which actions belong in a policy's action space, so
    a new action type that nobody classified would silently read as
    agent-reachable.
    """
    handled = set(
        HarborEnvironment(tasks=str(BUNDLED_TASKS_DIR), mode="local")._handlers
    )

    assert AGENT_ACTIONS | CONTROL_ACTIONS == handled
    assert not (AGENT_ACTIONS & CONTROL_ACTIONS)
    # Everything classified is also accepted on the wire, and nothing else is.
    for action_type in handled:
        HarborAction(action_type=action_type)
    with pytest.raises(ValueError):
        HarborAction(action_type="reset")


def test_sandbox_paths_expose_harbor_layout_as_env() -> None:
    env = SandboxPaths(workdir="/workspace").as_env()

    assert env["HARBOR_WORKDIR"] == "/workspace"
    assert env["HARBOR_TESTS_DIR"] == "/tests"
    assert env["HARBOR_LOGS_DIR"] == "/logs/verifier"


def test_expand_env_refs_resolves_and_blanks_missing_values(monkeypatch) -> None:
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "s3cret")
    monkeypatch.delenv("HARBOR_TEST_ABSENT", raising=False)

    resolved = expand_env_refs(
        {"A": "${HARBOR_TEST_TOKEN}", "B": "$HARBOR_TEST_ABSENT"}
    )

    assert resolved == {"A": "s3cret", "B": ""}


def test_local_sandbox_refuses_image_backed_tasks(tmp_path: Path) -> None:
    task = HarborTask.load(
        write_task(tmp_path, "needs-image", dockerfile="FROM python:3.12-slim\n")
    )
    sandbox = LocalSandbox()
    try:
        with pytest.raises(SandboxError, match="HARBOR_MODE=docker"):
            sandbox.start(task)
    finally:
        sandbox.close()


def test_local_sandbox_cleans_up_its_root(tmp_path: Path) -> None:
    task = HarborTask.load(
        write_task(tmp_path, "seeded", seed_files={"a.py": "x = 1\n"})
    )
    sandbox = LocalSandbox()
    sandbox.start(task)
    root = sandbox.root

    assert (Path(sandbox.paths.workdir) / "a.py").is_file()
    sandbox.close()
    assert not root.exists()


# ---------------------------------------------------------------------------
# environment: the bundled example task, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    environment = HarborEnvironment(mode="local")
    yield environment
    environment.close()


def test_reset_serves_the_task_instruction(env: HarborEnvironment) -> None:
    obs = env.reset(task_id=EXAMPLE_TASK_ID)

    assert obs.task_id == EXAMPLE_TASK_ID
    assert obs.task_name == "openenv/fix-sum-bug"
    assert obs.mode == "local"
    assert "stats.py" in obs.instruction
    assert obs.done is False
    assert env.state.available_tasks == [EXAMPLE_TASK_ID]


def test_seed_files_land_in_the_working_directory(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    obs = env.step(HarborAction(action_type="read", path="stats.py"))

    assert obs.success is True
    assert "def total(values)" in obs.output


def test_verifier_reward_is_forwarded_verbatim(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    obs = env.step(HarborAction(action_type="evaluate"))

    assert obs.reward == pytest.approx(BUGGY_REWARD)
    assert obs.done is True
    assert obs.info["reward_source"] == "reward.json"
    assert obs.info["reward_metrics"]["total_empty"] == 0.0
    assert env.state.evaluated is True
    assert env.state.reward == pytest.approx(BUGGY_REWARD)


def test_agent_edits_change_the_reward(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    fixed = textwrap.dedent(
        """
        def total(values):
            return sum(values)


        def mean(values):
            if not values:
                return 0.0
            return total(values) / len(values)
        """
    ).strip()
    write = env.step(HarborAction(action_type="write", path="stats.py", content=fixed))
    assert write.success is True

    assert env.step(HarborAction(action_type="evaluate")).reward == pytest.approx(1.0)


def test_oracle_solution_earns_full_reward(env: HarborEnvironment) -> None:
    """`solve` is how a task set is validated: the oracle must score 1.0."""
    env.reset(task_id=EXAMPLE_TASK_ID)
    solved = env.step(HarborAction(action_type="solve"))
    assert solved.success is True

    assert env.step(HarborAction(action_type="evaluate")).reward == pytest.approx(1.0)


def test_exec_reports_failures_without_ending_the_episode(
    env: HarborEnvironment,
) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    obs = env.step(HarborAction(action_type="exec", command="exit 3"))

    assert obs.success is False
    assert obs.exit_code == 3
    assert obs.done is False
    assert obs.reward is None


def test_agent_cannot_reach_the_verifier(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)

    escape = env.step(HarborAction(action_type="read", path="../tests/grade.py"))
    assert escape.success is False
    assert "escapes the working directory" in escape.error

    # tests/ is staged only when the verifier runs, so it is not there yet.
    listing = env.step(HarborAction(action_type="exec", command="ls"))
    assert listing.output.split() == ["stats.py"]


def test_planted_reward_files_are_discarded(env: HarborEnvironment) -> None:
    """A reward the agent writes itself must not survive into the score.

    The path is taken from the sandbox rather than `$HARBOR_LOGS_DIR`, which is
    deliberately hidden from agent commands — reading it from the environment
    would make the plant fail and the test pass for the wrong reason.
    """
    env.reset(task_id=EXAMPLE_TASK_ID)
    logs_dir = env._sandbox.paths.logs_verifier
    env.step(
        HarborAction(
            action_type="exec",
            command=(
                f"mkdir -p {logs_dir!r} && "
                f"echo '{{\"reward\": 1.0}}' > {logs_dir!r}/reward.json"
            ),
        )
    )
    planted = Path(logs_dir) / "reward.json"
    assert planted.is_file(), "the plant itself failed; the test would be vacuous"

    assert env.step(HarborAction(action_type="evaluate")).reward == pytest.approx(
        BUGGY_REWARD
    )


def test_unscorable_episode_reports_no_reward(tmp_path: Path) -> None:
    """A verifier that writes nothing must not be turned into a 0.0."""
    write_task(
        tmp_path,
        "silent",
        seed_files={"noop.txt": "hi\n"},
        test_script="#!/bin/bash\necho 'graded nothing'\n",
    )
    environment = HarborEnvironment(tasks=str(tmp_path), mode="local")
    try:
        environment.reset(task_id="silent")
        obs = environment.step(HarborAction(action_type="evaluate"))
    finally:
        environment.close()

    assert obs.reward is None
    assert obs.done is True
    assert obs.success is False
    assert "wrote no reward file" in obs.error
    assert obs.info["reward_source"] == "missing"


def test_missing_verifier_is_reported(tmp_path: Path) -> None:
    write_task(tmp_path, "ungraded", seed_files={"a.txt": "x\n"})
    environment = HarborEnvironment(tasks=str(tmp_path), mode="local")
    try:
        environment.reset(task_id="ungraded")
        obs = environment.step(HarborAction(action_type="evaluate"))
    finally:
        environment.close()

    assert obs.success is False
    assert "no tests/test.sh" in obs.error


def test_absolute_path_verifier_gets_an_actionable_hint(tmp_path: Path) -> None:
    write_task(
        tmp_path,
        "hardcoded",
        seed_files={"a.txt": "x\n"},
        test_script="#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n",
    )
    environment = HarborEnvironment(tasks=str(tmp_path), mode="local")
    try:
        environment.reset(task_id="hardcoded")
        obs = environment.step(HarborAction(action_type="evaluate"))
    finally:
        environment.close()

    assert obs.reward is None
    assert "HARBOR_MODE=docker" in obs.error


def test_step_before_reset_is_reported(env: HarborEnvironment) -> None:
    obs = env.step(HarborAction(action_type="exec", command="ls"))

    assert obs.success is False
    assert "call reset() first" in obs.error


def test_reset_without_a_task_id_lists_the_options(tmp_path: Path) -> None:
    write_task(tmp_path, "one")
    write_task(tmp_path, "two")
    environment = HarborEnvironment(tasks=str(tmp_path), mode="local")
    try:
        with pytest.raises(ValueError, match="needs a task_id"):
            environment.reset()
    finally:
        environment.close()


def test_failed_reset_does_not_leave_the_previous_episode_in_state(
    env: HarborEnvironment,
) -> None:
    """A reset that raises must not leave `state` describing a dead episode.

    reset() tears the old sandbox down before it can fail, so reporting the
    previous task afterwards would tell a trainer an episode is running when
    nothing is.
    """
    env.reset(task_id=EXAMPLE_TASK_ID)
    env.step(HarborAction(action_type="evaluate"))
    assert env.state.task_id == EXAMPLE_TASK_ID
    assert env.state.evaluated

    with pytest.raises(KeyError):
        env.reset(task_id="no-such-task")

    assert env.state.task_id == ""
    assert env.state.task_name == ""
    assert env.state.workdir == ""
    assert env.state.reward is None
    assert not env.state.evaluated
    # The catalog listing survives — it does not belong to any one episode.
    assert env.state.task_count == 1


def test_reset_starts_a_clean_episode(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    env.step(HarborAction(action_type="write", path="scratch.txt", content="dirty"))
    first_workdir = env.state.workdir

    env.reset(task_id=EXAMPLE_TASK_ID)

    assert env.state.workdir != first_workdir
    assert env.state.step_count == 0
    assert env.step(HarborAction(action_type="exec", command="ls")).output.split() == [
        "stats.py"
    ]


# ---------------------------------------------------------------------------
# wire contract between client and server
# ---------------------------------------------------------------------------


def test_action_schema_rejects_unknown_action_types() -> None:
    with pytest.raises(ValueError):
        HarborAction(action_type="rm_rf")


def test_client_parses_what_the_server_serializes(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    observation = env.step(HarborAction(action_type="evaluate"))
    payload = serialize_observation(observation)

    result = HarborEnv(base_url="http://localhost:8000")._parse_result(payload)

    assert isinstance(result.observation, HarborObservation)
    assert result.reward == pytest.approx(BUGGY_REWARD)
    assert result.done is True
    assert result.observation.info["reward_source"] == "reward.json"


def test_client_parses_state(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)

    state = HarborEnv(base_url="http://localhost:8000")._parse_state(
        env.state.model_dump()
    )

    assert isinstance(state, HarborState)
    assert state.task_id == EXAMPLE_TASK_ID
    assert state.mode == "local"


def test_client_step_payload_round_trips(tmp_path: Path) -> None:
    action = HarborAction(action_type="write", path="a.py", content="x = 1\n")

    payload = HarborEnv(base_url="http://localhost:8000")._step_payload(action)

    assert HarborAction.model_validate(payload) == action


# ---------------------------------------------------------------------------
# task policies: network, resources, user
# ---------------------------------------------------------------------------

_POLICY_TASK = textwrap.dedent(
    """
    schema_version = "1.4"

    [task]
    name = "tests/policy"
    description = "fixture"

    [environment]
    network_mode = "no-network"
    cpus = 2
    memory_mb = 512

    [agent]
    user = "agent"

    [verifier]
    user = "root"
    """
).strip()


def test_task_parses_network_resource_and_user_policies(tmp_path: Path) -> None:
    write_task(tmp_path, "policy", config=_POLICY_TASK, seed_files={"a.py": "x = 1\n"})
    task = TaskCatalog(tmp_path).get("policy")

    assert task.network.baseline == "no-network"
    assert task.network.restricted
    assert task.resources.cpus == 2
    assert task.resources.memory_mb == 512
    assert task.agent_user == "agent"
    assert task.verifier_user == "root"


def test_unknown_network_mode_is_rejected_not_downgraded(tmp_path: Path) -> None:
    """Silently treating an unrecognized mode as `public` is the bug to avoid."""
    config = _POLICY_TASK.replace('network_mode = "no-network"', 'network_mode = "vpn"')
    write_task(tmp_path, "weird", config=config, seed_files={"a.py": "x = 1\n"})

    with pytest.raises(TaskFormatError, match="network_mode"):
        TaskCatalog(tmp_path).get("weird")


def test_deprecated_allow_internet_false_still_restricts(tmp_path: Path) -> None:
    config = textwrap.dedent(
        """
        schema_version = "1.4"

        [task]
        name = "tests/legacy"

        [environment]
        allow_internet = false
        """
    ).strip()
    write_task(tmp_path, "legacy", config=config, seed_files={"a.py": "x = 1\n"})

    assert TaskCatalog(tmp_path).get("legacy").network.baseline == "no-network"


def test_local_backend_refuses_policies_it_cannot_enforce(tmp_path: Path) -> None:
    """Fail closed: a no-network task must not silently get the host network."""
    write_task(tmp_path, "policy", config=_POLICY_TASK, seed_files={"a.py": "x = 1\n"})
    task = TaskCatalog(tmp_path).get("policy")
    sandbox = LocalSandbox()

    with pytest.raises(SandboxError) as excinfo:
        sandbox.start(task)

    message = str(excinfo.value)
    assert "network_mode" in message
    assert "resource limits" in message
    assert "a declared user" in message
    assert "HARBOR_MODE=docker" in message
    sandbox.close()


def test_docker_limits_translate_harbor_policies() -> None:
    """no-network → an isolated network; cpus/memory → real container caps."""
    task = HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID)
    restricted = replace(
        task,
        network=NetworkPolicy(baseline="no-network"),
        resources=ResourceLimits(cpus=2, memory_mb=512),
    )

    limits = _container_limits(restricted)

    assert limits["network_mode"] == "none"
    assert limits["cpu_quota"] == 200_000
    assert limits["cpu_period"] == 100_000
    assert limits["mem_limit"] == "512m"
    # An unrestricted task gets no constraints imposed on it.
    assert _container_limits(task) == {}


def test_allowlist_and_gpu_tasks_are_refused_rather_than_approximated() -> None:
    task = HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID)

    with pytest.raises(SandboxError, match="allowlist"):
        _container_limits(replace(task, network=NetworkPolicy(baseline="allowlist")))
    with pytest.raises(SandboxError, match="GPU"):
        _container_limits(replace(task, resources=ResourceLimits(gpus=1)))


def test_mixed_phase_network_modes_take_the_restrictive_one() -> None:
    """One container, one network — never resolve the conflict permissively."""
    task = HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID)
    mixed = replace(
        task, network=NetworkPolicy(baseline="public", verifier="no-network")
    )

    assert _container_limits(mixed)["network_mode"] == "none"


# ---------------------------------------------------------------------------
# the agent's view of the server
# ---------------------------------------------------------------------------


def test_reset_does_not_disclose_the_task_directory(env: HarborEnvironment) -> None:
    """The observation reaches the agent; the task path points at solution/."""
    observation = env.reset(task_id=EXAMPLE_TASK_ID)

    assert "path" not in observation.info["task"]
    assert str(BUNDLED_TASKS_DIR) not in json.dumps(observation.info)
    # `state` rides the same socket, so it must not carry the path either.
    assert str(BUNDLED_TASKS_DIR) not in json.dumps(env.state.model_dump())


def test_local_exec_does_not_inherit_server_secrets(
    env: HarborEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task commands are attacker-controlled; the server's tokens are not theirs."""
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret")
    env.reset(task_id=EXAMPLE_TASK_ID)

    observation = env.step(HarborAction(action_type="exec", command="env"))

    assert "hf_super_secret" not in observation.output
    # PATH still gets through, or nothing would run at all.
    assert "PATH=" in observation.output


# ---------------------------------------------------------------------------
# episode termination and the control-action split
# ---------------------------------------------------------------------------


def test_episode_stays_terminal_after_evaluate(env: HarborEnvironment) -> None:
    """`done` must never walk back from True to False."""
    env.reset(task_id=EXAMPLE_TASK_ID)
    graded = env.step(HarborAction(action_type="evaluate"))
    assert graded.done

    for action in (
        HarborAction(action_type="exec", command="ls"),
        HarborAction(action_type="read", path="stats.py"),
        HarborAction(action_type="evaluate"),
    ):
        observation = env.step(action)
        assert observation.done, f"{action.action_type} reported done=False"
        assert not observation.success
        assert observation.reward is None
        assert "episode ended" in observation.error


def test_control_actions_can_be_refused_server_side(tmp_path: Path) -> None:
    """The agent/orchestration split, enforced rather than documented."""
    environment = HarborEnvironment(mode="local", allow_control_actions=False)
    try:
        environment.reset(task_id=EXAMPLE_TASK_ID)

        for action_type in sorted(CONTROL_ACTIONS):
            observation = environment.step(HarborAction(action_type=action_type))
            assert not observation.success
            assert "orchestration control" in observation.error
            assert not observation.done

        # Agent actions are unaffected.
        assert environment.step(HarborAction(action_type="exec", command="ls")).success
    finally:
        environment.close()


def test_agent_commands_are_not_told_where_the_verifier_lives(
    env: HarborEnvironment,
) -> None:
    """`/tests` holds the grading logic and `/solution` holds the answer.

    Exporting their locations into the agent's environment hands a policy the
    map even when the directories are not staged yet.
    """
    env.reset(task_id=EXAMPLE_TASK_ID)

    observation = env.step(
        HarborAction(
            action_type="exec",
            command="echo [$HARBOR_SOLUTION_DIR][$HARBOR_TESTS_DIR][$HARBOR_LOGS_DIR]",
        )
    )

    assert observation.output.strip() == "[][][]"
    # The agent still knows where it is working.
    assert (
        "HARBOR_WORKDIR"
        in env.step(HarborAction(action_type="exec", command="env")).output
    )


def test_verifier_still_receives_its_paths(env: HarborEnvironment) -> None:
    """Withholding paths from the agent must not break the verifier."""
    env.reset(task_id=EXAMPLE_TASK_ID)

    graded = env.step(HarborAction(action_type="evaluate"))

    assert graded.reward == pytest.approx(BUGGY_REWARD)
    assert graded.info["reward_source"] == "reward.json"


def test_planted_files_do_not_survive_into_the_staged_verifier(tmp_path: Path) -> None:
    """The agent shares the filesystem and can pre-create the tests directory.

    Anything it leaves there must be wiped before the real `tests/` lands, or a
    policy could plant a file the verifier goes on to read.
    """
    write_task(
        tmp_path,
        "plantable",
        seed_files={"a.py": "x = 1\n"},
        test_script=(
            "#!/bin/bash\n"
            'mkdir -p "$HARBOR_LOGS_DIR"\n'
            'if [ -f "$HARBOR_TESTS_DIR/planted.txt" ]; then\n'
            '  echo "1.0" > "$HARBOR_LOGS_DIR/reward.txt"\n'
            "else\n"
            '  echo "0.0" > "$HARBOR_LOGS_DIR/reward.txt"\n'
            "fi\n"
            "exit 0\n"
        ),
    )
    environment = HarborEnvironment(tasks=str(tmp_path), mode="local")
    try:
        environment.reset(task_id="plantable")
        tests_dir = Path(environment._sandbox.paths.tests)
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "planted.txt").write_text("gotcha", encoding="utf-8")

        graded = environment.step(HarborAction(action_type="evaluate"))

        assert graded.reward == 0.0, "a planted file survived into the verifier"
        assert not (tests_dir / "planted.txt").exists()
    finally:
        environment.close()


def test_declared_user_gets_ownership_of_what_it_must_write() -> None:
    """`[agent].user` is useless if the agent cannot write its own workdir.

    The Harbor directories are created by the image's default account, so a
    non-root phase inherits a tree it cannot touch — `write` fails and the
    verifier cannot save its reward. Ownership is handed over explicitly.
    """
    task = replace(
        HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID),
        agent_user="nobody",
        verifier_user="nobody",
    )
    sandbox = _RecordingSandbox()

    sandbox.start(task)
    sandbox.run_verifier(task)

    assert ("nobody", sandbox.paths.workdir) in sandbox.chowned
    assert ("nobody", sandbox.paths.logs_verifier) in sandbox.chowned


class _RecordingSandbox(Sandbox):
    """A sandbox that records the calls the phases make, without executing."""

    mode = "recording"

    def __init__(self) -> None:
        self.paths = SandboxPaths(workdir="/app")
        self.task_env = {}
        self.chowned: list[tuple[str, str]] = []
        self.read_users: list[str | None] = []
        self.write_users: list[str | None] = []
        self.agent_user: str | None = None

    def _boot(self, task: HarborTask) -> SandboxPaths:
        return self.paths

    def exec(self, command, *, timeout_s, env=None, workdir=None, user=None, **kw):
        return ExecResult(0, "")

    def chown(self, user: str | None, *paths: str) -> None:
        if user:
            self.chowned.extend((user, path) for path in paths)

    def real_path(self, path: str) -> str:
        return path

    def read_text(self, path: str, user: str | None = None) -> str | None:
        self.read_users.append(user)
        return "contents"

    def write_text(self, path: str, content: str, user: str | None = None) -> None:
        self.write_users.append(user)

    def upload_dir(self, source, destination: str) -> None:
        return None

    def close(self) -> None:
        return None


def test_malformed_resource_limits_are_rejected_not_ignored(tmp_path: Path) -> None:
    """A typo must not silently buy the task unlimited resources.

    Treating an invalid value as "absent" would also stop the local backend
    refusing the task, so the failure compounds.
    """
    for bad in ("0", "-4", '"lots"'):
        config = textwrap.dedent(
            f"""
            schema_version = "1.4"

            [task]
            name = "tests/bad"

            [environment]
            cpus = {bad}
            """
        ).strip()
        write_task(tmp_path, "bad", config=config, seed_files={"a.py": "x = 1\n"})
        catalog = TaskCatalog(tmp_path)
        catalog.refresh()

        with pytest.raises(TaskFormatError, match="cpus"):
            catalog.get("bad")


def test_storage_limits_are_refused_rather_than_faked() -> None:
    """Docker accepts `storage_opt` on drivers that never enforce it.

    A task that believes it is capped and is not is worse than one that is told
    up front we cannot cap it.
    """
    task = HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID)

    with pytest.raises(SandboxError, match="quota"):
        _container_limits(replace(task, resources=ResourceLimits(storage_mb=32)))


def test_phase_overrides_replace_the_baseline_they_override() -> None:
    """A baseline both phases override must not still take effect."""
    policy = NetworkPolicy(baseline="no-network", agent="public", verifier="public")

    assert policy.modes == {"public"}
    assert not policy.restricted

    task = HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID)
    assert "network_mode" not in _container_limits(replace(task, network=policy))


def test_agent_supplied_timeout_cannot_exceed_the_configured_ceiling(
    tmp_path: Path,
) -> None:
    """`timeout_s` arrives on the wire, so it is policy-controlled."""
    recorded: list[float] = []

    class _Recording(LocalSandbox):
        def exec(self, command, *, timeout_s, env=None, workdir=None, **kw):
            recorded.append(timeout_s)
            return ExecResult(0, "")

    environment = HarborEnvironment(
        mode="local", command_timeout_s=5.0, sandbox_factory=lambda mode: _Recording()
    )
    try:
        environment.reset(task_id=EXAMPLE_TASK_ID)
        recorded.clear()
        environment.step(
            HarborAction(action_type="exec", command="sleep 1", timeout_s=86_400.0)
        )
    finally:
        environment.close()

    assert recorded == [5.0]


def test_agent_reads_and_writes_run_as_the_declared_user() -> None:
    """`read` used `cat` as root and `write` extracted a root-owned tar.

    Either would let a task that declared an unprivileged agent reach files
    that agent could never have touched with `exec`.
    """
    task = replace(
        HarborTask.load(BUNDLED_TASKS_DIR / EXAMPLE_TASK_ID), agent_user="nobody"
    )
    del task  # the sandbox is driven directly below
    sandbox = _RecordingSandbox()
    environment = HarborEnvironment(mode="local", sandbox_factory=lambda mode: sandbox)
    try:
        environment.reset(task_id=EXAMPLE_TASK_ID)
        sandbox.agent_user = "nobody"
        environment.step(HarborAction(action_type="read", path="stats.py"))
        environment.step(
            HarborAction(action_type="write", path="stats.py", content="x = 1\n")
        )
    finally:
        environment.close()

    assert sandbox.read_users == ["nobody"]
    assert sandbox.write_users == ["nobody"]


def test_symlinks_cannot_walk_read_write_out_of_the_workdir(
    env: HarborEnvironment,
) -> None:
    """`resolve_within` is lexical, so it cannot see a link the agent planted.

    `exec ln -s /tests t` followed by `read path="t/test.sh"` would otherwise
    walk straight out of the working directory and read the grader.
    """
    env.reset(task_id=EXAMPLE_TASK_ID)
    tests_dir = Path(env._sandbox.paths.tests)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "grade.py").write_text("# grader internals\n", encoding="utf-8")
    env.step(HarborAction(action_type="exec", command=f"ln -sfn {tests_dir} link"))

    leaked = env.step(HarborAction(action_type="read", path="link/grade.py"))
    planted = env.step(
        HarborAction(action_type="write", path="link/test.sh", content="exit 0")
    )

    assert not leaked.success and "escapes" in leaked.error
    assert not planted.success and "escapes" in planted.error
    assert not (tests_dir / "test.sh").exists()


def test_legitimate_paths_survive_symlink_canonicalization(
    env: HarborEnvironment,
) -> None:
    """The working directory itself often sits behind a link.

    On macOS it is under `/var`, which is a symlink to `/private/var`; comparing
    a resolved path against an unresolved base would reject everything.
    """
    env.reset(task_id=EXAMPLE_TASK_ID)

    assert env.step(HarborAction(action_type="read", path="stats.py")).success
    # A write may legitimately create directories that do not exist yet.
    assert env.step(
        HarborAction(action_type="write", path="pkg/mod/new.py", content="x = 1\n")
    ).success
    assert env.step(HarborAction(action_type="read", path="pkg/mod/new.py")).success


def test_local_mode_flags_a_declared_image_it_is_not_providing(
    env: HarborEnvironment,
) -> None:
    """A seed-file task is reproducible locally, but its toolchain may not be.

    `fix-sum-bug` declares `docker_image = "python:3.12-slim"`; running it on
    the host's Python is close enough to be useful and different enough to
    mis-grade, so the mismatch is surfaced rather than left silent.
    """
    observation = env.reset(task_id=EXAMPLE_TASK_ID)

    assert "python:3.12-slim" in observation.info["toolchain_warning"]
    assert "HARBOR_MODE=docker" in observation.info["toolchain_warning"]


@pytest.mark.parametrize("value", ["2.0", "2e0"])
def test_whole_number_float_resource_limits(tmp_path: Path, value: str) -> None:
    task = HarborTask.load(
        write_task(
            tmp_path, "float-resource", config=f"[environment]\ncpus = {value}\n"
        )
    )
    assert task.resources.cpus == 2


@pytest.mark.parametrize("value", ["2.5", "nan", "inf", "-inf", "true"])
def test_invalid_numeric_resource_limits(tmp_path: Path, value: str) -> None:
    task_dir = write_task(
        tmp_path, "invalid-resource", config=f"[environment]\ncpus = {value}\n"
    )
    with pytest.raises(TaskFormatError, match="cpus"):
        HarborTask.load(task_dir)


def test_docker_user_write_creates_nested_parents(tmp_path: Path, monkeypatch) -> None:
    sandbox = DockerSandbox()
    local = LocalSandbox(root=tmp_path)
    users = []

    def exec_as_user(command, *, timeout_s, user):
        users.append(user)
        return local.exec(command, timeout_s=timeout_s)

    monkeypatch.setattr(sandbox, "exec", exec_as_user)
    target = tmp_path / "nested dir" / "it's a file.txt"
    sandbox.write_text(str(target), "hello 'world'\n", user="nobody")
    assert target.read_text() == "hello 'world'\n"
    assert users == ["nobody"]
