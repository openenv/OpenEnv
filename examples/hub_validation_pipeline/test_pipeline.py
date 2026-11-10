"""Admission and execution regressions at the report/runner boundaries."""

import asyncio
import hashlib
import json
import shutil
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from examples.hub_validation_pipeline import demo, pipeline
from openenv.harbor.models import HarborRolloutResult
from openenv.validation.parsers.harbor_toml import HarborTaskParser
from openenv.validation.runner import source_digest


@pytest.fixture
def evidence(tmp_path):
    return make_evidence(tmp_path)


def make_evidence(tmp_path, canary_count=2):
    dataset = demo.create_tasks(tmp_path / "input")
    subject = tmp_path / "subject"
    shutil.copytree(dataset / "good", subject)
    canaries = subject / "validation/canaries"
    if canaries.exists():
        shutil.rmtree(canaries)
    canaries.mkdir(parents=True)
    for index in range(canary_count):
        (canaries / f"wrong-{index}.sh").write_text(f"#!/bin/bash\necho {index}\n")
    original = source_digest(subject)
    contract = json.loads(
        Path(pipeline.__file__).with_name("contract.json").read_text()
    )
    if canary_count:
        contract["canaries"] = "validation/canaries"
    else:
        contract.pop("canaries", None)
    pipeline.write_json(
        subject / "validation/openenv.json",
        {**contract, "original_source_digest": original},
    )
    manifest = HarborTaskParser().parse(subject).model_dump(mode="json")
    report = {
        "report_schema_version": "2",
        "target": str(subject),
        "source_digest": source_digest(subject),
        "signature": "task.toml",
        "manifest": manifest,
        "policy_version": "v2",
        "lane": "local",
        "levels_run": [1, 2, 3],
        "verdict": "warn",
        "results": [
            {"check_id": key, "status": "pass", "duration_s": 0}
            for key in sorted(pipeline.PILOT_CHECKS)
        ]
        + (
            [{"check_id": "semantic.canary_floor", "status": "pass", "duration_s": 0}]
            if canary_count
            else []
        )
        + [
            {
                "check_id": "runtime.host_containment",
                "status": "skip",
                "duration_s": 0,
                "evidence": ["out of profile: native process boundary"],
            },
        ],
    }
    native = {
        "schema_version": "2",
        "profile": pipeline.PILOT,
        "task_effective_digest": report["source_digest"],
        "original_source_digest": original,
        "image_ref": "sha256:" + "a" * 64,
        "retained_image_tag": "openenv-validation:" + "a" * 32,
        "reward_key": "reward",
        "floor_reward": 0,
        "oracle_reward": 1,
        "cleanup": {"verified": True},
        "trials": [],
    }
    artifacts = tmp_path / "evidence"
    roles = ["nop", "oracle", "oracle-replay"] + [
        f"canary-{index}" for index in range(1, canary_count + 1)
    ]
    for index, role in enumerate(roles):
        rewards = {"reward": int(role in {"oracle", "oracle-replay"}), "aux": 0}
        trial = {
            "role": role,
            "image_ref": native["image_ref"],
            "retained_image_tag": native["retained_image_tag"],
            "container_id": f"{index + 1:064x}",
            "unmodified_task_digest": report["source_digest"],
            "task_overlay": {}
            if role == "nop"
            else {
                "environment.docker_image": native["retained_image_tag"],
            },
            "cleanup": {"verified": True},
            "configuration": {
                "user": "",
                "workdir": "/app",
                "host": {"Memory": 512 * 1024**2},
                "network_mode": "compose-default",
                "mounts": [],
            },
            "rewards": rewards,
            "raw_rewards": {},
        }
        if role != "nop":
            trial.update(oracle_exit_observed_at_agent_end=True, oracle_exit_code=0)
        if role.startswith("canary-"):
            canary = sorted(canaries.iterdir())[index - 3]
            trial["task_overlay"]["solution/solve.sh"] = {
                "canary": canary.relative_to(subject).as_posix(),
                "sha256": hashlib.sha256(canary.read_bytes()).hexdigest(),
            }
        native["trials"].append(trial)
        set_raw(artifacts, trial, "reward.json", pipeline.dumps(rewards) + "\n")
    save_bundle(artifacts, report, native)
    return subject, report, artifacts, original, native


def set_raw(directory, trial, filename, text):
    name = f"harbor-trials/{trial['role']}/{filename}"
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    trial["raw_rewards"][filename] = {
        "path": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def save_bundle(directory, report, native):
    pipeline.write_json(directory / "report.json", report)
    pipeline.write_json(directory / "harbor-evidence.json", native)
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(directory)}\n"
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )


def admit(evidence, profile=pipeline.PILOT):
    subject, report, artifacts, original, _ = evidence
    return pipeline.admission(report, 0, subject, artifacts, profile, original)


def test_explicit_pilot_admits_but_full_gate_stays_incomplete(evidence):
    admitted = admit(evidence)
    assert admitted["out_of_profile_skips"] == ["runtime.host_containment"]
    assert admitted["rfc008_complete"] is False
    with pytest.raises(ValueError, match="coverage"):
        admit(evidence, "full")


def test_full_profile_requires_every_policy_check_to_pass(evidence):
    _, report, artifacts, _, native = evidence
    report["verdict"] = "pass"
    report["results"] = [
        {"check_id": key, "status": "pass", "duration_s": 0}
        for key in pipeline.load_policy("v2").entries_for_lane(pipeline.Lane.LOCAL)
    ]
    save_bundle(artifacts, report, native)
    assert admit(evidence, "full")["rfc008_complete"] is True
    report["results"][-1]["status"] = "skip"
    with pytest.raises(ValueError, match="coverage"):
        admit(evidence, "full")


@pytest.mark.parametrize("profile", [pipeline.PILOT, "full"])
def test_local_validator_requests_semantic_level(tmp_path, monkeypatch, profile):
    calls = []
    report = tmp_path / "report.json"

    class Process:
        returncode = 0

        def communicate(self, timeout):
            report.write_text('{"validator":"result"}')
            return "", ""

    def start(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    assert pipeline.validate_task(tmp_path, report, profile, 10) == (
        0,
        {"validator": "result"},
    )
    command = calls[0][0]
    assert command[command.index("--level") + 1] == "semantic"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "skip", "fail", "unknown"])
def test_exit_zero_cannot_hide_bad_coverage(evidence, fault):
    report = evidence[1]
    if fault == "missing":
        report["results"].pop(0)
    elif fault == "duplicate":
        report["results"].append(deepcopy(report["results"][0]))
    elif fault == "skip":
        report["results"][0]["status"] = "skip"
    elif fault == "fail":
        report["results"][-1]["status"] = "fail"
    else:
        report["results"].append(
            {"check_id": "runtime.invented", "status": "pass", "duration_s": 0}
        )
    with pytest.raises(ValueError):
        admit(evidence)


@pytest.mark.parametrize(
    "check",
    [
        "runtime.episode_determinism",
        "semantic.oracle_max",
        "semantic.floor_gap",
        "semantic.canary_floor",
    ],
)
@pytest.mark.parametrize("status", [None, "skip", "fail", "error"])
def test_semantic_coverage_is_required(evidence, check, status):
    report = evidence[1]
    row = next(row for row in report["results"] if row["check_id"] == check)
    if status is None:
        report["results"].remove(row)
    else:
        row["status"] = status
    with pytest.raises(ValueError):
        admit(evidence)


@pytest.mark.parametrize("canary_count", [0, 1, 8])
def test_declared_canary_count_sets_the_trial_plan(tmp_path, canary_count):
    sample = make_evidence(tmp_path, canary_count)
    assert len(sample[-1]["trials"]) == 3 + canary_count
    assert admit(sample)["profile"] == "harbor-semantic-v1"


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "reordered",
        "duplicate-container",
        "image",
        "source",
        "failed",
        "cleanup",
        "top-cleanup",
        "configuration",
        "missing-floor",
        "missing-tag",
    ],
)
def test_every_trial_must_complete_with_bound_identity(evidence, fault):
    _, report, artifacts, _, native = evidence
    trials = native["trials"]
    if fault == "missing":
        trials.pop()
    elif fault == "reordered":
        trials[1], trials[2] = trials[2], trials[1]
    elif fault == "duplicate-container":
        trials[-1]["container_id"] = trials[0]["container_id"]
    elif fault == "image":
        trials[-1]["image_ref"] = "sha256:" + "b" * 64
    elif fault == "source":
        trials[-1]["unmodified_task_digest"] = "b" * 64
    elif fault == "failed":
        trials[-1]["failure"] = "RewardArtifactDrift"
    elif fault == "cleanup":
        trials[-1]["cleanup"]["verified"] = False
    elif fault == "top-cleanup":
        native["cleanup"]["verified"] = False
    elif fault == "configuration":
        trials[-1]["configuration"]["workdir"] = "/different"
    elif fault == "missing-floor":
        native.pop("floor_reward")
    else:
        native.pop("retained_image_tag")
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError):
        admit(evidence)


@pytest.mark.parametrize(
    "fault", ["reward-map", "exit", "unobserved", "marker", "unbound-marker"]
)
def test_oracle_replay_checks_every_reward_and_trusted_exit(evidence, fault):
    _, report, artifacts, _, native = evidence
    replay = native["trials"][2]
    if fault == "reward-map":
        replay["rewards"]["aux"] = 1
        set_raw(artifacts, replay, "reward.json", pipeline.dumps(replay["rewards"]))
    elif fault == "exit":
        replay["oracle_exit_code"] = 1
        set_raw(artifacts, replay, "oracle-exit-code.txt", "1\n")
    elif fault == "unobserved":
        replay["oracle_exit_observed_at_agent_end"] = False
    else:
        set_raw(artifacts, replay, "oracle-exit-code.txt", "1\n")
    save_bundle(artifacts, report, native)
    if fault == "unbound-marker":
        inventory = artifacts / "SHA256SUMS"
        inventory.write_text(
            "".join(
                row
                for row in inventory.read_text().splitlines(keepends=True)
                if not row.rstrip().endswith("oracle-exit-code.txt")
            )
        )
    with pytest.raises(ValueError):
        admit(evidence)


@pytest.mark.parametrize("fault", ["reward", "path", "digest", "missing", "extra"])
def test_canary_reward_and_overlay_match_declared_bytes(evidence, fault):
    _, report, artifacts, _, native = evidence
    canary = native["trials"][-1]
    overlay = canary["task_overlay"]
    if fault == "reward":
        canary["rewards"]["reward"] = 1
        set_raw(artifacts, canary, "reward.json", pipeline.dumps(canary["rewards"]))
    elif fault == "path":
        overlay["solution/solve.sh"]["canary"] = "solution/solve.sh"
    elif fault == "digest":
        overlay["solution/solve.sh"]["sha256"] = "b" * 64
    elif fault == "missing":
        overlay.pop("solution/solve.sh")
    else:
        overlay["tests/test.sh"] = "changed"
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError):
        admit(evidence)


@pytest.mark.parametrize(
    "raw",
    [
        '{"reward":true}',
        '{"reward":"1"}',
        '{"reward":NaN}',
        '{"reward":1,"aux":2}',
        '{"reward":1,"reward":0}',
        "{}",
    ],
)
def test_raw_reward_maps_fail_closed(evidence, raw):
    _, report, artifacts, _, native = evidence
    set_raw(artifacts, native["trials"][1], "reward.json", raw)
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError):
        admit(evidence)


@pytest.mark.parametrize(
    "fault",
    ["old-profile", "old-schema", "missing-schema", "operator-profile", "runtime-only"],
)
def test_stale_native_contract_is_rejected(evidence, fault):
    _, report, artifacts, _, native = evidence
    if fault == "old-profile":
        native["profile"] = "harbor-oracle-floor-v1"
    elif fault == "old-schema":
        native["schema_version"] = "1"
    elif fault == "missing-schema":
        native.pop("schema_version")
    elif fault == "runtime-only":
        report["levels_run"] = [1, 2]
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError):
        admit(
            evidence,
            "harbor-oracle-floor-v1" if fault == "operator-profile" else pipeline.PILOT,
        )


def test_admission_recomputes_oracle_floor_from_raw_rewards(evidence):
    _, report, artifacts, _, native = evidence
    trial = native["trials"][0]
    trial["rewards"]["reward"] = 1
    set_raw(artifacts, trial, "reward.json", pipeline.dumps(trial["rewards"]))
    native["floor_reward"] = 1
    # Even a self-consistent bundle with stale semantic PASS results is rejected.
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError, match="oracle/floor contract"):
        admit(evidence)


@pytest.mark.parametrize(
    "floor,oracle,margin,tolerance",
    [
        (Decimal("0.9"), 1, Decimal("0.100000000000000001"), 0),
        (0, Decimal("0.95"), 1, Decimal("0.1")),
    ],
)
def test_exact_operator_margin_overrides_rounded_semantic_pass(
    evidence, monkeypatch, floor, oracle, margin, tolerance
):
    subject, report, artifacts, _, native = evidence
    path = subject / "validation/openenv.json"
    contract = pipeline.loads(path.read_text())
    contract["reward"].update(floor_margin=margin, oracle_tolerance=tolerance)
    pipeline.write_json(path, contract)
    report["manifest"]["reward"].update(
        floor_margin=float(margin), oracle_tolerance=float(tolerance)
    )
    # Exercise admission independently of the parser's representability guard.
    monkeypatch.setattr(
        pipeline,
        "read_harbor_contract",
        lambda root: SimpleNamespace(canaries="validation/canaries"),
    )
    report["source_digest"] = source_digest(subject)
    native["task_effective_digest"] = report["source_digest"]
    for trial in native["trials"]:
        trial["unmodified_task_digest"] = report["source_digest"]
        trial["rewards"]["reward"] = (
            oracle if trial["role"] in {"oracle", "oracle-replay"} else floor
        )
        set_raw(artifacts, trial, "reward.json", pipeline.dumps(trial["rewards"]))
    native.update(floor_reward=floor, oracle_reward=oracle)
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError, match="oracle/floor contract"):
        admit(evidence)


def test_bundled_report_must_equal_the_cli_result(evidence):
    _, report, artifacts, _, native = evidence
    bundled = deepcopy(report)
    bundled["target"] = "different-subject"
    save_bundle(artifacts, bundled, native)
    with pytest.raises(ValueError, match="bundled report differs"):
        admit(evidence)


def test_raw_output_cannot_redirect_to_another_trials_file(evidence):
    _, report, artifacts, _, native = evidence
    native["trials"][2]["raw_rewards"] = deepcopy(native["trials"][1]["raw_rewards"])
    save_bundle(artifacts, report, native)
    with pytest.raises(ValueError, match="verified artifact inventory"):
        admit(evidence)


def test_tampered_artifact_or_subject_cannot_be_reused(evidence):
    subject, _, artifacts, _, _ = evidence
    (artifacts / "harbor-evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        admit(evidence)
    (subject / "instruction.md").write_text("changed")
    with pytest.raises(ValueError, match="changed after validation"):
        admit(evidence)


def approved_task(evidence, output):
    record = {**admit(evidence), "subject": str(evidence[0]), "task_id": "good-task"}
    return pipeline.materialize(record, output / "approved/tasks/good")


def test_execution_pins_only_declared_overlay_and_preserves_validated_source(
    evidence, tmp_path
):
    subject = evidence[0]
    original = (subject / "task.toml").read_bytes()
    record = approved_task(evidence, tmp_path / "output")
    assert (subject / "task.toml").read_bytes() == original
    assert source_digest(Path(record["execution_path"])) == record["execution_digest"]
    assert (
        record["execution_overlay"]["environment"]["docker_image"]
        == record["image_ref"]
    )


def test_copy_omits_unhashed_files_and_never_dereferences_links(evidence, tmp_path):
    source = evidence[0]
    hidden = source / ".worktrees"
    hidden.mkdir()
    (hidden / "host-file").symlink_to("/etc/hosts")
    expected = source_digest(source)
    destination = tmp_path / "copy"
    pipeline.copy_subject(source, destination, expected)
    assert not (destination / ".worktrees").exists()
    (source / "escaped").symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="symbolic links"):
        pipeline.copy_subject(source, tmp_path / "unsafe", expected)


def test_only_approved_tasks_reach_existing_model_runner(
    evidence, tmp_path, monkeypatch
):
    from openenv.harbor import runner

    output = tmp_path / "output"
    record = approved_task(evidence, output)
    call = AsyncMock(
        return_value=[
            HarborRolloutResult(task_id=record["execution_path"], reward=0, ok=True)
        ]
    )
    monkeypatch.setattr(runner, "run_batch", call)
    rows = pipeline.evaluate(
        [record],
        output,
        llm_url="http://model/v1",
        model="test",
        harness="opencode",
        expose="direct",
        timeout_s=10,
        step_limit=4,
    )
    assert len(rows) == 1 and rows[0]["result"]["reward"] == 0
    assert call.call_args.kwargs["task_indices"] == [0]
    assert call.call_args.kwargs["agent_step_limit"] == 4
    # An unapproved sibling must stop the runner before another model call.
    shutil.copytree(record["execution_path"], output / "approved/tasks/unapproved")
    from openenv.harbor.tasks import resolve_task_dirs

    resolve_task_dirs(str(output / "approved"), refresh=True)
    with pytest.raises(ValueError, match="corpus disagree"):
        pipeline.evaluate(
            [record],
            output,
            llm_url="http://model/v1",
            model="test",
            harness="opencode",
            expose="direct",
            timeout_s=10,
            step_limit=4,
        )
    assert call.await_count == 1


def test_model_image_guard_runs_before_agent_execution(evidence, tmp_path, monkeypatch):
    record = approved_task(evidence, tmp_path / "output")
    hooks = []
    trial = SimpleNamespace(
        config=SimpleNamespace(
            task=SimpleNamespace(path=Path(record["execution_path"]))
        ),
        agent_environment=SimpleNamespace(session_id="oe-good__env"),
        add_hook=lambda event, callback: hooks.append(callback),
    )
    actual = {"Image": "sha256:" + "b" * 64}
    monkeypatch.setattr(
        pipeline,
        "docker",
        lambda *args: "container" if args[0] == "ps" else json.dumps([actual]),
    )
    projects = set()
    pipeline.model_guard([record], projects)(trial)
    assert projects == {"oe-good__env"}
    with pytest.raises(ValueError, match="image differs"):
        asyncio.run(hooks[0](None))
