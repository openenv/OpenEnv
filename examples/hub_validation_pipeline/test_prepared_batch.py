"""Offline cohort binding, spend gates and one-shot controller polling."""

import hashlib
import json
import shutil
import socket
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.hub_validation_pipeline import cloud, full_run, prepared_batch, spend
from examples.hub_validation_pipeline.test_real_tasks import write_panel

SECRETS = dict(
    HF_TOKEN="local-results",
    HF_INFERENCE_TOKEN="local-inference",
    HF_SANDBOX_TOKEN="local-execution",
)


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Live operations forbidden"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    monkeypatch.setattr(huggingface_hub, "get_token", forbidden)
    repo = tmp_path / "repo"
    (repo / "src/openenv").mkdir(parents=True)
    (repo / "src/openenv/__init__.py").write_text("# source\n")
    (repo / "examples/hub_validation_pipeline").mkdir(parents=True)
    (repo / "examples/hub_validation_pipeline/cloud.py").write_bytes(
        Path(cloud.__file__).read_bytes()
    )
    shutil.copytree(
        Path(cloud.__file__).parents[1] / "coding_audit",
        repo / "examples/coding_audit",
        ignore=shutil.ignore_patterns("test_*", "__pycache__", ".pytest_cache"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE", "uv.lock"):
        (repo / name).write_text(name + "\n")
    monkeypatch.setattr(
        cloud, "_git_provenance", lambda root: {"commit": "a" * 40, "dirty": False}
    )
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "task.toml").write_text("version = '1.0'\n")
    (task / "tests/test.sh").write_text("printf 1\n")
    (task / "solution/solve.sh").write_text("true\n")
    prepared = tmp_path / "prepared"
    configs, plans, rows = [], [], []
    for index in range(4):
        (task / "instruction.md").write_text(f"Distinct task {index}\n")
        config = cloud._loads(
            (Path(__file__).parents[1] / "coding_audit/config.json").read_bytes()
        )
        config["cloud"].update(namespace="owner", timeout_seconds=60)
        path = prepared / f"task-{index}" / "plan.json"
        cloud.prepare(task, config, path.parent, repo_root=repo)
        configs.append(config)
        plans.append(path)
        rows.append(
            dict(
                task_id=f"task-{index}",
                plan_path=str(path.relative_to(prepared)),
                plan_sha256=full_run._hash(path)["sha256"],
            )
        )
    source = prepared / "preparation-manifest.json"
    source.write_bytes(
        cloud._json_bytes({"tasks": rows, "evaluation_protocol": "test"})
    )
    ledger = tmp_path / "local-spend.json"
    spend.init(ledger, "500")
    output = tmp_path / "batch"
    api = Mock()
    now, sleeps, events = [0], [], []
    counts, launched, active = Counter(), [], set()

    def sleep(seconds):
        assert 0 <= seconds <= 30
        sleeps.append(seconds)
        now[0] += seconds

    actual_submit = cloud.submit

    def submit(
        path, *, api, secrets, expected_plan_sha256, spend_ledger, reservation_id
    ):
        assert secrets == SECRETS
        assert full_run._hash(path)["sha256"] == expected_plan_sha256
        durable = cloud._loads((output / "manifest.json").read_bytes())
        assert spend_ledger == ledger
        assert reservation_id == durable["shared_spend"]["reservation_id"]
        index = plans.index(path)
        assert durable["tasks"][index]["status"] == "pending_or_unknown"
        if index:
            assert durable["tasks"][index - 1]["job_id"] == f"job-{index - 1}"
            assert durable["tasks"][index - 1]["status"] == "accepted_unmeasured"
        assert durable["shared_spend"]["status"] == "reserved"
        assert (
            spend.summary(ledger)["reserved_usd_rounded_up"]
            == durable["estimate"]["reservation_usd_rounded_up"]
        )
        job_id = f"job-{index}"
        launched.append(job_id)
        active.add(job_id)
        assert len(active) <= 2
        events.append(("submit", job_id, now[0]))
        plan = cloud._loads(path.read_bytes())
        receipt = dict(
            status="submitted",
            run_id=plan["run_id"],
            job_id=job_id,
            job_url=f"https://huggingface.co/jobs/owner/{job_id}",
            launch_spec=plan["mcp_request"]["args"],
        )
        (path.parent / "submission.json").write_bytes(cloud._json_bytes(receipt))
        return receipt

    def inspect_job(*, job_id, namespace, token):
        assert job_id in launched
        assert namespace == "owner" and token == SECRETS["HF_SANDBOX_TOKEN"]
        counts[job_id] += 1
        stage = "RUNNING" if counts[job_id] == 1 else "COMPLETED"
        if stage == "COMPLETED":
            active.remove(job_id)
        events.append(("inspect", job_id, now[0]))
        return SimpleNamespace(
            id=job_id,
            docker_image=cloud.WORKER_IMAGE,
            owner=SimpleNamespace(name="owner"),
            status=SimpleNamespace(stage=stage),
        )

    launch = Mock(side_effect=submit)
    api.inspect_job.side_effect = inspect_job
    monkeypatch.setattr(cloud, "submit", launch)
    return SimpleNamespace(
        source=source,
        output=output,
        ledger=ledger,
        configs=configs,
        plans=plans,
        rows=rows,
        api=api,
        actual_submit=actual_submit,
        launch=launch,
        now=now,
        sleeps=sleeps,
        events=events,
        arguments=dict(
            prepared_manifest=source,
            output=output,
            spend_ledger=ledger,
            reservation_id="cohort-v1",
            budget_usd="500",
            api=api,
            secrets=SECRETS,
            sleep=sleep,
            clock=lambda: now[0],
        ),
    )


def saved(cohort):
    return cloud._loads((cohort.output / "manifest.json").read_bytes())


def rewrite_manifest(cohort):
    cohort.source.write_bytes(cloud._json_bytes({"tasks": cohort.rows}))


def test_prepare_only_has_no_credential_network_or_ledger_io(cohort, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("No credentials/ledger IO"))
    monkeypatch.setattr(cloud, "_runtime_secrets", forbidden)
    monkeypatch.setattr(spend, "reserve", forbidden)
    cohort.ledger.unlink()
    result = prepared_batch.run(**cohort.arguments)
    assert result == saved(cohort)
    assert result["status"] == "prepared"
    assert result["accepted"] is result["results_collected"] is False
    assert result["estimate"] == full_run.estimate(cohort.configs)
    assert result["prepared_manifest"] == full_run._hash(cohort.source)
    assert result["shared_spend"]["status"] == "not_reserved"
    assert not cohort.ledger.exists()
    cohort.launch.assert_not_called()
    cohort.api.inspect_job.assert_not_called()
    forbidden.assert_not_called()


def test_exact_aggregate_is_reserved_first_and_only_two_controllers_run(cohort):
    result = prepared_batch.run(**cohort.arguments, submit=True)
    assert result == saved(cohort)
    assert result["status"] == "controllers_completed_unmeasured"
    assert result["accepted"] is result["results_collected"] is False
    assert all(
        task["status"] == "accepted_unmeasured"
        and task["controller_status"] == "COMPLETED"
        and task["accepted"] is False
        for task in result["tasks"]
    )
    assert cohort.sleeps == [30, 30]
    assert cohort.events == [
        ("submit", "job-0", 0),
        ("submit", "job-1", 0),
        ("inspect", "job-0", 0),
        ("inspect", "job-1", 0),
        ("inspect", "job-0", 30),
        ("inspect", "job-1", 30),
        ("submit", "job-2", 30),
        ("submit", "job-3", 30),
        ("inspect", "job-2", 30),
        ("inspect", "job-3", 30),
        ("inspect", "job-2", 60),
        ("inspect", "job-3", 60),
    ]
    fingerprint = hashlib.sha256(
        "\0".join(task["bundle_sha256"] for task in result["tasks"]).encode()
    ).hexdigest()[:16]
    assert result["shared_spend"]["reservation_id"] == "cohort-v1:" + fingerprint
    ledger = json.loads(cohort.ledger.read_text())
    assert (
        ledger["reservations"]["cohort-v1:" + fingerprint]["reserved_usd"]
        == result["estimate"]["reservation_usd_rounded_up"]
    )
    text = (cohort.output / "manifest.json").read_text()
    assert all(
        value not in text
        for value in (*SECRETS.values(), "launch_spec", "command", "coding-config")
    )


@pytest.mark.parametrize("global_cap", [False, True])
def test_aggregate_caps_stop_all_submissions(cohort, monkeypatch, global_cap):
    if global_cap:
        spend.reserve(cohort.ledger, "prior", "500", "Prior unresolved work")
    else:
        cohort.arguments["budget_usd"] = full_run.estimate(cohort.configs[:3])[
            "reservation_usd_rounded_up"
        ]
    credentials = Mock(side_effect=AssertionError("No token IO before budget gate"))
    monkeypatch.setattr(cloud, "_runtime_secrets", credentials)
    with pytest.raises(
        RuntimeError, match="during reserve" if global_cap else "during budget"
    ):
        prepared_batch.run(**cohort.arguments, submit=True)
    assert saved(cohort)["status"] == "failed"
    cohort.launch.assert_not_called()
    cohort.api.inspect_job.assert_not_called()
    credentials.assert_not_called()


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "bad-json",
        "duplicate-key",
        "changed-plan",
        "bad-bundle",
        "duplicate-task",
        "duplicate-run",
        "absolute",
        "escape",
        "symlink",
        "receipt",
        "dangling-receipt",
    ],
)
def test_invalid_cohort_stops_before_spend_or_credentials(cohort, monkeypatch, defect):
    if defect == "empty":
        cohort.rows.clear()
    elif defect == "changed-plan":
        cohort.plans[-1].write_text("changed")
    elif defect == "bad-bundle":
        (cohort.plans[-1].parent / "bundle.zip").write_bytes(b"changed")
    elif defect == "duplicate-task":
        cohort.rows[-1]["task_id"] = cohort.rows[0]["task_id"]
    elif defect == "duplicate-run":
        shutil.rmtree(cohort.plans[-1].parent)
        shutil.copytree(cohort.plans[0].parent, cohort.plans[-1].parent)
        cohort.rows[-1]["plan_sha256"] = cohort.rows[0]["plan_sha256"]
    elif defect == "absolute":
        cohort.rows[-1]["plan_path"] = str(cohort.plans[-1])
    elif defect == "escape":
        cohort.rows[-1]["plan_path"] = "../task/plan.json"
    elif defect == "symlink":
        outside = cohort.source.parent.parent / "outside"
        shutil.copytree(cohort.plans[-1].parent, outside)
        shutil.rmtree(cohort.plans[-1].parent)
        cohort.plans[-1].parent.symlink_to(outside, target_is_directory=True)
    elif defect in {"receipt", "dangling-receipt"}:
        receipt = cohort.plans[-1].parent / "submission.json"
        if defect == "receipt":
            receipt.write_text('{"status":"launch_unknown"}')
        else:
            receipt.symlink_to(receipt.parent / "missing")
    rewrite_manifest(cohort)
    if defect == "bad-json":
        cohort.source.write_text("{")
    elif defect == "duplicate-key":
        cohort.source.write_text('{"tasks":[],"tasks":[]}')
    forbidden = Mock(side_effect=AssertionError("No paid calls/credentials"))
    monkeypatch.setattr(spend, "reserve", forbidden)
    monkeypatch.setattr(cloud, "_runtime_secrets", forbidden)
    with pytest.raises(RuntimeError, match="during plans"):
        prepared_batch.run(**cohort.arguments, submit=True)
    forbidden.assert_not_called()
    cohort.launch.assert_not_called()


@pytest.mark.parametrize("which", ["plan", "manifest"])
def test_change_after_estimate_is_caught_before_reservation(cohort, monkeypatch, which):
    estimate = full_run.estimate

    def change(configs):
        result = estimate(configs)
        (cohort.source if which == "manifest" else cohort.plans[-1]).write_text(
            "changed"
        )
        return result

    monkeypatch.setattr(full_run, "estimate", change)
    reserve = Mock(side_effect=AssertionError("No reservation"))
    monkeypatch.setattr(spend, "reserve", reserve)
    with pytest.raises(RuntimeError, match="during budget"):
        prepared_batch.run(**cohort.arguments, submit=True)
    reserve.assert_not_called()
    cohort.launch.assert_not_called()


@pytest.mark.parametrize("stage", ["ERROR", "CANCELED", "DELETED", "NEW_STATE"])
def test_terminal_failure_and_unknown_stop_further_launches(cohort, stage):
    cohort.api.inspect_job.side_effect = lambda **kw: SimpleNamespace(
        id=kw["job_id"],
        docker_image=cloud.WORKER_IMAGE,
        owner=SimpleNamespace(name="owner"),
        status=SimpleNamespace(stage=stage),
    )
    with pytest.raises(RuntimeError, match="during poll"):
        prepared_batch.run(**cohort.arguments, submit=True)
    result = saved(cohort)
    assert cohort.launch.call_count == 2
    assert result["tasks"][0]["controller_status"] == (
        stage if stage in prepared_batch.STAGES else "UNKNOWN"
    )
    assert [task.get("job_id") for task in result["tasks"]] == [
        "job-0",
        "job-1",
        None,
        None,
    ]
    assert all((path.parent / "submission.json").exists() for path in cohort.plans[:2])


@pytest.mark.parametrize("field", ["id", "docker_image", "owner", "api-error"])
def test_inspection_failure_or_wrong_controller_stops(cohort, field):
    inspect = cohort.api.inspect_job.side_effect

    def bad(**kwargs):
        if field == "api-error":
            raise RuntimeError("private-inspection-error")
        job = inspect(**kwargs)
        if field == "owner":
            job.owner.name = "other"
        else:
            setattr(job, field, "wrong")
        return job

    cohort.api.inspect_job.side_effect = bad
    with pytest.raises(RuntimeError, match="during poll") as error:
        prepared_batch.run(**cohort.arguments, submit=True)
    assert cohort.launch.call_count == 2
    assert saved(cohort)["tasks"][0]["controller_status"] == "UNKNOWN"
    assert (
        "private-inspection-error"
        not in str(error.value) + (cohort.output / "manifest.json").read_text()
    )


def test_timeout_uses_hard_controller_cap_plus_startup(cohort):
    cohort.api.inspect_job.side_effect = lambda **kw: SimpleNamespace(
        id=kw["job_id"],
        docker_image=cloud.WORKER_IMAGE,
        owner=SimpleNamespace(name="owner"),
        status=SimpleNamespace(stage="SCHEDULING"),
    )
    with pytest.raises(RuntimeError, match="during poll"):
        prepared_batch.run(**cohort.arguments, submit=True)
    assert cohort.now[0] == 60 + full_run.RATEBOOK["startup_allowance_seconds"]
    assert cohort.sleeps == [30] * 12
    assert cohort.launch.call_count == 2
    assert saved(cohort)["tasks"][0]["controller_status"] == "TIMEOUT"


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_default_namespace_still_checks_owner_from_submission_url(cohort, wrong_owner):
    config = cohort.configs[0]
    config["cloud"].pop("namespace")
    shutil.rmtree(cohort.plans[0].parent)
    root = cohort.source.parent.parent
    cloud.prepare(
        root / "task", config, cohort.plans[0].parent, repo_root=root / "repo"
    )
    cohort.rows[:] = cohort.rows[:1]
    cohort.rows[0]["plan_sha256"] = full_run._hash(cohort.plans[0])["sha256"]
    rewrite_manifest(cohort)

    def inspect_job(*, job_id, namespace, token):
        assert namespace is None and token == SECRETS["HF_SANDBOX_TOKEN"]
        return SimpleNamespace(
            id=job_id,
            docker_image=cloud.WORKER_IMAGE,
            owner=SimpleNamespace(name="other" if wrong_owner else "owner"),
            status=SimpleNamespace(stage="COMPLETED"),
        )

    cohort.api.inspect_job.side_effect = inspect_job
    if wrong_owner:
        with pytest.raises(RuntimeError, match="during poll"):
            prepared_batch.run(**cohort.arguments, submit=True)
    else:
        assert (
            prepared_batch.run(**cohort.arguments, submit=True)["status"]
            == "controllers_completed_unmeasured"
        )


@pytest.mark.parametrize("receipt_kind", ["unknown", "known-id", "partial"])
def test_uncertain_launch_preserves_every_receipt_and_never_retries(
    cohort, receipt_kind
):
    launch = cohort.launch.side_effect

    def fail(path, **kwargs):
        if path != cohort.plans[1]:
            return launch(path, **kwargs)
        receipt = dict(
            status="launch_unknown", run_id=cloud._loads(path.read_bytes())["run_id"]
        )
        if receipt_kind == "known-id":
            receipt.update(
                job_id="job-uncertain",
                job_url="https://huggingface.co/jobs/owner/job-uncertain",
            )
        (path.parent / "submission.json").write_bytes(
            b"{" if receipt_kind == "partial" else cloud._json_bytes(receipt)
        )
        raise RuntimeError("private-launch-error")

    cohort.launch.side_effect = fail
    with pytest.raises(RuntimeError, match="during submit") as error:
        prepared_batch.run(**cohort.arguments, submit=True)
    result = saved(cohort)
    assert cohort.launch.call_count == 2
    assert result["tasks"][0]["job_id"] == "job-0"
    assert result["tasks"][1]["status"] == "pending_or_unknown"
    if receipt_kind == "known-id":
        assert result["tasks"][1]["job_id"] == "job-uncertain"
    assert (cohort.plans[1].parent / "submission.json").exists()
    assert (
        "private-launch-error"
        not in str(error.value) + (cohort.output / "manifest.json").read_text()
    )
    with pytest.raises(FileExistsError):
        prepared_batch.run(**cohort.arguments, submit=True)
    cohort.arguments["output"] = cohort.output.parent / "different-output"
    with pytest.raises(RuntimeError, match="during plans"):
        prepared_batch.run(**cohort.arguments, submit=True)
    assert cohort.launch.call_count == 2


def test_cli_requires_all_binding_arguments_and_defaults_to_prepare(cohort, capsys):
    arguments = [
        "--prepared-manifest",
        str(cohort.source),
        "--output",
        str(cohort.output),
        "--spend-ledger",
        str(cohort.ledger),
        "--reservation-id",
        "cohort",
        "--budget-usd",
        "500",
    ]
    for index in range(0, len(arguments), 2):
        with pytest.raises(SystemExit):
            prepared_batch.main(arguments[:index] + arguments[index + 2 :])
    assert not cohort.output.exists()
    prepared_batch.main(arguments)
    assert "prepared:" in capsys.readouterr().out
    cohort.launch.assert_not_called()
    with pytest.raises(FileExistsError):
        prepared_batch.main(arguments)


def test_replacement_after_last_budget_check_cannot_launch(cohort, monkeypatch):
    path = cohort.plans[0]
    replacement = cohort.source.parent / "replacement"
    config = dict(cohort.configs[0])
    config["cloud"] = dict(config["cloud"], timeout_seconds=86400)
    cloud.prepare(
        cohort.source.parent.parent / "task",
        config,
        replacement,
        repo_root=cohort.source.parent.parent / "repo",
    )
    original = cloud._runtime_secrets
    calls = 0

    def replace_during_submit(secrets):
        nonlocal calls
        calls += 1
        if calls == 2:
            for name in ("plan.json", "bundle.zip"):
                shutil.copyfile(replacement / name, path.parent / name)
        return original(secrets)

    monkeypatch.setattr(cloud, "_runtime_secrets", replace_during_submit)
    monkeypatch.setattr(cloud, "submit", cohort.actual_submit)
    with pytest.raises(RuntimeError, match="during submit"):
        prepared_batch.run(**cohort.arguments, submit=True)
    assert calls == 2
    cohort.api.run_job.assert_not_called()
    cohort.api.create_commit.assert_not_called()
    assert not (path.parent / "submission.json").exists()
    assert saved(cohort)["shared_spend"]["status"] == "reserved"


@pytest.mark.parametrize("runner", ["prepared", "native"])
def test_initial_parse_swap_cannot_underprice_reviewed_plan(
    cohort, monkeypatch, runner
):
    path = cohort.plans[0]
    cheap = {
        name: (path.parent / name).read_bytes() for name in ("plan.json", "bundle.zip")
    }
    replacement = cohort.source.parent / "replacement"
    config = dict(cohort.configs[0])
    config["cloud"] = dict(config["cloud"], timeout_seconds=86400)
    cloud.prepare(
        cohort.source.parent.parent / "task",
        config,
        replacement,
        repo_root=cohort.source.parent.parent / "repo",
    )
    expensive = {name: (replacement / name).read_bytes() for name in cheap}
    for name, content in expensive.items():
        (path.parent / name).write_bytes(content)
    cohort.rows[:] = cohort.rows[:1]
    cohort.rows[0]["plan_sha256"] = full_run._hash(path)["sha256"]
    rewrite_manifest(cohort)
    original = cloud._read_plan
    swapped = False

    def swap_at_parse(candidate, **kwargs):
        nonlocal swapped
        if Path(candidate).resolve() == path and not swapped:
            swapped = True
            for name, content in cheap.items():
                (path.parent / name).write_bytes(content)
            try:
                return original(candidate, **kwargs)
            finally:
                for name, content in expensive.items():
                    (path.parent / name).write_bytes(content)
        return original(candidate, **kwargs)

    monkeypatch.setattr(cloud, "_read_plan", swap_at_parse)
    cohort.api.dataset_info.return_value = SimpleNamespace(private=True, sha="a" * 40)
    cohort.api.list_repo_files.return_value = []
    cohort.api.create_commit.return_value = SimpleNamespace(oid="b" * 40)
    cohort.api.run_job.return_value = SimpleNamespace(
        id="job-0", url="https://huggingface.co/jobs/owner/job-0"
    )
    cap = full_run.estimate(cohort.configs[:1])["reservation_usd_rounded_up"]
    if runner == "prepared":
        monkeypatch.setattr(cloud, "submit", cohort.actual_submit)

        def invocation():
            return prepared_batch.run(
                **dict(cohort.arguments, budget_usd=cap), submit=True
            )
    else:
        monkeypatch.setattr(
            full_run.real_tasks,
            "_prepare",
            lambda *args, **kwargs: {
                "tasks": [{"task_id": "toy", "bundle_plan": str(path)}]
            },
        )
        monkeypatch.setattr(full_run, "_catalog", lambda *args: {})
        monkeypatch.setattr(
            cloud,
            "submit",
            lambda candidate, **kwargs: cohort.actual_submit(
                candidate, api=cohort.api, secrets=SECRETS, **kwargs
            ),
        )

        def invocation():
            return full_run.run(
                cohort.output,
                cohort.source,
                cohort.source,
                cap,
                panel_path=write_panel(cohort.source.parent),
                submit=True,
                spend_ledger=cohort.ledger,
                reservation_id="native-toy",
            )

    with pytest.raises(RuntimeError, match="during plans"):
        invocation()
    assert swapped
    assert full_run._hash(path)["sha256"] == cohort.rows[0]["plan_sha256"]
    assert spend.summary(cohort.ledger)["reserved_usd_exact"] == "0"
    cohort.api.run_job.assert_not_called()
    cohort.api.create_commit.assert_not_called()


def test_preparation_manifest_digest_binds_the_bytes_parsed(cohort, monkeypatch):
    original = cohort.source.read_bytes()
    replacement = cloud._json_bytes({"tasks": cohort.rows[:1]})
    read_bytes = Path.read_bytes
    changed = False

    def replace_only_while_reading(path):
        nonlocal changed
        if path.resolve() == cohort.source and not changed:
            changed = True
            path.write_bytes(replacement)
            try:
                return read_bytes(path)
            finally:
                path.write_bytes(original)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", replace_only_while_reading)
    with pytest.raises(RuntimeError, match="during budget"):
        prepared_batch.run(**cohort.arguments, submit=True)
    assert changed and read_bytes(cohort.source) == original
    cohort.launch.assert_not_called()
    assert spend.summary(cohort.ledger)["reserved_usd_exact"] == "0"


def test_blocking_inspection_is_interrupted_by_deadline(cohort):
    def blocked(**kwargs):
        time.sleep(5)
        pytest.fail("inspection was not interrupted")

    cohort.api.inspect_job.side_effect = blocked
    start = time.monotonic()
    result = prepared_batch._status(
        {"job_id": "owned"},
        {"image": cloud.WORKER_IMAGE},
        "owner",
        start + 0.05,
        cohort.api,
        SECRETS,
        time.monotonic,
    )
    assert result == "TIMEOUT"
    assert time.monotonic() - start < 1


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/jobs//job-0",
        "https://foreign.invalid/jobs/owner/job-0",
        "https://huggingface.co/jobs/owner/different-job",
    ],
)
def test_malformed_controller_owner_url_never_bypasses_identity(cohort, url):
    submit = cohort.launch.side_effect

    def malformed(*args, **kwargs):
        receipt = submit(*args, **kwargs)
        receipt["job_url"] = url
        return receipt

    cohort.launch.side_effect = malformed
    with pytest.raises(RuntimeError, match="during submit"):
        prepared_batch.run(**cohort.arguments, submit=True)
    cohort.api.inspect_job.assert_not_called()
    assert cohort.launch.call_count == 1
