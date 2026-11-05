"""Runtime scheduling, capability preflight and lifetime regression tests."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from openenv.validation.policy import load_policy, PolicyError
from openenv.validation.providers import ProviderError, StartupError
from openenv.validation.providers.docker import DockerValidationProvider
from openenv.validation.report import CheckResult
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.collector import RuntimeCollectionInterrupted
from openenv.validation.runtime.contracts import ReplayEvidence, RuntimeEvidence
from openenv.validation.runtime.replay import REPLAY_BUDGET_SECONDS
from openenv.validation.runtime.scheduler import execute_graders, order_graders
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence, exchange, FakeRuntimeProvider


FIXTURE = Path(__file__).parents[1] / "fixtures/validation/runtime/served_probe"


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "subject"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("__pycache__"))
    # Lifecycle tests use the minimal declaration; discovery tests opt in below.
    path = root / "openenv.yaml"
    document = yaml.safe_load(path.read_text())
    document["validation"]["capabilities"].update(
        declared_tools=[], rubric_tree=False, task_api=False
    )
    document["validation"]["capabilities"].pop("declared_task_count", None)
    path.write_text(yaml.safe_dump(document))
    return root


@pytest.fixture
def discovery_package(package):
    path = package / "openenv.yaml"
    document = yaml.safe_load(path.read_text())
    document["validation"]["capabilities"].update(
        task_api=True, declared_task_count={"train": 100}, rubric_tree=True
    )
    path.write_text(yaml.safe_dump(document))
    return package


@pytest.fixture
def baseline_only(monkeypatch):
    """Isolate original-subject failure handling from separately tested replays."""
    monkeypatch.setattr(
        "openenv.validation.runner.collect_replays",
        lambda provider, running, spec, plan, evidence, **kwargs: evidence,
    )


def measured_episode(seed=42):
    rows = []
    for step in (0, 1):
        operation = "reset" if step == 0 else "step"
        rows.append(
            exchange(
                operation,
                {
                    "type": operation,
                    "data": {"episode_id": "validation-probe", "seed": seed}
                    if operation == "reset"
                    else {"increment": 1},
                },
                {
                    "type": "observation",
                    "data": {
                        "observation": {"counter": step},
                        "reward": float(step),
                        "done": bool(step),
                    },
                },
            )
        )
        rows.append(
            exchange(
                "state",
                {"type": "state"},
                {
                    "type": "state",
                    "data": {"episode_id": "validation-probe", "step_count": step},
                },
            )
        )
    result = evidence(
        *rows,
        observation_schema={
            "type": "object",
            "required": ["counter", "reward", "done"],
            "properties": {
                "counter": {"type": "integer"},
                "done": {"type": "boolean"},
                "reward": {"type": "number"},
            },
        },
    )
    return replace(
        result,
        telemetry_json=json.dumps(
            {
                "schema_version": 1,
                "seed": {"requested": True, "accepted": True, "value": seed},
                "trajectory": {
                    "schema_version": 1,
                    "source": "openenv-server",
                    "complete": True,
                    "records": [
                        {
                            "operation": row.operation,
                            "request": json.loads(row.request_json),
                            "response": json.loads(row.response_json),
                        }
                        for row in rows
                    ],
                },
            }
        ),
    )


def discovery_episode():
    from test_runtime_discovery import node, telemetry

    result = measured_episode()
    snapshot = json.loads(result.telemetry_json)
    snapshot.update(telemetry([node(score=1.0)]))
    return replace(
        result,
        tools_json='{"tools":[]}',
        tasks_json=json.dumps(
            {
                "splits": [{"name": "train"}],
                "counts": {"train": 100},
                "previews": {"train": ["task-0", "task-1"]},
            }
        ),
        telemetry_json=json.dumps(snapshot),
    )


def test_runtime_collects_primary_and_session_replays_then_cleans_up(
    package, monkeypatch, tmp_path
):
    provider = FakeRuntimeProvider()
    calls = []

    def collect(*args, **kwargs):
        calls.append((args, kwargs))
        return measured_episode(seed=args[1].reset.seed)

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    monkeypatch.setattr(
        "openenv.validation.runtime.replay.collect_runtime_evidence", collect
    )
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    results = {r.check_id: r.status for r in report.results}
    assert len(calls) == 3
    assert [args[1].reset.seed for args, _ in calls] == [42, 42, 43]
    assert len(provider.builds) == len(provider.launches) == 1
    assert provider.subject.stopped
    assert report.report_schema_version == "2"
    assert report.levels_run == [Level.STATIC, Level.RUNTIME]
    assert report.verdict.value == "warn"
    for name in (
        "startup",
        "reward_well_formed",
        "observation_schema",
        "state_contract",
        "seed_control",
        "trajectory_record",
    ):
        assert results[f"runtime.{name}"] is CheckStatus.PASS
    assert results["runtime.network_policy"] is CheckStatus.SKIP
    assert results["runtime.episode_determinism"] is CheckStatus.SKIP
    assert any(
        "fresh_container" in reason
        for result in report.results
        if result.check_id == "runtime.episode_determinism"
        for reason in result.evidence
    )
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is True
    assert json.loads((bundle / "runtime-plan.json").read_text())["reset"]["seed"] == 42


def test_runtime_collection_caps_declared_episode_timeout_at_total_budget(
    package, monkeypatch, baseline_only
):
    path = package / "openenv.yaml"
    path.write_text(
        path.read_text().replace("episode_timeout_s: 30.0", "episode_timeout_s: 600.0")
    )
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        return measured_episode()

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    run_validation(package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider())

    assert calls[0]["episode_timeout_s"] == REPLAY_BUDGET_SECONDS


def test_skip_build_has_no_provider_side_effects(package):
    provider = FakeRuntimeProvider()
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, skip_build=True
    )
    assert not provider.builds and not provider.launches
    assert report.levels_run == [Level.STATIC]
    assert all(
        r.status is CheckStatus.SKIP
        for r in report.results
        if r.check_id.startswith("runtime.")
    )


@pytest.mark.parametrize(
    "phase",
    [
        "success",
        "build",
        "build_interrupt",
        "start",
        "collection",
        "interrupt",
        "teardown",
    ],
)
def test_image_cleanup_runs_after_containers_even_when_execution_fails(
    package, monkeypatch, tmp_path, phase, baseline_only
):
    provider = DockerValidationProvider()
    subject = FakeRuntimeProvider().subject
    owner = None
    events = []

    def build(root, execution, *, image_owner):
        nonlocal owner
        owner = image_owner
        events.append("build")
        if phase == "build":
            raise StartupError("build failed after image export")
        if phase == "build_interrupt":
            raise KeyboardInterrupt
        return "sha256:" + "a" * 64

    def start(spec):
        events.append("start")
        if phase == "start":
            raise StartupError("startup failed")
        return subject

    def collect(*args, **kwargs):
        if phase == "interrupt":
            raise KeyboardInterrupt
        result = measured_episode()
        if phase == "collection":
            result = replace(result, failure_reason="step failed (TimeoutError)")
        return result

    def cleanup(image_owner, expected_id):
        events.append("image cleanup")
        assert image_owner == owner
        assert (expected_id is None) == (phase in {"build", "build_interrupt"})
        assert subject.stopped or phase in {
            "build",
            "build_interrupt",
            "start",
            "teardown",
        }
        return {"required": True, "completed": True, "image_removed": True}

    if phase == "teardown":

        def failed_stop():
            events.append("stop")
            raise ProviderError("container teardown failed")

        monkeypatch.setattr(subject, "stop", failed_stop)

    monkeypatch.setattr(provider, "build", build)
    monkeypatch.setattr(provider, "start", start)
    monkeypatch.setattr(provider, "cleanup_image", cleanup)
    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    bundle = tmp_path / "bundle"
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=provider,
        cleanup_images=True,
        artifacts_dir=bundle,
    )
    startup = next(r for r in report.results if r.check_id == "runtime.startup")
    expected = (
        CheckStatus.PASS
        if phase == "success"
        else CheckStatus.ERROR
        if phase in {"interrupt", "build_interrupt", "teardown"}
        else CheckStatus.FAIL
    )
    assert startup.status is expected
    assert events[-1] == "image cleanup"
    recorded = json.loads((bundle / "cleanup.json").read_text())
    assert recorded["image"]["completed"]
    if phase == "teardown":
        assert events[-2:] == ["stop", "image cleanup"]
        assert recorded["completed"] is False
    if phase in {"interrupt", "build_interrupt"}:
        assert startup.evidence == ["validation interrupted"]


@pytest.mark.parametrize("phase", ["build", "start"])
def test_image_cleanup_error_preserves_primary_failure(
    package, monkeypatch, tmp_path, phase
):
    provider = DockerValidationProvider()

    def build(*args, **kwargs):
        if phase == "build":
            raise StartupError("original build failure")
        return "sha256:" + "a" * 64

    def start(*args):
        raise StartupError("original start failure")

    def cleanup(*args):
        raise ProviderError("private cleanup detail")

    monkeypatch.setattr(provider, "build", build)
    monkeypatch.setattr(provider, "start", start)
    monkeypatch.setattr(provider, "cleanup_image", cleanup)
    bundle = tmp_path / "bundle"
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=provider,
        cleanup_images=True,
        artifacts_dir=bundle,
    )
    startup = next(r for r in report.results if r.check_id == "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert startup.evidence == [
        f"original {phase} failure",
        "image cleanup failed (ProviderError)",
    ]
    cleanup = json.loads((bundle / "cleanup.json").read_text())
    assert cleanup["completed"] is cleanup["image"]["completed"] is False
    assert (
        "private cleanup detail" not in json.dumps(cleanup) + report.model_dump_json()
    )


def test_image_cleanup_rejects_non_docker_provider_before_build(package):
    provider = FakeRuntimeProvider()
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, cleanup_images=True
    )
    assert not provider.builds
    startup = next(r for r in report.results if r.check_id == "runtime.startup")
    assert startup.status is CheckStatus.SKIP
    assert "Docker provider" in startup.evidence[0]


@pytest.mark.parametrize("change", ["network", "gpu", "build"])
def test_unsupported_capability_is_refused_before_build(package, change):
    import yaml

    source = package / "openenv.yaml"
    data = yaml.safe_load(source.read_text())
    provider = FakeRuntimeProvider()
    if change == "network":
        data["validation"]["network"] = {"mode": "no-network"}
    elif change == "gpu":
        data["validation"]["resources"]["gpus"] = 1
    else:
        provider.capabilities = frozenset()
    source.write_text(yaml.safe_dump(data))
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert not provider.builds and not provider.launches
    assert (
        next(r for r in report.results if r.check_id == "runtime.startup").status
        is CheckStatus.SKIP
    )


@pytest.mark.parametrize("default_provider", [False, True])
def test_declared_credentials_skip_before_provider_access(
    package, monkeypatch, default_provider
):
    import yaml

    source = package / "openenv.yaml"
    data = yaml.safe_load(source.read_text())
    data["validation"]["execution"]["requires_credentials"] = True
    source.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("OPENAI_API_KEY", "host-credential-must-not-be-used")
    provider = FakeRuntimeProvider()

    def unexpected_provider():
        raise AssertionError("credential deferral must precede provider creation")

    monkeypatch.setattr(
        "openenv.validation.providers.docker.DockerValidationProvider",
        unexpected_provider,
    )
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=None if default_provider else provider,
    )

    assert not provider.builds and not provider.launches
    assert report.verdict.value == "warn"
    assert report.levels_run == [Level.STATIC]
    runtime_results = {
        r.check_id: r for r in report.results if r.check_id.startswith("runtime.")
    }
    assert all(r.status is CheckStatus.SKIP for r in runtime_results.values())
    assert runtime_results["runtime.startup"].evidence == [
        "credential_delivery is deferred for this release"
    ]
    for name in ("reward_well_formed", "observation_schema", "state_contract"):
        assert "runtime.startup" in runtime_results[f"runtime.{name}"].evidence[0]
    assert "host-credential-must-not-be-used" not in report.model_dump_json()


def test_credential_deferral_does_not_hide_invalid_runtime_plan(package):
    import yaml

    source = package / "openenv.yaml"
    data = yaml.safe_load(source.read_text())
    data["validation"]["execution"]["requires_credentials"] = True
    source.write_text(yaml.safe_dump(data))
    (package / "validation/runtime.json").write_text('{"actions": []}')
    provider = FakeRuntimeProvider()

    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)

    assert report.verdict.value == "fail"
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.FAIL
    assert "runtime plan schema validation failed" in result.evidence[0]
    assert not provider.builds and not provider.launches


@pytest.mark.parametrize("declare_no_credentials", [False, True])
def test_llm_judge_without_credentials_can_run(
    package, monkeypatch, declare_no_credentials
):
    import yaml

    source = package / "openenv.yaml"
    data = yaml.safe_load(source.read_text())
    data["validation"]["capabilities"]["llm_judged"] = True
    data["validation"]["reward"]["variance_tolerance"] = 0.1
    data["validation"]["judge"] = {"model": "offline-judge", "version": "1"}
    if declare_no_credentials:
        data["validation"]["execution"]["requires_credentials"] = False
    source.write_text(yaml.safe_dump(data))
    provider = FakeRuntimeProvider()
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence",
        lambda *args, **kwargs: measured_episode(),
    )

    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)

    assert len(provider.builds) == len(provider.launches) == 1
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.PASS
    assert provider.subject.stopped


def test_explicit_v1_rejected_before_runtime(package):
    provider = FakeRuntimeProvider()
    with pytest.raises(PolicyError, match="policy v2"):
        run_validation(
            package,
            max_level=Level.RUNTIME,
            policy=load_policy("v1"),
            provider=provider,
        )
    assert not provider.builds


def test_invalid_plan_is_visible_failure(package):
    (package / "validation/runtime.json").write_text('{"actions": []}')
    provider = FakeRuntimeProvider()
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert report.verdict.value == "fail"
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert "runtime plan schema validation failed" in result.evidence[0]
    assert "missing" in result.evidence[0]
    assert not provider.builds


def test_startup_failure_does_not_masquerade_as_skips(package):
    provider = FakeRuntimeProvider()

    def failed_build(*args):
        raise StartupError("subject build failed")

    provider.build = failed_build
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.FAIL
    assert result.evidence == ["subject build failed"]
    assert report.verdict.value == "fail"


def test_provider_failure_diagnostics_are_visible_and_bounded(package):
    provider = FakeRuntimeProvider()

    def failed_build(*args):
        raise ProviderError("provider deadline elapsed: " + "x" * 5000)

    provider.build = failed_build
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.ERROR
    assert result.evidence[0].startswith("provider deadline elapsed: ")
    assert len(result.evidence[0]) == 4096


@pytest.mark.parametrize("mutation", ["content", "symlink", "unreadable"])
def test_source_change_withdraws_dependent_runtime_results(
    package, monkeypatch, tmp_path, mutation, baseline_only, discovery_package
):
    provider = FakeRuntimeProvider()
    original_digest = source_digest(package)

    def unreadable_source(*args):
        raise OSError("private-source-path")

    def collect(*args, **kwargs):
        if mutation == "content":
            (package / "changed.txt").write_text("changed during collection")
        elif mutation == "symlink":
            (package / "changed.txt").symlink_to(tmp_path / "private-source-path")
        else:
            monkeypatch.setattr(
                "openenv.validation.runner.source_digest", unreadable_source
            )
        return discovery_episode()

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    results = {result.check_id: result for result in report.results}
    assert results["runtime.startup"].status is CheckStatus.ERROR
    reason = (
        "package source changed during validation"
        if mutation == "content"
        else "package source could not be verified after validation"
    )
    assert results["runtime.startup"].evidence == [reason]
    for name in (
        "reward_well_formed",
        "observation_schema",
        "state_contract",
        "tool_declaration_accuracy",
        "task_declaration_accuracy",
        "rubric_introspectable",
        "reward_attribution",
    ):
        result = results[f"runtime.{name}"]
        assert result.status is CheckStatus.SKIP
        assert "runtime.startup" in result.evidence[0]
        assert reason in result.evidence[0]
    assert report.verdict.value == "fail"
    assert report.source_digest == original_digest
    assert json.loads((bundle / "report.json").read_text()) == report.model_dump(
        mode="json"
    )
    assert json.loads((bundle / "cleanup.json").read_text()) == {
        "required": True,
        "completed": True,
    }
    assert len(json.loads((bundle / "collector-trace.json").read_text())) == 4
    assert "private-source-path" not in report.model_dump_json()
    assert provider.subject.stopped


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_collector_crash_or_cancel_always_tears_down(package, monkeypatch, failure):
    provider = FakeRuntimeProvider()

    def explode(*args, **kwargs):
        raise failure("token=must-not-appear")

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", explode)
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert provider.subject.stopped
    assert report.verdict.value == "fail"
    assert "must-not-appear" not in report.model_dump_json()


@pytest.mark.parametrize("failure", [StartupError, ProviderError])
def test_teardown_failure_preserves_primary_provider_error(
    package, monkeypatch, failure
):
    provider = FakeRuntimeProvider()

    def failed_inspect():
        raise failure("subject inspection failed")

    def failed_stop():
        raise RuntimeError("token=must-not-appear")

    monkeypatch.setattr(provider.subject, "inspect", failed_inspect)
    monkeypatch.setattr(provider.subject, "stop", failed_stop)
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.ERROR
    assert result.evidence == ["subject inspection failed", "subject teardown failed"]
    assert report.verdict.value == "fail"
    assert "must-not-appear" not in report.model_dump_json()


@pytest.mark.parametrize("collection_state", ["complete", "failed", "partial"])
@pytest.mark.parametrize("teardown_error", [RuntimeError, KeyboardInterrupt])
def test_teardown_failure_preserves_collection_outcome(
    package, monkeypatch, tmp_path, collection_state, teardown_error, baseline_only
):
    provider = FakeRuntimeProvider()
    collected = measured_episode()
    if collection_state == "failed":
        collected = RuntimeEvidence(
            failure_phase="schema", failure_reason="schema request failed"
        )
    elif collection_state == "partial":
        collected = replace(
            collected,
            exchanges=collected.exchanges[:2],
            failure_phase="step",
            failure_reason="step request failed",
        )

    def failed_stop():
        raise teardown_error("token=must-not-appear")

    monkeypatch.setattr(provider.subject, "stop", failed_stop)
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence", lambda *a, **k: collected
    )
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    result = next(r for r in report.results if r.check_id == "runtime.startup")
    assert result.status is CheckStatus.ERROR
    assert result.evidence == [
        collected.failure_reason or "subject built and reached its control endpoint",
        "subject teardown failed",
    ]
    if collection_state == "complete":
        assert result.measured == {
            "provider": provider.name,
            "image_ref": "sha256:" + "a" * 64,
        }
    assert report.verdict.value == "fail"
    assert json.loads((bundle / "cleanup.json").read_text()) == {
        "required": True,
        "completed": False,
        "reason": f"subject teardown failed ({teardown_error.__name__})",
    }
    assert len(json.loads((bundle / "collector-trace.json").read_text())) == len(
        collected.exchanges
    )
    saved_evidence = json.loads((bundle / "collector-evidence.json").read_text())
    assert saved_evidence["failure_reason"] == collected.failure_reason
    assert json.loads((bundle / "report.json").read_text()) == report.model_dump(
        mode="json"
    )
    assert "must-not-appear" not in report.model_dump_json()


def test_bad_static_bounds_do_not_suppress_independent_runtime(
    package, monkeypatch, baseline_only
):
    path = package / "openenv.yaml"
    path.write_text(path.read_text().replace("floor_margin: 0.5", "floor_margin: 0.01"))
    provider = FakeRuntimeProvider()
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence",
        lambda *a, **k: measured_episode(),
    )
    report = run_validation(package, max_level=Level.RUNTIME, provider=provider)
    assert report.results[0].status is CheckStatus.FAIL
    assert (
        next(r for r in report.results if r.check_id == "runtime.state_contract").status
        is CheckStatus.PASS
    )


@pytest.mark.parametrize("interrupted", [False, True])
def test_bundle_cleanup_includes_failed_replay_teardown(
    package, monkeypatch, tmp_path, interrupted
):
    provider = FakeRuntimeProvider()
    combined = replace(
        measured_episode(),
        replays=(
            ReplayEvidence(
                "container",
                RuntimeEvidence(failure_reason="fresh container teardown unconfirmed"),
                cleanup_complete=False,
            ),
        ),
    )

    def replays(*args, **kwargs):
        if interrupted:
            raise RuntimeCollectionInterrupted(combined)
        return combined

    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence",
        lambda *a, **k: measured_episode(),
    )
    monkeypatch.setattr("openenv.validation.runner.collect_replays", replays)
    bundle = tmp_path / "bundle"
    result = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    assert provider.subject.stopped
    assert (
        next(row for row in result.results if row.check_id == "runtime.startup").status
        is CheckStatus.ERROR
    )
    assert json.loads((bundle / "cleanup.json").read_text()) == {
        "required": True,
        "completed": False,
    }
    replay = json.loads((bundle / "replays.json").read_text())["samples"][0]
    assert replay["cleanup_complete"] is False


@pytest.mark.parametrize(
    "fault,failed_check",
    [
        (None, None),
        ("invalid_reward", "runtime.reward_well_formed"),
        ("replay_divergence", "runtime.episode_determinism"),
    ],
)
def test_replay_cleanup_failure_prevents_pass_and_preserves_runtime_findings(
    package, monkeypatch, tmp_path, fault, failed_check
):
    baseline = measured_episode()
    changed_rows = list(baseline.exchanges)
    response = json.loads(changed_rows[2].response_json)
    if fault == "invalid_reward":
        response["data"]["reward"] = 2.0
    elif fault == "replay_divergence":
        response["data"]["observation"]["counter"] = 99
    changed_rows[2] = replace(changed_rows[2], response_json=json.dumps(response))
    telemetry = json.loads(baseline.telemetry_json)
    telemetry["trajectory"]["records"][2]["response"] = response
    changed = replace(
        baseline, exchanges=tuple(changed_rows), telemetry_json=json.dumps(telemetry)
    )
    if fault == "invalid_reward":
        baseline = changed
    combined = replace(
        baseline,
        replays=(
            ReplayEvidence("session", baseline),
            ReplayEvidence("container", changed, cleanup_complete=False),
            ReplayEvidence("seed", measured_episode(seed=43)),
        ),
    )
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence", lambda *a, **k: baseline
    )
    monkeypatch.setattr(
        "openenv.validation.runner.collect_replays", lambda *a, **k: combined
    )
    bundle = tmp_path / "bundle"
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=FakeRuntimeProvider(),
        artifacts_dir=bundle,
    )
    checks = {row.check_id: row for row in report.results}
    assert checks["runtime.startup"].status is CheckStatus.ERROR
    assert "replay subject teardown failed" in checks["runtime.startup"].evidence
    if failed_check:
        assert checks[failed_check].status is CheckStatus.FAIL
    else:
        determinism = checks["runtime.episode_determinism"]
        assert determinism.status is CheckStatus.SKIP
        assert determinism.measured["completed_replays"] == 3
        assert determinism.evidence == ["fresh container cleanup was not confirmed"]
        assert checks["runtime.reward_well_formed"].status is CheckStatus.PASS
    assert checks["runtime.state_contract"].status is CheckStatus.PASS
    assert report.verdict.value == "fail"
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is False


def test_semantic_ceiling_does_not_claim_semantic_execution(package):
    report = run_validation(package, max_level=Level.SEMANTIC, skip_build=True)
    assert report.levels_run == [Level.STATIC]
    assert any(
        r.check_id == "semantic.oracle_max" and r.status is CheckStatus.SKIP
        for r in report.results
    )


def test_runner_checks_empty_tools_and_omits_undeclared_optional_capabilities(
    package, monkeypatch, baseline_only
):
    observed = []

    def collect(*args, **kwargs):
        observed.append(kwargs)
        return replace(measured_episode(), tools_json='{"tools":[]}')

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    result = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    checks = {row.check_id: row for row in result.results}
    assert checks["runtime.tool_declaration_accuracy"].status is CheckStatus.PASS
    assert observed[0]["collect_tools"] is True
    assert observed[0]["collect_tasks"] is False
    for name in (
        "task_declaration_accuracy",
        "rubric_introspectable",
        "reward_attribution",
    ):
        assert "runtime." + name not in checks


def test_runner_collects_and_grades_claimed_discovery_and_rubric_evidence(
    discovery_package, monkeypatch, baseline_only
):
    observed = []

    def collect(*args, **kwargs):
        observed.append(kwargs)
        return discovery_episode()

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    result = run_validation(
        discovery_package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    checks = {row.check_id: row for row in result.results}
    assert observed[0]["collect_tasks"] is True
    for name in (
        "tool_declaration_accuracy",
        "task_declaration_accuracy",
        "rubric_introspectable",
        "reward_attribution",
    ):
        assert checks["runtime." + name].status is CheckStatus.PASS


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("task_api", [False, True])
def test_parsed_declaration_presence_controls_collection_and_findings(
    package, monkeypatch, baseline_only, declared, task_api
):
    path = package / "openenv.yaml"
    document = yaml.safe_load(path.read_text())
    capabilities = document["validation"]["capabilities"]
    capabilities["task_api"] = task_api
    for field, empty in (("declared_tools", []), ("declared_task_count", {})):
        if declared:
            capabilities[field] = empty
        else:
            capabilities.pop(field, None)
    path.write_text(yaml.safe_dump(document))
    observed = []

    def collect(*args, **kwargs):
        observed.append(kwargs)
        # Omitted counts are unknown even when a working API advertises tasks.
        return replace(
            discovery_episode(),
            tools_json='{"tools":[]}',
            tasks_json='{"splits":[],"counts":{},"previews":{}}'
            if declared
            else discovery_episode().tasks_json,
        )

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    result = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    parsed = result.manifest.capabilities.model_fields_set
    assert ("declared_tools" in parsed) is declared
    assert ("declared_task_count" in parsed) is declared
    restored = type(result).model_validate_json(result.model_dump_json())
    restored_fields = restored.manifest.capabilities.model_fields_set
    assert ("declared_tools" in restored_fields) is declared
    assert ("declared_task_count" in restored_fields) is declared
    assert observed[0]["collect_tools"] is declared
    assert observed[0]["collect_tasks"] is (declared or task_api)
    checks = {row.check_id: row for row in result.results}
    assert ("runtime.tool_declaration_accuracy" in checks) is declared
    assert ("runtime.task_declaration_accuracy" in checks) is (declared or task_api)
    if declared:
        assert checks["runtime.tool_declaration_accuracy"].status is CheckStatus.PASS
        assert checks["runtime.task_declaration_accuracy"].status is CheckStatus.PASS
    elif task_api:
        check = checks["runtime.task_declaration_accuracy"]
        assert check.status is CheckStatus.SKIP
        assert check.evidence == ["task counts were not declared"]


@pytest.mark.parametrize("failed_collection", [False, True])
def test_claims_without_evidence_are_incomplete_or_fail_not_pass(
    discovery_package, monkeypatch, baseline_only, failed_collection
):
    reason = "discovery failed (ValueError)" if failed_collection else None
    measured = replace(
        measured_episode(),
        telemetry_json=None,
        telemetry_error=reason,
        tools_error=reason,
        tasks_error=reason,
    )
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence", lambda *a, **k: measured
    )
    result = run_validation(
        discovery_package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    checks = {row.check_id: row for row in result.results}
    for name in (
        "tool_declaration_accuracy",
        "task_declaration_accuracy",
        "rubric_introspectable",
    ):
        assert checks["runtime." + name].status is (
            CheckStatus.FAIL if failed_collection else CheckStatus.SKIP
        )
    assert checks["runtime.reward_attribution"].status is CheckStatus.SKIP


def test_all_claimed_discovery_checks_name_the_missing_startup_dependency(
    discovery_package,
):
    result = run_validation(discovery_package, max_level=Level.RUNTIME, skip_build=True)
    checks = {row.check_id: row for row in result.results}
    for name in (
        "tool_declaration_accuracy",
        "task_declaration_accuracy",
        "rubric_introspectable",
        "reward_attribution",
    ):
        check = checks["runtime." + name]
        assert check.status is CheckStatus.SKIP
        dependencies = "runtime.startup"
        if name == "reward_attribution":
            dependencies += ", runtime.rubric_introspectable"
        assert check.evidence == [f"unmet dependency: {dependencies}"]


def test_task_count_claim_is_checked_even_without_task_api_flag(
    package, monkeypatch, baseline_only
):
    path = package / "openenv.yaml"
    document = yaml.safe_load(path.read_text())
    document["validation"]["capabilities"]["declared_task_count"] = {"train": 2}
    path.write_text(yaml.safe_dump(document))
    observed = []

    def collect(*args, **kwargs):
        observed.append(kwargs)
        return measured_episode()

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    result = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    task_check = next(
        row
        for row in result.results
        if row.check_id == "runtime.task_declaration_accuracy"
    )
    assert task_check.status is CheckStatus.SKIP
    assert "missing prerequisite" in task_check.evidence[0]
    assert observed[0]["collect_tasks"] is True


def grader(check_id, depends_on=(), *, status=CheckStatus.PASS, requires=frozenset()):
    return SimpleNamespace(
        check_id=check_id,
        depends_on=depends_on,
        requires_provider=requires,
        run=lambda _: CheckResult(check_id=check_id, status=status, duration_s=0),
    )


def test_scheduler_orders_dependencies_and_keeps_independent_checks():
    first = grader("runtime.z", status=CheckStatus.FAIL)
    dependent = grader("runtime.a", ("runtime.z",))
    independent = grader("runtime.other")
    assert [g.check_id for g in order_graders([dependent, first])] == [
        "runtime.z",
        "runtime.a",
    ]
    results = {
        r.check_id: r for r in execute_graders([dependent, first, independent], None)
    }
    assert results["runtime.a"].status is CheckStatus.SKIP
    assert "runtime.z" in results["runtime.a"].evidence[0]
    assert results["runtime.other"].status is CheckStatus.PASS


def test_scheduler_detects_cycle_and_missing_prerequisites():
    with pytest.raises(PolicyError, match="cycle"):
        order_graders(
            [grader("runtime.a", ("runtime.b",)), grader("runtime.b", ("runtime.a",))]
        )
    (result,) = execute_graders(
        [
            grader(
                "runtime.a",
                ("runtime.missing",),
                requires=frozenset({ProviderCapability.EXEC}),
            )
        ],
        None,
    )
    assert result.status is CheckStatus.SKIP
    assert "runtime.missing" in result.evidence[0]
    assert "exec" in result.evidence[1]


def test_grader_cannot_substitute_a_different_result_id():
    wrong = grader("runtime.wrong")
    wrong.check_id = "runtime.expected"
    (result,) = execute_graders([wrong], None)
    assert result.check_id == "runtime.expected"
    assert result.status is CheckStatus.ERROR


def test_digest_rejects_symlinks_without_reading_the_target(tmp_path):
    root = tmp_path / "subject"
    root.mkdir()
    (root / "link").symlink_to(tmp_path / "missing-secret")
    with pytest.raises(ValueError, match="symbolic"):
        source_digest(root)


def test_artifacts_encode_invalid_numbers_and_redact_secrets(package, tmp_path):
    report = run_validation(package, max_level=Level.RUNTIME, skip_build=True)
    raw = evidence(exchange("step", {}, {"token": "secret", "reward": float("nan")}))
    write_runtime_bundle(tmp_path / "bundle", report, evidence=raw)
    payload = (tmp_path / "bundle/collector-trace.json").read_text()
    assert "secret" not in payload
    parsed = json.loads(
        payload, parse_constant=lambda x: pytest.fail(f"invalid JSON number {x}")
    )
    assert parsed[0]["response_json"]["reward"] == {"invalid_number": "nan"}


@pytest.mark.parametrize("prefix_length", [0, 1, 4])
def test_collection_failure_is_reported_once_and_dependents_skip(
    package, monkeypatch, prefix_length
):
    collected = replace(
        measured_episode(),
        exchanges=measured_episode().exchanges[:prefix_length],
        failure_phase="step",
        failure_reason="step failed (TimeoutError)",
    )
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence", lambda *a, **k: collected
    )
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    checks = {result.check_id: result for result in report.results}
    assert report.verdict.value == "fail"
    assert checks["runtime.startup"].status is CheckStatus.FAIL
    assert checks["runtime.startup"].evidence == [collected.failure_reason]
    for name in ("reward_well_formed", "observation_schema", "state_contract"):
        assert checks[f"runtime.{name}"].status is CheckStatus.SKIP
        assert checks[f"runtime.{name}"].evidence == [
            "unmet dependencies: runtime.startup"
        ]
