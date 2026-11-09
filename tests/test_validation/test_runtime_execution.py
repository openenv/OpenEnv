"""Runtime scheduling, capability preflight and lifetime regression tests."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from openenv.validation.policy import load_policy, PolicyError
from openenv.validation.providers import CleanupError, ProviderError, StartupError
from openenv.validation.report import CheckResult
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.runtime.artifacts import write_runtime_bundle
from openenv.validation.runtime.collector import (
    RuntimeCollectionInterrupted,
    RuntimeCollectionUnsupported,
)
from openenv.validation.runtime.contracts import (
    HttpProbe,
    ReplayEvidence,
    RuntimeEvidence,
)
from openenv.validation.runtime.replay import REPLAY_BUDGET_SECONDS
from openenv.validation.runtime.scheduler import execute_graders, order_graders
from openenv.validation.types import CheckStatus, Level, ProviderCapability
from support.runtime import evidence, exchange, FakeRuntimeProvider
from test_host_containment import contained
from test_network_policy_grader import evidence as network_evidence
from test_resource_bounds import bounded


FIXTURE = Path(__file__).parents[1] / "fixtures/validation/runtime/served_probe"


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "subject"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


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


def test_runtime_collects_primary_isolation_and_session_replays_then_cleans_up(
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
    # Primary, same-session pair, fresh isolation session, then repeatability controls.
    assert [kwargs.get("episodes", 1) for _, kwargs in calls] == [1, 2, 1, 1, 1]
    assert [args[1].reset.seed for args, _ in calls] == [42, 42, 42, 42, 43]
    assert len({kwargs["validation_token"] for _, kwargs in calls[:3]}) == 1
    assert [kwargs.get("request_telemetry", True) for _, kwargs in calls[:3]] == [
        True,
        False,
        False,
    ]
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

    assert [call["episode_timeout_s"] for call in calls] == [
        REPLAY_BUDGET_SECONDS,
        600.0,
        600.0,
    ]


@pytest.mark.parametrize("capability", ["json_depth", "json_numeric_range"])
def test_initial_unsupported_json_skips_and_keeps_the_received_schema(
    package, monkeypatch, tmp_path, capability
):
    prefix = RuntimeEvidence(
        schema_response_json=(
            '{"observation":{"minimum":1e9999999999999999999}}'
            if capability == "json_numeric_range"
            else '{"observation":' + "[" * 20000 + "0" + "]" * 20000 + "}"
        ),
        failure_phase="schema",
        failure_reason=f"schema JSON exceeds validator support ({capability})",
    )

    def collect(*args, **kwargs):
        raise RuntimeCollectionUnsupported(prefix, capability=capability)

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    provider = FakeRuntimeProvider()
    report = run_validation(
        package,
        max_level=Level.RUNTIME,
        provider=provider,
        artifacts_dir=tmp_path / "bundle",
    )
    startup = next(
        result for result in report.results if result.check_id == "runtime.startup"
    )
    assert startup.status is CheckStatus.SKIP
    assert capability in startup.evidence[0]
    for name in ("reward_well_formed", "observation_schema", "state_contract"):
        assert (
            next(r for r in report.results if r.check_id == f"runtime.{name}").status
            is CheckStatus.SKIP
        )
    assert provider.subject.stopped
    metadata = json.loads((tmp_path / "bundle/collector-evidence.json").read_text())
    assert metadata["schema_response_available"] is True
    assert metadata["schema_response_parse_failed"] is True


@pytest.mark.parametrize("unsupported_call", [2, 3], ids=["sequential", "fresh"])
@pytest.mark.parametrize("capability", ["json_depth", "json_numeric_range"])
@pytest.mark.parametrize("fault", [None, "reward", "observation", "state", "oracle"])
def test_late_json_limit_preserves_primary_grading(
    package, monkeypatch, tmp_path, unsupported_call, capability, fault
):
    original = measured_episode()
    original = replace(
        original,
        schema_response_json=json.dumps(
            {"observation": json.loads(original.observation_schema_json)}
        ),
    )
    primary = original
    if fault:
        index = 1 if fault in {"state", "oracle"} else 0
        rows = list(original.exchanges)
        response = json.loads(rows[index].response_json)
        data = response["data"]
        if fault == "reward":
            data["reward"] = 2
        elif fault == "observation":
            data["observation"]["counter"] = "wrong-type"
        elif fault == "state":
            data["episode_id"] = "wrong-episode"
        else:
            data["private"] = (package / "oracle/solve.sh").read_text()
        rows[index] = replace(rows[index], response_json=json.dumps(response))
        primary = replace(original, exchanges=tuple(rows))
    prefix = replace(
        original,
        exchanges=(original.exchanges * 2)[:6]
        if unsupported_call == 2
        else original.exchanges[:2],
        failure_phase="step",
        failure_reason=f"step JSON exceeds validator support ({capability})",
    )
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == unsupported_call:
            raise RuntimeCollectionUnsupported(prefix, capability=capability)
        if len(calls) == 1:
            return primary
        return replace(
            original, exchanges=original.exchanges * kwargs.get("episodes", 1)
        )

    probes = (HttpProbe("/oracle/solve.sh", 404),)
    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    monkeypatch.setattr("openenv.validation.runner._oracle_probes", lambda *a: probes)
    bundle = tmp_path / "bundle"
    provider = FakeRuntimeProvider()
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    results = {r.check_id: r for r in report.results}
    assert results["runtime.startup"].status is CheckStatus.PASS
    assert [call.get("episodes", 1) for call in calls] == [1, 2, 1]
    for name, fault_name in (
        ("reward_well_formed", "reward"),
        ("observation_schema", "observation"),
        ("state_contract", "state"),
    ):
        expected = CheckStatus.FAIL if fault == fault_name else CheckStatus.PASS
        assert results[f"runtime.{name}"].status is expected
    assert results["runtime.oracle_containment"].status is (
        CheckStatus.FAIL if fault == "oracle" else CheckStatus.SKIP
    )
    assert results["runtime.episode_isolation"].status is CheckStatus.SKIP
    assert report.verdict.value == ("warn" if fault is None else "fail")
    assert provider.subject.stopped
    phase = "sequential" if unsupported_call == 2 else "fresh"
    metadata = json.loads((bundle / f"isolation-{phase}-evidence.json").read_text())
    assert metadata["complete"] is False
    assert metadata["failure_phase"] == "step"
    assert metadata["failure_reason"] == prefix.failure_reason
    trace = json.loads((bundle / f"isolation-{phase}-trace.json").read_text())
    assert len(trace) == len(prefix.exchanges)
    assert len(json.loads((bundle / "collector-trace.json").read_text())) == 4
    assert json.loads((bundle / "oracle-http.json").read_text())[0]["status"] == 404
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is True


@pytest.mark.parametrize("unsupported_call", [2, 3])
@pytest.mark.parametrize(
    "outcome", ["measured", "provider_error", "cleanup_error", "stop_error"]
)
def test_late_json_limit_keeps_measurements_and_error_precedence(
    package, monkeypatch, tmp_path, unsupported_call, outcome
):
    provider = FakeRuntimeProvider(
        capabilities=frozenset(
            {
                ProviderCapability.IMAGE_BUILD,
                ProviderCapability.EXEC,
                ProviderCapability.NETWORK_POLICY,
                ProviderCapability.INSPECTED_ISOLATION,
                ProviderCapability.RESOURCE_ENFORCEMENT,
            }
        )
    )
    original = measured_episode()
    primary = original
    if outcome == "stop_error":
        row = original.exchanges[0]
        data = json.loads(row.response_json)
        data["data"]["reward"] = 2
        primary = replace(
            original,
            exchanges=(
                replace(row, response_json=json.dumps(data)),
                *original.exchanges[1:],
            ),
        )
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == unsupported_call:
            raise RuntimeCollectionUnsupported(
                replace(
                    original,
                    failure_phase="step",
                    failure_reason="step JSON exceeds validator support (json_depth)",
                ),
                capability="json_depth",
            )
        return primary if calls == 1 else original

    measurements = []

    def measure(name, value):
        def run():
            measurements.append(name)
            if name == "resources" and outcome in {"provider_error", "cleanup_error"}:
                error = CleanupError if outcome == "cleanup_error" else ProviderError
                raise error(
                    "resource helper cleanup failed"
                    if outcome == "cleanup_error"
                    else "resource measurement unavailable"
                )
            return value

        return run

    for name, value in (
        ("network", network_evidence("public")),
        ("containment", contained()),
        ("resources", bounded()),
    ):
        monkeypatch.setattr(
            provider.subject, f"measure_{name}", measure(name, value), raising=False
        )

    def stop():
        provider.subject.stopped = True
        if outcome == "stop_error":
            raise RuntimeError("token=must-not-appear")

    monkeypatch.setattr(provider.subject, "stop", stop)
    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    monkeypatch.setattr("openenv.validation.runner._oracle_probes", lambda *a: ())
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    results = {r.check_id: r for r in report.results}
    startup = results["runtime.startup"]
    assert startup.status is (
        CheckStatus.ERROR
        if outcome in {"cleanup_error", "stop_error"}
        else CheckStatus.PASS
    )
    assert measurements == ["network", "containment", "resources"]
    assert calls == 3 and provider.subject.stopped
    if outcome == "cleanup_error":
        assert startup.evidence == ["resource helper cleanup failed"]
        assert results["runtime.reward_well_formed"].status is CheckStatus.SKIP
    else:
        for name in ("network_policy", "host_containment"):
            assert results[f"runtime.{name}"].status is CheckStatus.PASS
        assert results["runtime.resource_bounds"].status is (
            CheckStatus.SKIP if outcome == "provider_error" else CheckStatus.PASS
        )
        assert results["runtime.reward_well_formed"].status is (
            CheckStatus.FAIL if outcome == "stop_error" else CheckStatus.PASS
        )
    if outcome == "stop_error":
        assert startup.evidence == [
            "subject built and reached its control endpoint",
            "subject teardown failed",
        ]
    assert json.loads((bundle / "cleanup.json").read_text())["completed"] is (
        outcome != "stop_error"
    )
    assert "must-not-appear" not in report.model_dump_json()


def test_cancellation_after_recovered_sequential_limit_retains_both_prefixes(
    package, monkeypatch, tmp_path
):
    original = measured_episode()
    unsupported = replace(
        original,
        exchanges=original.exchanges[:2],
        failure_phase="step",
        failure_reason="step JSON exceeds validator support (json_depth)",
    )
    interrupted = replace(
        original,
        exchanges=original.exchanges[:1],
        failure_phase="state",
        failure_reason="state failed (KeyboardInterrupt)",
    )
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeCollectionUnsupported(unsupported, capability="json_depth")
        if calls == 3:
            raise RuntimeCollectionInterrupted(interrupted)
        return original

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    provider = FakeRuntimeProvider()
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    startup = next(r for r in report.results if r.check_id == "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert startup.evidence == ["validation interrupted"]
    assert calls == 3 and provider.subject.stopped
    for phase, prefix in (("sequential", unsupported), ("fresh", interrupted)):
        metadata = json.loads((bundle / f"isolation-{phase}-evidence.json").read_text())
        assert metadata["failure_reason"] == prefix.failure_reason
        assert len(
            json.loads((bundle / f"isolation-{phase}-trace.json").read_text())
        ) == len(prefix.exchanges)


@pytest.mark.parametrize("interrupted_call", [2, 3])
def test_interrupted_isolation_retains_every_completed_session(
    package, monkeypatch, tmp_path, interrupted_call
):
    original = measured_episode()
    prefix = replace(
        original,
        exchanges=original.exchanges[:2],
        failure_phase="step",
        failure_reason="step failed (KeyboardInterrupt)",
    )
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == interrupted_call:
            raise RuntimeCollectionInterrupted(prefix)
        return original

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    bundle = tmp_path / "bundle"
    provider = FakeRuntimeProvider()
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    assert provider.subject.stopped
    startup = next(r for r in report.results if r.check_id == "runtime.startup")
    assert startup.status is CheckStatus.ERROR
    assert len(json.loads((bundle / "collector-trace.json").read_text())) == 4
    sequential = json.loads((bundle / "isolation-sequential-trace.json").read_text())
    fresh = json.loads((bundle / "isolation-fresh-trace.json").read_text())
    assert len(sequential) == (2 if interrupted_call == 2 else 4)
    assert len(fresh) == (0 if interrupted_call == 2 else 2)
    metadata = json.loads((bundle / "isolation-fresh-evidence.json").read_text())
    assert metadata["complete"] is False
    assert metadata["failure_phase"] == (
        "not_started" if interrupted_call == 2 else "step"
    )


def test_interrupted_oracle_collection_retains_completed_probes(
    package, monkeypatch, tmp_path
):
    manifest = package / "openenv.yaml"
    manifest.write_text(
        manifest.read_text().replace("location: oracle/solve.sh", "location: oracle")
    )
    (package / "oracle/00_first.txt").write_text("completed-probe-prefix")
    monkeypatch.setattr(
        "openenv.validation.runner.collect_runtime_evidence",
        lambda *args, **kwargs: measured_episode(),
    )

    def get(base_url, path, *args):
        if path == "oracle/00_first.txt":
            return HttpProbe("/" + path, 403, b"completed-probe-prefix")
        raise KeyboardInterrupt

    monkeypatch.setattr("openenv.validation.runner._get", get)
    provider = FakeRuntimeProvider()
    bundle = tmp_path / "bundle"
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
    )
    assert provider.subject.stopped
    assert (
        next(r for r in report.results if r.check_id == "runtime.startup").status
        is CheckStatus.ERROR
    )
    probes = json.loads((bundle / "oracle-http.json").read_text())
    assert len(probes) == 2
    assert probes[0]["body"] == "completed-probe-prefix"
    assert probes[0]["status"] == 403
    assert probes[1]["path"] == "/oracle/solve.sh"
    assert probes[1]["failure"] == "KeyboardInterrupt"


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
    package, monkeypatch, tmp_path, mutation, baseline_only
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
        return measured_episode()

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
    for name in ("reward_well_formed", "observation_schema", "state_contract"):
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


@pytest.mark.parametrize("unsupported_call", [2, 3])
def test_unsupported_isolation_keeps_startup_and_grades_completed_evidence(
    package, monkeypatch, unsupported_call
):
    partial = replace(
        measured_episode(),
        failure_phase="step",
        failure_reason="step JSON exceeds validator support (json_numeric_range)",
    )
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == unsupported_call:
            raise RuntimeCollectionUnsupported(partial)
        return measured_episode()

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    results = {result.check_id: result for result in report.results}

    assert results["runtime.startup"].status is CheckStatus.PASS
    assert results["runtime.reward_well_formed"].status is CheckStatus.PASS


def test_unsupported_sequential_still_collects_fresh_and_grades_prefix(
    package, monkeypatch
):
    original = measured_episode()
    after_reset = list(original.exchanges)
    after_reset[2] = replace(
        after_reset[2],
        response_json=after_reset[2].response_json.replace(
            '"counter": 1', '"counter": 2'
        ),
    )
    partial = replace(
        original,
        exchanges=original.exchanges + tuple(after_reset),
        failure_phase="state",
        failure_reason="state JSON exceeds validator support (json_depth)",
    )
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeCollectionUnsupported(partial)
        return original

    monkeypatch.setattr("openenv.validation.runner.collect_runtime_evidence", collect)
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    results = {result.check_id: result for result in report.results}

    assert calls == 3
    assert results["runtime.episode_isolation"].status is CheckStatus.FAIL
