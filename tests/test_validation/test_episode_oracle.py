"""`runtime.episode_isolation` and `runtime.oracle_containment` judge collected evidence."""

import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
from conftest import FIXTURES
from openenv.validation import runner
from openenv.validation.graders.runtime.isolation import EpisodeIsolationGrader
from openenv.validation.graders.runtime.oracle import OracleContainmentGrader
from openenv.validation.runner import run_validation
from openenv.validation.runtime.contracts import HttpProbe, IsolationEvidence
from openenv.validation.types import CheckStatus, Level
from support.runtime import evidence, exchange, FakeRuntimeProvider


def episode(counter_start=0, extra=None):
    rows = [
        exchange("reset", {"type": "reset"}, {"type": "observation", "data": {"c": 0}}),
        exchange("state", {"type": "state"}, {"type": "state", "data": {"n": 0}}),
    ]
    for step in (1, 2):
        data = {"c": counter_start + step}
        if extra:
            data["note"] = extra
        rows.append(
            exchange("step", {"type": "step"}, {"type": "observation", "data": data})
        )
        rows.append(
            exchange("state", {"type": "state"}, {"type": "state", "data": {"n": step}})
        )
    return rows


def isolation_subject(*, reference=None, sequential=None, fresh=None):
    return SimpleNamespace(
        runtime_evidence=evidence(*(reference or episode())),
        isolation_evidence=IsolationEvidence(
            sequential=evidence(*(sequential or episode() + episode())),
            fresh=evidence(*(fresh or episode())),
        ),
    )


def grade_isolation(subject):
    return EpisodeIsolationGrader().run(subject)


def test_reset_episode_matching_a_fresh_one_passes():
    result = grade_isolation(isolation_subject())
    assert result.status is CheckStatus.PASS, result.evidence


def test_sticky_state_after_reset_fails_at_first_divergence():
    result = grade_isolation(isolation_subject(sequential=episode() + episode(2)))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2 (step)" in result.evidence[0]
    assert result.remediation


def test_nondeterministic_subject_is_unproven_not_failed():
    result = grade_isolation(
        isolation_subject(fresh=episode(extra="t=2"), sequential=episode() + episode(5))
    )
    assert result.status is CheckStatus.SKIP
    assert "two fresh episodes differ" in result.evidence[0]


def test_first_episode_variation_on_the_reset_session_is_unproven_not_a_leak():
    # A per-session value: the reset session's episodes agree with each other,
    # but not with the two fresh sessions, so the reset itself leaked nothing.
    varied = episode(extra="session 2")
    result = grade_isolation(isolation_subject(sequential=varied + varied))
    assert result.status is CheckStatus.SKIP
    assert "two fresh episodes differ at exchange 2 (step)" in result.evidence[0]


STEP = '{"type": "observation", "data": {"c": 1}}'


def first_step_subject(after_reset, fresh=STEP):
    """Episode B's first step response as raw text; other episodes send `fresh`."""

    def with_first_step(raw):
        rows = episode()
        rows[2] = replace(rows[2], response_json=raw)
        return rows

    return isolation_subject(
        reference=with_first_step(fresh),
        sequential=with_first_step(fresh) + with_first_step(after_reset),
        fresh=with_first_step(fresh),
    )


@pytest.mark.parametrize(
    "after_reset",
    [
        '{"data": {"c": 1}, "type": "observation"}',
        '{\n  "type" : "observation",\n  "data" : {"c": 1}\n}',
        '{"type": "\\u006fbservation", "data": {"\\u0063": 1}}',
    ],
    ids=["key_order", "whitespace", "escapes"],
)
def test_equivalent_json_encodings_are_the_same_response(after_reset):
    result = grade_isolation(first_step_subject(after_reset))
    assert result.status is CheckStatus.PASS, result.evidence


def test_equivalent_decimal_spellings_are_the_same_response():
    result = grade_isolation(
        first_step_subject(
            '{"type":"observation","data":{"c":1.00e400}}',
            '{"type":"observation","data":{"c":1.0e400}}',
        )
    )
    assert result.status is CheckStatus.PASS


@pytest.mark.parametrize(
    "fresh, after_reset",
    [
        (STEP, '{"type": "observation", "data": {"c": true}}'),
        (STEP, '{"type": "observation", "data": {"c": 1.0}}'),
        (
            '{"type": "observation", "data": {"c": [1, 2]}}',
            '{"type": "observation", "data": {"c": [2, 1]}}',
        ),
        (STEP, '{"type": "observation", "data": {"c": 1}'),
        (
            '{"type":"observation","data":{"c":1e400}}',
            '{"type":"observation","data":{"c":Infinity}}',
        ),
        (
            '{"type":"observation","data":{"c":1.0000000000000001}}',
            '{"type":"observation","data":{"c":1.0}}',
        ),
    ],
    ids=[
        "bool_for_int",
        "float_for_int",
        "array_order",
        "malformed",
        "nonfinite_constant",
        "decimal_precision",
    ],
)
def test_value_type_order_and_malformed_differences_remain(fresh, after_reset):
    result = grade_isolation(first_step_subject(after_reset, fresh))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2 (step)" in result.evidence[0]


def incomplete_subject(*rows, reason):
    """A same-session collection that stopped with `reason` after `rows`."""
    subject = isolation_subject()
    subject.isolation_evidence = replace(
        subject.isolation_evidence,
        sequential=replace(evidence(*rows), failure_reason=reason),
    )
    return subject


@pytest.mark.parametrize(
    "rows, reason",
    [
        (episode()[:2], "step failed (TimeoutError)"),
        (episode(), "reset failed (ConnectionClosedError)"),
        (episode() + episode()[:2], "step failed (ValueError)"),
    ],
    ids=["in_episode_a", "at_second_reset", "in_episode_b"],
)
def test_incomplete_same_session_experiment_is_unproven_not_a_leak(rows, reason):
    # A deadline, the trace size bound or a lost connection records no response.
    result = grade_isolation(incomplete_subject(*rows, reason=reason))
    assert result.status is CheckStatus.SKIP
    assert result.evidence == [f"the same-session episodes did not complete: {reason}"]


@pytest.mark.parametrize(
    "rows, reason, where",
    [
        (episode() + episode(2)[:3], "state failed (TimeoutError)", "exchange 2"),
        (
            episode()
            + [exchange("reset", {"type": "reset"}, {"type": "error", "data": {}})],
            "reset failed (ValueError)",
            "exchange 0",
        ),
        (
            episode()
            + episode()[:2]
            + [replace(episode()[2], response_json='{"type": "observation"')],
            "step failed (JSONDecodeError)",
            "exchange 2",
        ),
    ],
    ids=["leak", "error_response", "malformed_response"],
)
def test_differing_response_before_the_failure_still_fails(rows, reason, where):
    result = grade_isolation(incomplete_subject(*rows, reason=reason))
    assert result.status is CheckStatus.FAIL
    assert f"differs from a fresh one at {where}" in result.evidence[0]


def test_failed_fresh_reference_skips():
    subject = isolation_subject()
    subject.isolation_evidence = replace(
        subject.isolation_evidence,
        fresh=replace(evidence(), failure_reason="reset failed (ValueError)"),
    )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.SKIP
    assert "the fresh reference episode failed" in result.evidence[0]


def test_missing_isolation_evidence_or_second_episode_is_a_skip():
    assert (
        grade_isolation(
            SimpleNamespace(runtime_evidence=None, isolation_evidence=None)
        ).status
        is CheckStatus.SKIP
    )
    result = grade_isolation(isolation_subject(sequential=episode()))
    assert result.status is CheckStatus.SKIP
    assert "no second episode" in result.evidence[0]


CANARY = "openenv-oracle-canary-3f9c1e7a52d4"


def oracle_subject(tmp_path, *, responses=(), http=(), location="oracle/solve.sh"):
    (tmp_path / "oracle").mkdir(exist_ok=True)
    (tmp_path / "oracle" / "solve.sh").write_text(f'#!/bin/sh\necho "{CANARY}"\n')
    rows = [
        exchange("step", {"type": "step"}, {"type": "observation", "data": data})
        for data in responses
    ]
    return SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(
            capabilities=SimpleNamespace(oracle=SimpleNamespace(location=location))
        ),
        runtime_evidence=replace(evidence(*rows), schema_response_json="{}"),
        isolation_evidence=None,
        oracle_http=tuple(http),
    )


def grade_oracle(subject):
    return OracleContainmentGrader().run(subject)


def test_contained_oracle_passes(tmp_path):
    subject = oracle_subject(
        tmp_path,
        responses=[{"c": 1}],
        http=[HttpProbe("/oracle/solve.sh", 404, b'{"detail":"Not Found"}')],
    )
    result = grade_oracle(subject)
    assert result.status is CheckStatus.PASS, result.evidence


def test_oracle_in_an_observation_fails_even_when_json_escaped(tmp_path):
    leaked = {"metadata": {"hint": f'#!/bin/sh\necho "{CANARY}"\n'}}
    result = grade_oracle(oracle_subject(tmp_path, responses=[leaked]))
    assert result.status is CheckStatus.FAIL
    assert result.evidence == [
        "oracle content is readable through the API: step response"
    ]


def test_served_oracle_fails(tmp_path):
    body = f'#!/bin/sh\necho "{CANARY}"\n'.encode()
    result = grade_oracle(
        oracle_subject(tmp_path, http=[HttpProbe("/oracle/solve.sh", 200, body)])
    )
    assert result.status is CheckStatus.FAIL
    assert "GET /oracle/solve.sh" in result.evidence[0]


@pytest.mark.parametrize("location", ["../outside.sh", "oracle/missing.sh"])
def test_unreadable_oracle_is_a_skip(tmp_path, location):
    result = grade_oracle(oracle_subject(tmp_path, location=location))
    assert result.status is CheckStatus.SKIP
    assert "not readable" in result.evidence[0]


def test_oracle_without_distinctive_content_is_a_skip(tmp_path):
    subject = oracle_subject(tmp_path)
    (tmp_path / "oracle" / "solve.sh").write_text("exit 0\n")
    result = grade_oracle(subject)
    assert result.status is CheckStatus.SKIP


def test_runner_runs_both_experiments_and_probes_the_oracle_path(tmp_path, monkeypatch):
    package = tmp_path / "subject"
    shutil.copytree(FIXTURES / "runtime" / "served_probe", package)
    calls = []

    def collect(*args, **kwargs):
        calls.append(kwargs.get("episodes", 1))
        return replace(
            evidence(*(episode() * kwargs.get("episodes", 1))),
            schema_response_json="{}",
        )

    probes = []
    monkeypatch.setattr(runner, "collect_runtime_evidence", collect)
    monkeypatch.setattr(
        runner,
        "_get",
        lambda base_url, location, *bounds: (
            probes.append(location) or HttpProbe("/" + location, 404)
        ),
    )
    report = run_validation(
        package, max_level=Level.RUNTIME, provider=FakeRuntimeProvider()
    )
    results = {r.check_id: r for r in report.results}
    assert calls == [1, 2, 1]
    assert probes == ["oracle/solve.sh"]
    assert results["runtime.episode_isolation"].status is CheckStatus.PASS
    assert results["runtime.oracle_containment"].status is CheckStatus.PASS
