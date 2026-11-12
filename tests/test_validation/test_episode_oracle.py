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


@pytest.mark.parametrize("changed_leaf", [False, True])
def test_deep_valid_json_encoding_is_not_episode_state(changed_leaf):
    raw = '{"type":"observation","data":{"c":' + "[" * 500 + "0" + "]" * 500 + "}}"
    after = raw.replace(":", ": ")
    if changed_leaf:
        after = after.replace("0", "1")
    result = grade_isolation(first_step_subject(after, raw))
    assert result.status is (CheckStatus.FAIL if changed_leaf else CheckStatus.PASS), (
        result.evidence
    )


def test_json_beyond_parser_depth_is_explicitly_unproven(monkeypatch):
    def depth_exceeded(text):
        raise RecursionError("decoder nesting limit")

    monkeypatch.setattr(
        "openenv.validation.graders.runtime.isolation.loads", depth_exceeded
    )
    raw = '{"data": [0]}'
    result = grade_isolation(first_step_subject(raw.replace(":", ": "), raw))
    assert result.status is CheckStatus.SKIP
    assert "JSON nesting exceeds the parser depth" in result.evidence[0]


def test_json_beyond_decimal_range_is_unproven():
    raw = '{"data":1e9999999999999999999}'
    result = grade_isolation(first_step_subject(raw.replace(":", ": "), raw))
    assert result.status is CheckStatus.SKIP


UNSUPPORTED_JSON = [
    pytest.param('{"data":1e9999999999999999999}', id="numeric_range"),
    pytest.param('{"data":' + "[" * 10_000 + "0" + "]" * 10_000 + "}", id="depth"),
    pytest.param('{"data":' + "7" * 5_000 + "}", id="integer_digits"),
]


def with_response(rows, index, raw):
    rows = list(rows)
    rows[index] = replace(rows[index], response_json=raw)
    return rows


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
@pytest.mark.parametrize("session", ["reference", "first", "after_reset", "fresh"])
@pytest.mark.parametrize("failed_prefix", [False, True])
def test_supported_reset_leak_survives_later_unsupported_json(
    raw, session, failed_prefix
):
    traces = {
        name: episode() for name in ("reference", "first", "after_reset", "fresh")
    }
    traces["after_reset"] = with_response(
        traces["after_reset"], 0, '{"type":"observation","data":{"c":1}}'
    )
    traces[session] = with_response(traces[session], 4, raw)
    subject = isolation_subject(
        reference=traces["reference"],
        sequential=traces["first"] + traces["after_reset"],
        fresh=traces["fresh"],
    )
    if failed_prefix:
        subject.isolation_evidence = replace(
            subject.isolation_evidence,
            sequential=replace(
                subject.isolation_evidence.sequential,
                exchanges=subject.isolation_evidence.sequential.exchanges[:-1],
                failure_reason="state failed (TimeoutError)",
            ),
        )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.FAIL, result.evidence
    assert "exchange 0 (reset)" in result.evidence[0]


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
def test_supported_reset_leak_survives_a_later_unsupported_fresh_prefix(raw):
    subject = isolation_subject(
        sequential=episode() + with_response(episode(), 0, '{"data":"sticky"}')
    )
    subject.isolation_evidence = replace(
        subject.isolation_evidence,
        fresh=replace(
            evidence(*with_response(episode()[:5], 4, raw)),
            failure_phase="step",
            failure_reason="step JSON exceeds validator support",
        ),
    )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.FAIL, result.evidence
    assert "exchange 0 (reset)" in result.evidence[0]


def test_supported_transport_failed_fresh_prefix_remains_unproven():
    subject = isolation_subject(
        sequential=episode() + with_response(episode(), 0, '{"data":"sticky"}')
    )
    subject.isolation_evidence = replace(
        subject.isolation_evidence,
        fresh=replace(
            evidence(*episode()[:5]), failure_reason="state failed (TimeoutError)"
        ),
    )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.SKIP, result.evidence
    assert "the fresh reference episode failed" in result.evidence[0]


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
@pytest.mark.parametrize("encoding", ["identical", "whitespace", "escaped_key"])
def test_unsupported_json_is_neither_equality_nor_a_leak(raw, encoding):
    after = raw
    if encoding == "whitespace":
        after = raw.replace(":", ": ")
    elif encoding == "escaped_key":
        after = raw.replace('"data"', '"\\u0064ata"')
    result = grade_isolation(first_step_subject(after, raw))
    assert result.status is CheckStatus.SKIP, result.evidence


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
@pytest.mark.parametrize("session", ["reference", "first", "after_reset", "fresh"])
def test_unsupported_value_at_the_only_differing_position_cannot_prove_a_leak(
    raw, session
):
    traces = {
        name: episode() for name in ("reference", "first", "after_reset", "fresh")
    }
    traces["after_reset"] = with_response(traces["after_reset"], 2, '{"data":1}')
    traces[session] = with_response(traces[session], 2, raw)
    result = grade_isolation(
        isolation_subject(
            reference=traces["reference"],
            sequential=traces["first"] + traces["after_reset"],
            fresh=traces["fresh"],
        )
    )
    assert result.status is CheckStatus.SKIP, result.evidence


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
@pytest.mark.parametrize("drift_at", [0, 2])
def test_supported_fresh_instability_still_vetoes_leak_with_unknown_elsewhere(
    raw, drift_at
):
    subject = isolation_subject(
        sequential=episode() + with_response(episode(), 0, '{"data":"sticky"}'),
        fresh=with_response(
            with_response(episode(), drift_at, '{"data":"fresh drift"}'), 4, raw
        ),
    )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.SKIP, result.evidence
    assert "two fresh episodes differ" in result.evidence[0]


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
def test_unknown_json_does_not_turn_missing_trace_coverage_into_a_leak(raw):
    short = with_response(episode()[:-1], 4, raw)
    result = grade_isolation(isolation_subject(sequential=episode() + short))
    assert result.status is CheckStatus.SKIP, result.evidence


def test_supported_complete_trace_length_divergence_still_fails():
    result = grade_isolation(isolation_subject(sequential=episode() + episode()[:-1]))
    assert result.status is CheckStatus.FAIL, result.evidence
    assert "trace length" in result.evidence[0]


@pytest.mark.parametrize("raw", UNSUPPORTED_JSON)
def test_known_fresh_input_drift_is_not_hidden_by_an_unknown_response(raw):
    fresh = with_response(episode(), 4, raw)
    fresh[4] = replace(fresh[4], request_json='{"type":"step","data":{"different":1}}')
    result = grade_isolation(
        isolation_subject(
            sequential=episode() + with_response(episode(), 0, '{"data":"sticky"}'),
            fresh=fresh,
        )
    )
    assert result.status is CheckStatus.SKIP, result.evidence
    assert "two fresh episodes differ at exchange 4 (step)" in result.evidence[0]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"type":"step","data":{"different":1}}', id="changed_input"),
        *UNSUPPORTED_JSON,
    ],
)
@pytest.mark.parametrize("changed_input_at", [0, 2])
def test_leak_needs_supported_equal_input_prefix(raw, changed_input_at):
    after = with_response(episode(), 2, '{"data":"different"}')
    after[changed_input_at] = replace(after[changed_input_at], request_json=raw)
    result = grade_isolation(isolation_subject(sequential=episode() + after))
    assert result.status is CheckStatus.SKIP, result.evidence


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


@pytest.mark.parametrize(
    "private,public,status",
    [
        ("1.0000000000000001", "1.0", CheckStatus.PASS),
        ("1e400", "2e400", CheckStatus.PASS),
        ("1e400", "Infinity", CheckStatus.SKIP),
        ("1", "true", CheckStatus.PASS),
        ("1", "1.0", CheckStatus.PASS),
        ("1.0000000000000001", '"1.0000000000000001"', CheckStatus.PASS),
        ("1.0000000000000001", "1.0000000000000001", CheckStatus.FAIL),
        ("1.00e400", "1.0e400", CheckStatus.FAIL),
    ],
)
def test_oracle_structured_numbers_are_exact_and_typed(
    tmp_path, private, public, status
):
    from openenv.validation.runtime.contracts import WireExchange

    subject = oracle_subject(tmp_path, http=[HttpProbe("/oracle/solve.sh", 404)])
    (tmp_path / "oracle/solve.sh").write_text('{"private_answer":' + private + "}")
    subject.runtime_evidence = replace(
        subject.runtime_evidence,
        exchanges=(
            WireExchange("step", "{}", '{"data": {"private_answer": ' + public + "}}"),
        ),
    )
    result = grade_oracle(subject)
    assert result.status is status, result.evidence


@pytest.mark.parametrize(
    "public,status",
    [
        ('{"data":1234567890123456}', CheckStatus.PASS),
        ('{"data":"1234567890123456"}', CheckStatus.FAIL),
        ("1234567890123456", CheckStatus.PASS),
        ("plain output: 1234567890123456", CheckStatus.FAIL),
    ],
)
def test_string_oracle_is_not_a_json_number(tmp_path, public, status):
    from openenv.validation.runtime.contracts import WireExchange

    subject = oracle_subject(tmp_path, http=[HttpProbe("/oracle/solve.sh", 404)])
    (tmp_path / "oracle/solve.sh").write_text('"1234567890123456"')
    subject.runtime_evidence = replace(
        subject.runtime_evidence, exchanges=(WireExchange("step", "{}", public),)
    )
    result = grade_oracle(subject)
    assert result.status is status, result.evidence


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


def test_json_limit_on_only_one_side_is_unproven():
    result = grade_isolation(first_step_subject('{"data":1e9999999999999999999}'))
    assert result.status is CheckStatus.SKIP
    assert "JSON numeric range exceeds Decimal support" in result.evidence[0]


def test_json_limit_in_one_control_preserves_a_later_supported_leak():
    first = episode()
    first[2] = replace(first[2], response_json='{"data":1e9999999999999999999}')
    result = grade_isolation(isolation_subject(sequential=first + episode(2)))
    assert result.status is CheckStatus.FAIL
    assert "exchange 4 (step)" in result.evidence[0]


def test_json_limit_after_a_confirmed_leak_does_not_hide_the_leak():
    after_reset = episode(2)
    after_reset[4] = replace(
        after_reset[4], response_json='{"data":1e9999999999999999999}'
    )
    result = grade_isolation(isolation_subject(sequential=episode() + after_reset))
    assert result.status is CheckStatus.FAIL
    assert "exchange 2 (step)" in result.evidence[0]


def test_failed_fresh_trace_without_unsupported_values_is_unproven():
    subject = isolation_subject(sequential=episode() + episode(2))
    subject.isolation_evidence = replace(
        subject.isolation_evidence,
        fresh=replace(
            subject.isolation_evidence.fresh,
            failure_phase="state",
            failure_reason="state JSON exceeds validator support (json_depth)",
        ),
    )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.SKIP
    assert "the fresh reference episode failed" in result.evidence[0]


def test_json_limit_in_control_at_the_only_difference_is_unproven():
    first, after_reset = episode(), episode()
    first[2] = replace(first[2], response_json='{"data":1e9999999999999999999}')
    after_reset[2] = replace(after_reset[2], response_json='{"data":3}')
    result = grade_isolation(isolation_subject(sequential=first + after_reset))
    assert result.status is CheckStatus.SKIP
    assert "JSON numeric range exceeds Decimal support" in result.evidence[0]


def test_incomplete_fresh_trace_after_a_confirmed_leak_does_not_hide_the_leak():
    subject = isolation_subject(sequential=episode() + episode(2))
    fresh = episode()[:4]
    fresh[3] = replace(
        fresh[3],
        response_json='{"type":"state","data":'
        + "[" * 10_000
        + "0"
        + "]" * 10_000
        + "}",
    )
    subject.isolation_evidence = replace(
        subject.isolation_evidence,
        fresh=replace(
            evidence(*fresh),
            failure_phase="state",
            failure_reason="state JSON exceeds validator support (json_depth)",
        ),
    )
    result = grade_isolation(subject)
    assert result.status is CheckStatus.FAIL
    assert "exchange 2 (step)" in result.evidence[0]
