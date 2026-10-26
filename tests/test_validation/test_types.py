from openenv.validation.signature import WELL_KNOWN_FILES
from openenv.validation.types import (
    CheckStatus,
    Lane,
    Level,
    Severity,
    SignatureKind,
    Verdict,
)


def test_levels_are_ordered_by_cost():
    assert Level.STATIC < Level.RUNTIME < Level.SEMANTIC < Level.STATISTICAL


def test_skip_and_error_are_statuses_advisory_is_a_severity():
    statuses = {s.value for s in CheckStatus}
    severities = {s.value for s in Severity}
    assert "skip" in statuses and "skip" not in severities
    assert "error" in statuses and "error" not in severities
    assert "advisory" in severities and "advisory" not in statuses


def test_verdicts():
    assert {v.value for v in Verdict} == {"pass", "warn", "fail"}


def test_lanes():
    assert {lane.value for lane in Lane} == {"local", "hub"}


def test_well_known_files_lists_implemented_parsers():
    assert WELL_KNOWN_FILES == {SignatureKind.OPENENV_SERVED: "openenv.yaml"}
