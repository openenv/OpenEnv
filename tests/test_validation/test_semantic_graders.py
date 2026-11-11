"""Semantic graders compare binding-measured rewards with the declared contract."""

from fractions import Fraction
from pathlib import Path

import pytest
from openenv.validation.graders import Subject
from openenv.validation.graders.semantic import (
    CanaryFloorGrader,
    FloorGapGrader,
    OracleMaxGrader,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.runtime.contracts import OracleEvidence
from openenv.validation.types import CheckStatus

FIXTURE = Path(__file__).parent.parent / "fixtures/validation/served_min_pass"


def subject(evidence, **reward):
    manifest = NormalizedManifest.model_validate_json(
        (FIXTURE / "normalized_manifest.json").read_text()
    )
    manifest = manifest.model_copy(
        update={"reward": manifest.reward.model_copy(update=reward)}
    )
    return Subject(
        root=FIXTURE,
        manifest=manifest,
        image_ref=None,
        running=None,
        outputs_dir=FIXTURE,
        oracle_evidence=evidence,
    )


def evidence(floor="0", oracle=("1", "1"), exits=(0, 0), canaries=()):
    return OracleEvidence(
        floor=Fraction(floor),
        oracle=tuple(Fraction(value) for value in oracle),
        oracle_exit_codes=exits,
        canaries=tuple((name, Fraction(value)) for name, value in canaries),
    )


@pytest.mark.parametrize(
    "oracle,exits,tolerance,status",
    [
        (("1", "1"), (0, 0), 0.0, CheckStatus.PASS),
        (("0.95", "0.95"), (0, 0), 0.05, CheckStatus.PASS),
        (("0.95", "0.95"), (0, 0), 0.0, CheckStatus.FAIL),
        (("1", "1"), (3, 3), 0.0, CheckStatus.FAIL),
    ],
)
def test_oracle_max(oracle, exits, tolerance, status):
    found = OracleMaxGrader().run(
        subject(evidence(oracle=oracle, exits=exits), oracle_tolerance=tolerance)
    )
    assert found.status is status
    assert (found.remediation is None) is (status is CheckStatus.PASS)


@pytest.mark.parametrize(
    "floor,oracle,status",
    [
        ("0", ("1", "1"), CheckStatus.PASS),
        # Doing nothing earns the maximum: the motivating defect class.
        ("1", ("1", "1"), CheckStatus.FAIL),
        # The oracle barely beats doing nothing, though max - floor clears the margin.
        ("0.4", ("0.5", "0.5"), CheckStatus.FAIL),
        # Exact rationals: a tiny floor never rounds into the margin.
        ("1e-300", ("0.5", "0.5"), CheckStatus.FAIL),
    ],
)
def test_floor_gap(floor, oracle, status):
    found = FloorGapGrader().run(subject(evidence(floor=floor, oracle=oracle)))
    assert found.status is status


def test_canaries_must_not_earn_more_than_doing_nothing():
    grader = CanaryFloorGrader()
    held = grader.run(subject(evidence(canaries=[("empty.sh", "0")])))
    assert held.status is CheckStatus.PASS
    leaked = grader.run(subject(evidence(canaries=[("hardcoded.sh", "0.25")])))
    assert leaked.status is CheckStatus.FAIL
    assert "hardcoded.sh" in leaked.evidence[0]


@pytest.mark.parametrize(
    "grader", [OracleMaxGrader(), FloorGapGrader(), CanaryFloorGrader()]
)
def test_missing_evidence_is_an_error(grader):
    assert grader.run(subject(None)).status is CheckStatus.ERROR
    assert grader.depends_on == ("runtime.episode_determinism",)


@pytest.mark.parametrize(
    "reward_range,oracle,tolerance,status",
    [
        ((0, 100), "95", 0.05, CheckStatus.PASS),
        ((0, 100), "94.999999999999999999999", 0.05, CheckStatus.FAIL),
        ((50, 150), "145", 0.05, CheckStatus.PASS),
        ((0, 0.01), "0.009", 0.05, CheckStatus.FAIL),
    ],
)
def test_oracle_tolerance_is_a_fraction_of_range(
    reward_range, oracle, tolerance, status
):
    found = OracleMaxGrader().run(
        subject(
            evidence(oracle=(oracle, oracle)),
            range=reward_range,
            oracle_tolerance=tolerance,
        )
    )
    assert found.status is status
    assert found.measured["oracle_tolerance"] == tolerance


@pytest.mark.parametrize(
    "floor,oracle,status",
    [
        ("90", "100", CheckStatus.PASS),
        ("99.9", "100", CheckStatus.FAIL),
        ("90", "99", CheckStatus.FAIL),
        ("90.000000000000000000001", "100", CheckStatus.FAIL),
    ],
)
def test_floor_margin_is_a_fraction_of_range(floor, oracle, status):
    found = FloorGapGrader().run(
        subject(
            evidence(floor=floor, oracle=(oracle, oracle)),
            range=(0, 100),
            floor_margin=0.1,
        )
    )
    assert found.status is status
    assert found.measured["declared_margin"] == 0.1
