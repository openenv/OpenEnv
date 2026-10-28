"""Durable report identities do not replace the source used for validation."""

import hashlib
import json
from pathlib import Path

import pytest
from openenv.validation.runner import run_validation, source_digest
from openenv.validation.types import Level


SOURCE = Path(__file__).parents[1] / "fixtures/validation/runtime/served_probe"


@pytest.mark.parametrize("level", [Level.STATIC, Level.RUNTIME])
def test_report_identity_is_final_before_bundle_serialization(tmp_path, level):
    target = "spaces/openenv-testing/probe@" + "a" * 40
    digest = source_digest(SOURCE)
    report = run_validation(
        SOURCE,
        max_level=level,
        skip_build=True,
        report_target=target,
        artifacts_dir=tmp_path,
    )
    bundled = json.loads((tmp_path / "report.json").read_text())
    assert report.target == target
    assert report.source_digest == digest == source_digest(SOURCE)
    assert bundled == report.model_dump(mode="json")
    for line in (tmp_path / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
