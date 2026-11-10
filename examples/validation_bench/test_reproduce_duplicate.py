"""Offline checks of the duplicate-key defect reproducer."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "bench_reproduce_duplicate",
    Path(__file__).parent / "exp2/reproduce_duplicate.py",
)
reproducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproducer)


@pytest.mark.parametrize(
    "classification, oracle_changed, reproduced",
    [
        ("supported", True, True),
        ("inconclusive", True, False),
        ("unsupported", True, False),
        ("supported", False, False),
    ],
)
def test_defect_requires_supported_replay_and_oracle_change(
    monkeypatch, tmp_path, capsys, classification, oracle_changed, reproduced
):
    row = {
        "replay": {
            "classification": classification,
            "status_mismatches": [
                {
                    "check_id": "runtime.oracle_containment",
                    "original": "fail",
                    "replayed": "pass",
                }
            ]
            if oracle_changed
            else [],
            "limits": [],
        },
        "integrity": {"all_checksums_match": True},
    }
    monkeypatch.setattr(reproducer, "run_case", lambda *args: row)
    monkeypatch.setattr(
        reproducer.sys, "argv", ["reproduce_duplicate.py", "--output", str(tmp_path)]
    )

    assert reproducer.main() == (0 if reproduced else 1)
    assert json.loads(capsys.readouterr().out)["reproduced"] is reproduced
