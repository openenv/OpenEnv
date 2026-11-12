"""Offline checks of the incomplete-evidence harness; no real HF calls."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "bench_incomplete", Path(__file__).with_name("incomplete.py")
)
incomplete = importlib.util.module_from_spec(spec)
spec.loader.exec_module(incomplete)


def test_unexpected_direct_hf_start_is_a_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        incomplete.HFSandboxValidationProvider, "start", lambda self, spec: object()
    )

    row = incomplete.run_case(tmp_path, "hf_missing_runtime_metadata", 17, 0)

    assert row["observations"]["direct_start_gate"] == "UNEXPECTED_START"
    assert row["observations"]["jobs_created"] == 0
    assert row["mismatches"] == [
        {
            "field": "direct_start_gate",
            "expected": "ProviderError",
            "observed": "UNEXPECTED_START",
        }
    ]


@pytest.mark.parametrize("inventory", ["empty", "truncated"])
def test_bundle_requires_a_complete_checksum_inventory(tmp_path, inventory):
    row = incomplete.run_case(tmp_path, "hf_missing_runtime_metadata", 17, 0)
    assert row["integrity"]["all_checksums_match"] is True
    bundle = Path(row["bundle"])
    report = incomplete.ValidationReportV2.model_validate_json(
        (bundle / "report.json").read_text()
    )
    manifest = bundle / "SHA256SUMS"
    entries = manifest.read_text().splitlines(keepends=True)
    manifest.write_text(
        ""
        if inventory == "empty"
        else "".join(line for line in entries if "collector-trace.json" not in line)
    )
    (bundle / "collector-trace.json").write_text("corrupt unlisted trace\n")

    integrity = incomplete.verify_bundle(bundle, report)

    assert all(entry["match"] for entry in integrity["files"])
    assert integrity["all_checksums_match"] is False
