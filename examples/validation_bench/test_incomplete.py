"""Offline checks of the incomplete-evidence harness; no real HF calls."""

import importlib.util
from pathlib import Path


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
