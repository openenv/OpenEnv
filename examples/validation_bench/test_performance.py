"""Offline failure matching with inert worker and Docker command doubles."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "bench_performance", Path(__file__).with_name("performance.py")
)
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


@pytest.mark.parametrize(
    "evidence",
    [
        "build_failure",
        "create_failure",
        "start_failure",
        "missing_log",
        "wrong_log",
        "other_subject",
        "complete",
    ],
)
def test_startup_fault_requires_create_start_and_fixture_log(
    monkeypatch, tmp_path, evidence
):
    row = {
        "id": "startup-failure",
        "run_id": "offline-startup-failure",
        "kind": "failure",
        "arm": "full",
        "fault": "startup_failure",
    }
    args = SimpleNamespace(
        context=performance.ROOT / "tests/fixtures/validation/runtime/served_probe",
        output=tmp_path,
        study=tmp_path / "unused-study.json",
        lab=tmp_path / "unused-lab",
    )

    def command(argv, *args, **kwargs):
        if argv[:2] == ["docker", "build"]:
            return 1, "", "no space left on device"
        assert argv[:2] == ["docker", "ps"]
        return 0, "", ""

    def worker(*popen_args, **popen_kwargs):
        if evidence == "build_failure":
            metrics = performance.run_trial(row, args.context, tmp_path)
            assert any(
                error["phase"] == "build"
                and "no space left on device" in error["message"]
                for error in metrics["errors"]
            )
        else:
            name = "openenv-validation-double-startup"
            metrics = {
                **row,
                "verdict": "fail",
                "checks": {"runtime.startup": "fail"},
                "errors": [],
                "subject_names": [name],
                "commands": [
                    {
                        "argv": ["docker", "create", "--name", name, "image"],
                        "exit_code": 1 if evidence == "create_failure" else 0,
                    },
                    {
                        "argv": ["docker", "start", name],
                        "exit_code": 1 if evidence == "start_failure" else 0,
                    },
                ],
            }
            directory = tmp_path / "runs" / row["id"]
            performance.save(directory / "metrics.json", metrics)
            if evidence != "missing_log":
                log_name = "unrelated-subject" if evidence == "other_subject" else name
                (directory / f"{log_name}.log").write_text(
                    "unrelated startup error"
                    if evidence == "wrong_log"
                    else "RuntimeError: Deliberate startup failure from the validation test fixture\n"
                )
        return SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda **kw: 0)

    monkeypatch.setattr(performance.docker, "_command", command)
    monkeypatch.setattr(performance.subprocess, "Popen", worker)

    result = performance.failure_subprocess(row, args)

    assert result["checks"]["runtime.startup"] == "fail"
    assert result["verdict"] == "fail"
    assert result["parent_cleanup"]["assert_empty"] is True
    assert result["milestone_observed"] is False
    assert result["expected_failure_match"] is (evidence == "complete")
