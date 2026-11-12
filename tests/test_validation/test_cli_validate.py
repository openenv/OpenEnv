import subprocess
import sys
from pathlib import Path

from conftest import FIXTURES
from openenv.validation.report import ValidationReport, ValidationReportV2

REPO_ROOT = Path(__file__).parent.parent.parent


def _validate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "openenv.cli", "validate", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_valid_package_exits_zero():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verdict: WARN" in result.stdout


def test_echo_env_exits_zero():
    result = _validate(
        str(REPO_ROOT / "envs" / "echo_env"),
        "--level",
        "static",
        "--skip-build",
        "--json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = ValidationReport.model_validate_json(result.stdout)
    # Static graders other than static.manifest are named SKIPs: WARN, exit 0.
    assert report.verdict.value == "warn"
    assert any(r.check_id == "static.manifest" for r in report.results)


def test_json_report_is_schema_valid():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build", "--json"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = ValidationReport.model_validate_json(result.stdout)
    # Static graders other than static.manifest are named SKIPs: WARN, exit 0.
    assert report.verdict.value == "warn"


def test_output_writes_the_json_report(tmp_path):
    out = tmp_path / "report.json"
    result = _validate(
        str(FIXTURES / "served_min_pass"),
        "--level",
        "static",
        "--skip-build",
        "--output",
        str(out),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ValidationReport.model_validate_json(out.read_text())


def test_output_write_failure_is_an_internal_error(tmp_path):
    out = tmp_path / "missing" / "report.json"
    result = _validate(
        str(FIXTURES / "served_min_pass"),
        "--level",
        "static",
        "--skip-build",
        "--output",
        str(out),
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "Internal error:" in result.stderr


def test_failing_manifest_exits_one():
    result = _validate(
        str(FIXTURES / "broken_manifest"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "static.manifest" in result.stdout


def test_ambiguous_package_exits_two():
    result = _validate(
        str(FIXTURES / "ambiguous_package"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ambiguous" in result.stderr


def test_unrecognized_package_exits_two():
    result = _validate(
        str(FIXTURES / "unrecognized_package"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unrecognized" in result.stderr


def test_harbor_without_trusted_contract_fails_manifest_validation():
    result = _validate(
        str(FIXTURES / "harbor_task_min"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "static.manifest" in result.stdout
    assert "Harbor contract" in result.stdout


def test_nonexistent_path_exits_two(tmp_path):
    result = _validate(str(tmp_path / "nope"))
    assert result.returncode == 2, result.stdout + result.stderr


def test_unknown_level_is_an_internal_error():
    result = _validate(str(FIXTURES / "served_min_pass"), "--level", "cosmic")
    assert result.returncode == 3, result.stdout + result.stderr


def test_unknown_policy_is_an_internal_error():
    result = _validate(str(FIXTURES / "served_min_pass"), "--policy", "nope")
    assert result.returncode == 3, result.stdout + result.stderr


def test_unpinned_judge_fails_static_manifest():
    result = _validate(
        str(FIXTURES / "unpinned_judge"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 1
    assert "judge pin" in result.stdout


def test_served_package_cannot_claim_harbor_execution(tmp_path):
    source = (FIXTURES / "served_min_pass/openenv.yaml").read_text()
    (tmp_path / "openenv.yaml").write_text(
        source + "  execution:\n    kind: harbor_task\n    agent_boundary: process\n"
    )

    result = _validate(str(tmp_path), "--json")

    assert result.returncode == 1, result.stdout + result.stderr
    report = ValidationReportV2.model_validate_json(result.stdout)
    manifest_result = next(r for r in report.results if r.check_id == "static.manifest")
    assert manifest_result.status.value == "fail"
    assert "openenv_ws" in " ".join(manifest_result.evidence)
    assert "Internal error" not in result.stderr
