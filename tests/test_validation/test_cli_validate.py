import subprocess
import sys
from pathlib import Path

from conftest import FIXTURES
from openenv.validation.report import ValidationReport

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
    assert "Verdict: PASS" in result.stdout


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
    assert report.verdict.value == "pass"
    assert any(r.check_id == "static.manifest" for r in report.results)


def test_json_report_is_schema_valid():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build", "--json"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = ValidationReport.model_validate_json(result.stdout)
    assert report.verdict.value == "pass"


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


def test_format_without_a_parser_exits_two():
    result = _validate(
        str(FIXTURES / "harbor_task_min"), "--level", "static", "--skip-build"
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unrecognized" in result.stderr


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
