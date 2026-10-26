# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from openenv.cli.__main__ import app
from openenv.cli._validation import (
    validate_multi_mode_deployment,
    validate_running_environment,
)
from openenv.validation import ValidationReport, write_report
from typer.testing import CliRunner


runner = CliRunner()


class _MockResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


def _write_minimal_valid_env(
    env_dir: Path,
    *,
    main_signature: str = "def main():",
    main_invocation: str = "main()",
) -> None:
    (env_dir / "server").mkdir(parents=True)

    (env_dir / "openenv.yaml").write_text(
        "spec_version: 1\nname: test_env\ntype: space\nruntime: fastapi\napp: server.app:app\nport: 8000\n"
    )
    (env_dir / "uv.lock").write_text("")
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["openenv>=0.2.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )
    (env_dir / "server" / "app.py").write_text(
        f"{main_signature}\n    return None\n\nif __name__ == '__main__':\n    {main_invocation}\n"
    )


def test_validate_running_environment_success() -> None:
    def _fake_get(url: str, timeout: float) -> _MockResponse:
        if url.endswith("/openapi.json"):
            return _MockResponse(
                200,
                {
                    "info": {"version": "1.0.0"},
                    "paths": {
                        "/health": {},
                        "/metadata": {},
                        "/schema": {},
                        "/mcp": {},
                        "/reset": {},
                        "/step": {},
                        "/state": {},
                    },
                },
            )
        if url.endswith("/health"):
            return _MockResponse(200, {"status": "healthy"})
        if url.endswith("/metadata"):
            return _MockResponse(200, {"name": "EchoEnv", "description": "Echo env"})
        if url.endswith("/schema"):
            return _MockResponse(
                200,
                {"action": {"type": "object"}, "observation": {}, "state": {}},
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def _fake_post(url: str, json: dict, timeout: float) -> _MockResponse:
        if url.endswith("/mcp"):
            return _MockResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
            )
        raise AssertionError(f"Unexpected POST url: {url}")

    with patch("openenv.cli._validation.requests.get", side_effect=_fake_get):
        with patch("openenv.cli._validation.requests.post", side_effect=_fake_post):
            report = validate_running_environment("http://localhost:8000")

    assert report["passed"] is True
    assert report["standard_version"] == "1.0.0"
    assert report["mode"] == "simulation"
    assert report["validation_type"] == "running_environment"
    assert report["summary"]["passed_count"] == 6
    assert report["summary"]["total_count"] == 6
    assert report["summary"]["failed_criteria"] == []


def test_validate_running_environment_failure() -> None:
    def _fake_get(url: str, timeout: float) -> _MockResponse:
        if url.endswith("/openapi.json"):
            return _MockResponse(
                200,
                {
                    "info": {"version": "1.0.0"},
                    "paths": {
                        "/health": {},
                        "/metadata": {},
                        "/schema": {},
                        "/mcp": {},
                    },
                },
            )
        if url.endswith("/health"):
            return _MockResponse(200, {"status": "healthy"})
        if url.endswith("/metadata"):
            return _MockResponse(500, {"detail": "boom"})
        if url.endswith("/schema"):
            return _MockResponse(
                200,
                {"action": {"type": "object"}, "observation": {}, "state": {}},
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def _fake_post(url: str, json: dict, timeout: float) -> _MockResponse:
        if url.endswith("/mcp"):
            return _MockResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
            )
        raise AssertionError(f"Unexpected POST url: {url}")

    with patch("openenv.cli._validation.requests.get", side_effect=_fake_get):
        with patch("openenv.cli._validation.requests.post", side_effect=_fake_post):
            report = validate_running_environment("http://localhost:8000")

    assert report["passed"] is False
    metadata_checks = [c for c in report["criteria"] if c["id"] == "metadata_endpoint"]
    assert metadata_checks
    assert metadata_checks[0]["passed"] is False
    assert report["summary"]["passed_count"] == 5
    assert report["summary"]["total_count"] == 6
    assert report["summary"]["failed_criteria"] == ["metadata_endpoint"]


def test_validate_command_runtime_target_outputs_json() -> None:
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "passed": True,
        "criteria": [],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ) as mock_validate:
        result = runner.invoke(app, ["validate", "https://example.com", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == mock_report
    mock_validate.assert_called_once_with("https://example.com", timeout_s=5.0)


def test_validate_command_runtime_target_without_json_outputs_human_readable() -> None:
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "standard_profile": "openenv-http/1.x",
        "mode": "simulation",
        "passed": True,
        "criteria": [
            {
                "id": "health_endpoint",
                "description": "GET /health returns healthy status",
                "passed": True,
            }
        ],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ):
        result = runner.invoke(app, ["validate", "https://example.com"])

    assert result.exit_code == 0
    assert "Validation report for https://example.com" in result.output
    assert "PASS  health_endpoint" in result.output
    assert "Verdict: PASS" in result.output


def test_validate_command_runtime_target_failed_criterion_displays_expected_and_actual() -> (
    None
):
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "standard_profile": "openenv-http/1.x",
        "mode": "simulation",
        "passed": False,
        "criteria": [
            {
                "id": "metadata_endpoint",
                "description": "GET /metadata returns name and description",
                "passed": False,
                "details": "Failed to validate metadata schema",
                "expected": {"status_code": 200, "fields": ["name", "description"]},
                "actual": {"status_code": 500, "detail": "Internal error"},
            }
        ],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ):
        result = runner.invoke(app, ["validate", "https://example.com"])

    assert result.exit_code == 1
    assert "FAIL  metadata_endpoint" in result.output
    assert "Failed to validate metadata schema" in result.output
    assert (
        "expected: {'status_code': 200, 'fields': ['name', 'description']}"
        in result.output
    )
    assert "actual: {'status_code': 500, 'detail': 'Internal error'}" in result.output
    assert "Verdict: FAIL" in result.output


def test_validate_command_runtime_target_with_output_writes_file(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "report.json"
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "passed": True,
        "criteria": [],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ):
        result = runner.invoke(
            app, ["validate", "https://example.com", "--output", str(out_file)]
        )

    assert result.exit_code == 0
    assert out_file.exists()
    assert json.loads(out_file.read_text()) == mock_report
    assert "Validation report for https://example.com" in result.output


def test_validate_command_runtime_target_with_output_and_json(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "report.json"
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "passed": True,
        "criteria": [],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "https://example.com",
                "--output",
                str(out_file),
                "--json",
            ],
        )

    assert result.exit_code == 0
    assert out_file.exists()
    assert json.loads(out_file.read_text()) == mock_report
    assert json.loads(result.output) == mock_report


def test_validate_command_runtime_target_output_write_failure(
    tmp_path: Path,
) -> None:
    out_file = tmp_path / "non_existent_dir" / "report.json"
    mock_report = {
        "target": "https://example.com",
        "validation_type": "running_environment",
        "standard_version": "1.0.0",
        "passed": True,
        "criteria": [],
    }

    with patch(
        "openenv.cli.commands.validate.validate_running_environment",
        return_value=mock_report,
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "https://example.com",
                "--output",
                str(out_file),
            ],
        )

    assert result.exit_code == 3
    assert "Internal error:" in result.output


def test_validate_command_local_path_without_validation_block_fails(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openenv.cli",
            "validate",
            str(env_dir),
            "--level",
            "static",
            "--skip-build",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "static.manifest" in result.stdout
    assert "validation" in result.stdout


def test_multi_mode_rejects_environment_package_as_runtime_dependency(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["openenv-echo-env>=0.1.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )

    is_valid, issues = validate_multi_mode_deployment(env_dir)

    assert is_valid is False
    assert "Missing required dependency: openenv>=0.2.0" in issues


def test_multi_mode_accepts_dockerfile_managed_openenv_runtime(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["fastapi>=0.115.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )
    (env_dir / "server" / "Dockerfile").write_text(
        'RUN pip install --no-cache-dir --no-deps "openenv>=0.2.2"\n'
    )

    is_valid, issues = validate_multi_mode_deployment(env_dir)

    assert is_valid is True
    assert issues == []


def test_multi_mode_accepts_main_call_with_arguments(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(
        env_dir,
        main_signature="def main(port: int = 8000):",
        main_invocation="main(port=8000)",
    )

    is_valid, issues = validate_multi_mode_deployment(env_dir)

    assert is_valid is True
    assert not any("main() function not callable" in issue for issue in issues)


def test_multi_mode_rejects_nested_main_guard(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main():\n    return None\n\n"
        "def wrapper():\n"
        "    if __name__ == '__main__':\n"
        "        main()\n"
    )

    is_valid, issues = validate_multi_mode_deployment(env_dir)

    assert is_valid is False
    assert any("main() function not callable" in issue for issue in issues)


def test_multi_mode_accepts_later_top_level_main_guard(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main():\n    return None\n\n"
        "if __name__ == '__main__':\n"
        "    print('starting')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    is_valid, issues = validate_multi_mode_deployment(env_dir)

    assert is_valid is True
    assert issues == []


def test_multi_mode_syntax_error_fallback_requires_dunder_main(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main(:\n    return None\n\nif __name__ ==\n    main(\n"
    )

    is_valid, issues = validate_multi_mode_deployment(env_dir)

    assert is_valid is False
    assert any("main() function not callable" in issue for issue in issues)


def test_validate_command_rejects_mixed_path_and_url(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(
        app,
        ["validate", str(env_dir), "--url", "http://localhost:8000"],
    )

    assert result.exit_code != 0
    assert "Cannot combine a local path argument with --url" in result.output


def test_write_report_validation_report_only() -> None:
    assert write_report.__annotations__["report"] is ValidationReport
    with pytest.raises(AttributeError):
        write_report({"target": "http://example.com"})  # type: ignore[arg-type]


def test_sigterm_during_local_validation_takes_the_interrupt_path(
    tmp_path: Path,
) -> None:
    """SIGTERM during local validation takes the interrupt path."""
    import os
    import signal
    import time

    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    observed = []

    def outside_handler(signum, frame):
        # Catches SIGTERM if the CLI does not handle it.
        observed.append("uncaught SIGTERM")

    def fake_run_validation(*args, **kwargs):
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        except KeyboardInterrupt:
            observed.append("interrupt")
        raise RuntimeError("stop after observing the signal")

    previous = signal.signal(signal.SIGTERM, outside_handler)
    try:
        with patch("openenv.cli.commands.validate.run_validation", fake_run_validation):
            runner.invoke(app, ["validate", str(env_dir), "--level", "runtime"])
        assert observed == ["interrupt"]
        assert signal.getsignal(signal.SIGTERM) is outside_handler
    finally:
        signal.signal(signal.SIGTERM, previous)
