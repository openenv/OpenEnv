# SPDX-License-Identifier: BSD-3-Clause

"""
Validation utilities for multi-mode deployment readiness.

This module provides functions to check if environments are properly
configured for multi-mode deployment (Docker, direct Python, notebooks, clusters).
"""

import ast
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _make_criterion(
    criterion_id: str,
    description: str,
    passed: bool,
    *,
    required: bool = True,
    details: str | None = None,
    expected: Any | None = None,
    actual: Any | None = None,
) -> dict[str, Any]:
    """Create a standard criterion result payload."""
    criterion: dict[str, Any] = {
        "id": criterion_id,
        "description": description,
        "passed": passed,
        "required": required,
    }
    if details is not None:
        criterion["details"] = details
    if expected is not None:
        criterion["expected"] = expected
    if actual is not None:
        criterion["actual"] = actual
    return criterion


def _normalize_runtime_url(base_url: str) -> str:
    """Normalize and validate a runtime target URL."""
    target = base_url.strip()
    if not target:
        raise ValueError("Runtime URL cannot be empty")

    if "://" not in target:
        target = f"http://{target}"

    parsed = urlparse(target)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid runtime URL: {base_url}")

    return target.rstrip("/")


def _runtime_standard_profile(api_version: str) -> str:
    """Resolve the runtime standard profile for an API version."""
    if api_version.startswith("1."):
        return "openenv-http/1.x"
    return "openenv-http/unknown"


def _build_summary(criteria: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact pass/fail summary for a criteria list."""
    total_count = len(criteria)
    passed_count = sum(1 for criterion in criteria if criterion.get("passed", False))
    failed_criteria = [
        criterion.get("id", "unknown")
        for criterion in criteria
        if not criterion.get("passed", False)
    ]
    required_criteria = [
        criterion for criterion in criteria if criterion.get("required", True)
    ]
    required_total_count = len(required_criteria)
    required_passed_count = sum(
        1 for criterion in required_criteria if criterion.get("passed", False)
    )

    return {
        "passed_count": passed_count,
        "total_count": total_count,
        "failed_criteria": failed_criteria,
        "required_passed_count": required_passed_count,
        "required_total_count": required_total_count,
    }


def validate_running_environment(
    base_url: str, timeout_s: float = 5.0
) -> dict[str, Any]:
    """
    Validate a running OpenEnv server against runtime API standards.

    The returned JSON report contains an overall pass/fail result and
    per-criterion outcomes that can be consumed in CI.
    """
    normalized_url = _normalize_runtime_url(base_url)
    criteria: list[dict[str, Any]] = []

    report: dict[str, Any] = {
        "target": normalized_url,
        "validation_type": "running_environment",
        "standard_version": "unknown",
        "standard_profile": "openenv-http/unknown",
        "mode": "unknown",
        "passed": False,
        "summary": {},
        "criteria": criteria,
    }

    openapi_paths: dict[str, Any] = {}
    api_version = "unknown"

    # Criterion: OpenAPI endpoint reachable with a declared version.
    try:
        openapi_response = requests.get(
            f"{normalized_url}/openapi.json", timeout=timeout_s
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "openapi_version_available",
                "GET /openapi.json returns OpenAPI info.version",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "info.version": "string"},
            )
        )
    else:
        try:
            openapi_json = openapi_response.json()
        except ValueError:
            openapi_json = None

        openapi_ok = (
            openapi_response.status_code == 200
            and isinstance(openapi_json, dict)
            and isinstance(openapi_json.get("info"), dict)
            and isinstance(openapi_json["info"].get("version"), str)
        )

        if openapi_ok:
            api_version = str(openapi_json["info"]["version"])
            openapi_paths = openapi_json.get("paths", {})
            criteria.append(
                _make_criterion(
                    "openapi_version_available",
                    "GET /openapi.json returns OpenAPI info.version",
                    True,
                    expected={"status_code": 200, "info.version": "string"},
                    actual={
                        "status_code": openapi_response.status_code,
                        "info.version": api_version,
                    },
                )
            )
        else:
            criteria.append(
                _make_criterion(
                    "openapi_version_available",
                    "GET /openapi.json returns OpenAPI info.version",
                    False,
                    details="Response missing required OpenAPI info.version field",
                    expected={"status_code": 200, "info.version": "string"},
                    actual={
                        "status_code": openapi_response.status_code,
                        "body_type": (
                            type(openapi_json).__name__
                            if openapi_json is not None
                            else "non_json"
                        ),
                    },
                )
            )

    report["standard_version"] = api_version
    report["standard_profile"] = _runtime_standard_profile(api_version)

    # Criterion: Health endpoint.
    try:
        health_response = requests.get(f"{normalized_url}/health", timeout=timeout_s)
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "health_endpoint",
                "GET /health returns healthy status",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "status": "healthy"},
            )
        )
    else:
        try:
            health_json = health_response.json()
        except ValueError:
            health_json = None

        health_ok = (
            health_response.status_code == 200
            and isinstance(health_json, dict)
            and health_json.get("status") == "healthy"
        )
        criteria.append(
            _make_criterion(
                "health_endpoint",
                "GET /health returns healthy status",
                health_ok,
                expected={"status_code": 200, "status": "healthy"},
                actual={
                    "status_code": health_response.status_code,
                    "status": (
                        health_json.get("status")
                        if isinstance(health_json, dict)
                        else None
                    ),
                },
            )
        )

    # Criterion: Metadata endpoint has required fields.
    try:
        metadata_response = requests.get(
            f"{normalized_url}/metadata", timeout=timeout_s
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "metadata_endpoint",
                "GET /metadata returns name and description",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "fields": ["name", "description"]},
            )
        )
    else:
        try:
            metadata_json = metadata_response.json()
        except ValueError:
            metadata_json = None

        metadata_ok = (
            metadata_response.status_code == 200
            and isinstance(metadata_json, dict)
            and isinstance(metadata_json.get("name"), str)
            and isinstance(metadata_json.get("description"), str)
        )
        criteria.append(
            _make_criterion(
                "metadata_endpoint",
                "GET /metadata returns name and description",
                metadata_ok,
                expected={"status_code": 200, "fields": ["name", "description"]},
                actual={
                    "status_code": metadata_response.status_code,
                    "name": (
                        metadata_json.get("name")
                        if isinstance(metadata_json, dict)
                        else None
                    ),
                    "description": (
                        metadata_json.get("description")
                        if isinstance(metadata_json, dict)
                        else None
                    ),
                },
            )
        )

    # Criterion: Schema endpoint returns action/observation/state.
    try:
        schema_response = requests.get(f"{normalized_url}/schema", timeout=timeout_s)
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "schema_endpoint",
                "GET /schema returns action, observation, and state schemas",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={
                    "status_code": 200,
                    "fields": ["action", "observation", "state"],
                },
            )
        )
    else:
        try:
            schema_json = schema_response.json()
        except ValueError:
            schema_json = None

        schema_ok = (
            schema_response.status_code == 200
            and isinstance(schema_json, dict)
            and isinstance(schema_json.get("action"), dict)
            and isinstance(schema_json.get("observation"), dict)
            and isinstance(schema_json.get("state"), dict)
        )
        criteria.append(
            _make_criterion(
                "schema_endpoint",
                "GET /schema returns action, observation, and state schemas",
                schema_ok,
                expected={
                    "status_code": 200,
                    "fields": ["action", "observation", "state"],
                },
                actual={
                    "status_code": schema_response.status_code,
                    "has_action": (
                        isinstance(schema_json.get("action"), dict)
                        if isinstance(schema_json, dict)
                        else False
                    ),
                    "has_observation": (
                        isinstance(schema_json.get("observation"), dict)
                        if isinstance(schema_json, dict)
                        else False
                    ),
                    "has_state": (
                        isinstance(schema_json.get("state"), dict)
                        if isinstance(schema_json, dict)
                        else False
                    ),
                },
            )
        )

    # Criterion: MCP endpoint is reachable.
    try:
        mcp_response = requests.post(
            f"{normalized_url}/mcp", json={}, timeout=timeout_s
        )
    except requests.RequestException as exc:
        criteria.append(
            _make_criterion(
                "mcp_endpoint",
                "POST /mcp is reachable and returns JSON-RPC payload",
                False,
                details=f"Request failed: {type(exc).__name__}: {exc}",
                expected={"status_code": 200, "jsonrpc": "2.0"},
            )
        )
    else:
        try:
            mcp_json = mcp_response.json()
        except ValueError:
            mcp_json = None

        mcp_ok = (
            mcp_response.status_code == 200
            and isinstance(mcp_json, dict)
            and mcp_json.get("jsonrpc") == "2.0"
        )
        criteria.append(
            _make_criterion(
                "mcp_endpoint",
                "POST /mcp is reachable and returns JSON-RPC payload",
                mcp_ok,
                expected={"status_code": 200, "jsonrpc": "2.0"},
                actual={
                    "status_code": mcp_response.status_code,
                    "jsonrpc": (
                        mcp_json.get("jsonrpc") if isinstance(mcp_json, dict) else None
                    ),
                },
            )
        )

    # Criterion: mode endpoint contract consistency via OpenAPI paths.
    if isinstance(openapi_paths, dict) and openapi_paths:
        has_reset = "/reset" in openapi_paths
        has_step = "/step" in openapi_paths
        has_state = "/state" in openapi_paths

        if has_reset:
            report["mode"] = "simulation"
            mode_ok = has_step and has_state
            expected_paths = {"/reset": True, "/step": True, "/state": True}
        else:
            report["mode"] = "production"
            mode_ok = not has_step and not has_state
            expected_paths = {"/reset": False, "/step": False, "/state": False}

        criteria.append(
            _make_criterion(
                "mode_endpoint_consistency",
                "OpenAPI endpoint set matches OpenEnv mode contract",
                mode_ok,
                expected=expected_paths,
                actual={
                    "/reset": has_reset,
                    "/step": has_step,
                    "/state": has_state,
                },
            )
        )
    else:
        criteria.append(
            _make_criterion(
                "mode_endpoint_consistency",
                "OpenAPI endpoint set matches OpenEnv mode contract",
                False,
                details="Cannot determine mode without OpenAPI paths",
                expected={"openapi.paths": "present"},
                actual={"openapi.paths": "missing"},
            )
        )

    report["passed"] = all(
        criterion["passed"] for criterion in criteria if criterion.get("required", True)
    )
    report["summary"] = _build_summary(criteria)
    return report


def _has_main_guard_call(app_content: str) -> bool:
    """Return True when the module calls main() under a __main__ guard."""
    try:
        tree = ast.parse(app_content)
    except SyntaxError:
        return (
            "__name__" in app_content
            and "__main__" in app_content
            and "main(" in app_content
        )

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If) or not _is_main_guard(node.test):
            continue

        if any(_contains_main_call(guarded_node) for guarded_node in node.body):
            return True

    return False


def _is_main_guard(test: ast.expr) -> bool:
    """Return True for `if __name__ == "__main__"` tests."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _contains_main_call(node: ast.AST) -> bool:
    """Return True when an AST node contains a direct `main(...)` call."""
    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "main"
        for candidate in ast.walk(node)
    )


_OPENENV_RUNTIME_DEP_RE = re.compile(r"^openenv(?:\s*(?:$|[<>=!~@;])|\[)")
_LEGACY_OPENENV_CORE_DEP_RE = re.compile(r"^openenv-core(?:\s*(?:$|[<>=!~@;])|\[)")
_OPENENV_DOCKER_INSTALL_RE = re.compile(
    r"(?<![a-z0-9_.-])openenv(?:\s*(?:$|[<>=!~@;])|\[)"
)


def _dockerfile_installs_openenv_runtime(env_path: Path) -> bool:
    """Return True when a Docker deployment installs OpenEnv outside pyproject."""
    for dockerfile_path in (
        env_path / "server" / "Dockerfile",
        env_path / "Dockerfile",
    ):
        if not dockerfile_path.exists():
            continue

        try:
            dockerfile = dockerfile_path.read_text(encoding="utf-8")
        except OSError:
            continue

        for line in dockerfile.splitlines():
            stripped = line.strip().lower()
            if not stripped or stripped.startswith("#"):
                continue
            if _OPENENV_DOCKER_INSTALL_RE.search(stripped):
                return True
            if "openenv-core" in stripped:
                return True

    return False


def validate_multi_mode_deployment(env_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that an environment is ready for multi-mode deployment.

    Checks:
    1. pyproject.toml exists
    2. uv.lock exists
    3. pyproject.toml has [project.scripts] with server entry point
    4. server/app.py has a main() function
    5. Required dependencies are present

    Returns:
        `tuple` of `(is_valid, issues)` where `is_valid` is a `bool` and `issues` is a
        `list` of issue strings found during validation.
    """
    issues = []

    # Check pyproject.toml exists
    pyproject_path = env_path / "pyproject.toml"
    if not pyproject_path.exists():
        issues.append("Missing pyproject.toml")
        return False, issues

    # Check uv.lock exists
    lockfile_path = env_path / "uv.lock"
    if not lockfile_path.exists():
        issues.append("Missing uv.lock - run 'uv lock' to generate it")

    # Parse pyproject.toml
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except Exception as e:
        issues.append(f"Failed to parse pyproject.toml: {e}")
        return False, issues

    # Check [project.scripts] section
    scripts = pyproject.get("project", {}).get("scripts", {})
    if "server" not in scripts:
        issues.append("Missing [project.scripts] server entry point")

    # Check server entry point format
    server_entry = scripts.get("server", "")
    if server_entry and ":main" not in server_entry:
        issues.append(
            f"Server entry point should reference main function, got: {server_entry}"
        )

    # Check required dependencies
    deps = [dep.lower() for dep in pyproject.get("project", {}).get("dependencies", [])]
    has_openenv = any(_OPENENV_RUNTIME_DEP_RE.match(dep) for dep in deps)
    has_legacy_core = any(_LEGACY_OPENENV_CORE_DEP_RE.match(dep) for dep in deps)
    has_dockerfile_core = _dockerfile_installs_openenv_runtime(env_path)

    if not (has_openenv or has_legacy_core or has_dockerfile_core):
        issues.append("Missing required dependency: openenv>=0.2.0")

    # Check server/app.py exists
    server_app = env_path / "server" / "app.py"
    if not server_app.exists():
        issues.append("Missing server/app.py")
    else:
        # Check for main() function (flexible - with or without parameters)
        app_content = server_app.read_text(encoding="utf-8")
        if "def main(" not in app_content:
            issues.append("server/app.py missing main() function")

        # Check if main() is callable
        if not _has_main_guard_call(app_content):
            issues.append(
                "server/app.py main() function not callable (missing if __name__ == '__main__')"
            )

    return len(issues) == 0, issues


def get_deployment_modes(env_path: Path) -> dict[str, bool]:
    """
    Check which deployment modes are supported by the environment.

    Returns:
        `dict` mapping deployment mode names to whether they are supported.
    """
    modes = {
        "docker": False,
        "openenv_serve": False,
        "uv_run": False,
        "python_module": False,
    }

    # Check Docker (Dockerfile may be in server/ or at env root)
    modes["docker"] = (env_path / "server" / "Dockerfile").exists() or (
        env_path / "Dockerfile"
    ).exists()

    # Check multi-mode deployment readiness
    is_valid, _ = validate_multi_mode_deployment(env_path)
    if is_valid:
        modes["openenv_serve"] = True
        modes["uv_run"] = True
        modes["python_module"] = True

    return modes


def format_validation_report(env_name: str, is_valid: bool, issues: list[str]) -> str:
    """
    Format a validation report for display.

    Returns:
        `str`: formatted validation report.
    """
    if is_valid:
        return f"[OK] {env_name}: Ready for multi-mode deployment"

    report = [f"[FAIL] {env_name}: Not ready for multi-mode deployment", ""]
    report.append("Issues found:")
    for issue in issues:
        report.append(f"  - {issue}")

    return "\n".join(report)


def build_local_validation_json_report(
    env_name: str,
    env_path: Path,
    is_valid: bool,
    issues: list[str],
    deployment_modes: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build a JSON report for local environment validation."""
    criteria = [
        _make_criterion(
            "multi_mode_deployment_readiness",
            "Environment structure is ready for multi-mode deployment",
            is_valid,
            details="No issues found" if is_valid else f"{len(issues)} issue(s) found",
            actual={"issues": issues},
        )
    ]

    if deployment_modes:
        for mode, supported in deployment_modes.items():
            criteria.append(
                _make_criterion(
                    f"deployment_mode_{mode}",
                    f"Deployment mode '{mode}' is supported",
                    supported,
                    required=False,
                )
            )

    return {
        "target": str(env_path),
        "environment": env_name,
        "validation_type": "local_environment",
        "standard_version": "local",
        "standard_profile": "openenv-local",
        "passed": is_valid,
        "summary": _build_summary(criteria),
        "criteria": criteria,
        "issues": issues,
        "deployment_modes": deployment_modes or {},
    }
