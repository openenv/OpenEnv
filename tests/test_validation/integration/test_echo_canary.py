"""Run the unchanged reference Echo environment through the installed CLI."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from test_runtime_cli import _invoke_cli


pytestmark = pytest.mark.docker


def test_reference_echo_validates_reset_and_unscored_steps(tmp_path):
    configured = os.environ.get("OPENENV_VALIDATION_ECHO_CONTEXT")
    if not configured:
        if os.environ.get("OPENENV_REQUIRE_DOCKER") == "1":
            pytest.fail("Required Echo canary needs OPENENV_VALIDATION_ECHO_CONTEXT")
        pytest.skip("Run the validation lab to stage the exact-source Echo canary")
    context = Path(configured)
    evidence_root = Path(os.environ["OPENENV_VALIDATION_ARTIFACTS"])
    provenance = json.loads((evidence_root / "echo-canary-source.json").read_text())
    assert provenance["source_directory"] == "envs/echo_env"
    assert provenance["subject_imported_on_host"] is False
    assert provenance["source_hashes"]
    for name, expected in provenance["source_hashes"].items():
        assert (
            hashlib.sha256((context / "echo_env" / name).read_bytes()).hexdigest()
            == expected
        )

    result, report, checks, artifacts = _invoke_cli(context, tmp_path, "echo_canary")
    expected = {
        "runtime.startup": "pass",
        "runtime.state_contract": "pass",
        "runtime.reward_well_formed": "fail",
        "runtime.observation_schema": "pass",
    }
    (evidence_root / "cli/echo_canary/compatibility-findings.json").write_text(
        json.dumps(
            {
                "source_directory": "envs/echo_env",
                "source_digest": report["source_digest"],
                "image_ref": checks["runtime.startup"]["measured"].get("image_ref"),
                "expected": expected,
                "observed": {
                    check_id: {
                        "status": checks[check_id]["status"],
                        "evidence": checks[check_id]["evidence"],
                    }
                    for check_id in expected
                },
                "interpretation": "Echo is valid under the core API, but null step rewards fail Level Two validation.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert report["levels_run"] == [1, 2]
    assert report["manifest"]["name"] == "echo_env"
    assert checks["static.manifest"]["status"] == "pass"
    assert checks["runtime.startup"]["status"] == "pass"
    assert checks["runtime.state_contract"]["status"] == "pass"
    assert artifacts["cleanup.json"]["required"] is True
    trace = artifacts["collector-trace.json"]
    assert [row["operation"] for row in trace] == [
        "reset",
        "state",
        "step",
        "state",
        "step",
        "state",
    ]
    states = [
        row["response_json"]["data"] for row in trace if row["operation"] == "state"
    ]
    assert [state["episode_id"] for state in states] == ["echo-canary"] * 3
    assert [state["step_count"] for state in states] == [0, 1, 2]
    steps = [
        row["response_json"]["data"] for row in trace if row["operation"] == "step"
    ]
    for step, tool_name, message in zip(
        steps,
        ("echo_message", "echo_with_length"),
        ("echo canary first", "echo canary second"),
        strict=True,
    ):
        observation = step["observation"]
        assert observation["tool_name"] == tool_name
        assert observation["error"] is None
        assert message in json.dumps(observation["result"])
        assert step["reward"] is None
        assert step["done"] is False

    assert result.returncode == 1
    assert report["verdict"] == "fail"
    assert checks["runtime.reward_well_formed"]["status"] == "fail"
    assert checks["runtime.observation_schema"]["status"] == "pass"
    assert "tool_name" not in trace[0]["response_json"]["data"]["observation"]
    metadata = artifacts["collector-evidence.json"]
    assert metadata["complete"] is True
    assert metadata["failure_phase"] is None
    assert "tool_name" in metadata["observation_schema"]["required"]
    assert metadata["reset_schema_available"] is True
    assert "tool_name" not in metadata["reset_observation_schema"].get("required", [])
