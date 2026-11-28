# SPDX-License-Identifier: BSD-3-Clause
"""Runtime coverage for the Pathway app's serving contract."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from pathway_analysis_env.models import PathwayState


ENV_DIR = Path(__file__).resolve().parents[2] / "envs" / "pathway_analysis_env"


def test_declared_app_imports_from_environment_root():
    target = yaml.safe_load((ENV_DIR / "openenv.yaml").read_text())["app"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from uvicorn.importer import import_from_string; "
            f"import_from_string({target!r})",
        ],
        cwd=ENV_DIR,
        env={
            **os.environ,
            "ENABLE_WEB_INTERFACE": "false",
            "PYTHONPATH": os.pathsep.join(
                [str(ENV_DIR.parent), str(ENV_DIR.parents[1] / "src")]
            ),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_http_serves_pathway_state(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    environment = importlib.import_module(
        "pathway_analysis_env.server.pathway_environment"
    )
    monkeypatch.setattr(environment, "OUTPUT_TRACE_DIR", tmp_path)
    app_module = importlib.import_module("pathway_analysis_env.server.app")
    with TestClient(app_module.app) as client:
        schema = client.get("/schema")
        state = client.get("/state")
    assert schema.status_code == state.status_code == 200
    assert schema.json()["state"] == PathwayState.model_json_schema()
    assert set(state.json()) == set(PathwayState.model_fields)
    assert state.json()["pipeline_mode"] is True
    assert state.json()["de_run"] is False
