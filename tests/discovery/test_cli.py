# SPDX-License-Identifier: BSD-3-Clause

import json
import subprocess
from pathlib import Path

import pytest
from openenv.cli.__main__ import app
from typer.testing import CliRunner


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_catalog_build_then_discover_is_a_read_only_metadata_workflow(
    tmp_path: Path, newline: str
):
    repository = tmp_path / "repository"
    environment = repository / "envs" / "echo_env"
    environment.mkdir(parents=True)
    (environment / "openenv.yaml").write_text(
        "name: echo_env\nspec_version: 1\napp: server.app:app\n"
    )
    (environment / "pyproject.toml").write_text(
        '[project]\nname="openenv-echo-env"\ndependencies=["openenv>=0.3.1"]\n'
    )
    (environment / "README.md").write_bytes(
        newline.join(
            [
                "---",
                "description: Echo messages for client smoke testing.",
                "tags: [frontmatter]",
                "---",
                "# Echo",
                "",
            ]
        ).encode()
    )
    (environment / "__init__.py").write_text(
        "raise RuntimeError('discovery must not execute this package')\n"
    )
    for arguments in (
        ["init", "-q"],
        ["add", "."],
        ["commit", "-qm", "Metadata fixture"],
    ):
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Catalog tests",
                "-c",
                "user.email=catalog@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            check=True,
            capture_output=True,
        )
    catalog = tmp_path / "catalog.json"
    runner = CliRunner()
    built = runner.invoke(
        app,
        [
            "catalog",
            "build",
            "--repository",
            str(repository),
            "--repository-uri",
            "https://github.com/example/environments.git",
            "--publisher",
            "example.org",
            "--output",
            str(catalog),
        ],
    )
    assert built.exit_code == 0, built.output
    result = runner.invoke(
        app,
        ["discover", "client smoke test", "--catalog", str(catalog), "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["results"][0]["data"]["source"]["path"] == "envs/echo_env"
    assert data["results"][0]["data"]["license"] == "unknown"
    assert data["results"][0]["tags"] == ["frontmatter"]
    assert data["metadataOnly"] is True


def test_discover_reports_bad_input_instead_of_empty_success(tmp_path: Path):
    source = tmp_path / "bad.json"
    source.write_text('{"entries":[{"identifier":"urn:air:example.org:openenv:echo"}]}')
    result = CliRunner().invoke(
        app, ["discover", "tool test", "--catalog", str(source), "--json"]
    )
    assert result.exit_code == 1
    assert "catalog" in result.output.lower()
    assert '"results": []' not in result.output
