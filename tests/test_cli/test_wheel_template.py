"""Packaging checks for the environment created by ``openenv init``."""

import subprocess
from pathlib import Path
from zipfile import ZipFile


def test_wheel_includes_template_dockerignore(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    (wheel,) = tmp_path.glob("openenv-*.whl")
    with ZipFile(wheel) as package:
        dockerignore = "openenv/cli/templates/openenv_env/.dockerignore"
        assert dockerignore in package.namelist()
        assert ".env" in package.read(dockerignore).decode()
