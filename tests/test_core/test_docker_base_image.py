# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv-base Docker image.

These tests verify that the openenv-base image has the correct dependencies
and console scripts available.

Build the base image first:
    docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .

Run these tests with:
    PYTHONPATH=src:envs uv run pytest tests/test_core/test_docker_base_image.py -v
"""

import shutil
import subprocess

import pytest


@pytest.fixture
def check_docker_available():
    """Check if Docker is available."""
    if not shutil.which("docker"):
        pytest.skip("Docker is not installed")

    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, text=True
        )
        if result.returncode != 0:
            pytest.skip("Docker daemon is not running")
    except subprocess.TimeoutExpired:
        pytest.skip("Docker daemon not responding")
    except Exception as e:
        pytest.skip(f"Cannot access Docker: {e}")


@pytest.fixture
def check_base_image_exists(check_docker_available):
    """Check if the openenv-base image exists."""
    result = subprocess.run(
        ["docker", "images", "-q", "openenv-base:latest"],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        pytest.skip(
            "Docker image 'openenv-base:latest' not found. "
            "Build it with: docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile ."
        )


@pytest.mark.docker
class TestOpenEnvBaseImage:
    """Tests for the openenv-base Docker image.

    These tests verify that console scripts from installed packages
    are available in the PATH.
    """

    def test_uvicorn_command_available(self, check_base_image_exists):
        """Test that uvicorn command is available in openenv-base image.

        This verifies that console_scripts from installed packages
        are properly copied from the builder stage.
        """
        result = subprocess.run(
            ["docker", "run", "--rm", "openenv-base:latest", "uvicorn", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"uvicorn command not found in openenv-base image.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "uvicorn" in result.stdout.lower() or "Running uvicorn" in result.stdout

    def test_fastapi_command_available(self, check_base_image_exists):
        """Test that fastapi CLI command is available in openenv-base image.

        This verifies that console_scripts from installed packages
        are properly copied from the builder stage.
        """
        result = subprocess.run(
            ["docker", "run", "--rm", "openenv-base:latest", "fastapi", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"fastapi command not found in openenv-base image.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "fastapi" in result.stdout.lower() or "Usage" in result.stdout

    def test_uv_command_available(self, check_base_image_exists):
        """Test that uv command is available (baseline check).

        This test should PASS since uv is already copied in the Dockerfile.
        This serves as a baseline to verify the test infrastructure works.
        """
        result = subprocess.run(
            ["docker", "run", "--rm", "openenv-base:latest", "uv", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"uv command not found in openenv-base image.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "uv" in result.stdout.lower()

    def test_python_can_import_fastapi(self, check_base_image_exists):
        """Test that Python can import fastapi module.

        This verifies that the packages are installed correctly,
        even if console scripts are missing.
        """
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "openenv-base:latest",
                "python",
                "-c",
                "import fastapi; print(fastapi.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Failed to import fastapi in openenv-base image.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # Should print a version number
        assert result.stdout.strip(), "fastapi version output is empty"

    def test_python_can_import_uvicorn(self, check_base_image_exists):
        """Test that Python can import uvicorn module.

        This verifies that the packages are installed correctly,
        even if console scripts are missing.
        """
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "openenv-base:latest",
                "python",
                "-c",
                "import uvicorn; print(uvicorn.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Failed to import uvicorn in openenv-base image.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # Should print a version number
        assert result.stdout.strip(), "uvicorn version output is empty"
