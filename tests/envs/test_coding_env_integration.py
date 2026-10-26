# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for CodingEnv with Docker.

These tests require Docker to be running and the coding-env image to be built:
    docker build -t coding-env:latest -f envs/coding_env/server/Dockerfile .

Run with:
    PYTHONPATH=src:envs uv run pytest tests/envs/test_coding_env_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

# Skip if Docker is not available or image not built
docker_available = pytest.mark.skipif(
    os.environ.get("SKIP_DOCKER_TESTS", "1") == "1",
    reason="Docker tests disabled. Set SKIP_DOCKER_TESTS=0 to enable.",
)

from coding_env import CodeAction, CodingEnv


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def coding_env_client():
    """Create a CodingEnv client from Docker image.

    This fixture is module-scoped to avoid starting/stopping containers
    for each test, which is slow.
    """
    client = CodingEnv.from_docker_image("coding-env:latest")
    yield client
    client.close()


# ============================================================================
# Integration Tests
# ============================================================================


@docker_available
class TestCodingEnvDocker:
    """Integration tests that run against the Docker container."""

    def test_reset(self, coding_env_client):
        """Test that reset returns a valid observation."""
        result = coding_env_client.reset()

        assert result.observation is not None
        assert result.observation.exit_code == 0
        assert result.observation.stderr == ""

    def test_step_simple_print(self, coding_env_client):
        """Test executing a simple print statement."""
        coding_env_client.reset()

        result = coding_env_client.step(CodeAction(code="print('Hello, World!')"))

        assert result.observation.exit_code == 0
        assert "Hello, World!" in result.observation.stdout
        assert result.reward is not None

    def test_step_calculation(self, coding_env_client):
        """Test executing a calculation."""
        coding_env_client.reset()

        result = coding_env_client.step(
            CodeAction(code="x = 5 + 3\nprint(f'Result: {x}')")
        )

        assert result.observation.exit_code == 0
        assert "Result: 8" in result.observation.stdout

    def test_step_import_math(self, coding_env_client):
        """Test importing and using the math module."""
        coding_env_client.reset()

        result = coding_env_client.step(
            CodeAction(code="import math\nprint(f'Pi: {math.pi:.4f}')")
        )

        assert result.observation.exit_code == 0
        assert "Pi: 3.1416" in result.observation.stdout

    def test_step_multiline(self, coding_env_client):
        """Test executing multi-line code."""
        coding_env_client.reset()

        code = """
for i in range(1, 4):
    print(f'{i} squared is {i**2}')
"""
        result = coding_env_client.step(CodeAction(code=code))

        assert result.observation.exit_code == 0
        assert "1 squared is 1" in result.observation.stdout
        assert "2 squared is 4" in result.observation.stdout
        assert "3 squared is 9" in result.observation.stdout

    def test_error_division_by_zero(self, coding_env_client):
        """Test that division by zero returns an error."""
        coding_env_client.reset()

        result = coding_env_client.step(CodeAction(code="x = 1 / 0"))

        assert result.observation.exit_code == 1
        assert (
            "ZeroDivisionError" in result.observation.stderr
            or result.observation.stderr != ""
        )

    def test_error_undefined_variable(self, coding_env_client):
        """Test that undefined variable returns an error."""
        coding_env_client.reset()

        result = coding_env_client.step(CodeAction(code="print(undefined_variable)"))

        assert result.observation.exit_code == 1

    def test_error_syntax_error(self, coding_env_client):
        """Test that syntax error returns an error."""
        coding_env_client.reset()

        result = coding_env_client.step(CodeAction(code="print('Hello'"))

        assert result.observation.exit_code == 1

    def test_state_tracking(self, coding_env_client):
        """Test that state is properly tracked."""
        coding_env_client.reset()

        state = coding_env_client.state()
        assert state.episode_id is not None
        assert state.step_count == 0

        coding_env_client.step(CodeAction(code="x = 1"))
        state = coding_env_client.state()
        assert state.step_count == 1

        coding_env_client.step(CodeAction(code="y = 2"))
        state = coding_env_client.state()
        assert state.step_count == 2

    def test_reward_safe_code(self, coding_env_client):
        """Test that safe code receives a positive or zero reward."""
        coding_env_client.reset()

        result = coding_env_client.step(CodeAction(code="x = 5"))

        assert result.reward is not None
        assert result.reward >= 0  # Safe code should not be penalized

    def test_reward_dangerous_code(self, coding_env_client):
        """Test that dangerous code receives a negative reward."""
        coding_env_client.reset()

        result = coding_env_client.step(CodeAction(code="import os"))

        assert result.reward is not None
        assert result.reward < 0  # Dangerous code should be penalized

    def test_variable_persistence_within_episode(self, coding_env_client):
        """Test that variables persist within an episode."""
        coding_env_client.reset()

        # Define a variable
        coding_env_client.step(CodeAction(code="my_var = 42"))

        # Use the variable in a subsequent step
        result = coding_env_client.step(CodeAction(code="print(my_var)"))

        assert result.observation.exit_code == 0
        assert "42" in result.observation.stdout

    def test_reset_clears_variables(self, coding_env_client):
        """Test that reset clears variables from previous episode."""
        # Define a variable
        coding_env_client.reset()
        coding_env_client.step(CodeAction(code="my_var = 42"))

        # Reset and try to use the variable
        coding_env_client.reset()
        result = coding_env_client.step(CodeAction(code="print(my_var)"))

        # Should fail because my_var is no longer defined
        assert result.observation.exit_code == 1
