# SPDX-License-Identifier: BSD-3-Clause

"""Test that PythonCodeActEnv.reset() properly resets executor state."""

import os
import sys
from pathlib import Path

import pytest


# Add the project root and src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Skip entire module if smolagents is not installed (optional dependency)
pytest.importorskip("smolagents", reason="smolagents is not installed")

from envs.coding_env.models import CodeAction
from envs.coding_env.server.python_codeact_env import PythonCodeActEnv


@pytest.fixture
def make_env():
    """Build envs and stop their worker processes after the test."""
    envs = []

    def factory(**kwargs) -> PythonCodeActEnv:
        envs.append(PythonCodeActEnv(**kwargs))
        return envs[-1]

    yield factory
    for env in envs:
        env.close()


def test_reset_clears_executor_state(make_env):
    """Test that reset() clears functions and variables defined in
    previous execution."""
    env = make_env()

    # Initial reset
    obs = env.reset()
    assert obs.exit_code == 0
    assert env.state.step_count == 0

    # Define a function in the executor
    action1 = CodeAction(code="def my_function():\n    return 'Hello from function'\n")
    obs1 = env.step(action1)
    assert obs1.exit_code == 0

    # Call the function to verify it exists
    action2 = CodeAction(code="result = my_function()\nprint(result)")
    obs2 = env.step(action2)
    assert obs2.exit_code == 0
    assert "Hello from function" in obs2.stdout

    # Reset the environment
    obs_reset = env.reset()
    assert obs_reset.exit_code == 0
    assert env.state.step_count == 0

    # Try to call the function again - should fail because executor was reset
    action3 = CodeAction(code="result = my_function()\nprint(result)")
    obs3 = env.step(action3)

    # Should get an error because my_function is no longer defined
    assert obs3.exit_code == 1  # Error exit code
    assert "my_function" in obs3.stderr or "NameError" in obs3.stderr


def test_reset_clears_variables(make_env):
    """Test that reset() clears variables defined in previous execution."""
    env = make_env()

    # Initial reset
    env.reset()

    # Define a variable
    action1 = CodeAction(code="my_variable = 42\n")
    obs1 = env.step(action1)
    assert obs1.exit_code == 0

    # Use the variable to verify it exists
    action2 = CodeAction(code="print(my_variable)")
    obs2 = env.step(action2)
    assert obs2.exit_code == 0
    assert "42" in obs2.stdout

    # Reset the environment
    env.reset()

    # Try to use the variable again - should fail
    action3 = CodeAction(code="print(my_variable)")
    obs3 = env.step(action3)

    # Should get an error because my_variable is no longer defined
    assert obs3.exit_code == 1
    assert "my_variable" in obs3.stderr or "NameError" in obs3.stderr


def test_reset_clears_imports(make_env):
    """Test that reset() clears imported modules from previous execution."""
    env = make_env()

    # Initial reset
    env.reset()

    # Import a module and define an alias
    action1 = CodeAction(code="import math as m\n")
    obs1 = env.step(action1)
    assert obs1.exit_code == 0

    # Use the alias to verify it exists
    action2 = CodeAction(code="print(m.pi)")
    obs2 = env.step(action2)
    assert obs2.exit_code == 0
    assert "3.14" in obs2.stdout

    # Reset the environment
    env.reset()

    # Try to use the alias again - should fail
    action3 = CodeAction(code="print(m.pi)")
    obs3 = env.step(action3)

    # Should get an error because 'm' is no longer defined
    assert obs3.exit_code == 1
    assert (
        "NameError" in obs3.stderr
        or "'m'" in obs3.stderr
        or "variable `m` is not defined" in obs3.stderr
    )


def test_json_import_is_available_by_default(make_env):
    """Test that the coding env permits common JSON serialization code."""
    env = make_env()
    env.reset()

    action = CodeAction(
        code="import json\nprint(json.dumps({'status': 'ok', 'n': 3}, sort_keys=True))"
    )
    obs = env.step(action)

    assert obs.exit_code == 0
    assert obs.stdout.strip() == '{"n": 3, "status": "ok"}'


def test_reset_preserves_step_count_reset(make_env):
    """Test that reset() properly resets step count."""
    env = make_env()

    # Initial reset
    env.reset()
    assert env.state.step_count == 0

    # Execute some steps
    for i in range(5):
        action = CodeAction(code=f"print({i})")
        env.step(action)

    assert env.state.step_count == 5

    # Reset should reset step count
    env.reset()
    assert env.state.step_count == 0

    # Execute another step
    action = CodeAction(code="print('test')")
    env.step(action)
    assert env.state.step_count == 1


def test_reset_changes_episode_id(make_env):
    """Test that reset() generates a new episode ID."""
    env = make_env()

    # Initial reset
    env.reset()
    episode_id_1 = env.state.episode_id

    # Execute some steps
    action = CodeAction(code="print('test')")
    env.step(action)

    # Reset and get new episode ID
    env.reset()
    episode_id_2 = env.state.episode_id

    # Episode IDs should be different
    assert episode_id_1 != episode_id_2
