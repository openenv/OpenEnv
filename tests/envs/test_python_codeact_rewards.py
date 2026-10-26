# SPDX-License-Identifier: BSD-3-Clause

"""Test that PythonCodeActEnv properly computes rewards via transform pipeline."""

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


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def env():
    """Provides a fresh PythonCodeActEnv for each test."""
    environment = PythonCodeActEnv()
    environment.reset()
    return environment


@pytest.fixture
def env_with_variable(env):
    """Environment with a variable already defined."""
    env.step(CodeAction(code="test_var = 42"))
    return env


# ============================================================================
# Parametrized Tests - Reward Computation
# ============================================================================


@pytest.mark.parametrize(
    "code,expected_reward,expected_exit_code,description",
    [
        # Safe + concise code
        ("x = 5", 0.1, 0, "safe + concise"),
        ("print('Hello')", 0.1, 0, "safe + concise print"),
        ("y = 10 + 5", 0.1, 0, "safe + concise calculation"),
        # Safe + verbose code (>100 chars, no concise bonus)
        ("x = " + " + ".join(str(i) for i in range(50)), 0.0, 0, "safe + verbose"),
        # Dangerous + concise (-1.0 safety + 0.1 concise = -0.9)
        # NOTE: These actually fail at execution, so exit_code=1
        ("import os", -0.9, 1, "dangerous + concise"),
        ("eval('1+1')", -0.9, 1, "dangerous eval"),
        ("exec('x=1')", -0.9, 1, "dangerous exec"),
        ("with open('f.txt') as f: pass", -0.9, 1, "dangerous open"),
        # Dangerous + verbose (-1.0 safety, no concise bonus)
        ("import os\n" + "x = 1\n" * 50, -1.0, 1, "dangerous + verbose"),
        # Syntax error + concise (0.0 safe - 0.2 syntax + 0.1 concise = -0.1)
        ("print('unclosed", -0.1, 1, "syntax error + concise"),
        # Syntax error + verbose (0.0 safe - 0.2 syntax = -0.2)
        (
            "x = " + " + ".join(str(i) for i in range(50)) + "\nprint('unclosed",
            -0.2,
            1,
            "syntax error + verbose",
        ),
    ],
    ids=lambda x: (
        x if isinstance(x, str) and len(x) < 20 else None
    ),  # Use description for test IDs
)
def test_reward_computation(
    env, code, expected_reward, expected_exit_code, description
):
    """Test reward computation for various code patterns.

    Parametrized test covering:
    - Safe code (concise and verbose)
    - Dangerous patterns (import os, eval, exec, open)
    - Syntax errors
    - Combinations of safety and quality transforms

    Uses pytest.approx() for all float comparisons since rewards are computed
    via floating point addition in the transform pipeline (transforms.py line 101).
    """
    action = CodeAction(code=code)
    obs = env.step(action)

    assert obs.reward == pytest.approx(expected_reward, rel=1e-9), (
        f"{description}: expected reward {expected_reward}, got {obs.reward}"
    )
    assert obs.exit_code == expected_exit_code, (
        f"{description}: expected exit_code {expected_exit_code}, got {obs.exit_code}"
    )


# ============================================================================
# Metadata Tests
# ============================================================================


def test_metadata_contains_last_code(env):
    """Test that step() includes executed code in observation metadata.

    This is CRITICAL for the transform pipeline to evaluate code and assign rewards.
    Without metadata["last_code"], transforms cannot access the code and rewards
    will always be None.
    """
    code = "print('Hello, World!')"
    action = CodeAction(code=code)
    obs = env.step(action)

    assert "last_code" in obs.metadata, (
        "metadata must contain 'last_code' for transform pipeline to evaluate code"
    )
    assert obs.metadata["last_code"] == code, (
        f"metadata['last_code'] should be '{code}', got '{obs.metadata.get('last_code')}'"
    )


@pytest.mark.parametrize(
    "code,should_have_violation",
    [
        ("import os", True),
        ("eval('1+1')", True),
        ("open('file.txt')", True),
        ("print('safe')", False),
        ("x = 1 + 2", False),
    ],
)
def test_metadata_safety_violations(env, code, should_have_violation):
    """Test that metadata correctly tracks safety violations."""
    action = CodeAction(code=code)
    obs = env.step(action)

    assert "last_code" in obs.metadata
    assert obs.metadata["last_code"] == code

    if should_have_violation:
        assert "safety_violation" in obs.metadata, (
            f"Code '{code}' should have safety_violation in metadata"
        )
    else:
        assert "safety_violation" not in obs.metadata, (
            f"Code '{code}' should NOT have safety_violation in metadata"
        )


# ============================================================================
# Consistency and State Tests
# ============================================================================


def test_reward_not_none_for_safe_code(env):
    """Test that safe code always receives a non-None reward."""
    action = CodeAction(code="print('Hello')")
    obs = env.step(action)

    assert obs.reward is not None, "Safe code should receive a reward (not None)"
    assert obs.exit_code == 0, "Safe code should execute successfully"


def test_reward_consistency_across_steps(env):
    """Test that rewards are computed consistently across multiple steps."""
    for i in range(5):
        action = CodeAction(code=f"x = {i}")
        obs = env.step(action)

        assert obs.reward is not None, f"Step {i}: Reward should not be None"
        assert obs.reward == pytest.approx(0.1, rel=1e-9), (
            f"Step {i}: Should get consistent 0.1 reward, got {obs.reward}"
        )


def test_reset_preserves_transform_functionality(env):
    """Test that reset() doesn't break reward computation."""
    # First episode
    action1 = CodeAction(code="x = 1")
    obs1 = env.step(action1)
    assert obs1.reward == pytest.approx(0.1, rel=1e-9)

    # Reset and start new episode
    env.reset()
    action2 = CodeAction(code="y = 2")
    obs2 = env.step(action2)
    assert obs2.reward == pytest.approx(0.1, rel=1e-9), (
        "Reward computation should work after reset"
    )


# ============================================================================
# Fixture Composition Tests
# ============================================================================


def test_using_composed_fixture(env_with_variable):
    """Test using an environment that builds on base fixture."""
    action = CodeAction(code="print(test_var)")
    obs = env_with_variable.step(action)

    assert obs.exit_code == 0
    assert "42" in obs.stdout
    assert obs.reward == pytest.approx(0.1, rel=1e-9)


@pytest.mark.parametrize(
    "code,expected_output",
    [
        ("print(test_var)", "42"),
        ("print(test_var * 2)", "84"),
        ("print(test_var + 8)", "50"),
    ],
)
def test_fixture_with_parametrization(env_with_variable, code, expected_output):
    """Test combining fixtures with parametrization."""
    action = CodeAction(code=code)
    obs = env_with_variable.step(action)

    assert obs.exit_code == 0
    assert expected_output in obs.stdout
    assert obs.reward == pytest.approx(0.1, rel=1e-9)


# ============================================================================
# Edge Cases and Special Patterns
# ============================================================================


@pytest.mark.parametrize(
    "dangerous_pattern",
    [
        "import os",
        "import subprocess",
        "eval('x')",
        "exec('x=1')",
        "__import__('os')",
        "open('file.txt')",
    ],
)
def test_all_dangerous_patterns_detected(env, dangerous_pattern):
    """Test that all dangerous patterns are correctly detected and penalized."""
    action = CodeAction(code=dangerous_pattern)
    obs = env.step(action)

    # Concise dangerous code gets -0.9 (-1.0 safety + 0.1 concise)
    assert obs.reward == pytest.approx(-0.9, rel=1e-9), (
        f"Pattern '{dangerous_pattern}' should get -0.9 reward, got {obs.reward}"
    )
    assert "safety_violation" in obs.metadata


def test_multiline_code_with_mixed_patterns(env):
    """Test code with both safe and dangerous patterns (dangerous wins)."""
    code = """
x = 5
y = 10
import os
z = x + y
"""
    action = CodeAction(code=code)
    obs = env.step(action)

    # Should be flagged as dangerous even with safe code mixed in
    assert obs.reward < 0, "Code with dangerous import should have negative reward"
    assert "safety_violation" in obs.metadata
