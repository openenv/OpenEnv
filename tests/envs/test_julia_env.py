# SPDX-License-Identifier: BSD-3-Clause

"""Test the JuliaCodeActEnv and JuliaExecutor functionality."""

import os
import shutil
import sys
from pathlib import Path

import pytest

# Add the project root and src to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))


# Skip tests if Julia is not installed
julia_available = shutil.which("julia") is not None
julia_skip_reason = "Julia is not installed"


class TestJuliaModelsImport:
    """Test that julia_env models can be imported correctly."""

    def test_import_models(self):
        """Test that models can be imported."""
        from julia_env import JuliaAction, JuliaObservation, JuliaState

        # Verify they are the expected types (Pydantic models)
        assert hasattr(JuliaAction, "model_fields")
        assert hasattr(JuliaObservation, "model_fields")
        assert hasattr(JuliaState, "model_fields")

    def test_julia_action_fields(self):
        """Test JuliaAction fields."""
        from julia_env import JuliaAction

        action = JuliaAction(core_code="println(1)")
        assert action.core_code == "println(1)"
        assert action.test_code is None

        action_with_test = JuliaAction(
            core_code="function add(a, b)\n  return a + b\nend",
            test_code="using Test\n@test add(1, 2) == 3",
        )
        assert action_with_test.core_code == "function add(a, b)\n  return a + b\nend"
        assert action_with_test.test_code == "using Test\n@test add(1, 2) == 3"

    def test_julia_observation_fields(self):
        """Test JuliaObservation default values."""
        from julia_env import JuliaObservation

        obs = JuliaObservation()
        assert obs.stdout == ""
        assert obs.stderr == ""
        assert obs.exit_code == 0
        assert obs.tests_passed == 0
        assert obs.tests_failed == 0
        assert obs.code_compiles is True
        assert obs.done is False
        assert obs.reward is None

    def test_julia_state_fields(self):
        """Test JuliaState fields."""
        from julia_env import JuliaState

        state = JuliaState()
        assert state.episode_id is None
        assert state.step_count == 0
        assert state.last_exit_code == 0
        assert state.last_code_compiles is True
        assert state.total_tests_passed == 0
        assert state.total_tests_failed == 0


class TestJuliaClientImport:
    """Test that julia_env client can be imported correctly."""

    def test_import_client(self):
        """Test that JuliaEnv client can be imported."""
        from julia_env import JuliaEnv

        # Verify it's an EnvClient subclass
        from openenv.core.env_client import EnvClient

        assert issubclass(JuliaEnv, EnvClient)


class TestJuliaExecutorImport:
    """Test that JuliaExecutor can be imported correctly."""

    def test_import_julia_executor(self):
        """Test that JuliaExecutor can be imported from julia_env.server."""
        from julia_env.server.julia_executor import JuliaExecutor

        executor = JuliaExecutor(use_process_pool=False)
        assert hasattr(executor, "run")
        assert hasattr(executor, "enable_process_pool")
        assert hasattr(executor, "shutdown_pool")
        assert hasattr(executor, "get_pool_metrics")


class TestJuliaServerImport:
    """Test that julia_env server can be imported correctly."""

    def test_import_codeact_env(self):
        """Test that JuliaCodeActEnv can be imported."""
        from julia_env.server import JuliaCodeActEnv

        # Verify it's an Environment subclass
        from openenv.core.env_server.interfaces import Environment

        assert issubclass(JuliaCodeActEnv, Environment)

    def test_import_transforms(self):
        """Test that transforms can be imported."""
        from julia_env.server import create_safe_julia_transform

        transform = create_safe_julia_transform()
        assert callable(transform)


def test_process_pool_failed_init_cleans_up(monkeypatch):
    """__del__ must not raise when __init__ fails before the pool is set up."""
    import gc

    from julia_env.server.julia_process_pool import JuliaProcessPool

    def fail(self):
        raise RuntimeError("Julia executable not found")

    unraisable = []
    monkeypatch.setattr(JuliaProcessPool, "_find_julia_executable", fail)
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    with pytest.raises(RuntimeError):
        JuliaProcessPool(size=1)
    gc.collect()

    assert unraisable == []


@pytest.mark.skipif(not julia_available, reason=julia_skip_reason)
class TestJuliaCodeActEnv:
    """Test JuliaCodeActEnv functionality (requires Julia)."""

    def test_reset(self):
        """Test that reset() returns an empty observation."""
        from julia_env.server import JuliaCodeActEnv

        env = JuliaCodeActEnv(use_process_pool=False)
        obs = env.reset()

        assert obs.exit_code == 0
        assert obs.stdout == ""
        assert obs.stderr == ""
        assert env.state.step_count == 0

    def test_step_simple_print(self):
        """Test executing simple Julia code."""
        from julia_env import JuliaAction
        from julia_env.server import JuliaCodeActEnv

        env = JuliaCodeActEnv(use_process_pool=False)
        env.reset()

        action = JuliaAction(core_code='println("Hello, Julia!")')
        obs = env.step(action)

        assert "Hello, Julia!" in obs.stdout
        assert obs.exit_code == 0
        assert obs.code_compiles is True

    def test_step_with_tests_pass(self):
        """Test executing Julia code with passing tests."""
        from julia_env import JuliaAction
        from julia_env.server import JuliaCodeActEnv

        env = JuliaCodeActEnv(use_process_pool=False)
        env.reset()

        action = JuliaAction(
            core_code="""
            function add(a, b)
                return a + b
            end
            """,
            test_code="""
            using Test
            @testset "add function tests" begin
                @test add(1, 2) == 3
                @test add(0, 0) == 0
            end
            """,
        )
        obs = env.step(action)

        assert obs.code_compiles is True
        assert obs.tests_passed == 2
        assert obs.tests_failed == 0
        assert obs.reward > 0

    def test_step_with_tests_fail(self):
        """Test executing Julia code with failing tests."""
        from julia_env import JuliaAction
        from julia_env.server import JuliaCodeActEnv

        env = JuliaCodeActEnv(use_process_pool=False)
        env.reset()

        action = JuliaAction(
            core_code="""
            function add(a, b)
                return a - b  # Intentional bug
            end
            """,
            test_code="""
            using Test
            @testset "add function tests" begin
                @test add(1, 2) == 3  # This will fail
            end
            """,
        )
        obs = env.step(action)

        assert obs.tests_passed == 0
        assert obs.tests_failed == 1

    def test_step_compilation_error(self):
        """Test executing Julia code with syntax error."""
        from julia_env import JuliaAction
        from julia_env.server import JuliaCodeActEnv

        env = JuliaCodeActEnv(use_process_pool=False)
        env.reset()

        action = JuliaAction(core_code='println("missing closing quote)')
        obs = env.step(action)

        assert obs.exit_code != 0
        assert obs.code_compiles is False

    def test_reset_changes_episode_id(self):
        """Test that reset() generates a new episode ID."""
        from julia_env.server import JuliaCodeActEnv

        env = JuliaCodeActEnv(use_process_pool=False)
        env.reset()
        episode_id_1 = env.state.episode_id

        env.reset()
        episode_id_2 = env.state.episode_id

        assert episode_id_1 != episode_id_2


@pytest.mark.skipif(not julia_available, reason=julia_skip_reason)
class TestJuliaExecutor:
    """Test JuliaExecutor functionality (requires Julia)."""

    def test_run_simple(self):
        """Test running simple Julia code."""
        from julia_env.server.julia_executor import JuliaExecutor

        executor = JuliaExecutor(use_process_pool=False)
        result = executor.run('println("Hello")')

        assert "Hello" in result.stdout
        assert result.exit_code == 0

    def test_run_math(self):
        """Test running Julia math code."""
        from julia_env.server.julia_executor import JuliaExecutor

        executor = JuliaExecutor(use_process_pool=False)
        result = executor.run("println(2 + 2)")

        assert "4" in result.stdout
        assert result.exit_code == 0

    def test_run_syntax_error(self):
        """Test running Julia code with syntax error."""
        from julia_env.server.julia_executor import JuliaExecutor

        executor = JuliaExecutor(use_process_pool=False)
        result = executor.run('println("unclosed string)')

        assert result.exit_code != 0
        assert result.stderr != ""
