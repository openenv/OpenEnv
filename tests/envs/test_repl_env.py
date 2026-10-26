# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the REPL Environment."""

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Skip entire module if smolagents is not installed
pytest.importorskip("smolagents", reason="smolagents is not installed")

from repl_env import LocalREPLEnv, LocalRLMRunner, REPLEnv
from repl_env.models import CodeBlockResult, REPLAction, REPLObservation, REPLState
from repl_env.recursive_controller import create_server_recursive_controller
from repl_env.rubrics import ExactMatchRubric, REPLRubric
from repl_env.server.python_executor import PythonExecutor
from repl_env.server.repl_environment import REPLEnvironment


class TestPythonExecutor:
    """Tests for the PythonExecutor class."""

    def test_basic_execution(self):
        """Test basic code execution."""
        executor = PythonExecutor()
        result = executor.execute("x = 1 + 1")
        assert result["success"]
        assert executor.get_variable("x") == 2

    def test_stdout_capture(self):
        """Test stdout is captured correctly."""
        executor = PythonExecutor()
        result = executor.execute("print('hello world')")
        assert result["success"]
        assert "hello world" in result["stdout"]

    def test_server_package_import_from_env_root(self, monkeypatch):
        """Importing `server.repl_environment` from env root should work."""
        env_root = Path(__file__).resolve().parents[2] / "envs" / "repl_env"
        monkeypatch.syspath_prepend(str(env_root))

        for module_name in [
            "server",
            "server.python_executor",
            "server.repl_environment",
        ]:
            sys.modules.pop(module_name, None)

        module = importlib.import_module("server.repl_environment")
        assert hasattr(module, "REPLEnvironment")

    def test_server_app_imports_from_env_root_without_path_rewrite(self):
        """Importing server.app from env root should work without bundled-src hacks."""
        env_root = Path(__file__).resolve().parents[2] / "envs" / "repl_env"
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        result = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                ".",
                "python",
                "-c",
                (
                    "import importlib; "
                    "importlib.import_module('server.app'); "
                    "print('ok')"
                ),
            ],
            cwd=env_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip().splitlines()[-1] == "ok"

    def test_stderr_capture(self):
        """Test stderr is captured correctly via exception handling.

        Note: smolagents.LocalPythonExecutor blocks direct sys.stderr access,
        so we test stderr capture through exception handling instead.
        """
        executor = PythonExecutor()
        # Exceptions are captured in stderr
        result = executor.execute("raise RuntimeError('error message')")
        assert not result["success"]
        assert (
            "error message" in result["stderr"]
            or "error message" in result["exception"]
        )

    def test_exception_handling(self):
        """Test exception handling."""
        executor = PythonExecutor()
        result = executor.execute("raise ValueError('test error')")
        assert not result["success"]
        assert result["exception"] is not None
        assert "ValueError" in result["exception"]
        assert "test error" in result["exception"]

    def test_persistent_namespace(self):
        """Test that namespace persists across executions."""
        executor = PythonExecutor()
        executor.execute("x = 10")
        executor.execute("y = x * 2")
        assert executor.get_variable("x") == 10
        assert executor.get_variable("y") == 20

    def test_context_loading(self):
        """Test context loading."""
        executor = PythonExecutor()
        executor.set_context("Hello World")
        assert executor.get_variable("context") == "Hello World"

    def test_list_variables(self):
        """Test listing variables."""
        executor = PythonExecutor()
        executor.execute("a = 1")
        executor.execute("b = 2")
        variables = executor.list_variables()
        assert "a" in variables
        assert "b" in variables

    def test_output_truncation(self):
        """Test output truncation."""
        executor = PythonExecutor(max_output_length=100)
        result = executor.execute("print('x' * 500)")
        assert len(result["stdout"]) < 200  # Should be truncated

    def test_inject_function(self):
        """Test function injection."""
        executor = PythonExecutor()

        def custom_func(x):
            return x * 2

        executor.inject_function("double", custom_func)
        result = executor.execute("result = double(5)")
        assert result["success"]
        assert executor.get_variable("result") == 10

    def test_reset(self):
        """Test namespace reset."""
        executor = PythonExecutor()
        executor.execute("x = 10")
        executor.reset()
        assert executor.get_variable("x") is None


class TestRecursiveController:
    """Tests for the recursive controller composition."""

    def test_direct_controller(self):
        controller = create_server_recursive_controller(
            lambda messages, model=None: "ok",
            max_depth=1,
            max_iterations=4,
        )
        try:
            assert controller.llm_query_fn("hello") == "ok"
            assert controller.rlm_query_fn is None
        finally:
            controller.close()


class TestREPLEnvironment:
    """Tests for the REPLEnvironment class."""

    def test_reset_without_context(self):
        """Test reset without context."""
        env = REPLEnvironment()
        obs = env.reset()
        assert not obs.done
        assert obs.iteration == 0
        assert obs.context_length == 0
        assert "answer" in obs.available_variables

    def test_reset_with_context(self):
        """Test reset with context."""
        env = REPLEnvironment(context="Hello World")
        obs = env.reset()
        assert not obs.done
        assert obs.context_length == 11
        assert "context" in obs.available_variables
        assert obs.context_preview == "Hello World"

    def test_reset_with_task_prompt(self):
        """Test reset with task prompt."""
        env = REPLEnvironment(task_prompt="Count the chars")
        obs = env.reset()
        assert obs.metadata["task_prompt"] == "Count the chars"

    def test_step_basic(self):
        """Test basic step execution."""
        env = REPLEnvironment(context="test context")
        env.reset()
        obs = env.step(REPLAction(code="result = len(context)"))
        assert obs.result.success
        assert "result" in obs.available_variables
        assert obs.iteration == 1

    def test_step_with_error(self):
        """Test step with code error."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code="raise ValueError('test')"))
        assert not obs.result.success
        assert obs.result.exception is not None
        assert not obs.done

    def test_final_pattern_basic(self):
        """Test FINAL() pattern."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code="print('FINAL(42)')"))
        assert obs.done
        assert obs.metadata["final_answer"] == "42"

    @pytest.mark.parametrize(
        "code, expected",
        [
            # Nested function calls inside FINAL(...).
            ("print('FINAL(f(x))')", "f(x)"),
            # Tuple as the final answer.
            ("print('FINAL((1, 2, 3))')", "(1, 2, 3)"),
            # Math expression with multiple nested parens (e2b_repl_example).
            (
                "print('FINAL(2^(2^(2^(2))) = 65536)')",
                "2^(2^(2^(2))) = 65536",
            ),
            # Dict containing a tuple value.
            ("print(\"FINAL({'a': (1, 2)})\")", "{'a': (1, 2)}"),
            # Output after FINAL must not bleed into the extracted answer.
            ("print('FINAL(42)\\nresult: (ok)')", "42"),
        ],
    )
    def test_final_pattern_nested_parentheses(self, code, expected):
        """FINAL(...) extraction must handle nested parentheses (rlm #75)."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code=code))
        assert obs.done
        assert obs.metadata["final_answer"] == expected

    def test_final_var_pattern(self):
        """Test FINAL_VAR() pattern."""
        env = REPLEnvironment()
        env.reset()
        env.step(REPLAction(code="my_answer = 'the answer is 42'"))
        obs = env.step(REPLAction(code="print('FINAL_VAR(my_answer)')"))
        assert obs.done
        assert obs.metadata["final_answer"] == "the answer is 42"

    def test_answer_dict_pattern(self):
        """Test Prime Intellect style answer dict."""
        env = REPLEnvironment()
        env.reset()
        env.step(REPLAction(code="answer['content'] = 'my answer'"))
        obs = env.step(REPLAction(code="answer['ready'] = True"))
        assert obs.done
        assert obs.metadata["final_answer"] == "my answer"

    def test_explicit_final(self):
        """Test explicit is_final=True."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(
            REPLAction(code="", is_final=True, final_answer="explicit answer")
        )
        assert obs.done
        assert env.state.final_answer == "explicit answer"

    def test_max_iterations(self):
        """Test max iterations limit."""
        env = REPLEnvironment(max_iterations=2)
        env.reset()
        env.step(REPLAction(code="x = 1"))
        obs = env.step(REPLAction(code="x = 2"))
        assert obs.done
        assert "Maximum iterations" in obs.result.stdout

    def test_state_property(self):
        """Test state property."""
        env = REPLEnvironment(context="test")
        env.reset()
        state = env.state
        assert state.context == "test"
        assert state.iteration == 0

    def test_state_not_initialized(self):
        """Test state raises error when not initialized."""
        env = REPLEnvironment()
        with pytest.raises(RuntimeError):
            _ = env.state

    def test_rubric_reward_on_success(self):
        """Test rubric reward when final answer matches expected."""

        rubric = REPLRubric(outcome=ExactMatchRubric())
        env = REPLEnvironment(rubric=rubric)
        env.reset(expected_answer="done")
        obs = env.step(REPLAction(code="print('FINAL(done)')"))
        assert obs.done
        assert obs.reward == 1.0

    def test_rubric_reward_on_wrong_answer(self):
        """Test rubric reward when final answer does not match expected."""

        rubric = REPLRubric(outcome=ExactMatchRubric())
        env = REPLEnvironment(rubric=rubric)
        env.reset(expected_answer="correct")
        obs = env.step(REPLAction(code="print('FINAL(wrong)')"))
        assert obs.done
        assert obs.reward == 0.0

    def test_rubric_reward_on_error(self):
        """Test rubric process reward on code error."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code="raise ValueError()"))
        assert obs.reward == -0.05  # default CodeExecutionRubric error_penalty

    def test_close(self):
        """Test close cleans up resources."""
        env = REPLEnvironment()
        env.reset()
        env.close()
        assert env._state is None
        assert env._executor is None

    def test_get_metadata(self):
        """Test get_metadata returns correct info."""
        env = REPLEnvironment(max_iterations=50)
        metadata = env.get_metadata()
        assert metadata.name == "repl_env"
        assert metadata.version == "0.1.0"

    def test_llm_functions_injected(self):
        """Test LLM functions are injected when provided."""

        def mock_query(prompt):
            return f"Response to: {prompt}"

        def mock_batch(prompts):
            return [f"Response to: {p}" for p in prompts]

        env = REPLEnvironment(llm_query_fn=mock_query, llm_batch_fn=mock_batch)
        env.reset()

        # Test llm_query
        obs = env.step(REPLAction(code="result = llm_query('Hello')"))
        assert obs.result.success
        obs = env.step(REPLAction(code="print(result)"))
        assert "Response to: Hello" in obs.result.stdout

        # Test llm_batch
        obs = env.step(REPLAction(code="results = llm_batch(['A', 'B'])"))
        assert obs.result.success

        # Test documented aliases and helper surface
        obs = env.step(REPLAction(code="deep = rlm_query('Hello again')"))
        assert obs.result.success
        obs = env.step(REPLAction(code="batched = rlm_query_batched(['X', 'Y'])"))
        assert obs.result.success
        obs = env.step(REPLAction(code="vars_now = SHOW_VARS()"))
        assert obs.result.success
        obs = env.step(REPLAction(code="print(vars_now)"))
        assert "deep" in obs.result.stdout

    def test_server_backed_recursive_runtime(self, monkeypatch):
        """Test HF-backed runtime installs a real recursive subcall function."""

        def fake_build_hf_chat_fn(hf_token, llm_model=None):
            def fake_chat(messages, model=None):
                joined = "\n".join(message["content"] for message in messages)
                if "Return child value" in joined:
                    return "```repl\nvalue = 'child'\nprint(FINAL(value))\n```"
                return "unreachable"

            return fake_chat

        monkeypatch.setattr(
            REPLEnvironment,
            "_build_hf_chat_fn",
            staticmethod(fake_build_hf_chat_fn),
        )

        env = REPLEnvironment(rlm_max_depth=3, rlm_max_iterations=4)
        obs = env.reset(hf_token="fake-token")
        assert "rlm_query" in obs.available_variables

        obs = env.step(REPLAction(code="result = rlm_query('Return child value')"))
        assert obs.result.success
        obs = env.step(REPLAction(code="print(result)"))
        assert "child" in obs.result.stdout


class TestModels:
    """Tests for the data models."""

    def test_repl_action_defaults(self):
        """Test REPLAction default values."""
        action = REPLAction(code="x = 1")
        assert action.code == "x = 1"
        assert action.is_final is False
        assert action.final_answer is None

    def test_repl_action_final(self):
        """Test REPLAction with final flag."""
        action = REPLAction(code="", is_final=True, final_answer="42")
        assert action.is_final is True
        assert action.final_answer == "42"

    def test_code_block_result(self):
        """Test CodeBlockResult model."""
        result = CodeBlockResult(
            stdout="output",
            stderr="",
            locals_snapshot={"x": "1"},
            execution_time=0.01,
            success=True,
        )
        assert result.stdout == "output"
        assert result.success is True

    def test_repl_observation(self):
        """Test REPLObservation model."""
        obs = REPLObservation(
            result=CodeBlockResult(
                stdout="test",
                stderr="",
                locals_snapshot={},
                execution_time=0.0,
                success=True,
            ),
            context_length=100,
            available_variables=["x", "y"],
            iteration=5,
            max_iterations=30,
            done=False,
            reward=0.0,
        )
        assert obs.context_length == 100
        assert len(obs.available_variables) == 2
        assert obs.iteration == 5

    def test_repl_state(self):
        """Test REPLState model."""
        state = REPLState(
            episode_id="test-123",
            step_count=5,
            context="hello",
            task_prompt="count",
            iteration=3,
            max_iterations=30,
        )
        assert state.episode_id == "test-123"
        assert state.context == "hello"


class TestLocalREPLEnv:
    """Tests for the explicit local REPL helper."""

    def test_local_mode_basic(self):
        """Test basic local mode execution."""

        with LocalREPLEnv() as env:
            result = env.reset()
            assert not result.done
            assert result.observation.iteration == 0

            result = env.execute("x = 42")
            assert result.observation.result.success

            result = env.execute("print(f'FINAL({x})')")
            assert result.done
            assert env.state().final_answer == "42"

    def test_local_mode_with_context(self):
        """Test local mode with context."""

        with LocalREPLEnv() as env:
            result = env.reset(context="Hello World", task_prompt="Count chars")
            assert result.observation.context_length == 11
            assert "context" in result.observation.available_variables

            result = env.execute("count = len(context)")
            assert result.observation.result.success

    def test_local_mode_with_llm_functions(self):
        """Test local mode with LLM functions."""

        def mock_query(prompt):
            return f"Response: {prompt[:20]}"

        def mock_batch(prompts):
            return [mock_query(p) for p in prompts]

        with LocalREPLEnv(llm_query_fn=mock_query, llm_batch_fn=mock_batch) as env:
            result = env.reset(context="Test")
            assert "llm_query" in result.observation.available_variables
            assert "llm_batch" in result.observation.available_variables
            assert "SHOW_VARS" in result.observation.available_variables

            result = env.execute("r = llm_query('Hello')")
            assert result.observation.result.success

            result = env.execute("print(r)")
            assert "Response: Hello" in result.observation.result.stdout

            result = env.execute("r2 = rlm_query('World')")
            assert result.observation.result.success

            result = env.execute("vars_now = SHOW_VARS()")
            assert result.observation.result.success

            result = env.execute("print(vars_now)")
            assert "r2" in result.observation.result.stdout

    def test_submit_final_answer(self):
        """Test submit_final_answer() method."""

        with LocalREPLEnv() as env:
            env.reset()
            result = env.submit_final_answer("my answer")
            assert result.done
            assert env.state().final_answer == "my answer"

    def test_state_method(self):
        """Test state() method."""

        with LocalREPLEnv() as env:
            env.reset(context="test", task_prompt="do something")
            state = env.state()
            assert state.context == "test"
            assert state.task_prompt == "do something"

    def test_list_variables(self):
        """Test list_variables() method."""

        with LocalREPLEnv() as env:
            env.reset()
            env.execute("my_var = 123")
            variables = env.list_variables()
            assert "my_var" in variables

    def test_context_manager(self):
        """Test context manager properly closes."""

        env = LocalREPLEnv()
        with env:
            env.reset()
            env.execute("x = 1")
        with pytest.raises(RuntimeError):
            env.state()


class TestLocalRLMRunner:
    """Tests for the local recursive runner."""

    def test_recursive_subcall(self):
        """Test rlm_query spawns a child runner and returns its final answer."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Return the answer 42" in joined:
                return "```repl\nchild_answer = '42'\nprint(FINAL(child_answer))\n```"
            return (
                "```repl\n"
                "result = rlm_query('Return the answer 42')\n"
                "print(result)\n"
                "print(FINAL(result))\n"
                "```"
            )

        runner = LocalRLMRunner(mock_chat, max_iterations=4, max_depth=3)
        result = runner.run("Root context", "Ask a recursive child for the answer")
        assert result.final_answer == "42"

    def test_recursive_batched_subcall(self):
        """Test rlm_query_batched spawns multiple child runners."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Return A" in joined:
                return "```repl\nvalue = 'A'\nprint(FINAL(value))\n```"
            if "Return B" in joined:
                return "```repl\nvalue = 'B'\nprint(FINAL(value))\n```"
            return (
                "```repl\n"
                "parts = rlm_query_batched(['Return A', 'Return B'])\n"
                "combined = ''.join(parts)\n"
                "print(FINAL(combined))\n"
                "```"
            )

        runner = LocalRLMRunner(mock_chat, max_iterations=4, max_depth=3)
        result = runner.run("Root context", "Ask recursive children for two parts")
        assert result.final_answer == "AB"

    def test_multiple_code_blocks_all_executed(self):
        """Test that all code blocks in a single response are executed before checking FINAL.

        Matches official RLM behavior: the model writes exploration code first
        and FINAL last, expecting all blocks to run in the same namespace.
        """

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "REPL output" in joined:
                return "```repl\nprint(FINAL(total))\n```"
            # Three blocks — setup, compute, then FINAL
            return (
                "```repl\na = 10\nprint(a)\n```\n"
                "```repl\nb = 20\nprint(b)\n```\n"
                "```repl\ntotal = a + b\nprint(FINAL(total))\n```"
            )

        runner = LocalRLMRunner(mock_chat, max_iterations=4, max_depth=1)
        result = runner.run("Root context", "Add numbers across blocks")
        assert result.final_answer == "30"

    def test_max_children_total_limit(self):
        """Test recursive child spawning respects max_children_total."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Child prompt 1" in joined:
                return "```repl\nprint(FINAL('one'))\n```"
            if "Child prompt 2" in joined:
                return "```repl\nprint(FINAL('two'))\n```"
            return (
                "```repl\n"
                "a = rlm_query('Child prompt 1')\n"
                "b = rlm_query('Child prompt 2')\n"
                "print(FINAL(b))\n"
                "```"
            )

        runner = LocalRLMRunner(
            mock_chat,
            max_iterations=4,
            max_depth=3,
            max_children_total=1,
        )
        result = runner.run("Root context", "Try to spawn too many children")
        assert "max_children_total exceeded" in (result.final_answer or "")

    def test_max_children_per_batch_limit(self):
        """Test batched recursive child spawning is capped."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Return A" in joined:
                return "```repl\nprint(FINAL('A'))\n```"
            if "Return B" in joined:
                return "```repl\nprint(FINAL('B'))\n```"
            if "Return C" in joined:
                return "```repl\nprint(FINAL('C'))\n```"
            return (
                "```repl\n"
                "parts = rlm_query_batched(['Return A', 'Return B', 'Return C'])\n"
                "print(FINAL(''.join(parts)))\n"
                "```"
            )

        runner = LocalRLMRunner(
            mock_chat,
            max_iterations=4,
            max_depth=3,
            max_children_per_batch=2,
        )
        result = runner.run("Root context", "Cap batch child count")
        assert result.final_answer == "AB"

    def test_result_truncation_limit(self):
        """Test recursive child results are truncated when configured."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Long child" in joined:
                return "```repl\nprint(FINAL('abcdefghijklmnopqrstuvwxyz'))\n```"
            return (
                "```repl\nresult = rlm_query('Long child')\nprint(FINAL(result))\n```"
            )

        runner = LocalRLMRunner(
            mock_chat,
            max_iterations=4,
            max_depth=3,
            result_truncation_limit=5,
        )
        result = runner.run("Root context", "Truncate child results")
        assert result.final_answer == "abcde"

    def test_child_trace_metadata(self):
        """Test child trace metadata is recorded on the run result."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Return traced child" in joined:
                return "```repl\nprint(FINAL('child-result'))\n```"
            return (
                "```repl\n"
                "result = rlm_query('Return traced child')\n"
                "print(FINAL(result))\n"
                "```"
            )

        runner = LocalRLMRunner(mock_chat, max_iterations=4, max_depth=3)
        result = runner.run("Root context", "Collect child trace")
        assert result.final_answer == "child-result"
        assert len(result.child_traces) == 1
        trace = result.child_traces[0]
        assert trace.depth == 1
        assert "Return traced child" in trace.prompt_preview
        assert trace.error is None

    def test_per_child_timeout(self):
        """Test child recursion returns a timeout error when time is exceeded.

        Uses cooperative timeout: checked between iterations, so the child
        must take multiple iterations to trigger the timeout.
        """

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Slow child" in joined:
                time.sleep(0.02)
                # Never finishes — keeps iterating until timeout
                return "```repl\nx = 1\nprint(x)\n```"
            return (
                "```repl\nresult = rlm_query('Slow child')\nprint(FINAL(result))\n```"
            )

        runner = LocalRLMRunner(
            mock_chat,
            max_iterations=100,
            max_depth=3,
            per_child_timeout_s=0.05,
        )
        result = runner.run("Root context", "Trigger a child timeout")
        assert "child timeout" in (result.final_answer or "")

    def test_subcall_callbacks(self):
        """Test official-style subcall lifecycle callbacks fire for real child runs."""

        starts = []
        completes = []

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            if "Child callback task" in joined:
                return "```repl\nprint(FINAL('callback-child'))\n```"
            return (
                "```repl\n"
                "result = rlm_query('Child callback task')\n"
                "print(FINAL(result))\n"
                "```"
            )

        runner = LocalRLMRunner(
            mock_chat,
            max_iterations=4,
            max_depth=3,
            on_subcall_start=lambda depth, model, prompt: starts.append(
                (depth, model, prompt)
            ),
            on_subcall_complete=lambda depth, model, duration, error: completes.append(
                (depth, model, duration, error)
            ),
        )
        result = runner.run("Root context", "Exercise callbacks")
        assert result.final_answer == "callback-child"
        assert len(starts) == 1
        assert starts[0][0] == 1
        assert "Child callback task" in starts[0][2]
        assert len(completes) == 1
        assert completes[0][0] == 1
        assert completes[0][2] >= 0.0
        assert completes[0][3] is None

    def test_default_answer_on_max_iterations(self):
        """Test that the runner makes a final LLM call when iterations are exhausted."""

        def mock_chat(messages, model=None):
            joined = "\n".join(message["content"] for message in messages)
            # On the final "run out of REPL iterations" call, provide an answer
            if "run out of REPL iterations" in joined:
                return "FINAL(fallback-answer)"
            # Otherwise just do computation without finishing
            return "```repl\nx = 42\nprint(x)\n```"

        runner = LocalRLMRunner(mock_chat, max_iterations=2, max_depth=1)
        result = runner.run("Root context", "Never finish on time")
        assert result.final_answer == "fallback-answer"
        assert result.iterations == 2


class TestREPLEnvRemoteClient:
    """Tests for the async OpenEnv REPL client."""

    @pytest.mark.asyncio
    async def test_async_execute_and_state(self, monkeypatch):

        env = REPLEnv(base_url="http://localhost:8000")

        async def fake_send_and_receive(message):
            if message["type"] == "reset":
                return {
                    "data": {
                        "observation": {
                            "result": {
                                "stdout": "ready",
                                "stderr": "",
                                "locals_snapshot": {},
                                "execution_time": 0.0,
                                "success": True,
                                "exception": None,
                            },
                            "context_preview": "Hello",
                            "context_length": 5,
                            "available_variables": ["answer", "context"],
                            "iteration": 0,
                            "max_iterations": 30,
                            "metadata": {"message": "Environment ready."},
                        },
                        "reward": 0.0,
                        "done": False,
                    }
                }
            if message["type"] == "step":
                assert message["data"]["code"] == "print('FINAL(42)')"
                return {
                    "data": {
                        "observation": {
                            "result": {
                                "stdout": "FINAL(42)",
                                "stderr": "",
                                "locals_snapshot": {},
                                "execution_time": 0.01,
                                "success": True,
                                "exception": None,
                            },
                            "context_preview": "Hello",
                            "context_length": 5,
                            "available_variables": ["answer", "context"],
                            "iteration": 1,
                            "max_iterations": 30,
                            "metadata": {"final_answer": "42"},
                        },
                        "reward": 1.0,
                        "done": True,
                    }
                }
            if message["type"] == "state":
                return {
                    "data": {
                        "episode_id": "episode-1",
                        "step_count": 1,
                        "context": "Hello",
                        "task_prompt": "Count chars",
                        "iteration": 1,
                        "max_iterations": 30,
                        "namespace_keys": ["answer", "context"],
                        "final_answer": "42",
                        "total_execution_time": 0.01,
                    }
                }
            if message["type"] == "close":
                return {"data": {}}
            raise AssertionError(f"Unexpected message type: {message['type']}")

        monkeypatch.setattr(env, "_send_and_receive", fake_send_and_receive)

        result = await env.reset(context="Hello", task_prompt="Count chars")
        assert result.observation.context_length == 5

        result = await env.execute("print('FINAL(42)')")
        assert result.done
        assert result.observation.metadata["final_answer"] == "42"

        state = await env.state()
        assert state.final_answer == "42"

    def test_sync_wrapper(self, monkeypatch):

        env = REPLEnv(base_url="http://localhost:8000").sync()

        async def fake_connect():
            return env.async_client

        async def fake_send_and_receive(message):
            if message["type"] == "reset":
                return {
                    "data": {
                        "observation": {
                            "result": {
                                "stdout": "ready",
                                "stderr": "",
                                "locals_snapshot": {},
                                "execution_time": 0.0,
                                "success": True,
                                "exception": None,
                            },
                            "context_preview": None,
                            "context_length": 0,
                            "available_variables": ["answer"],
                            "iteration": 0,
                            "max_iterations": 30,
                            "metadata": {},
                        },
                        "reward": 0.0,
                        "done": False,
                    }
                }
            if message["type"] == "step":
                return {
                    "data": {
                        "observation": {
                            "result": {
                                "stdout": "FINAL(done)",
                                "stderr": "",
                                "locals_snapshot": {},
                                "execution_time": 0.0,
                                "success": True,
                                "exception": None,
                            },
                            "context_preview": None,
                            "context_length": 0,
                            "available_variables": ["answer"],
                            "iteration": 1,
                            "max_iterations": 30,
                            "metadata": {"final_answer": "done"},
                        },
                        "reward": 1.0,
                        "done": True,
                    }
                }
            if message["type"] == "state":
                return {
                    "data": {
                        "episode_id": "episode-1",
                        "step_count": 1,
                        "context": "",
                        "task_prompt": "",
                        "iteration": 1,
                        "max_iterations": 30,
                        "namespace_keys": ["answer"],
                        "final_answer": "done",
                        "total_execution_time": 0.0,
                    }
                }
            if message["type"] == "close":
                return {"data": {}}
            raise AssertionError(f"Unexpected message type: {message['type']}")

        monkeypatch.setattr(env.async_client, "_connect_async", fake_connect)
        monkeypatch.setattr(
            env.async_client, "_send_and_receive", fake_send_and_receive
        )

        with env:
            result = env.reset()
            assert not result.done

            result = env.execute("print('FINAL(done)')")
            assert result.done

            state = env.state()
            assert state.final_answer == "done"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
