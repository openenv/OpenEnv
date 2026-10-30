# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for OpenClaw Environment."""

import tempfile
from pathlib import Path

import pytest

# Skip if dependencies aren't available
pytest.importorskip("fastmcp")

from envs.openclaw_env.server.openclaw_environment import OpenClawEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


class TestOpenClawEnvironment:
    """Test suite for OpenClawEnvironment."""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = OpenClawEnvironment(workspace_dir=tmpdir)
            environment.reset()
            yield environment

    def test_reset(self, env):
        """Test environment reset creates workspace."""
        obs = env.reset()

        assert obs.done is False
        assert obs.metadata["status"] == "ready"
        assert "workspace" in obs.metadata
        assert Path(obs.metadata["workspace"]).exists()

    def test_list_tools(self, env):
        """Test that all expected tools are available."""
        obs = env.step(ListToolsAction())

        tool_names = [t.name for t in obs.tools]

        # Check all expected tools are present
        expected_tools = [
            "read",
            "write",
            "edit",
            "exec",
            "web_search",
            "web_fetch",
            "memory_search",
            "memory_get",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"

    def test_exec_echo(self, env):
        """Test executing a simple echo command."""
        action = CallToolAction(
            tool_name="exec", arguments={"command": "echo 'hello world'"}
        )
        obs = env.step(action)

        assert obs.result is not None
        # The result contains structured_content with our dict
        result_data = obs.result.structured_content.get("result", obs.result.data)
        assert result_data["exit_code"] == 0
        assert "hello world" in result_data["stdout"]

    def test_write_and_read(self, env):
        """Test writing and reading a file."""
        # Write a file
        write_action = CallToolAction(
            tool_name="write",
            arguments={"path": "test.txt", "content": "Hello, OpenClaw!"},
        )
        write_obs = env.step(write_action)

        write_result = write_obs.result.structured_content.get(
            "result", write_obs.result.data
        )
        assert write_result["success"] is True

        # Read the file
        read_action = CallToolAction(tool_name="read", arguments={"path": "test.txt"})
        read_obs = env.step(read_action)

        read_result = read_obs.result.structured_content.get(
            "result", read_obs.result.data
        )
        assert read_result["content"] == "Hello, OpenClaw!"

    def test_edit_file(self, env):
        """Test editing a file."""
        # Write initial content
        env.step(
            CallToolAction(
                tool_name="write",
                arguments={
                    "path": "config.txt",
                    "content": "DEBUG = False\nVERBOSE = True",
                },
            )
        )

        # Edit the file
        edit_action = CallToolAction(
            tool_name="edit",
            arguments={
                "path": "config.txt",
                "old_string": "DEBUG = False",
                "new_string": "DEBUG = True",
            },
        )
        edit_obs = env.step(edit_action)

        edit_result = edit_obs.result.structured_content.get(
            "result", edit_obs.result.data
        )
        assert edit_result["success"] is True
        assert edit_result["replacements"] == 1

        # Verify the change
        read_obs = env.step(
            CallToolAction(tool_name="read", arguments={"path": "config.txt"})
        )
        read_result = read_obs.result.structured_content.get(
            "result", read_obs.result.data
        )
        assert "DEBUG = True" in read_result["content"]

    def test_read_nonexistent_file(self, env):
        """Test reading a file that doesn't exist."""
        action = CallToolAction(tool_name="read", arguments={"path": "nonexistent.txt"})
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert "error" in result

    def test_exec_with_workdir(self, env):
        """Test executing command in specific directory."""
        # Create a subdirectory with a file
        env.step(
            CallToolAction(
                tool_name="exec",
                arguments={
                    "command": "mkdir -p subdir && echo 'test' > subdir/file.txt"
                },
            )
        )

        # Execute in that directory
        action = CallToolAction(
            tool_name="exec", arguments={"command": "cat file.txt", "workdir": "subdir"}
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["exit_code"] == 0
        assert "test" in result["stdout"]

    def test_web_search_sandbox(self, env):
        """Test web search returns simulated results in sandbox mode."""
        action = CallToolAction(
            tool_name="web_search",
            arguments={"query": "python reinforcement learning", "count": 3},
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert "results" in result
        assert len(result["results"]) == 3
        assert "note" in result  # Sandbox mode indicator

    def test_web_fetch_sandbox(self, env):
        """Test web fetch returns simulated content in sandbox mode."""
        action = CallToolAction(
            tool_name="web_fetch",
            arguments={"url": "https://example.com", "extract_mode": "markdown"},
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert "content" in result
        assert result["url"] == "https://example.com"
        assert "note" in result  # Sandbox mode indicator

    def test_memory_search_empty(self, env):
        """Test memory search with no memory files."""
        action = CallToolAction(
            tool_name="memory_search", arguments={"query": "test query"}
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert "results" in result
        assert result["total_found"] == 0

    def test_memory_search_with_content(self, env):
        """Test memory search finds content in memory files."""
        # Create a memory file
        env.step(
            CallToolAction(
                tool_name="write",
                arguments={
                    "path": "MEMORY.md",
                    "content": "# Important Notes\n\nRemember to check the API endpoints.\n",
                },
            )
        )

        # Search for content
        action = CallToolAction(
            tool_name="memory_search", arguments={"query": "API endpoints"}
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["total_found"] >= 1
        assert "API endpoints" in result["results"][0]["snippet"]

    def test_step_count_increments(self, env):
        """Test that step count increments properly."""
        initial_count = env.state.step_count

        env.step(ListToolsAction())
        assert env.state.step_count == initial_count + 1

        env.step(CallToolAction(tool_name="exec", arguments={"command": "echo hi"}))
        assert env.state.step_count == initial_count + 2

    def test_episode_isolation(self, env):
        """Test that reset creates a fresh episode."""
        # Create a file in first episode
        env.step(
            CallToolAction(
                tool_name="write",
                arguments={"path": "episode1.txt", "content": "first episode"},
            )
        )

        # Reset to new episode
        obs = env.reset()
        assert Path(obs.metadata["workspace"]).exists()

        # File from first episode should not exist
        read_obs = env.step(
            CallToolAction(tool_name="read", arguments={"path": "episode1.txt"})
        )
        result = read_obs.result.structured_content.get("result", read_obs.result.data)
        assert "error" in result


class TestOpenClawEnvironmentEdgeCases:
    """Edge case tests for OpenClawEnvironment."""

    @pytest.fixture
    def env(self):
        """Create environment for edge case tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = OpenClawEnvironment(workspace_dir=tmpdir)
            environment.reset()
            yield environment

    def test_exec_timeout(self, env):
        """Test command timeout handling."""
        action = CallToolAction(
            tool_name="exec", arguments={"command": "sleep 10", "timeout": 1}
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert "error" in result
        assert "timed out" in result["error"].lower()

    def test_edit_string_not_found(self, env):
        """Test editing when old string doesn't exist."""
        env.step(
            CallToolAction(
                tool_name="write",
                arguments={"path": "test.txt", "content": "hello world"},
            )
        )

        action = CallToolAction(
            tool_name="edit",
            arguments={
                "path": "test.txt",
                "old_string": "not found",
                "new_string": "replacement",
            },
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["success"] is False

    def test_write_creates_directories(self, env):
        """Test that write creates parent directories."""
        action = CallToolAction(
            tool_name="write",
            arguments={"path": "deep/nested/dir/file.txt", "content": "nested content"},
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["success"] is True

        # Verify file exists
        read_obs = env.step(
            CallToolAction(
                tool_name="read", arguments={"path": "deep/nested/dir/file.txt"}
            )
        )
        read_result = read_obs.result.structured_content.get(
            "result", read_obs.result.data
        )
        assert read_result["content"] == "nested content"

    def test_read_with_offset_and_limit(self, env):
        """Test reading specific line ranges."""
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        env.step(
            CallToolAction(
                tool_name="write", arguments={"path": "lines.txt", "content": content}
            )
        )

        action = CallToolAction(
            tool_name="read", arguments={"path": "lines.txt", "offset": 50, "limit": 10}
        )
        obs = env.step(action)

        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["lines_read"] == 10
        assert "Line 50" in result["content"]
        assert "Line 59" in result["content"]


class TestOpenClawBoundaries:
    @pytest.fixture
    def env(self, tmp_path):
        environment = OpenClawEnvironment(workspace_dir=str(tmp_path))
        environment.reset()
        yield environment
        environment.close()

    @pytest.mark.parametrize(
        "kind", ["parent", "absolute_parent", "sibling", "symlink", "new_symlink_child"]
    )
    def test_rejects_workspace_escape(self, env, kind):
        outside = env._workspace.parent / (env._workspace.name + "_outside")
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_text("synthetic boundary sentinel")
        link = env._workspace / "link"
        link.symlink_to(outside, target_is_directory=True)
        paths = {
            "parent": f"../{outside.name}/sentinel.txt",
            "absolute_parent": str(
                env._workspace / ".." / outside.name / "sentinel.txt"
            ),
            "sibling": str(sentinel),
            "symlink": "link/sentinel.txt",
            "new_symlink_child": "link/new.txt",
        }
        with pytest.raises(ValueError, match="outside workspace"):
            env._resolve_path(paths[kind])
        assert sentinel.read_text() == "synthetic boundary sentinel"

    def test_allows_nested_and_internal_symlink_paths(self, env):
        nested = env._workspace / "nested"
        nested.mkdir()
        (env._workspace / "link").symlink_to(nested, target_is_directory=True)
        assert env._resolve_path("link/new.txt") == nested.resolve() / "new.txt"
        assert (
            env._resolve_path(str(nested / "new.txt")) == nested.resolve() / "new.txt"
        )
        assert env._resolve_path(None) == env._workspace.resolve()

    def test_exec_does_not_inherit_server_environment(self, env):
        from unittest.mock import patch

        # Replace the entire environment with synthetic values before spawning.
        with patch.dict(
            "os.environ",
            {"OPENCLAW_TEST_SECRET": "synthetic-secret", "PATH": "/usr/bin:/bin"},
            clear=True,
        ):
            obs = env.step(
                CallToolAction(
                    tool_name="exec",
                    arguments={
                        "command": 'printf "%s" "${OPENCLAW_TEST_SECRET-unset}"'
                    },
                )
            )
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["exit_code"] == 0
        assert result["stdout"] == "unset"

    def test_memory_search_does_not_follow_external_symlinks(self, env):
        outside = env._workspace.parent / "sentinel.md"
        outside.write_text("synthetic-memory-sentinel")
        (env._workspace / "MEMORY.md").symlink_to(outside)
        (env._workspace / "memory" / "linked.md").symlink_to(outside)
        obs = env.step(
            CallToolAction(
                tool_name="memory_search",
                arguments={"query": "synthetic-memory-sentinel"},
            )
        )
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["results"] == []

    @pytest.mark.parametrize("next_id", ["same-prefix-first", "same-prefix-second"])
    def test_reset_never_reuses_workspace(self, env, next_id):
        env.reset(episode_id="same-prefix-first")
        first = env._workspace
        (first / "MEMORY.md").write_text("previous episode sentinel")
        env.reset(episode_id=next_id)
        assert env._workspace != first
        assert not (env._workspace / "MEMORY.md").exists()
        assert env.state.episode_id == next_id

    @pytest.mark.parametrize(
        "tool,extra",
        [
            ("read", {}),
            ("write", {"content": "changed"}),
            ("edit", {"old_string": "sentinel", "new_string": "changed"}),
            ("memory_get", {}),
            ("exec", {"command": ":"}),
        ],
    )
    def test_tools_reject_escape_without_touching_sentinel(self, env, tool, extra):
        sentinel = env._workspace.parent / "sentinel.txt"
        sentinel.write_text("sentinel")
        key = "workdir" if tool == "exec" else "path"
        obs = env.step(
            CallToolAction(tool_name=tool, arguments={key: "../sentinel.txt", **extra})
        )
        assert obs.error is not None
        assert sentinel.read_text() == "sentinel"

    @pytest.mark.asyncio
    async def test_async_step_count_and_reset(self, env):
        from openenv.core.env_server.types import Action

        actions = [
            ListToolsAction(),
            CallToolAction(
                tool_name="write", arguments={"path": "test.txt", "content": "ok"}
            ),
            CallToolAction(tool_name="missing_tool", arguments={}),
            Action(),
        ]
        for count, action in enumerate(actions, 1):
            await env.step_async(action)
            assert env.state.step_count == count
        env.reset()
        assert env.state.step_count == 0

    @pytest.mark.parametrize("explicit", [False, True])
    def test_configured_workspace_preserves_episode_confinement(
        self, tmp_path, monkeypatch, explicit
    ):
        configured = tmp_path / "configured"
        override = tmp_path / "override"
        monkeypatch.setenv("OPENCLAW_WORKSPACE", str(configured))
        env = OpenClawEnvironment(workspace_dir=str(override) if explicit else None)
        other = OpenClawEnvironment(workspace_dir=str(override) if explicit else None)
        base = override if explicit else configured
        try:
            env.reset(episode_id="../../same-id")
            first = env._workspace
            (first / "MEMORY.md").write_text("previous episode")
            other.reset(episode_id="../../same-id")
            env.reset(episode_id="../../same-id")
            assert len({first, other._workspace, env._workspace}) == 3
            for workspace in [first, other._workspace, env._workspace]:
                assert workspace.parent == base.resolve()
            assert not (env._workspace / "MEMORY.md").exists()
            for path in [str(first / "MEMORY.md"), "../outside.txt"]:
                with pytest.raises(ValueError, match="outside workspace"):
                    env._resolve_path(path)
        finally:
            env.close()
            other.close()

    def test_workspace_defaults_to_system_temp(self, monkeypatch):
        monkeypatch.delenv("OPENCLAW_WORKSPACE", raising=False)
        env = OpenClawEnvironment()
        try:
            assert Path(env._workspace_base) == Path(tempfile.gettempdir())
        finally:
            env.close()

    @pytest.mark.parametrize("limit", [0, 2, 10])
    def test_memory_search_total_before_limit(self, env, limit):
        (env._workspace / "MEMORY.md").write_text("match one\nmatch two\nmatch three\n")
        obs = env.step(
            CallToolAction(
                tool_name="memory_search",
                arguments={"query": "match", "max_results": limit},
            )
        )
        assert obs.error is None
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["total_found"] == 3
        assert len(result["results"]) == min(limit, 3)
