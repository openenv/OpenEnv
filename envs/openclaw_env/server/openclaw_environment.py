# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenClaw Environment Implementation.

An MCP environment that provides OpenClaw's agentic tool capabilities for RL training.
This enables agents to learn workflows involving file operations, shell commands,
web research, and memory management.

All interactions happen through MCP tools:
- exec: Execute shell commands
- read: Read file contents
- write: Write to files
- edit: Make precise file edits
- web_search: Search the web
- web_fetch: Fetch URL content
- memory_search: Search memory files
- memory_get: Get memory snippets

Example:
    >>> from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
    >>> env = OpenClawEnvironment()
    >>> env.reset()
    >>>
    >>> # List available tools
    >>> obs = env.step(ListToolsAction())
    >>> print([t.name for t in obs.tools])
    >>>
    >>> # Execute a command
    >>> obs = env.step(CallToolAction(tool_name="exec", arguments={"command": "echo hello"}))
    >>> print(obs.result)
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State


# Maximum file size to read (50KB)
MAX_FILE_SIZE = 50 * 1024
# Maximum lines to read
MAX_LINES = 2000
# Command timeout in seconds
COMMAND_TIMEOUT = 30
# Workspace directory (created per episode)
WORKSPACE_BASE = tempfile.gettempdir()


class OpenClawEnvironment(MCPEnvironment):
    """
    An MCP environment providing OpenClaw's agentic capabilities.

    This environment exposes file system operations, shell execution,
    web tools, and memory management through MCP tools. It's designed
    for training agents on real-world agentic tasks.

    The environment maintains a fresh workspace per episode. A working directory
    is not a shell sandbox: commands retain the server user's filesystem and
    network permissions. Run only inside an externally isolated container.

    Security:
    - File tool paths are restricted to the workspace
    - Shell commands require external filesystem and network isolation
    - Web tools return simulated results; shell network access is not restricted

    Example:
        >>> from openenv.core.mcp_client import MCPToolClient
        >>>
        >>> with MCPToolClient(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     result = env.call_tool("exec", command="echo hello")
        ...     print(result)
    """

    def __init__(self, workspace_dir: Optional[str] = None):
        """
        Initialize the OpenClaw environment.

        Args:
            workspace_dir: Optional base directory for workspaces.
                If not provided, uses OPENCLAW_WORKSPACE or the system temp directory.
        """
        # Create MCP server and define tools inline
        mcp = FastMCP("openclaw_env")

        # Store reference to self for tool closures
        env = self
        self._workspace_base = (
            workspace_dir or os.environ.get("OPENCLAW_WORKSPACE") or WORKSPACE_BASE
        )
        self._workspace: Optional[Path] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # =========================================================================
        # File System Tools
        # =========================================================================

        @mcp.tool
        def read(
            path: str,
            offset: int = 1,
            limit: int = MAX_LINES,
        ) -> Dict[str, Any]:
            """
            Read contents of a file.

            Supports text files with optional line range. Output is truncated
            to MAX_LINES or 50KB, whichever is hit first.

            Args:
                path: Path to the file (relative to workspace or absolute)
                offset: Line number to start reading from (1-indexed)
                limit: Maximum number of lines to read

            Returns:
                Dictionary with 'content', 'lines_read', 'truncated', and 'path'
            """
            file_path = env._resolve_path(path)

            if not file_path.exists():
                return {"error": f"File not found: {path}"}

            if not file_path.is_file():
                return {"error": f"Not a file: {path}"}

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()

                total_lines = len(lines)
                start_idx = max(0, offset - 1)
                end_idx = min(start_idx + limit, total_lines)

                selected_lines = lines[start_idx:end_idx]
                content = "".join(selected_lines)

                # Truncate by size if needed
                truncated = False
                if len(content) > MAX_FILE_SIZE:
                    content = content[:MAX_FILE_SIZE]
                    truncated = True

                return {
                    "content": content,
                    "lines_read": len(selected_lines),
                    "total_lines": total_lines,
                    "truncated": truncated,
                    "path": str(file_path),
                }
            except Exception as e:
                return {"error": str(e)}

        @mcp.tool
        def write(path: str, content: str) -> Dict[str, Any]:
            """
            Write content to a file.

            Creates the file if it doesn't exist, overwrites if it does.
            Automatically creates parent directories.

            Args:
                path: Path to the file (relative to workspace or absolute)
                content: Content to write to the file

            Returns:
                Dictionary with 'success', 'path', and 'bytes_written'
            """
            file_path = env._resolve_path(path)

            try:
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return {
                    "success": True,
                    "path": str(file_path),
                    "bytes_written": len(content.encode("utf-8")),
                }
            except Exception as e:
                return {"error": str(e), "success": False}

        @mcp.tool
        def edit(
            path: str,
            old_string: str,
            new_string: str,
        ) -> Dict[str, Any]:
            """
            Edit a file by replacing exact text.

            The old_string must match exactly (including whitespace).
            Use this for precise, surgical edits.

            Args:
                path: Path to the file to edit
                old_string: Exact text to find and replace
                new_string: New text to replace the old text with

            Returns:
                Dictionary with 'success', 'replacements', and 'path'
            """
            file_path = env._resolve_path(path)

            if not file_path.exists():
                return {"error": f"File not found: {path}", "success": False}

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if old_string not in content:
                    return {
                        "error": "Old string not found in file",
                        "success": False,
                        "path": str(file_path),
                    }

                # Count replacements
                count = content.count(old_string)
                new_content = content.replace(old_string, new_string)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                return {
                    "success": True,
                    "replacements": count,
                    "path": str(file_path),
                }
            except Exception as e:
                return {"error": str(e), "success": False}

        # =========================================================================
        # Shell Execution Tool
        # =========================================================================

        @mcp.tool
        def exec(
            command: str,
            workdir: Optional[str] = None,
            timeout: int = COMMAND_TIMEOUT,
        ) -> Dict[str, Any]:
            """
            Execute a shell command.

            Commands run in the workspace directory by default.
            Use for running scripts, building projects, or system operations.

            Args:
                command: Shell command to execute
                workdir: Working directory (defaults to workspace)
                timeout: Timeout in seconds (default 30)

            Returns:
                Dictionary with 'stdout', 'stderr', 'exit_code', and 'command'
            """
            cwd = env._resolve_path(workdir) if workdir else env._workspace

            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={
                        "HOME": str(env._workspace),
                        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                        "TERM": "xterm",
                    },
                )

                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.returncode,
                    "command": command,
                }
            except subprocess.TimeoutExpired:
                return {
                    "error": f"Command timed out after {timeout} seconds",
                    "command": command,
                    "exit_code": -1,
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "command": command,
                    "exit_code": -1,
                }

        # =========================================================================
        # Web Tools (Simulated in sandbox mode)
        # =========================================================================

        @mcp.tool
        def web_search(
            query: str,
            count: int = 5,
        ) -> Dict[str, Any]:
            """
            Search the web using Brave Search API.

            Returns titles, URLs, and snippets for research tasks.
            Note: In sandbox mode, returns simulated results.

            Args:
                query: Search query string
                count: Number of results to return (1-10)

            Returns:
                Dictionary with 'results' list containing title, url, snippet
            """
            # In sandbox mode, return simulated results
            # In production, this would call the Brave Search API
            return {
                "query": query,
                "results": [
                    {
                        "title": f"Search result {i + 1} for: {query}",
                        "url": f"https://example.com/result{i + 1}",
                        "snippet": f"This is a simulated search result for '{query}'. "
                        f"In production, this would return real search results.",
                    }
                    for i in range(min(count, 10))
                ],
                "note": "Simulated results in sandbox mode",
            }

        @mcp.tool
        def web_fetch(
            url: str,
            extract_mode: str = "markdown",
            max_chars: int = 10000,
        ) -> Dict[str, Any]:
            """
            Fetch and extract readable content from a URL.

            Converts HTML to markdown or text for lightweight page access.
            Note: In sandbox mode, returns simulated content.

            Args:
                url: HTTP or HTTPS URL to fetch
                extract_mode: Extraction mode ("markdown" or "text")
                max_chars: Maximum characters to return

            Returns:
                Dictionary with 'content', 'url', and metadata
            """
            # In sandbox mode, return simulated content
            return {
                "url": url,
                "content": f"# Simulated Content\n\n"
                f"This is simulated content for URL: {url}\n\n"
                f"In production mode, this would fetch and extract "
                f"the actual page content in {extract_mode} format.",
                "extract_mode": extract_mode,
                "truncated": False,
                "note": "Simulated content in sandbox mode",
            }

        # =========================================================================
        # Memory/Context Tools
        # =========================================================================

        @mcp.tool
        def memory_search(
            query: str,
            max_results: int = 5,
        ) -> Dict[str, Any]:
            """
            Search memory files for relevant context.

            Semantically searches workspace memory files (MEMORY.md, memory/*.md)
            and returns matching snippets with file paths and line numbers.

            Args:
                query: Search query string
                max_results: Maximum number of results to return

            Returns:
                Dictionary with 'results' list containing path, lines, snippet
            """
            results = []
            memory_dir = env._workspace / "memory"

            # Search MEMORY.md if it exists
            memory_file = env._workspace / "MEMORY.md"
            if memory_file.exists():
                results.extend(env._search_file(memory_file, query))

            # Search memory/*.md files
            if memory_dir.exists():
                for md_file in memory_dir.glob("*.md"):
                    results.extend(env._search_file(md_file, query))

            # Simple relevance scoring (count query term occurrences)
            for result in results:
                result["score"] = result["snippet"].lower().count(query.lower())

            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            total_found = len(results)
            results = results[:max_results]

            return {
                "query": query,
                "results": results,
                "total_found": total_found,
            }

        @mcp.tool
        def memory_get(
            path: str,
            from_line: int = 1,
            lines: int = 50,
        ) -> Dict[str, Any]:
            """
            Get a snippet from a memory file.

            Safe snippet read from MEMORY.md or memory/*.md files.
            Use after memory_search to pull specific context.

            Args:
                path: Path to the memory file
                from_line: Starting line number (1-indexed)
                lines: Number of lines to read

            Returns:
                Dictionary with 'content', 'path', 'from_line', 'lines_read'
            """
            # Delegate to read tool with appropriate parameters
            return read(path=path, offset=from_line, limit=lines)

        # Pass the MCP server to the base class
        super().__init__(mcp)

    def _resolve_path(self, path: Optional[str]) -> Path:
        """
        Resolve a path relative to the workspace.

        Args:
            path: Path string (relative or absolute)

        Returns:
            Resolved Path object, constrained to workspace
        """
        workspace = self._workspace.resolve()
        resolved = (workspace / path).resolve() if path is not None else workspace
        if not resolved.is_relative_to(workspace):
            raise ValueError("Path is outside workspace")
        return resolved

    def _search_file(self, file_path: Path, query: str) -> List[Dict[str, Any]]:
        """
        Search a file for lines matching a query.

        Args:
            file_path: Path to the file to search
            query: Search query string

        Returns:
            List of result dicts with path, line_start, snippet
        """
        results = []
        query_lower = query.lower()

        try:
            file_path = self._resolve_path(str(file_path))
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if query_lower in line.lower():
                    # Get context (2 lines before and after)
                    start = max(0, i - 3)
                    end = min(len(lines), i + 2)
                    snippet = "".join(lines[start:end])

                    results.append(
                        {
                            "path": str(file_path.relative_to(self._workspace)),
                            "line_start": start + 1,
                            "line_end": end,
                            "snippet": snippet.strip(),
                        }
                    )
        except Exception:
            pass

        return results

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment.

        Creates a fresh workspace directory for the new episode.

        Args:
            seed: Optional random seed
            episode_id: Optional episode ID to use
            **kwargs: Additional reset options

        Returns:
            Observation indicating the environment is ready
        """
        # Generate episode ID
        ep_id = episode_id or str(uuid4())

        # Create workspace directory
        Path(self._workspace_base).mkdir(parents=True, exist_ok=True)
        self._workspace = Path(
            tempfile.mkdtemp(prefix="openclaw_env_", dir=self._workspace_base)
        ).resolve()

        # Create initial directory structure
        (self._workspace / "memory").mkdir(exist_ok=True)

        # Initialize state
        self._state = State(episode_id=ep_id, step_count=0)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "OpenClaw environment ready!",
                "workspace": str(self._workspace),
                "episode_id": ep_id,
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.

        This environment only supports MCP actions (ListToolsAction, CallToolAction).
        Any other action type returns an error observation.

        Args:
            action: The action to execute
            timeout_s: Optional timeout
            **kwargs: Additional arguments

        Returns:
            Observation with error for unknown action types
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions. Increments step count.

        Args:
            action: The MCP action to execute
            timeout_s: Optional timeout for the action
            **kwargs: Additional arguments

        Returns:
            Observation from the action execution
        """
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute an async step and increment the episode step count."""
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id, step_count, and workspace path
        """
        return self._state

    def close(self) -> None:
        """
        Clean up the environment.

        Optionally removes the workspace directory.
        """
        # Note: We don't delete the workspace by default to allow inspection
        # In production, you might want to clean up
        super().close()
