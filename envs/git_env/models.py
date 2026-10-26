#!/usr/bin/env python3

"""
envs/git_env/models.py
--------------------------------
Action/Observation types for the Git environment with Gitea server.
"""

from __future__ import annotations

from typing import Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class GitAction(Action):
    """
    Action for Git environment operations.

    This unified action class supports multiple operation types:
    - clone_repo: Clone a repository from Gitea to workspace
    - list_repos: List all available repositories
    - execute_git_command: Execute a git command in workspace

    Attributes:
        action_type: Type of operation ("clone_repo", "list_repos", "execute_git_command")
        repo_name: Name of repository (for clone_repo, execute_git_command)
        target_dir: Target directory for clone (optional)
        command: Git command to execute (for execute_git_command)
        working_dir: Working directory relative to workspace (for execute_git_command)
    """

    action_type: str = "list_repos"
    repo_name: str = ""
    target_dir: Optional[str] = None
    command: str = ""
    working_dir: str = ""


class GitObservation(Observation):
    """
    Result of executing a Git action.

    Attributes:
        success: Whether the action was successful
        message: Human-readable message about the result
        output: Command output or detailed result
        error: Error message if action failed
        repos: List of repositories (for list_repos action)
    """

    success: bool = False
    message: str = ""
    output: str = ""
    error: str = ""
    repos: list[dict[str, str]] = Field(default_factory=list)


class GitState(State):
    """
    State for Git environment.

    Attributes:
        episode_id: Unique identifier for the episode
        step_count: Number of steps taken
        gitea_ready: Whether Gitea server is accessible
        workspace_path: Path to the workspace directory
    """

    gitea_ready: bool = False
    workspace_path: str = "/workspace"
