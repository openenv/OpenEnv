#!/usr/bin/env python3

"""
Git Task Environment - Optimized for task-based isolation.

This module provides an optimized Git environment for scenarios where:
- Multiple tasks share the same base repository states
- Tasks need fast reset() to reproducible states
- Each task has an isolated workspace
- A shared Gitea service provides repository storage
"""

import uuid

from openenv.core.env_server import Action, Environment, Observation

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import GitAction, GitObservation, GitState
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import GitAction, GitObservation, GitState
from openenv.core.tools.git_server_client import GitServerClient


class GitTaskEnvironment(Environment):
    """
    Git Environment optimized for task-based isolation.

    This environment connects to a shared Gitea service and provides:
    - Fast reset() via git operations (no server restart)
    - Isolated workspace per environment instance
    - Shared repository cache across tasks
    - Reproducible base states from specific commits

    Architecture:
        Shared Gitea Service (external)
            ↓
        GitTaskEnvironment instances (many)
            ↓
        Isolated workspaces (/workspace)

    Args:
        gitea_url: URL of shared Gitea service (e.g., "http://gitea:3000")
        username: Gitea username for authentication
        password: Gitea password for authentication
        workspace_dir: Directory for git operations (default: /workspace)
        task_repos: Dict mapping task names to (repo_name, commit) tuples
                   for pre-configuring task base states

    Example (Basic):
        >>> env = GitTaskEnvironment(gitea_url="http://localhost:3000")
        >>> obs = env.reset()
        >>> # Clone and work
        >>> from ..models import GitAction
        >>> obs = env.step(GitAction(action_type="clone_repo", repo_name="my-repo"))
        >>> obs = env.step(GitAction(action_type="execute_git_command", command="status", working_dir="my-repo"))

    Example (Task-based):
        >>> # Pre-configure tasks with specific repo states
        >>> env = GitTaskEnvironment(
        ...     gitea_url="http://localhost:3000",
        ...     task_repos={
        ...         "task1": ("my-repo", "abc123"),  # Specific commit
        ...         "task2": ("my-repo", "def456"),  # Different commit
        ...     }
        ... )
        >>> # Reset to task1 base state
        >>> obs = env.reset(task_id="task1")  # Fast! Just git reset
        >>> # Work on task...
        >>> # Reset to task2 base state
        >>> obs = env.reset(task_id="task2")  # Fast reset to different state
    """

    def __init__(
        self,
        gitea_url: str,
        username: str,
        password: str,
        workspace_dir: str = "/workspace",
        task_repos: dict[str, tuple[str, str]] | None = None,
    ):
        """Initialize Git Task Environment."""
        super().__init__()
        self.workspace_dir = workspace_dir
        self.task_repos = task_repos or {}

        # Initialize Git server client (connects to external Gitea)
        self._git_client = GitServerClient(
            gitea_url=gitea_url,
            username=username,
            password=password,
            workspace_dir=workspace_dir,
        )

        # Initialize state
        self._state = GitState(workspace_path=workspace_dir)
        self._current_task_id: str | None = None

        # Wait for Gitea to be ready
        if self._git_client.wait_for_ready():
            self._state.gitea_ready = True
        else:
            print("Warning: Gitea server not ready")
            self._state.gitea_ready = False

    def reset(self, task_id: str | None = None) -> Observation:
        """
        Reset environment to clean state.

        This is optimized for task-based workflows:
        - If task_id specified and configured: fast reset to that task's base state
        - If workspace exists: git reset --hard (very fast, <1s)
        - Otherwise: clone from Gitea (slower, ~5-10s)

        Args:
            task_id: Optional task identifier for task-specific base states

        Returns:
            Initial observation indicating environment is ready
        """
        # Initialize fresh state
        self._state = GitState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            gitea_ready=self._git_client.is_ready,
            workspace_path=self.workspace_dir,
        )

        self._current_task_id = task_id

        # If task_id provided and configured, set up task base state
        if task_id and task_id in self.task_repos:
            repo_name, commit = self.task_repos[task_id]

            try:
                if self._git_client.workspace_exists(repo_name):
                    # Fast path: workspace exists, just reset
                    self._git_client.reset_workspace(repo_name, commit)
                    message = f"Reset to task '{task_id}' base state (repo: {repo_name}@{commit})"
                else:
                    # Slower path: clone fresh
                    self._git_client.clone_to_workspace(repo_name, commit=commit)
                    message = (
                        f"Initialized task '{task_id}' (repo: {repo_name}@{commit})"
                    )

                current_commit = self._git_client.get_current_commit(repo_name)

                return GitObservation(
                    success=True,
                    message=message,
                    output=f"Workspace: {self.workspace_dir}/{repo_name}\nCommit: {current_commit}\nTask: {task_id}",
                )
            except Exception as e:
                return GitObservation(
                    success=False,
                    message=f"Failed to reset task '{task_id}'",
                    error=str(e),
                )

        # Default reset: just ready state, no pre-configured repos
        return GitObservation(
            success=True,
            message="Git task environment ready.",
            output=f"Workspace: {self.workspace_dir}\nGitea: {self._git_client.gitea_url}\nUse GitAction with action_type='clone_repo' to clone repositories.",
        )

    def step(self, action: Action) -> Observation:
        """
        Execute a Git action and return observation.

        Supported action types:
        - "clone_repo": Clone repository to workspace
        - "execute_git_command": Execute git command
        - "list_repos": List available repositories

        Args:
            action: GitAction to execute

        Returns:
            GitObservation with execution results
        """
        if not isinstance(action, GitAction):
            raise ValueError(f"Expected GitAction, got {type(action)}")

        # Update step count
        self._state.step_count += 1

        # Route to appropriate handler based on action_type
        try:
            if action.action_type == "clone_repo":
                return self._handle_clone_repo(action)
            elif action.action_type == "list_repos":
                return self._handle_list_repos(action)
            elif action.action_type == "execute_git_command":
                return self._handle_git_command(action)
            else:
                return GitObservation(
                    success=False,
                    message=f"Action not supported in task mode: {type(action).__name__}",
                    error="Use shared Gitea for repository migration/creation",
                )
        except Exception as e:
            return GitObservation(
                success=False, message=f"Action failed: {str(e)}", error=str(e)
            )

    def _handle_clone_repo(self, action: GitAction) -> GitObservation:
        """Handle repository clone action."""
        try:
            # Determine commit to use
            commit = "main"  # Default

            # If this repo is part of current task config, use that commit
            if self._current_task_id and self._current_task_id in self.task_repos:
                task_repo, task_commit = self.task_repos[self._current_task_id]
                if task_repo == action.repo_name:
                    commit = task_commit

            clone_path = self._git_client.clone_to_workspace(
                action.repo_name, action.target_dir, commit=commit
            )

            return GitObservation(
                success=True,
                message=f"Successfully cloned {action.repo_name}",
                output=f"Cloned to: {clone_path}\nCommit: {commit}",
            )
        except Exception as e:
            return GitObservation(
                success=False,
                message=f"Failed to clone repository: {action.repo_name}",
                error=str(e),
            )

    def _handle_list_repos(self, action: GitAction) -> GitObservation:
        """Handle list repositories action."""
        try:
            repos = self._git_client.list_repositories()

            # Format output
            if not repos:
                output = "No repositories available."
            else:
                output = "Available repositories:\n"
                for repo in repos:
                    output += f"  - {repo['name']}: {repo['clone_url']}\n"
                    if repo.get("description"):
                        output += f"    {repo['description']}\n"

            return GitObservation(
                success=True,
                message=f"Found {len(repos)} repositories",
                output=output,
                repos=repos,
            )
        except Exception as e:
            return GitObservation(
                success=False, message="Failed to list repositories", error=str(e)
            )

    def _handle_git_command(self, action: GitAction) -> GitObservation:
        """Handle git command execution action."""
        try:
            exit_code, stdout, stderr = self._git_client.execute_git_command(
                action.command, action.working_dir
            )

            success = exit_code == 0
            message = f"Git command {'succeeded' if success else 'failed'}"

            return GitObservation(
                success=success, message=message, output=stdout, error=stderr
            )
        except Exception as e:
            return GitObservation(
                success=False,
                message=f"Failed to execute git command: {action.command}",
                error=str(e),
            )

    @property
    def state(self) -> GitState:
        """Get current environment state."""
        return self._state
