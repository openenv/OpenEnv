<!-- openenv-source: git_env -->
# Git Environment

A Git server environment using Gitea that provides isolated Git repository management optimized for task-based RL training. Perfect for training agents on Git operations with fast reset capabilities.

## Overview

The Git Environment connects to a **shared external Gitea service** for optimal task-based isolation. **Perfect for**: RL training, task-based workflows, parallel execution

### Architecture

```
┌────────────────────────────────────┐
│ Shared Gitea (start once)          │
│ Port 3000                          │
│ - Pre-migrated repositories        │
└──────────────┬─────────────────────┘
               │ HTTP API
      ┾────────┼────────┾
      │        │        │
  ┌───▼──┐ ┌──▼───┐ ┌──▼───┐
  │Env 1 │ │Env 2 │ │Env 3 │
  │Task A│ │Task B│ │Task A│
  │@abc  │ │@def  │ │@abc  │
  └──────┘ └──────┘ └──────┘
  Isolated workspaces
```

## Quick Start

```python
from envs.git_env import GitAction, GitEnv

# Create environment from Docker image
git_env = GitEnv.from_docker_image("git-env:latest").sync()

# Reset environment
result = git_env.reset()
print(result.observation.message)

# List available repositories (pre-migrated to shared Gitea)
result = git_env.step(GitAction(action_type="list_repos"))
for repo in result.observation.repos:
    print(f"{repo['name']}: {repo['clone_url']}")

# Clone to workspace
result = git_env.step(GitAction(action_type="clone_repo", repo_name="OpenEnv"))
print(result.observation.output)  # Cloned to: /workspace/OpenEnv

# Execute git commands
result = git_env.step(GitAction(
    action_type="execute_git_command",
    command="status",
    working_dir="OpenEnv"
))
print(result.observation.output)

# Cleanup
git_env.close()
```

## Setup and Running the Example

Complete setup (run these steps in order):

```bash
# 0. Configure environment variables
cp .env.example .env
# Edit .env and set your Gitea credentials if needed

# 1. Start shared Gitea service (one-time)
./scripts/setup_shared_gitea.sh

# 2. Migrate a test repository to Gitea (one-time)
docker exec openenv-gitea curl -X POST \
  http://localhost:3000/api/v1/repos/migrate \
  -u gitea:gitea123 \
  -H 'Content-Type: application/json' \
  -d '{
    "clone_addr": "https://github.com/huggingface/OpenEnv",
    "repo_name": "OpenEnv",
    "repo_owner": "gitea",
    "service": "github"
  }'

# 3. Build Docker images
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t git-env:latest -f envs/git_env/server/Dockerfile .

# 4. Install Python dependencies
uv pip install -e .

# 5. Run the example (loads credentials from .env)
python3 examples/local_git_env.py
```

**Note**:
- Steps 1-3 are one-time setup
- Make sure `.env` file exists with your Gitea credentials
- After initial setup, you only need step 5 to run the example

## Environment Details

### Actions

**GitAction**: Unified action class for all Git operations

```python
@dataclass
class GitAction(Action):
    action_type: str           # Operation type
    repo_name: str            # Repository name (for clone/execute)
    target_dir: Optional[str] # Target directory (for clone)
    command: str              # Git command (for execute)
    working_dir: str          # Working directory (for execute)
```

**Supported action_type values:**

#### "clone_repo" - Clone repository to workspace
```python
GitAction(action_type="clone_repo", repo_name="OpenEnv")
GitAction(action_type="clone_repo", repo_name="OpenEnv", target_dir="custom-dir")
```

#### "list_repos" - List available repositories
```python
GitAction(action_type="list_repos")
```

#### "execute_git_command" - Execute git command
```python
GitAction(
    action_type="execute_git_command",
    command="status",
    working_dir="OpenEnv"
)
```

### Observation

**GitObservation**: Contains results of Git operations

```python
@dataclass
class GitObservation(Observation):
    success: bool          # Whether operation succeeded
    message: str           # Human-readable message
    output: str            # Command output or detailed result
    error: str             # Error message if failed
    repos: list[dict]      # List of repositories (for list_repos)
```

### State

**GitState**: Tracks environment state

```python
@dataclass
class GitState(State):
    episode_id: str           # Unique episode identifier
    step_count: int           # Number of steps taken
    gitea_ready: bool         # Whether Gitea is accessible
    workspace_path: str       # Path to workspace directory
```

## Advanced: Task-Based Training

For RL training scenarios where you need fast resets to specific repository states, you can configure task-specific base states in the environment. This is done by setting environment variables before starting containers:

```bash
# Example: Configure tasks for your training setup
docker run \
  -e GITEA_URL=http://host.docker.internal:3000 \
  -e TASK_REPOS='{"bug_fix": ["my-repo", "abc123"], "feature": ["my-repo", "def456"]}' \
  git-env:latest
```

Then in your training code, environments automatically reset to the configured state.

See [`examples/local_git_env.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/local_git_env.py) for a complete
synchronous example that resolves the Docker bootstrap handle with `.sync()`.

## Project Structure

```
git_env/
├── README.md                      # This file
├── __init__.py                    # Exports
├── models.py                      # Action, Observation, State definitions
├── client.py                      # GitEnv HTTP client
├── docker-compose.gitea.yml       # Shared Gitea service
└── server/
    ├── __init__.py
    ├── git_task_environment.py    # Task-optimized environment
    ├── app.py                     # FastAPI application
    └── Dockerfile                 # Lightweight container image
```

## Troubleshooting

### Gitea Not Ready

If environment can't connect to Gitea:
1. Ensure Gitea is running: `docker ps | grep gitea`
2. Check Gitea URL in environment: `GITEA_URL=http://gitea:3000`
3. Verify network connectivity: `docker network ls | grep openenv`

### Repository Not Found

Ensure repository is migrated to Gitea:
```bash
# List repos
curl -u gitea:gitea123 http://localhost:3000/api/v1/user/repos
```

### Slow Clone/Reset

- First clone is slower (~5-10s) - downloads from Gitea
- Subsequent resets are fast (<1s) - just git operations
- Use task-based mode with `task_repos` for optimal performance


## Security Notes

- **Never commit `.env` file** - it contains credentials (already in .gitignore)
- Use `.env.example` as a template and create your own `.env`
- Gitea credentials are for local development only
- For production, use proper secret management (Docker secrets, k8s secrets, etc.)
- All workspaces are isolated per container
- Only public repositories supported (no private repo auth)