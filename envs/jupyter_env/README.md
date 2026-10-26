---
title: Jupyter Environment Server
emoji: 📓
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - jupyter
  - coding
short_description: Stateful Jupyter notebook environment backed by E2B
---

# Jupyter Environment

`jupyter_env` exposes a stateful Jupyter-style Python notebook through MCP
tools. Each episode creates a fresh E2B Code Interpreter sandbox, so Python
variables, imports, files, and generated plots persist across tool calls until
the next reset.

Reset can also receive setup and verify scripts. Setup commands run immediately
after the sandbox is created. Verify commands are stored for the episode and run
when the agent calls `final_answer`.

## Tools

- `add_and_execute_code_cell(code)`: execute a new Python cell.
- `edit_and_execute_current_cell(code)`: replace and re-run the latest code cell.
- `execute_shell_command(command)`: run shell commands inside the sandbox.
- `get_notebook_state(include_images=False)`: summarize recent notebook cells.
- `final_answer(answer)`: record a final answer and run any configured verify
  commands.

## Quick Start

```python
from jupyter_env import JupyterEnv

with JupyterEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(
        setup=["pip install -q pandas"],
        verify=[
            "python - <<'PY'\nfrom pathlib import Path\nassert Path('/home/user/work/answer.txt').exists()\nPY"
        ],
    )
    print(env.call_tool("add_and_execute_code_cell", code="x = 21 * 2\nx"))
    print(env.call_tool("final_answer", answer="done"))
```

## Local Server

```bash
cd envs/jupyter_env
E2B_API_KEY=e2b_... uv run --project . server
```

The API and custom web UI are served on port 8000. The notebook UI is mounted
at `/web`.

## Docker

```bash
cd envs/jupyter_env
openenv build -t jupyter-env
docker run -p 8000:8000 -e E2B_API_KEY=e2b_... jupyter-env
```

## Configuration

- `E2B_API_KEY`: required when resetting an episode.
- `MAX_CONCURRENT_ENVS`: maximum concurrent WebSocket sessions. Defaults to `4`.
- `KAGGLE_DATA_DIR`: optional root directory for reset-time Kaggle file staging.

## Setup and Verify Commands

`reset()` accepts either `setup` / `verify` or `setup_scripts` /
`verify_scripts`.

```python
env.reset(
    setup=[
        "mkdir -p /home/user/work",
        "printf 'seed data' > /home/user/work/input.txt",
    ],
    verify=[
        "test -f /home/user/work/answer.txt",
        "python -m pytest /home/user/work/tests",
    ],
)
```

Setup failure ends the reset response with `done=True` and returns the captured
setup results. Verify commands run when `final_answer(answer)` is called. Reward
defaults to `passed_verify_commands / total_verify_commands`. A verify command
can override this by writing a float to:

```text
/home/user/logs/verifier/reward.txt
```

The file is deleted before the verify commands run, so only a verify command
can set it; anything the agent writes there earlier is discarded. The value
must be a finite number; a verify command is written by the task author, so
negative rewards and values above 1 are accepted too. A value that is not a
finite number is ignored, the pass rate is used, and the reason is recorded in
`state.reward_override_ignored`.

## Notes

This first version intentionally keeps sandbox provider selection local to the
environment and uses E2B as the concrete backend. A shared OpenEnv sandbox API
can be introduced later once the provider contract is RFC-backed.
