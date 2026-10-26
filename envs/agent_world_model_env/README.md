---
title: Agent World Model Environment Server 
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Agent World Model

AgentWorldModel-1K is a synthetic agentic environment suite containing **1,000 tool-use environments** with **10,000 tasks** for large-scale RL training. Each environment is a fully functional MCP server with tools, database state, and verification logic.


## Quick Start

You can interact with the AWM environments at Huggingface Space : [ChilleD/agent_world_model_env](https://huggingface.co/spaces/ChilleD/agent_world_model_env) 🤗.

### 1. Start the Server

```bash
# From the OpenEnv root directory
PYTHONPATH=src:envs uv run uvicorn envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899
```

### 2. Connect with the Client

```python
import asyncio
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

async def main():
    async with AWMEnv(base_url="http://localhost:8899") as env:
        # Reset to a scenario with a specific task
        result = await env.reset(scenario="e_commerce_33", task_idx=0)
        print(f"Task: {result.observation.task}")
        print(f"Tools available: {result.observation.num_tools}")
        print(f"Verifier support: {result.observation.has_verifier}")  # {sql: True, code: True}

        # List available tools
        tools = await env.list_tools()
        for tool in tools[:3]:
            print(f"  - {tool.name}: {tool.description}")

        # Call a tool
        obs = await env.call_tool("search_products", query="headphones")
        print(f"Result: {obs.tool_result}")

        # Run verification (can be called multiple times with different modes)
        result = await env.step(CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": "optional answer"}
        ))
        print(f"Reward type: {result.observation.reward_type}")
        print(f"Reward: {result.reward}")
        print(f"Verify result: {result.observation.verify_result}")

        # End episode (destroys subprocess; set keep_session=True to preserve files)
        result = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
        print(f"Episode done: {result.done}")

asyncio.run(main())
```

## Environment Details

### Actions

AWM supports two action types:

| Action | Description |
|--------|-------------|
| `ListToolsAction()` | List all available MCP tools for the current scenario |
| `CallToolAction(tool_name, arguments)` | Call a specific tool with arguments |

Special tool names:
- `"verify"` - Run verifier with `{verifier_mode: "sql"|"code", final_answer: "optional"}` arguments
- `"done"` - End the episode and destroy subprocess (does NOT run verifier)
- `"__list_scenarios__"` - List all 1,000 available scenarios and their tasks

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| `reward` | float | Reward value based on reward_type and config |
| `reward_type` | str | Outcome classification (see below) |
| `scenario` | str | Current scenario name |
| `task` | str | Task description in natural language |
| `task_idx` | int | Task index (0-9) |
| `has_verifier` | dict/None | Verifier support: `{sql: bool, code: bool}` or None |
| `num_tools` | int | Number of tools available |
| `tool_name` | str | Name of the tool called |
| `tool_result` | Any | Result from the tool call |
| `error` | str | Error message if any |
| `verify_result` | dict | Verification output after calling verify |
| `trajectory_path` | str | Path to saved trajectory JSON (after `done`) |
| `session_dir` | str | Path to session directory (only if `keep_session=True`) |

### Reward Types and Values

Default reward configuration:

| Type | Reward | Description |
|------|--------|-------------|
| `complete` | 1.0 | Task completed successfully (verifier passed) |
| `incomplete` | 0.1 | Task not completed (verifier failed) |
| `format_error` | -1.0 | Format error (maps from tool_not_found, invalid_args) |
| `tool_not_found` | -1.0 | Tool name not recognized |
| `invalid_args` | -1.0 | Tool arguments invalid |
| Other types | 0.0 | server_error, timeout, etc. |

You can customize rewards at reset:
```python
result = await env.reset(
    scenario="e_commerce_33",
    task_idx=0,
    reward_config={"complete": 1.0, "incomplete": 0.0, "format_error": 0.0}
)
```

## Session Artifacts

When calling `done(keep_session=True)`, the session directory is preserved with:

| File | Description |
|------|-------------|
| `trajectory.json` | Full episode trajectory (scenario, task, steps, each action/result) |
| `{scenario}.db` | SQLite database after agent interaction (final state) |
| `{scenario}_initial.db` | SQLite database snapshot before agent interaction |
| `server.py` | Patched Python code for the launched environment |
| `server.log` | Launched environment uvicorn logs (startup + HTTP requests) |

When `keep_session=False` (default), all files are cleaned up after the episode.

## Verifier Modes

AWM supports two verification modes, selected when calling the `verify` tool:

### Code Mode (Default, no LLM needed)

```python
result = await env.step(CallToolAction(
    tool_name="verify",
    arguments={"verifier_mode": "code", "final_answer": "optional answer"}
))
```

Executes a Python verifier function that compares initial and final database states. Deterministic and does not require LLM.

### SQL Mode (code-augmented LLM-as-a-Judge)

This mode is recommended for judge performance. You need to set the LLM credentials via environment variables before using this mode.

```python
# Set LLM credentials via environment variables
# OPENENV_AWM_LLM_BASE_URL, OPENENV_AWM_LLM_API_KEY, OPENENV_AWM_LLM_MODEL

result = await env.step(CallToolAction(
    tool_name="verify",
    arguments={"verifier_mode": "sql"}
))
```

Runs SQL queries to extract state changes, then uses an LLM judge to determine success.


## Listing Scenarios & Tasks

```python
async with AWMEnv(base_url="http://localhost:8899") as env:
    # List all 1,000 scenarios
    result = await env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))

    print(f"Total scenarios: {result.observation.total}")
    for scenario in result.observation.scenarios[:5]:
        print(f"  - {scenario['name']}: {scenario['num_tasks']} tasks")
        print(f"    Sample task: {scenario['tasks'][0][:80]}...")
```

## Server Monitoring

The server exposes a `/stats` endpoint for monitoring active sessions:

```bash
curl http://localhost:8899/stats
```

Returns: `total_sessions`, `max_idle_time_config`, `cleanup_interval_config`, `scenarios` breakdown, and `max_idle_s`.

A background cleanup daemon automatically kills sessions idle longer than `MAX_IDLE_TIME` (default 600s) when total sessions exceed `ALLOWED_IDLE_SESSIONS` (default 3000).

## Full Agent Interaction Example

See [`examples/agent_world_model/example_usage.py`](../../examples/agent_world_model/example_usage.py) for a complete example of an LLM-powered agent that:

1. Discovers available tools via `list_tools`
2. Iteratively calls tools to accomplish the task
3. Runs verification via `verify` tool (can use "sql" or "code" mode)
4. Ends episode via `done` action with `keep_session=True` to inspect artifacts

The example supports both a local server and the public Hugging Face Space, set `AWM_BASE_URL=https://chilled-agent-world-model-env.hf.space` (may be slow) to try without local setup.


## Large-Scale RL Training

AWM is designed for large-scale agentic RL. A single server supports thousands of concurrent WebSocket sessions, each with its own isolated environment subprocess.

### Simulated Stress Test

A stress test simulating large-scale RL is included:

```bash
# after server started, then in another terminal:
PYTHONPATH=src:envs uv run python examples/agent_world_model/example_stress_test.py \
    --scale 1024 --concurrency 64 --min-turns 3 --max-turns 20 \
    --think-min 3.0 --think-max 30.0
```

This launches 1024 parallel episodes, each with 3-20 multi-turn tool interactions and 3-30s simulated LLM rollout time per turn.

## AWM Server Configuration

Server configuration is in `server/config.py`, overridable via environment variables:

| Config | Default | Env Var | Description |
|--------|---------|---------|-------------|
| `MAX_CONCURRENT_ENVS` | 10000 | — | Max WebSocket sessions |
| `READY_TIMEOUT` | 180s | `OPENENV_AWM_READY_TIMEOUT` | Subprocess startup timeout |
| `MAX_PORT_RETRIES` | 5 | `OPENENV_AWM_MAX_PORT_RETRIES` | Port-retry attempts on startup failure |
| `RETRY_READY_TIMEOUT` | 30s | `OPENENV_AWM_RETRY_READY_TIMEOUT` | Shorter timeout for retry attempts |
| `READY_POLL_INTERVAL` | 0.5s | — | Polling interval during startup check |
| `MAX_IDLE_TIME` | 600s | `OPENENV_AWM_MAX_IDLE_TIME` | Idle session cleanup threshold |
| `ALLOWED_IDLE_SESSIONS` | 3000 | `OPENENV_AWM_ALLOWED_IDLE_SESSIONS` | Session count before idle cleanup triggers |
| `CLEANUP_INTERVAL` | 5s | `OPENENV_AWM_CLEANUP_INTERVAL` | Cleanup daemon scan interval |

## Warning

AWM treats verifier code and scenario code from the curated [AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) dataset as **trusted**. Verifier code (`server/_verifier_runner.py`) is run in a subprocess sandbox (rlimits, restricted builtins, import allowlist); scenario subprocesses run without per-process sandboxing and rely on the container as the outer isolation boundary. The codes are synthetically generated and carefully curated, however, there is no guarantee of absolute safety. We recommend only academic research use.

## Citation

More details can be found at:

| Resource | Link |
|----------|------|
| Hugging Face Space | [ChilleD/agent_world_model_env](https://huggingface.co/spaces/ChilleD/agent_world_model_env) |
| Paper | [arxiv.org/abs/2602.10090](https://arxiv.org/abs/2602.10090) |
| Synthesis Pipeline Code | [Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model) |
| AgentWorldModel-1K | [Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) |
| Arctic-AWM-4B | [Snowflake/Arctic-AWM-4B](https://huggingface.co/Snowflake/Arctic-AWM-4B) |
| Arctic-AWM-8B | [Snowflake/Arctic-AWM-8B](https://huggingface.co/Snowflake/Arctic-AWM-8B) |
| Arctic-AWM-14B | [Snowflake/Arctic-AWM-14B](https://huggingface.co/Snowflake/Arctic-AWM-14B) |

If you find this work useful, please kindly cite:

```bibtex
@article{wang2026agentworldmodelinfinity,
      title={Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning},
      author={Zhaoyang Wang and Canwen Xu and Boyi Liu and Yite Wang and Siwei Han and Zhewei Yao and Huaxiu Yao and Yuxiong He},
      year={2026},
      eprint={2602.10090},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.10090},
}
```
