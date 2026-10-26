# Simulation vs Production

An OpenEnv server runs in one of two modes:

- **Simulation mode** (the default) is for training and evaluation. The caller owns the episode through `reset()`, `step()` and `state()`, and gets a reward and a `done` signal back.
- **Production mode** serves the environment's MCP tools directly to an agent or another client, with no episode control.

The same environment class supports both. Simulation mode models trajectory time, production mode models service time.

## Routes and Boundaries

OpenEnv keeps two boundaries apart:

- **Infrastructure boundary**: `/ws`, `/reset`, `/step` and `/state`, used by the training or evaluation loop.
- **Agent boundary**: MCP tools over `/mcp`.

| Route | Simulation | Production |
|---|---|---|
| `/mcp` | yes | yes |
| `/ws` | yes | yes (infrastructure only) |
| `/reset`, `/step`, `/state` | yes | no |
| `/harness` | no | yes, when the environment is a `HarnessEnvironment` |

Production mode still registers `/ws`, but agents should never reach it. If agents can reach the service directly, restrict `/ws` at the network, auth or gateway layer.

## Setting the Mode

Pass the mode to `create_app`, or set `OPENENV_MODE` (`simulation` or `production`) when the app is created without one:

```python
from openenv.core.env_server.http_server import create_app

app = create_app(MyEnv, MyAction, MyObservation, mode="production")
```

```bash
OPENENV_MODE=production uvicorn server.app:app --host 0.0.0.0 --port 8000
```

With `HTTPEnvServer` directly, the mode is chosen when routes are registered:

```python
from fastapi import FastAPI

from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.types import ServerMode

app = FastAPI()
server = HTTPEnvServer(env=MyEnv, action_cls=MyAction, observation_cls=MyObservation)
server.register_routes(app, mode=ServerMode.PRODUCTION)  # default: ServerMode.SIMULATION
```

## MCP Tools in Each Mode

An MCP environment is still an OpenEnv environment, and it can run in either mode.

In **simulation mode**, tool calls are actions. Send them through `step()` so they count as steps, get rewards, can end the episode and land in the trajectory:

```python
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

obs = env.step(ListToolsAction())
obs = env.step(CallToolAction(tool_name="echo_message", arguments={"message": "Hello"}))
```

`examples/echo_mcp_demo.py` runs this pattern end to end.

In **production mode**, clients call the tools directly through `/mcp`, like any MCP server.

### `step(CallToolAction(...))` or `call_tool()`

Environment clients built on `MCPToolClient` (`EchoEnv`, `FinQAEnv`, …) also have `list_tools()` and `call_tool()`. These go to `/mcp`, not through `step()`, so they never produce a reward, a step count or `done`.

| | `step(CallToolAction(...))` | `await client.call_tool(name, **kwargs)` |
|---|---|---|
| Goes through | `step()` (simulation) | `/mcp` |
| Returns | a `CallToolObservation` (`reward`, `done`, metadata, `obs.result`), wrapped in a `StepResult` over HTTP | the tool's unwrapped return value |
| On a tool error | an observation you can inspect (`ToolError.error_type`) | raises `RuntimeError` |

Over HTTP, `step()` returns a `StepResult`: the observation is `result.observation` and the reward is `result.reward`. `obs.result` holds the tool's return value as the tool produced it, often a FastMCP `CallToolResult` (`.data`, `.content`, `.structured_content`), or a dict or plain value from other environments.

Use `step()` for training and evaluation, and `call_tool()` when you only need a tool's output. `MCPToolClient` only supports `mode="production"` and raises `ValueError` otherwise. For simulation over HTTP use the environment's `EnvClient` or `GenericEnvClient(base_url=..., mode="simulation")`. `list_tools()` returns an empty list when the request fails.

Tool calls sent through `step()` time out after 30 seconds by default (`MCP_TOOL_CALL_TIMEOUT`). `MCPEnvironment.step()` takes a `timeout_s`, so an environment whose tools wait on slow work, such as an LLM call, can pass a larger value from its own `step()`.

### Mode-Aware Tools

`MCPEnvironment` can expose different tools in each mode:

```python
class MyEnv(MCPEnvironment):
    def __init__(self):
        @self.tool(mode="simulation")
        def score_candidate(answer: str) -> str:
            return "Used inside the training loop"

        @self.tool(mode="production")
        def lookup_docs(query: str) -> str:
            return "Used by live MCP clients"
```

## Harness Environments: `WS /harness`

When the environment factory produces a [`HarnessEnvironment`](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md) (an agent harness such as Claude Code running inside the environment), production mode also registers a `/harness` WebSocket, so clients talk to the harness itself:

- Each connection gets its own session, which starts the harness and injects the environment's tools. The server answers with a `session_started` frame.
- Each `{"type": "message", "content": "..."}` frame is one conversational turn, streamed back as harness events that end with `turn_complete`.
- A turn is bounded by the harness config's `session_timeout_s`, and a failure ends the session with an `error` event.

[Evaluate Claude Code in an Environment](../tutorials/claude-code-harness) serves a harness this way.

## Debugging: "`step()` Is Not Called"

The WebSocket handler calls `step_async()` when the environment overrides it, and `step()` otherwise (the same for `reset()` and `reset_async()`). An async client can therefore run your action without ever hitting instrumentation you put only in `step()`. If an action seems to skip `step()`, check that:

1. You instrumented both `step()` and `step_async()`.
2. You are not using `call_tool()`, which goes through `/mcp` and never reaches `step()`.

## Related Reading

- [MCP Environments](../tutorials/mcp-environment), with [Echo](../environments/echo) and [FinQA](../environments/finqa) as examples
- [Core API](../reference/core)
- [RFC 002: Environment Spec](https://github.com/huggingface/OpenEnv/blob/main/rfcs/002-env-spec.md) and [RFC 005: Agentic Harnesses](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md)
