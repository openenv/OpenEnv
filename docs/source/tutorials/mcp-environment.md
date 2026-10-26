# MCP Environments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/mcp_environment.ipynb)

Most agentic work ends up needing the same thing: a way for the model to **call tools** and receive structured feedback, whether that is during RL training or offline evaluation. OpenEnv standardises that surface with **[MCP](https://modelcontextprotocol.io)** (Model Context Protocol), so the same tool interface works during training, eval, inference, and external serving. This tutorial covers the four paths you will walk in practice — wiring an MCP-backed environment into a training loop, using the same env for offline eval, inspecting the API underneath both of those, and building your own MCP environment when no existing one fits.

## Why MCP?

If your tools are just local Python functions, you do not need MCP — pass them to your trainer directly (TRL, torchforge, whatever) and you are done. MCP earns its complexity when the tool surface has to exist as a **process boundary**, not a function call:

- **The env runs elsewhere** — in a Docker container, a Hugging Face Space, a remote server. MCP is the transport that crosses that boundary.
- **You want to reuse someone else's env** — the OpenEnv catalog, third-party envs, and community hubs all expose their tools over MCP, so the same env works in your training run without rewriting its interface.
- **You want the env to be callable by other agents** — Claude Desktop, Cursor, inference servers, and any MCP-compatible client can plug into an MCP server. A private Python function doesn't get that for free.
- **You need tool discovery and schemas** — `list_tools()` + auto-generated JSON schemas are part of the protocol; models see them the same way they see any MCP server's tools.

In short: MCP is the answer when your env is more than a helper function in your training script — when the same tools have to be usable from training, inference, and external clients without maintaining three interfaces.

### The dual API boundary

Inside OpenEnv, MCP plays a specific role in a two-surface split:

- **Training / orchestration infrastructure** uses the Gym-style control plane — `reset()`, `step()`, `state()` — over WebSocket (`/ws`). This is what the trainer needs to roll out episodes, compute rewards, and enforce termination.
- **Agents** use MCP tools over the `/mcp` JSON-RPC endpoint. Tools are what the model calls to act on the world.

> [!NOTE]
> In simulation mode, MCP tool calls flow **through** `step()`. The trainer stays in control of timing, rewards, and termination; the MCP action types are just a standardised action schema. [Simulation vs Production](../guides/simulation-vs-production) covers the split in depth.

> [!NOTE]
> **MCP adoption in OpenEnv is still in flight.** [RFC 003](https://github.com/huggingface/OpenEnv/blob/main/rfcs/003-mcp-support.md) proposes MCP as the standard interface for *all* agent-facing actions, but it is still **In Review**. Today only a handful of envs are MCP-backed: `echo_env` and `finqa_env` inherit from the canonical `openenv.core.env_server.mcp_environment.MCPEnvironment`; `calendar_env` uses a local wrapper with the same shape. The majority (`textarena_env` / Wordle, `openspiel_env`, `chess_env`, `browsergym_env`, and most others) still use custom action types that you pass through `env.step(CustomAction(...))` without MCP plumbing. Before using the patterns in this tutorial against a specific env, check whether it inherits from an `MCPEnvironment` base; if not, the env's own action schema applies instead.

## Using MCP Tools in a Training Loop

An MCP-backed env is consumed like any other OpenEnv env from the trainer's side. At the atomic level, each agent action is:

```python
obs = env.step(CallToolAction(tool_name=..., arguments=...))
# obs.result       — runtime tool result object, or None on error
# obs.reward       — env's reward for this turn (may be None)
# obs.done         — episode terminated
```

That is the only MCP-specific piece. Everything around it — how the trainer generates actions, how tool schemas are surfaced to the model, how rewards are collected — belongs to your training framework, not to MCP.

### Framework-agnostic rollout loop

If you drive the rollout yourself (a custom loop, [torchforge](https://github.com/meta-pytorch/torchforge), an external agent server), you own the full generation path and call `env.step()` directly:

```python
obs = env.reset()
total_reward = 0.0
for turn in range(max_turns):
    tool_call = model.decide(obs)  # your agent picks a tool + args from the latest observation
    obs = env.step(
        CallToolAction(tool_name=tool_call.name, arguments=tool_call.arguments)
    )
    total_reward += obs.reward or 0.0
    if obs.done:
        break
```

Whatever policy / generation code you use, `env.step(CallToolAction(...))` is the only line that talks to the MCP env.

### TRL `environment_factory`

[TRL](https://huggingface.co/docs/trl)'s `GRPOTrainer` takes an `environment_factory` class whose public methods auto-register as discoverable tools — the trainer then handles the multi-turn generation loop for you. The [Wordle GRPO tutorial](wordle-grpo.md) shows the full recipe (wrapper class, reward function, `GRPOTrainer` construction) with a non-MCP env. For an MCP-backed env, only the tool method bodies change; they call through to `env.step(CallToolAction(...))`:

```python
def echo(self, message: str) -> str:
    """Echo back a message.

    Args:
        message: The message to echo.
    """
    step_result = self.env.step(
        CallToolAction(tool_name="echo_message", arguments={"message": message})
    )
    obs = step_result.observation
    self.reward = step_result.reward or obs.reward or 0.0
    result = obs.result
    return result.data if hasattr(result, "data") else result
```

`environment_factory` is a TRL API, not an MCP API. It works equally well with non-MCP envs (Wordle uses it with `TextArenaAction`), and MCP envs work equally well without it (the rollout-loop path above). They compose, but they are orthogonal.

The rest of this tutorial is for the other paths: the API **underneath** `env.step(CallToolAction(...))` (useful when you need the full observation or want to debug), **using the same env for offline eval**, and **building your own MCP environment** from scratch.

## Under the Hood: `CallToolAction` and `ListToolsAction`

The two MCP action types are `ListToolsAction` (discover what's available) and `CallToolAction` (invoke one). They behave like any other Gym action — pass them to `step()` and inspect the returned observation.

### Discovering tools

```python
from echo_env.server.echo_environment import EchoEnvironment
from openenv.core.env_server.mcp_types import ListToolsAction, ListToolsObservation

env = EchoEnvironment()
env.reset()

obs = env.step(ListToolsAction())
assert isinstance(obs, ListToolsObservation)

for tool in obs.tools:
    print(f"{tool.name}: {tool.description}")
```

Each `Tool` carries a `name`, a `description`, and an `input_schema` (JSON Schema) describing the accepted arguments. The schema is what lets a language-model agent know which parameters to fill in when it emits a tool call.

### Calling a tool

```python
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

obs = env.step(
    CallToolAction(
        tool_name="echo_message",
        arguments={"message": "Hello from MCP!"},
    )
)

assert isinstance(obs, CallToolObservation)
print(obs.tool_name)       # "echo_message"
print(obs.error)           # None
result = obs.result
print(result.data if hasattr(result, "data") else result)  # "Hello from MCP!"
```

`CallToolObservation.result` is typed as `Any` in OpenEnv. At runtime, FastMCP commonly returns a `fastmcp.client.client.CallToolResult` object with `.data`, `.structured_content`, and `.content` attributes, but JSON round-trips or custom environments can surface a plain dict or value instead. Treat `.data` as a convenience when it exists, not as an OpenEnv-defined wrapper type. `obs.error` carries **every** failure mode — transport errors, unknown tool names, malformed arguments, **and** exceptions raised from inside the tool function itself (as `ToolErrorType.EXECUTION_ERROR`). On an error, `obs.result` is `None`. Always branch on `obs.error is None` before reading a runtime result.

### Error handling

```python
obs = env.step(
    CallToolAction(tool_name="does_not_exist", arguments={}),
)

assert isinstance(obs, CallToolObservation)
print(obs.error.error_type)  # ToolErrorType.TOOL_NOT_FOUND
print(obs.error.message)     # human-readable message from FastMCP, e.g. "Unknown tool: 'does_not_exist'"
```

The `ToolError.error_type` enum (`TOOL_NOT_FOUND`, `INVALID_ARGS`, `EXECUTION_ERROR`, `TRANSPORT_ERROR`, `TIMEOUT`) lets training code distinguish between bugs in the agent, bugs in the environment, and transient infrastructure issues — which often warrant different reward signals.

### `step(CallToolAction(...))` vs `call_tool()`

Environment clients that inherit from `MCPToolClient` (such as `EchoEnv` and `FinQAEnv`) expose a shorter **async** `await env.call_tool("name", arg=value)` helper for a running environment server. It returns the tool's raw return value directly instead of a `CallToolObservation` — and it **raises `RuntimeError`** on any tool error (transport failure, unknown tool, invalid arguments, or a tool exception), so you cannot branch on `error_type` without a `try/except`. Use `step(CallToolAction(...))` when you need the whole observation (reward, done, metadata, or graceful error classification); reach for `call_tool()` in async production scripts where the raw result is all you care about and a failure is allowed to propagate. [Simulation vs Production](../guides/simulation-vs-production) compares the two.

> [!NOTE]
> `MCPToolClient` and its base `MCPClientBase` only support `mode="production"`; construction raises `ValueError` for other modes. For direct in-process training or eval snippets like the ones above, call `env.step(CallToolAction(...))` on the environment class itself.

## Using MCP Tools for Evaluation

The same mechanics work outside a training loop. For an offline eval — benchmarking a model's tool use on a static dataset, regression-testing a deployed agent, or scoring a policy — drop the trainer and drive the step loop yourself:

```python
from echo_env.server.echo_environment import EchoEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

env = EchoEnvironment()
env.reset()

results = []
for sample in eval_dataset:
    tool_call = model.decide(sample)   # your agent picks a tool + arguments
    obs = env.step(
        CallToolAction(tool_name=tool_call.name, arguments=tool_call.arguments),
    )
    results.append({
        "prompt": sample.prompt,
        "reply": (
            obs.result.data if obs.error is None and hasattr(obs.result, "data")
            else obs.result if obs.error is None
            else None
        ),
        "reward": obs.reward or 0.0,
        "error": obs.error,
    })
    env.reset()
```

Pair the loop with a scoring function of your choice — the [Rewards](../guides/rewards.md) guide covers common patterns (test-pass rate, LLM-as-judge quality, compliance gates) — and aggregate across the dataset. The eval harness integration in `src/openenv/core/evals/` is still evolving; until that bridge lands, this plain-Python loop is the canonical pattern.

## Building an MCP Environment

Reach for this path when no existing environment covers the tools your agent needs — e.g. a new coding sandbox, a game, a proprietary API wrapper. The provider side is small: subclass `MCPEnvironment`, create a `FastMCP` server, register tools with the `@mcp.tool` decorator, and pass the server to `super().__init__`. Here is the echo environment, trimmed from [`envs/echo_env/server/echo_environment.py`](https://github.com/huggingface/OpenEnv/blob/main/envs/echo_env/server/echo_environment.py) down to the parts this tutorial covers:

```python
from uuid import uuid4

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State


class EchoEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("echo_env")

        @mcp.tool
        def echo_message(message: str) -> str:
            """Echo back the provided message.

            Args:
                message: The message to echo back

            Returns:
                The same message that was provided
            """
            return message

        @mcp.tool
        def echo_with_length(message: str) -> dict:
            """Echo back the message with its length.

            Args:
                message: The message to echo back

            Returns:
                Dictionary with the message and its length
            """
            return {"message": message, "length": len(message)}

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return Observation(done=False, reward=0.0, metadata={"status": "ready"})

    def _step_impl(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        # Called for non-MCP actions. Echo exposes MCP tools only,
        # so anything that isn't ListToolsAction / CallToolAction is an error.
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}"},
        )

    @property
    def state(self) -> State:
        return self._state
```

A few things worth calling out:

- **Docstring → schema.** `FastMCP` inspects each tool's signature and Google-style docstring to build the `input_schema` automatically. The `Args:` block becomes parameter descriptions, and type hints become JSON types. No hand-written schemas.
- **Reserved names.** `reset`, `step`, `state`, and `close` are reserved and cannot be tool names — they belong to the infrastructure boundary. Trying to register a tool with one of those names raises at construction time.
- **`_step_impl` is required, `step` is not.** `MCPEnvironment.step` already routes `ListToolsAction` and `CallToolAction` through the FastMCP server for you. Your subclass only has to implement `_step_impl`, which the base class calls for any **non-MCP** action. In pure-MCP environments like Echo it just returns an error observation; in environments that mix tool calls with other action types (e.g. a terminal "submit" action) it's where that extra dispatch lives.
- **Rewards and `done` still work.** Because MCP actions flow through `step()`, you can compute rewards, flip `done`, and emit metadata just like in any other OpenEnv environment.

## Running the Demo End-to-End

The repo ships a self-contained walkthrough at [`examples/echo_mcp_demo.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/echo_mcp_demo.py). Run it directly from the repo root:

```bash
PYTHONPATH=src:envs uv run python examples/echo_mcp_demo.py
```

You will see the discovery call, two tool invocations, and an error case printed in sequence — the same four steps the "Under the hood" section covers, end-to-end against the real `EchoEnvironment`.

## Next Steps

- **End-to-end training recipe** — the [Wordle GRPO tutorial](wordle-grpo.md) walks through a full GRPO training run with `environment_factory`. The wrapper-class shape is the same for an MCP-backed env; inside each tool method, build a `CallToolAction(tool_name=..., arguments={...})` instead of Wordle's single-field `TextArenaAction(message=guess)`.
- **MCP in both modes**: [Simulation vs Production](../guides/simulation-vs-production) covers `step()` vs `step_async()`, the `call_tool()` convenience path, and common debugging questions.
- **A richer MCP environment** — [`envs/finqa_env/`](https://github.com/huggingface/OpenEnv/tree/main/envs/finqa_env) shows tool calls participating in episode progression, rewards, and terminal submission — not just a stateless echo.
- **Design rationale** — [RFC 003](https://github.com/huggingface/OpenEnv/blob/main/rfcs/003-mcp-support.md) explains why OpenEnv picked MCP as the agent boundary and how tool-calling and CodeAct styles share the same plumbing.
- **Serving tools to an external agent** — the `/mcp` JSON-RPC endpoint is available alongside `/ws` on any MCP environment server. Point an MCP-compatible client at it for production inference without going through the step loop. This direct path bypasses reward computation, step counts, and episode termination, and it exposes only registered MCP tools — not `reset`, `step`, or `state`.
