# ${env_name}: AgentEnv bundle

This package serves the vendored `${bundle}` bundle through
`agent_env.openenv.build_app`. The bundle keeps its original folder name at
`vendor/${bundle}`; the generated package name is `${env_name}`.

## Prerequisites

- Python >=3.11.4 and `${framework}` (declared in
  `pyproject.toml` and `server/requirements.txt`).
- Local Docker: a Docker CLI and a running daemon accessible to the server.
  The delegated server builds the bundle's environment images at startup and
  deploys episode containers using local Docker by default.
- A self-contained bundle: tasks, environment definitions, verifiers, data and
  dependencies must be included in the bundle or provided by agentenv-framework.
  Only the selected bundle is vendored; external plugin registrations and files
  elsewhere in the source repository are not included.

## Run locally

From this generated package directory, after setting up Docker:

```bash
uv sync
MAX_CONCURRENT_ENVS=8 uv run server --host 127.0.0.1 --port 8000
```

`MAX_CONCURRENT_ENVS` (default 8) caps the concurrent sessions. Each session
plays its own episode on its own deployment.

The `server` entry point runs `server/app.py`, which passes
`vendor/${bundle}` to `build_app`. Task selection, splits, tools and rewards
are defined by the bundle and agentenv-framework. The served environment is
`agent_env.openenv`'s `AgentEnvEnvironment`; read it, and `build_app`, at the
minimum supported version: [environment.py](${adapter}/environment.py) and
[app.py](${adapter}/app.py). Follow
[AgentEnv's OpenEnv documentation](https://www.agentenvframework.com/docs/openenv)
for its API and client usage.

## Client

`client.py` defines `${client}`, an `MCPToolClient`: the served environment
takes OpenEnv's MCP actions (`ListToolsAction`, `CallToolAction`), which
`__init__.py` also exports.

```python
from ${env_name} import ${client}

async with ${client}(base_url="http://127.0.0.1:8000") as env:
    await env.reset(split="all", index=0)
    tools = await env.list_tools()
    ended = await env.call_tool("end_turn", response="Done.")
    print(ended["done"], ended["reward"])
```

The tools are the bundle environment's MCP tools plus `end_turn`, which ends
the turn with its `response`. Its result carries the next turn's prompt, or,
once `done`, the task's reward. A step before `reset` raises.

`list_tools()` discovers fresh tools by default because tools can change after
reset, `end_turn`, or a bundle tool call. `get_tool()` and `has_tool()` also use
fresh discovery. Pass `list_tools(use_cache=True)` explicitly to reuse the last
discovery result.

## Deployment limitations

**Warning:** `server/Dockerfile` does not provide the Docker runtime required
by the delegated server. `openenv build` / `openenv push` alone do not produce
a working deployment, including on Hugging Face Spaces. Deployment requires
separately provisioning the server's image-build and episode-container runtime;
this importer does not configure it. Review AgentEnv's runtime requirements
before choosing a deployment host.

## Generated files

- `server/app.py`: delegated application and local server entry point.
- `vendor/${bundle}`: the selected bundle and its data.
- `pyproject.toml`: package dependencies and the `server` entry point.
- `server/requirements.txt`: server dependencies.
- `openenv.yaml`: generated OpenEnv manifest.
- `server/Dockerfile`: shared image template with the limitation described above.
- `client.py`, `__init__.py`: the client and the MCP action types it sends.
