# Getting Started

Install OpenEnv, load an environment, and run your first step.

## Install OpenEnv

```bash
pip install openenv
```

> [!NOTE]
> This installs the full OpenEnv runtime: the environment server, the client,
> the `openenv` CLI, the web interface, and MCP support. Environments depend on
> `openenv` directly.

### Optional dependencies

A few integrations ship as optional extras. Install them with
`pip install openenv[<extra>]`:

| Extra | Pulls in |
|-------|----------|
| `inspect` | The Inspect AI evaluation harness |
| `harbor` | The [Harbor integration](environments/harbor) and its sandbox backends (Python 3.12+) |
| `daytona`, `aca`, `modal`, `novita` | Cloud sandbox providers (see [Runtime Providers](guides/runtime-providers)) |

## Try an Environment

Use `AutoEnv` to load the Echo client from its Hugging Face Space. It asks for
confirmation before installing the environment package, then connects to the
running Space. Echo exposes its actions as MCP tools.

```python
from openenv import AutoEnv

env = AutoEnv.from_env(
    "openenv/echo_env",
    base_url="https://openenv-echo-env.hf.space",
)

with env.sync() as client:
    result = client.reset()
    print(result.observation.metadata["message"])  # "Echo environment ready!"

    message = client.call_tool("echo_message", message="Hello, OpenEnv!")
    print(message)  # "Hello, OpenEnv!"
```

For installed environment packages, `AutoEnv.from_env()` also accepts the common
name forms below. Without a `base_url`, these start a local Docker container and
require Docker and the environment image:

```python
from openenv import AutoEnv

AutoEnv.from_env("echo")
AutoEnv.from_env("echo-env")
AutoEnv.from_env("echo_env")
```

## Connect to a Running Environment

OpenEnv clients are async by default. Use the async client for production code,
parallel environment runs, and integrations with async frameworks.

```python
import asyncio

from echo_env import CallToolAction, EchoEnv


async def main():
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        await client.reset()

        result = await client.step(
            CallToolAction(
                tool_name="echo_message",
                arguments={"message": "Hello, World!"},
            )
        )
        print(result.reward)


asyncio.run(main())
```

For scripts and notebooks, use `.sync()`:

```python
from echo_env import CallToolAction, EchoEnv

with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello, World!"},
        )
    )
    print(result.observation.result)
```

## Use Containers or Local Servers

You can run an environment from a Docker image:

```python
import asyncio

from echo_env import EchoEnv


async def main():
    client = await EchoEnv.from_docker_image(
        "registry.hf.space/openenv-echo-env:latest"
    )
    async with client:
        result = await client.reset()
        print(result.observation)


asyncio.run(main())
```

If you run the Hugging Face Space image yourself with `docker run`, expose port
`7860` and connect to that port:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 \
    registry.hf.space/openenv-echo-env:latest
```

```python
from echo_env import EchoEnv

with EchoEnv(base_url="http://localhost:7860").sync() as client:
    result = client.reset()
```

Or connect to a local server:

```bash
cd path/to/echo-env
uv venv
source .venv/bin/activate
uv pip install -e .
uv run server --host 0.0.0.0 --port 8000
```

```python
from echo_env import EchoEnv

with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
```

The same client also runs environments on cloud sandboxes (Daytona, Modal, Novita, Azure Container Apps, Hugging Face) through runtime providers. `HFSandboxProvider` ships with core OpenEnv and needs a Hugging Face token for an account that can run Jobs (the sandboxes are billed as Jobs): it runs an environment's Space image in a Hugging Face sandbox ([example](guides/runtime-providers#hfsandboxprovider)). See the [Runtime Providers guide](guides/runtime-providers) to pick one.

## Next Steps

- [Train an agent](guides/training) with your training framework, or train a coding agent through [Harbor](environments/harbor)
- [Explore environments](environments)
- [Build your first environment](guides/first-environment)
- [Concepts](guides/concepts)
