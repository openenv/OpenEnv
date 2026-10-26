# Runtime Providers

A client connects to an environment server that is already running, or starts
one through a runtime provider.

## Connecting to a server

`AutoEnv.from_env` covers the common cases:

```python
from openenv import AutoEnv

# A Hugging Face Space (installs the environment's client package after asking)
env = AutoEnv.from_env("openenv/echo_env")

# A server you already run
env = AutoEnv.from_env("echo", base_url="http://your-server:8000")

# A local Docker container, started for you
env = AutoEnv.from_env("coding", docker_image="coding-env:latest", wait_timeout=60.0)
```

Pass `trust_remote_code=True` (or set `OPENENV_TRUST_REMOTE_CODE`) to install a
Hub environment's client without the prompt, or `skip_install=True` to connect
with a `GenericEnvClient` and install nothing. [Auto-Discovery](auto-discovery)
lists every option, and [Async vs Sync](async-sync) covers the client modes.
A sleeping Space wakes up on the first request, so the first call can take a
while. To keep your own copy running, duplicate the Space.

A private Space needs your Hugging Face token. Construct the client directly and
pass it in `headers`, which every client sends on each connection:

```python
from echo_env import EchoEnv
from huggingface_hub import get_token

env = EchoEnv(
    base_url="https://your-name-echo-env.hf.space",
    headers={"Authorization": f"Bearer {get_token()}"},
)
```

To run the server somewhere else, such as a cloud sandbox, pick a provider below.

## Providers

A runtime provider starts an environment server and returns a `base_url` that an
`EnvClient` connects to. Container providers implement the same
`ContainerProvider` contract, so switching from local Docker to a cloud sandbox
is a one-line change.

### Available providers

| Provider | Backend | Install | Status |
|----------|---------|---------|--------|
| `LocalDockerProvider` | Local Docker daemon | core | ✅ |
| `DockerSwarmProvider` | Docker Swarm cluster | core | ✅ |
| `UVProvider` | Local process via `uv` (no container) | core | ✅ |
| `DaytonaProvider` | Daytona cloud sandboxes | `pip install openenv[daytona]` | ✅ |
| `ACASandboxProvider` | Azure Container Apps Sandboxes | `pip install openenv[aca]` | ✅ |
| `ModalProvider` | Modal sandboxes | `pip install openenv[modal]` | ✅ |
| `NovitaSandboxProvider` | Novita AI sandboxes | `pip install openenv[novita]` | ✅ |
| `HFSandboxProvider` | Hugging Face sandboxes (billed as HF Jobs) | core | ✅ |

A `KubernetesProvider` is planned but not available yet.

Cloud-provider SDKs are optional extras, imported lazily, so installing core
OpenEnv pulls in no cloud SDK. The core providers (`LocalDockerProvider`,
`DockerSwarmProvider`, `UVProvider`) are re-exported from the runtime package;
cloud providers are imported from their module:

```python
from openenv.core.containers.runtime import LocalDockerProvider  # core
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider  # cloud
```

See the [Core API reference](../reference/core.md#container-providers) for each
provider's full API.

### Lifecycle

Container providers that store their source image on the provider can be owned
by the client. In this form, the client starts the provider on first connect,
waits for readiness, and stops the provider when the client closes:

```python
image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
provider = DaytonaProvider(image=image)

async with MyEnv(provider=provider) as env:
    result = await env.reset()
    ...
```

`ModalProvider`, `DaytonaProvider`, `ACASandboxProvider`, and `HFSandboxProvider`
support this provider-owned flow. Providers that require an explicit image at
`start_container()` time, such as `LocalDockerProvider` and
`DockerSwarmProvider`, should still be started manually and passed in with the
returned `base_url`:

```python
base_url = provider.start_container(image)
provider.wait_for_ready(base_url, timeout_s=180)
try:
    async with MyEnv(base_url=base_url, provider=provider) as env:
        result = await env.reset()
        ...
finally:
    provider.stop_container()
```

`UVProvider` is not a container provider: it runs the server as a local process
and exposes `.start()` / `.wait_for_ready()` / `.stop()` instead.

## Reusing one server for multiple sessions

After a client has connected, call `new_session()` to open another independent
environment session against the same running server:

```python
async with MyEnv(provider=provider) as env:
    first = await env.reset()
    child = await env.new_session()
    second = await child.reset()
```

Child sessions are owned by the parent client: closing the parent also closes
any children it created. You can still close a child earlier when you no longer
need it. Server capacity limits still apply, so `new_session()` can fail while
opening the child WebSocket when the server has reached `MAX_CONCURRENT_ENVS`.

## Running many environments in parallel

Scaling out (for example, many concurrent RL rollouts) is a main reason cloud
providers exist. The model is one provider and one client per environment: each
provider starts its own isolated sandbox, so they run independently. Launch them
concurrently with `asyncio.gather`, wrapping the blocking provider calls in
`asyncio.to_thread` since most cloud SDKs are synchronous:

```python
async def run_one(env_id: int, image) -> dict:
    provider = DaytonaProvider()
    base_url = await asyncio.to_thread(provider.start_container, image)
    try:
        await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        async with MyEnv(base_url=base_url, provider=provider) as env:
            result = await env.reset()
            return result.observation.metadata
    finally:
        await asyncio.to_thread(provider.stop_container)

image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
results = await asyncio.gather(*(run_one(i, image) for i in range(20)))
```

Full example: [`examples/daytona_tbench2_concurrent.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_concurrent.py)
spins up N sandboxes concurrently and reports per-stage timing.

## Per-provider setup

### ACASandboxProvider

Runs the server in an Azure Container Apps Sandbox. Install with
`pip install openenv[aca]`. Requires Azure credentials (`credential=None`
falls back to `DefaultAzureCredential`).

```python
from openenv.core.containers.runtime.aca_provider import ACASandboxProvider

provider = ACASandboxProvider(
    image="disk:my-env",
    subscription_id="<subscription-id>",
    resource_group="<resource-group>",
    sandbox_group="<sandbox-group>",
    region="eastus",   # used to derive the endpoint when endpoint=None
    endpoint=None,
    credential=None,   # defaults to DefaultAzureCredential()
    sdk_kwargs={},
)
```

### DaytonaProvider

Runs the server in a Daytona cloud sandbox. Install with
`pip install openenv[daytona]`. Requires the `DAYTONA_API_KEY` environment
variable.

```python
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider

image = DaytonaProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
provider = DaytonaProvider(image=image)
```

Full examples: [`examples/daytona_tbench2_simple.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_simple.py)
and [`examples/daytona_tbench2_concurrent.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_concurrent.py).

### DockerSwarmProvider

Deploys the server as a service on a Docker Swarm cluster. Initializes Swarm
automatically when it is not already active.

```python
from openenv.core.containers.runtime import DockerSwarmProvider

provider = DockerSwarmProvider()
```

### HFSandboxProvider

Runs the server in a Hugging Face sandbox, when your Hugging Face account is
the only cloud account you want to use. Included in core OpenEnv. Requires a
Hugging Face token (`HF_TOKEN` or `hf auth login`) for an account or
organization that can run [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs).
The sandbox hosts are billed as Jobs.

The image must provide a `server` command that serves OpenEnv on port 8000.
Images built from an OpenEnv environment, such as its Space image, already do.

```python
from coding_env import CodeAction, CodingEnv
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider

provider = HFSandboxProvider(image="hf.co/spaces/openenv/coding_env")

with CodingEnv(provider=provider).sync() as env:
    env.reset()
    result = env.step(CodeAction(code="print(40 + 2)"))
    print(result.observation.stdout)
```

`flavor` (default `cpu-basic`) picks the hardware and `env_vars` passes
environment variables to the server. Providers with the same image and flavor
in one process share a sandbox pool, which shuts down after 10 idle minutes.

Full example: [`examples/hf_sandbox_coding_env.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/hf_sandbox_coding_env.py).

### KubernetesProvider

🚧 Not yet implemented. The class exists as a placeholder for the planned
Kubernetes backend.

### LocalDockerProvider

Runs the server on the local Docker daemon. This is the default for
`from_docker_image`, so you rarely construct it explicitly.

```python
from openenv.core.containers.runtime import LocalDockerProvider

provider = LocalDockerProvider()
```

### ModalProvider

Runs the server in a Modal sandbox over an encrypted tunnel. Install with
`pip install openenv[modal]`. Requires a configured Modal account
(`modal setup`).

```python
from openenv.core.containers.runtime.modal_provider import ModalProvider

image = ModalProvider.image_from_dockerfile("envs/echo_env/server/Dockerfile")
provider = ModalProvider(app_name="openenv", image=image)
```

Full example: [`examples/modal_echo_env.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/modal_echo_env.py).

### NovitaSandboxProvider

Runs the server in a Novita AI sandbox. Install with
`pip install openenv[novita]`. Requires the `NOVITA_API_KEY` environment
variable (and optionally `NOVITA_DOMAIN` to select a region — the default is
`us-phx-1`).

```python
from openenv.core.containers.runtime.novita_provider import NovitaSandboxProvider

provider = NovitaSandboxProvider(image="ghcr.io/org/echo-env:latest")
```

A local Dockerfile works too, and builds a Novita template on first use:

```python
image = NovitaSandboxProvider.image_from_dockerfile(
    "envs/echo_env/server/Dockerfile"
)
provider = NovitaSandboxProvider(image=image)
```

Novita's template parser does not accept multi-stage build definitions, which is
the layout every in-repo environment uses, so `image_from_dockerfile` rewrites
the Dockerfile before handing it over: BuildKit `--mount` flags are stripped,
`ARG`/`--platform` in `FROM` lines are resolved, and a two-stage build whose
stages share one base image is replayed as a single stage. A Dockerfile that
does not fit those rules raises with the registry route as the alternative —
build with `openenv build`, push with `openenv push --registry`, and pass the
resulting registry reference.

Two things the provider pins that the image does not: the sandbox runs as
`root` (Novita's parser otherwise rewrites USER to a non-root `user`, which
cannot write the root-owned `/app` the server installs into), and the template's
start command is a keepalive rather than the image's `CMD`, which leaves port
8000 free for the server the provider launches itself. That launch writes a PID
file, so `wait_for_ready` can report a crashed server immediately instead of
waiting out the full timeout.

### UVProvider

Runs the server as a local process via `uv`, without a container. Useful for
developing an environment from a checkout.

```python
from openenv.core.containers.runtime import UVProvider

provider = UVProvider(project_path="path/to/env")
base_url = provider.start()
provider.wait_for_ready()
```
