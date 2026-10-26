# Auto-Discovery

`AutoEnv` and `AutoAction` load an environment's client and action classes by name, without importing its package yourself. They work with environment packages installed locally (`pip install -e envs/<name>`, or `pip install git+https://huggingface.co/spaces/<owner>/<space>`) and with environments hosted on Hugging Face Spaces.

To find an environment you don't know yet, see [catalog discovery](catalog-discovery).

## Quick start

```python
from openenv import AutoEnv, AutoAction

env = AutoEnv.from_env("coding-env")           # async client
CodeAction = AutoAction.from_env("coding-env")  # action class

with env.sync() as client:
    result = client.reset()
    result = client.step(CodeAction(code="print('Hello, OpenEnv!')"))
    print(result.observation.stdout)
```

Names are flexible: `"coding"`, `"coding-env"` and `"coding_env"` all resolve to the same environment. Unknown names raise a `ValueError` that suggests close matches.

## `AutoEnv.from_env(name, **kwargs)`

| Parameter | Description |
|-----------|-------------|
| `name` | Environment name (`"coding"`) or Hub repo ID (`"openenv/coding_env"`, `"username/env-name"`) |
| `base_url` | Connect to a running server at this URL |
| `docker_image` | Docker image to start (overrides the default) |
| `container_provider` | Container provider to start it with |
| `wait_timeout` | Container startup timeout in seconds (default `30.0`) |
| `env_vars` | Environment variables for the container |
| `trust_remote_code` | Install a Hub environment's package without the confirmation prompt (default `False`) |
| `skip_install` | Don't install the package. Connect with a `GenericEnvClient` to `base_url`, the running Space, or the Space's Docker image (default `False`) |
| `**kwargs` | Passed to the client class |

```python
env = AutoEnv.from_env(
    "coding",
    docker_image="my-coding-env:v2",
    wait_timeout=60.0,
    env_vars={"DEBUG": "1"},
)
```

`AutoEnv.from_hub()` is an alias. Other helpers:

- `AutoEnv.list_environments()` prints the installed environments.
- `AutoEnv.get_env_info(name)` returns a dict with `description`, `version`, `default_image`, `env_class` and `action_class`.
- `AutoEnv.get_env_class(name)` returns the client class, to instantiate yourself (for example with `from_docker_image(...)`).

## `AutoAction.from_env(name)`

Returns the action class (not an instance) for a local name or Hub repo ID. `AutoAction.from_hub()` is an alias, `AutoAction.list_actions()` prints the available action classes and `AutoAction.get_action_info(name)` returns a dict with `action_class` and `module`.

## Environments on Hugging Face Spaces

For a Hub repo ID such as `username/coding-env`, `AutoEnv` resolves the Space URL (`https://username-coding-env.hf.space`), checks that the Space is running, installs the environment package from `git+https://huggingface.co/spaces/username/coding-env` (with `uv pip` when available, otherwise `pip`) and connects to the Space.

Installing the package runs code from the internet, so `AutoEnv` asks for confirmation first. To skip the prompt, pass `trust_remote_code=True` or set `OPENENV_TRUST_REMOTE_CODE=1`. In a non-interactive shell without either, the install is refused. To connect without installing anything, use `skip_install=True`.

## How it works

`AutoEnv` finds installed `openenv-*` packages with `importlib.metadata`, reads each package's `openenv.yaml` and imports classes only when they are used. Client, action and observation classes are found by naming convention (`coding_env` → `CodingEnv`, `CodingAction`, `CodingObservation`). When an environment's classes don't follow it, the manifest names them:

```yaml
name: coding_env
version: "0.1.0"
description: "Coding environment for OpenEnv"
action: CodeAction
observation: CodeObservation
```

## See also

- [Your First Environment](first-environment) to create your own environment
- [Core API](../reference/core) for the low-level API
- [OpenEnv on the Hub](https://huggingface.co/openenv) for pre-built environments
