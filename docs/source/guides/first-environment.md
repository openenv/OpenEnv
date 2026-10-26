# Your First Environment

This walkthrough builds an environment from scratch: scaffold it with the `openenv` CLI, write the models, the environment logic, the server and the client, run it locally, then deploy it to a Hugging Face Space. For what each piece is, see [Core Concepts](concepts.md).

You need Python 3.10+, [`uv`](https://github.com/astral-sh/uv) and the OpenEnv library (`pip install openenv`). Docker is only needed to build the image or run it locally (`openenv build`, `from_env` and `from_docker_image`).

## 1. Scaffold with `openenv init`

```bash
openenv init my_env
cd my_env
```

Use `--output-dir` to create it somewhere else. The command generates a working environment that echoes back messages, plus a `uv.lock`:

```
my_env/
├── __init__.py                # Exports the client and models
├── README.md                  # Documentation, also used as the Space card
├── client.py                  # MyEnv: the client
├── models.py                  # MyAction, MyObservation
├── openenv.yaml               # Manifest
├── pyproject.toml             # Package metadata and dependencies
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI app built with create_app
    ├── my_env_environment.py  # MyEnvironment: reset(), step(), state
    ├── requirements.txt
    └── Dockerfile
```

Class names come from the environment name, with a trailing `_env` dropped: `my_env` gives `MyAction`, `MyObservation`, `MyEnvironment` and `MyEnv`. The steps below go through each file. Edit them to replace the echo logic with yours. If you're working inside the OpenEnv repo, move the folder under `envs/`.

> [!TIP]
> If you already have an ORS/OpenReward or Prime Intellect Verifiers environment, run `openenv import SOURCE --name my_env --output-dir DIR` instead. It detects the source type, vendors the source under the generated package and emits an OpenEnv wrapper with task/split and MCP-style tool actions. Non-secret data files and portable dependencies from the source tree are carried over.

## 2. Define the models

`models.py` declares what the agent sends (the action) and what it gets back (the observation). Subclass the base classes from `openenv.core.env_server.types`, not `pydantic.BaseModel` directly:

```python
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MyAction(Action):
    message: str = Field(..., description="Message to echo back")


class MyObservation(Observation):
    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
```

The base `Observation` already has `done`, `reward` and `metadata` fields. The template uses the core `State` class (`episode_id`, `step_count`). Subclass `State` if you need to track more.

## 3. Implement the environment

`server/my_env_environment.py` holds the logic. Subclass `Environment` and implement `reset()`, `step()` and the `state` property. Reward and termination go on the returned observation, `step()` does not return a tuple:

```python
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import MyAction, MyObservation


class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> MyObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return MyObservation(echoed_message="My Env environment ready!", done=False, reward=0.0)

    def step(self, action: MyAction) -> MyObservation:
        self._state.step_count += 1
        length = len(action.message)
        return MyObservation(
            echoed_message=action.message,
            message_length=length,
            done=False,
            reward=min(length * 0.1, 1.0),
        )

    @property
    def state(self) -> State:
        return self._state
```

The full method signatures are `reset(self, seed=None, episode_id=None, **kwargs)` and `step(self, action, timeout_s=None, **kwargs)`. Accept those arguments when your environment uses them. Set `SUPPORTS_CONCURRENT_SESSIONS = True` only if instances share no mutable state, so several clients can each get their own instance.

For anything beyond a one-line reward, compute it with a rubric: pass `rubric=...` to `super().__init__()`, call `self._reset_rubric()` in `reset()` and `self._apply_rubric(action, observation)` in `step()`. See [Rewards](rewards.md) and the [Rubrics tutorial](../tutorials/rubrics.md).

## 4. Create the server

`server/app.py` wraps the environment as a FastAPI app with `create_app`. Pass the environment **class**, not an instance. The server calls it to create one instance per WebSocket session:

```python
from openenv.core.env_server.http_server import create_app

from ..models import MyAction, MyObservation
from .my_env_environment import MyEnvironment

app = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="my_env",
    max_concurrent_envs=1,  # raise it to allow more concurrent sessions
)
```

If the environment takes constructor arguments, pass a factory function instead of the class:

```python
import os


def create_my_environment():
    return MyEnvironment(api_key=os.getenv("MY_API_KEY"))


app = create_app(create_my_environment, MyAction, MyObservation, env_name="my_env")
```

The generated files also wrap these imports in `try/except ImportError` so the server runs both as a package and from inside the folder (as in the Docker image). Keep that pattern when you edit them.

## 5. Implement the client

`client.py` subclasses `EnvClient` and converts between your models and the JSON sent over the WebSocket. Update these three methods when you change the models:

```python
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyAction, MyObservation


class MyEnv(EnvClient[MyAction, MyObservation, State]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"message": action.message}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs_data = payload.get("observation", {})
        observation = MyObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
```

## 6. Run it locally

From the environment folder, start the server on port 8000:

```bash
uv run --project . server
```

To use another port, run uvicorn directly: `uv run --project . uvicorn server.app:app --port 8001`.

In another terminal, connect with the client. The client is async by default. `.sync()` gives a synchronous wrapper:

```python
from my_env import MyAction, MyEnv

with MyEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    print(result.observation.echoed_message)  # "My Env environment ready!"

    result = client.step(MyAction(message="Hello!"))
    print(result.observation.echoed_message, result.reward)  # "Hello!" 0.6

    print(client.state())  # episode_id=... step_count=1
```

Save it as `try_it.py` and run `uv run --project . python try_it.py` so `my_env` is importable. See [Async vs Sync](async-sync.md) for the async version.

### Web UI

Start the server with `ENABLE_WEB_INTERFACE=true` and open `http://localhost:8000/web`:

```bash
ENABLE_WEB_INTERFACE=true uv run --project . server
```

You get a playground to reset the environment, fill an action form, step and read the observations, with no extra code. `openenv push` turns it on for Spaces. Optionally, override `render_web()` to draw the observation and `web_actions()` to offer one-click actions. See [Customizing the Web UI](customizing-web-ui.md).

### Validate

`openenv.yaml` is the environment's manifest. It names the app and port the server runs, and declares the contract your environment promises (reward range, resources, capabilities). The template generates:

```yaml
spec_version: 1
name: my_env
version: 0.1.0
type: space
runtime: fastapi
app: server.app:app
port: 8000
validation:
  reward:
    range: [0.0, 1.0]
    oracle_tolerance: 0.0
    floor_margin: 0.1
  resources:
    cpu: 1.0
    memory_mb: 1024
    disk_mb: 512
    episode_timeout_s: 60.0
  capabilities:
    verifier:
      kind: reward_channel
  types:
    tags: [demo]
```

Check it before deploying:

```bash
openenv validate --level static --skip-build
```

## 7. Deploy

Push the environment to a Hugging Face Space:

```bash
openenv push
```

`openenv push` logs you in if needed, enables the web UI and uploads the folder to the Space `<your-username>/my_env`, which builds the Docker image. To check that the image builds before pushing, run `openenv build` (needs Docker).

Others can then run your environment. `from_env` pulls the Space's image and starts it locally with Docker:

```python
from my_env import MyEnv

with MyEnv.from_env("your-username/my_env").sync() as client:
    result = client.reset()
```

[Deploying an Environment](../getting_started/environment-builder.md) covers the build and push options, Space variables and secrets, other registries and how to connect to a deployed environment.

## Next steps

- [Rewards](rewards.md) and [Rubrics](../tutorials/rubrics.md) to design the reward
- [MCP Environments](../tutorials/mcp-environment.md) to expose tools instead of a single action type
- [Training](training.md) to train a model on your environment
