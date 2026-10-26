# Hello World

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)

This tutorial runs an OpenEnv environment on your machine and then builds a small one from scratch. You will install OpenEnv, start an environment server, connect to it with a client, and write the four files every environment is made of: the models, the environment, the server app and the client.

Everything runs on a CPU. You do not need a GPU or Docker. The page and the [notebook](https://github.com/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb) have the same code, so you can follow either one top to bottom.

This page is based on the original OpenEnv tutorial by Sanyam Bhutani.

## How OpenEnv works

An OpenEnv environment runs as a server. Your code talks to it through a typed client over a WebSocket connection, so the environment can run in the same machine, in a Docker container or in a Hugging Face Space without changing the client code.

```
your code                                   environment server
---------                                   ------------------
client.reset()        ---- WebSocket ---->  Environment.reset()
client.step(action)   ---- /ws -------->    Environment.step(action)
client.state()        <--- JSON ---------   Environment.state
```

The API is the same for every environment:

- `reset()` starts an episode and returns the first observation.
- `step(action)` applies an action and returns the next observation, with its `reward` and a `done` flag.
- `state()` returns episode metadata, such as the episode id and the step count.

Actions, observations and state are Pydantic models, so both sides agree on their fields. [Core Concepts](../guides/concepts) covers the design in more detail.

## Setup

Install OpenEnv and the Echo environment, a minimal environment that ships in the OpenEnv repository:

```bash
pip install openenv "openenv-echo-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/echo_env"
```

The next cells start servers in the background. This helper launches a server with `uvicorn` and waits until its `/health` endpoint answers:

```python
import subprocess
import sys
import time

import requests


def start_server(app, port):
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port)]
    )
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError(f"{app} exited, is port {port} already in use?")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
                return process
        except requests.ConnectionError:
            pass
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f"{app} did not start on port {port}")
```

In a terminal you can start the same server with `uvicorn echo_env.server.app:app --port 8000` instead.

## Run an environment locally

Start the Echo server on port 8000:

```python
echo_server = start_server("echo_env.server.app:app", port=8000)
```

Echo exposes its actions as MCP tools. Connect with its client, reset the episode and call a tool. Clients are async by default and `.sync()` gives you a synchronous client for scripts and notebooks:

```python
from echo_env import EchoEnv

with EchoEnv(base_url="http://127.0.0.1:8000").sync() as client:
    result = client.reset()
    print(result.observation.metadata["message"])

    print([tool.name for tool in client.list_tools()])
    print(client.call_tool("echo_message", message="Hello, World!"))
```

```
Echo environment ready!
['echo_message', 'echo_with_length']
Hello, World!
```

Stop the server when you are done:

```python
echo_server.terminate()
echo_server.wait()
```

The same client connects to the hosted Echo Space with `base_url="https://openenv-echo-env.hf.space"`, and `AutoEnv.from_env()` loads any environment by name. [Getting Started](../getting-started) shows both, and [Environments](../environments) lists the environments you can use.

## The environment skeleton

Now build an environment of your own. It is a number guessing game: the environment picks a number between 1 and 10, the agent guesses, and the environment answers `higher`, `lower` or `correct`. A correct guess ends the episode with a reward of `1.0`. Running out of guesses ends it with `0.0`.

An environment is a Python package with four parts:

```
hello_env/
├── __init__.py
├── models.py              # Action and Observation types
├── client.py              # EnvClient used by your code
└── server/
    ├── __init__.py
    ├── environment.py     # Environment with reset(), step() and state
    └── app.py             # FastAPI app built with create_app()
```

Create the package folders:

```bash
mkdir -p hello_env/server
touch hello_env/__init__.py hello_env/server/__init__.py
```

Each of the next four code blocks is a file. Save it to the path in its first line.

### Models

The action is what the agent sends and the observation is what it gets back. `Observation` already defines `reward`, `done` and `metadata`, so you only add the fields of your environment:

```python
# hello_env/models.py
from openenv.core.env_server.types import Action, Observation


class GuessAction(Action):
    guess: int


class GuessObservation(Observation):
    hint: str = ""
    guesses_left: int = 0
```

### Environment

The environment holds the game logic. `reset()` starts an episode and `step()` returns an observation with the reward and the `done` flag. `state` returns the episode id and step count, using the core `State` model:

```python
# hello_env/server/environment.py
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from hello_env.models import GuessAction, GuessObservation

MAX_GUESSES = 4


class GuessEnvironment(Environment[GuessAction, GuessObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._target = 0
        self._done = False

    def reset(self, seed=None, episode_id=None, **kwargs) -> GuessObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._target = random.Random(seed).randint(1, 10)
        self._done = False
        return GuessObservation(hint="Guess a number between 1 and 10", guesses_left=MAX_GUESSES)

    def step(self, action: GuessAction, timeout_s=None, **kwargs) -> GuessObservation:
        if self._done:
            return GuessObservation(hint="the episode is over, call reset()", guesses_left=0, done=True)
        self._state.step_count += 1
        guesses_left = MAX_GUESSES - self._state.step_count
        if action.guess == self._target:
            self._done = True
            return GuessObservation(hint="correct", guesses_left=guesses_left, reward=1.0, done=True)
        self._done = guesses_left == 0
        hint = "higher" if action.guess < self._target else "lower"
        return GuessObservation(hint=hint, guesses_left=guesses_left, reward=0.0, done=self._done)

    @property
    def state(self) -> State:
        return self._state
```

`SUPPORTS_CONCURRENT_SESSIONS = True` tells the server that each WebSocket session can get its own instance of the environment.

### Server app

`create_app()` wraps the environment in a FastAPI app with the WebSocket endpoint, `/health`, `/schema` and the other endpoints a client needs. Pass the class, not an instance, so the server can create one environment per session:

```python
# hello_env/server/app.py
from openenv.core.env_server import create_app

from hello_env.models import GuessAction, GuessObservation
from hello_env.server.environment import GuessEnvironment

app = create_app(GuessEnvironment, GuessAction, GuessObservation, env_name="hello_env")
```

### Client

The client subclasses `EnvClient` and converts between your models and the JSON messages sent over the WebSocket. It implements three methods:

- `_step_payload()` turns an action into JSON.
- `_parse_result()` turns a step or reset response into a `StepResult`.
- `_parse_state()` turns a state response into a `State`.

```python
# hello_env/client.py
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from hello_env.models import GuessAction, GuessObservation


class GuessEnv(EnvClient[GuessAction, GuessObservation, State]):
    def _step_payload(self, action: GuessAction) -> dict:
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult[GuessObservation]:
        observation = GuessObservation(
            **payload["observation"], reward=payload.get("reward"), done=payload.get("done", False)
        )
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def _parse_state(self, payload: dict) -> State:
        return State(**payload)
```

### Run it

Start the server on port 8001:

```python
guess_server = start_server("hello_env.server.app:app", port=8001)
```

Play one episode with a binary search policy. This loop is the shape of every RL rollout: reset, pick an action from the observation, step, and read the reward:

```python
from hello_env.client import GuessEnv
from hello_env.models import GuessAction

with GuessEnv(base_url="http://127.0.0.1:8001").sync() as env:
    result = env.reset(seed=42)
    print(result.observation.hint)

    low, high = 1, 10
    while not result.done:
        guess = (low + high) // 2
        result = env.step(GuessAction(guess=guess))
        print(f"guess={guess} hint={result.observation.hint} reward={result.reward}")
        if result.observation.hint == "higher":
            low = guess + 1
        elif result.observation.hint == "lower":
            high = guess - 1

    print(env.state())
```

```
Guess a number between 1 and 10
guess=5 hint=lower reward=0.0
guess=2 hint=correct reward=1.0
episode_id='...' step_count=2
```

Stop the server:

```python
guess_server.terminate()
guess_server.wait()
```

You wrote these files by hand to see each part. `openenv init my_env` generates the same four parts (with slightly different file names) plus a `pyproject.toml`, an `openenv.yaml` manifest and a Dockerfile, ready to build and push to the Hub.

## Next steps

- [Your First Environment](../guides/first-environment) and [Deploying an Environment](../getting_started/environment-builder) take this skeleton to a packaged environment: `openenv init`, Docker, `openenv validate` and `openenv push`.
- [Environments](../environments) lists the environments you can use today.
- [MCP Environments](mcp-environment) covers tool-based environments like Echo.
- [Rewards](../guides/rewards) and [Rubrics](rubrics) cover how environments compute rewards.
- [Training with OpenEnv](../guides/training) lists every training framework that works with OpenEnv environments, with an example for each.

OpenEnv is openly governed. See [GOVERNANCE.md](https://github.com/huggingface/OpenEnv/blob/main/GOVERNANCE.md) for how decisions are made.
