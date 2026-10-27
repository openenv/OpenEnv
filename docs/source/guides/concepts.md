# Core Concepts

OpenEnv follows a client-server model inspired by Gymnasium's simple API.
Agents send structured actions to isolated environments and receive
observations, rewards, and episode status in return.

```
+-----------------+     HTTP/WebSocket     +-----------------+
|   Your Agent    | <--------------------> |   Environment   |
|   (Client)      |    step/reset/state    |    (Server)     |
+-----------------+                        +-----------------+
```

To build one step by step, follow [Your First Environment](first-environment.md).

## Key Abstractions

### Environment

An **Environment** is an isolated execution context where your agent can take
actions and receive observations. It subclasses `Environment` and implements
`reset()`, `step()` and the `state` property. It runs inside a server, which
creates one instance per client session.

### Action

An **Action** is a structured command that your agent sends to the environment.
Each environment defines its own action schema as a subclass of `Action`.

```python
from coding_env import CodeAction

action = CodeAction(code="print('Hello!')")
```

### Observation

An **Observation** is the response from the environment after taking an action.
It contains the current state visible to your agent. Every observation carries
`done` and `reward` fields, so `step()` returns an observation, not a tuple.

```python
result = client.step(action)
print(result.observation.stdout)  # "Hello!"
```

### State

The **State** is the episode's bookkeeping on the server side: at least
`episode_id` and `step_count`. Clients read it with `state()`.

### StepResult

A **StepResult** bundles together everything the client gets back from a step:

- `observation`: what the agent can see
- `reward`: numeric reward signal for training
- `done`: whether the episode has ended
- `metadata`: additional metadata returned alongside the observation

### Reward and Rubric

Rewards are computed **inside the environment**, not by external code. A
**Rubric** is a composable unit of reward computation passed to the environment.
Rubrics can be combined with `WeightedSum`, `Gate`, and `Sequential`, use LLM
judges for subjective criteria, and handle delayed rewards with
`TrajectoryRubric`. See [Rewards](rewards.md) and the
[Rubrics tutorial](../tutorials/rubrics.md).

### Client

A **Client** is how you connect to and interact with an environment. OpenEnv
provides both async and sync clients.

```python
from openenv import AutoEnv

# Async
async with AutoEnv.from_env("coding") as client:
    result = await client.reset()
    result = await client.step(action)

# Sync, with its own client (a client is locked to the mode it is first used in)
with AutoEnv.from_env("coding").sync() as client:
    result = client.reset()
    result = client.step(action)
```

## The Step Loop

```python
with env.sync() as client:
    result = client.reset()

    while not result.done:
        obs = result.observation
        action = decide_action(obs)
        result = client.step(action)
        learn(result.reward)
```

## Connection Methods

| Method | Use Case | Example |
|--------|----------|---------|
| HTTP URL | Remote servers, Hugging Face Spaces | `EnvClient(base_url="https://...")` |
| Docker | Local development | `EnvClient.from_docker_image("env:latest")` |
| Cloud / custom runtime | Run the server on a cloud sandbox | `EnvClient.from_docker_image("env:latest", provider=DaytonaProvider())` |
| Auto-discovery | Installed packages or known environments | `AutoEnv.from_env("echo")` |

See the [Runtime Providers guide](runtime-providers.md) for the available providers and how to pick one.

## Environment Anatomy

An environment is a Python package with a manifest (`openenv.yaml`), the
models, the client, and a `server/` folder with the environment, the FastAPI app
built with `create_app`, and a Dockerfile. The client, action and observation
classes are found by naming convention (`MyEnv`, `MyAction`, `MyObservation`),
so the manifest doesn't list them. `openenv init my_env` generates this layout.
[Your First Environment](first-environment.md) goes through each file.

## Next Steps

- [Your First Environment](first-environment.md)
- [Getting Started](../getting-started.md)
- [Auto-discovery](auto-discovery.md)
