# OpenEnv: Agentic Execution Environments

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs. OpenEnv provides a standard for interacting with agentic execution environments via simple Gymnasium style APIs - step(), reset(), state(). Users of agentic execution environments can interact with the environment during RL training loops using these simple APIs.

In addition to making it easier for researchers and RL framework writers, we also provide tools for environment creators making it easier for them to create richer environments and make them available over familiar protocols like HTTP and packaged using canonical technologies like docker. Environment creators can use the OpenEnv framework to create environments that are isolated, secure, and easy to deploy and use.

> [!NOTE]
> OpenEnv is in early development, see the note in the [project README](../../../README.md) before contributing. The [README](../../../README.md#community-support--acknowledgments) lists the technical committee that governs the project, and the [charter](../../../GOVERNANCE.md) explains how it works.

## Overview
`openenv.core` provides the foundational building blocks for creating and interacting with containerized environments over HTTP. It enables you to build agent environments that can be deployed as Docker containers and accessed via a simple HTTP API.

# OpenEnv Core

Core components for OpenEnv - a framework for building HTTP-based agentic environments.

## Features

- **EnvClient**: Async-first client for interacting with remote environments
- **SyncEnvClient**: Synchronous wrapper via `.sync()` for sync codebases
- **HTTPEnvServer**: FastAPI-based server wrapper for exposing environments over HTTP/WebSocket
- **Container Providers**: Pluggable architecture for running containers (Docker, Kubernetes, etc.)
- **Type System**: Strongly-typed Action/Observation/State interfaces
- **Web Interface**: Optional web UI for interacting with environments
- **Experimental Harness Helpers**: `openenv.core.harness` provides
  `ResourceSessionFactory`, `StepEnvSessionAdapter`, and `HarnessAdapter` for
  MCP-first training/evaluation harnesses

## Experimental Harness Helpers

OpenEnv now includes an additive harness-facing layer for cases where a rollout
driver should interact with environment resources through tools instead of
calling `reset()` / `step()` directly. These helpers are currently
experimental while RFC 005 remains in review, so import them from
`openenv.core.harness`.

Core pieces:

- `ResourceSessionFactory.create(task, seed, episode_id)` creates an isolated
  per-rollout session
- `StepEnvSessionAdapter` wraps an existing client and exposes session tools
- `SessionMCPBridge` exposes a session through MCP-style JSON-RPC
- `MCPHarnessAdapter` drives white-box rollouts where the trainer still owns
  model sampling, token IDs, and logprobs
- `CLIHarnessAdapter` supports opaque evaluation harnesses over the same session
  surface
- `build_harness_rollout_func(...)` builds a TRL-compatible `rollout_func`

This layer is additive. Existing environment clients and MCP environments keep
their current control-plane APIs unchanged.

## Installation

```bash
pip install "openenv"
```

For development:
```bash
pip install "openenv"
```

## Quick Start

### Creating an Environment Client

EnvClient is **async by default**. Use `async with` and `await` for all operations:

```python
import asyncio
from openenv.core import EnvClient, StepResult
from dataclasses import dataclass
from typing import Any


@dataclass
class MyAction:
    text: str


@dataclass
class MyObservation:
    response: str


class MyEnvClient(EnvClient[MyAction, MyObservation, Any]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"text": action.text}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs_data = payload["observation"]
        return StepResult(
            observation=MyObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> Any:
        return payload


# Async usage (recommended)
async def main():
    client = await MyEnvClient.from_docker_image("my-env:latest")
    async with client:
        result = await client.reset()
        step_result = await client.step(MyAction(text="hello"))


asyncio.run(main())

# Sync usage (via .sync() wrapper)
with MyEnvClient(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    step_result = client.step(MyAction(text="hello"))
```

### Creating an Environment Server

```python
from openenv.core.env_server import Environment, HTTPEnvServer, create_app
from dataclasses import dataclass


@dataclass
class MyAction:
    text: str


@dataclass
class MyObservation:
    response: str
    reward: float = 0.0
    done: bool = False


class MyEnvironment(Environment):
    def reset(self) -> MyObservation:
        return MyObservation(response="Ready")

    def step(self, action: MyAction) -> MyObservation:
        return MyObservation(response=f"Echo: {action.text}", reward=1.0, done=False)


# Create FastAPI app
env = MyEnvironment()
app = create_app(env, MyAction, MyObservation)

# Run with: uvicorn module:app --host 0.0.0.0 --port 8000
```

## Container Providers

OpenEnv Core supports multiple container providers:

### Local Docker Provider

```python
from openenv.core.containers.runtime import LocalDockerProvider

provider = LocalDockerProvider()
base_url = provider.start_container("my-env:latest")
provider.wait_for_ready(base_url)
# Use environment...
provider.stop_container()
```

### Kubernetes Provider (Coming Soon)

```python
from openenv.core.containers.runtime import KubernetesProvider

provider = KubernetesProvider(namespace="envs")
base_url = provider.start_container("my-env:latest")
# Use environment...
provider.stop_container()
```


## API Reference

### EnvClient

Async base class for environment clients. Key methods:

- `async connect()`: Establish WebSocket connection
- `async reset(**kwargs)`: Reset environment
- `async step(action)`: Execute action
- `async state()`: Get current state
- `async close()`: Close connection and cleanup
- `sync()`: Return a SyncEnvClient wrapper for synchronous usage

Abstract methods to implement:
- `_step_payload(action)`: Convert action to JSON
- `_parse_result(payload)`: Parse response to StepResult
- `_parse_state(payload)`: Parse state response

### SyncEnvClient

Synchronous wrapper around EnvClient. Use `client.sync()` to get one:

```python
sync_client = async_client.sync()
with sync_client:
    result = sync_client.reset()
    result = sync_client.step(action)
```

### HTTPEnvServer

Server wrapper with these methods:

- `register_routes(app)`: Register endpoints on FastAPI app
- `_deserialize_action(data)`: Convert JSON to Action
- `_serialize_observation(obs)`: Convert Observation to JSON

### Environment Interface

Base interface for environment implementations:

- `reset()`: Reset environment and return initial observation
- `step(action)`: Execute action and return observation
- `state`: Property returning current environment state

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the main OpenEnv repository for contribution guidelines.

## Links

- **Homepage**: https://github.com/huggingface/OpenEnv
- **Documentation**: https://github.com/huggingface/OpenEnv/blob/main/README.md
- **Bug Tracker**: https://github.com/huggingface/OpenEnv/issues
