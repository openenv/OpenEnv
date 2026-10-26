"""BrowserGym Environment for OpenEnv.

BrowserGym is a unified framework for web-based agent tasks that provides
access to multiple benchmarks under a single Gymnasium-compatible API.

Included Benchmarks:
- **MiniWoB++**: 100+ simple web tasks for training (no external infrastructure!)
- **WebArena**: 812 realistic evaluation tasks (requires backend setup)
- **VisualWebArena**: Visual web navigation tasks
- **WorkArena**: Enterprise task automation

Key Features:
- Unified API across all benchmarks
- Gymnasium-compatible interface
- Support for multiple observation types (text, visual, DOM)
- Action spaces for natural language commands
- Perfect for training (MiniWoB) and evaluation (WebArena)

Training Example (MiniWoB - works immediately):
    ```python
    from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

    # Create training environment - no backend setup needed!
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        environment={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": "click-test",
        }
    )

    # Train your agent
    for episode in range(1000):
        result = env.reset()
        while not result.done:
            action = agent.get_action(result.observation)
            result = env.step(action)

    env.close()
    ```

Evaluation Example (WebArena - requires backend):
    ```python
    from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

    # Create evaluation environment
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        environment={
            "BROWSERGYM_BENCHMARK": "webarena",
            "BROWSERGYM_TASK_NAME": "0",
            "SHOPPING": "http://your-server:7770",
            # ... other backend URLs
        }
    )

    # Evaluate your trained agent
    result = env.reset()
    # ... run evaluation
    env.close()
    ```
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .client import BrowserGymEnv
from .models import BrowserGymAction, BrowserGymObservation, BrowserGymState

if TYPE_CHECKING:
    from .harness import BrowserGymSessionFactory, build_browsergym_action_tool_call

__all__ = [
    "BrowserGymEnv",
    "BrowserGymAction",
    "BrowserGymObservation",
    "BrowserGymState",
    "BrowserGymSessionFactory",
    "build_browsergym_action_tool_call",
]

_LAZY_ATTRS = {
    "BrowserGymSessionFactory": (".harness", "BrowserGymSessionFactory"),
    "build_browsergym_action_tool_call": (
        ".harness",
        "build_browsergym_action_tool_call",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
