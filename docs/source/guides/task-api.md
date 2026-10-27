# The Task API

This guide explains the Task API: the optional discovery layer that lets a dataset-backed environment publish the *set of tasks* it can run, so a trainer or evaluator can enumerate them and choose which one each episode uses.

It exists to answer a common question: OpenEnv's step loop gives you one episode at a time, so how does a training script find out that an environment holds 7,595 test problems, and how does it ask for problem number 12?

## The Short Answer

The Task API has two halves:

- **Server side**: your environment optionally implements five methods — `list_splits()`, `list_tasks()`, `num_tasks()`, `get_task()`, `get_task_range()`. These are described by the `TaskProvider` protocol.
- **HTTP side**: when those methods exist, the environment server automatically exposes them as routes under `/{env_name}/…`. You do not register anything.

Task discovery is **metadata only**. It never starts an episode. Selecting a task happens the usual way, through `reset()`:

```python
env.list_splits()            # ["train", "test"]
env.num_tasks("test")        # 7595
env.get_task("test", 12)     # {"id": "test-12", "index": 12, "split": "test"}
env.reset(split="test", index=12)   # <- this is what starts the episode
```

The API is shaped to be compatible with ORS/OpenReward task and split conventions, which is why environments imported with `openenv import` get it for free.

## When You Need It

Implement the Task API when your environment is **dataset-backed** — when an episode means "run one row of a dataset" and a trainer needs to iterate, shard, or shuffle over those rows.

You do **not** need it for environments where every episode is generated or self-contained: Echo, Snake, a game, a REPL. Those environments simply omit the methods, and the routes report `501 Not Implemented`.

## The Protocol

`TaskProvider` is a `typing.Protocol`, not a base class. You do not inherit from it — you just define the methods on your `Environment` subclass and the server discovers them by name.

```python
from typing import Any, Optional

class TaskProvider(Protocol):
    def list_splits(self) -> list[Any]: ...
    def list_tasks(self, split: str) -> list[Any]: ...
    def num_tasks(self, split: str) -> int: ...
    def get_task(self, split: str, index: int) -> Any: ...
    def get_task_range(
        self,
        split: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> list[Any]: ...
```

Each method may be sync or `async`; the server awaits the result when it is awaitable, so you can back task discovery with async dataset clients.

A "task spec" is deliberately untyped (`Any`). Return whatever your environment needs — a dict, a Pydantic model, a dataclass. The server converts it to JSON, handling Pydantic models, dataclasses, and plain objects. In practice a positional stub such as `{"id": "test-12", "index": 12, "split": "test"}` is enough, because `reset()` is what actually loads the row.

### Two rules that are easy to miss

1. **Task methods must be side-effect-free.** They are discovery, not control. They must not mutate episode state, consume a stream, or advance a cursor.
2. **They must work on a freshly constructed environment.** Each task route builds a short-lived environment instance, calls the one method, and closes it again. Nothing you set up in `reset()` is available, so read configuration in `__init__` (or lazily inside the method) rather than relying on episode state.

## HTTP Routes

The routes are registered by `HTTPEnvServer` and appear in the server's OpenAPI schema under the `Task API` tag. All of them are namespaced by environment name except `/list_environments`.

| Method | Route | Body | Response |
|--------|-------|------|----------|
| `GET`  | `/list_environments` | — | `["latex_ocr_env"]` |
| `GET`  | `/{env_name}/splits` | — | `[{"name": "train", "type": "train"}, …]` |
| `POST` | `/{env_name}/tasks` | `{"split": "test"}` | `{"tasks": [...], "env_name": "latex_ocr_env"}` |
| `POST` | `/{env_name}/num_tasks` | `{"split": "test"}` | `{"num_tasks": 7595}` |
| `POST` | `/{env_name}/task` | `{"split": "test", "index": 12}` | `{"task": {...}}` |
| `POST` | `/{env_name}/task_range` | `{"split": "test", "start": 0, "stop": 32}` | `{"tasks": [...]}` |

The request bodies are `ListTasksRequest`, `NumTasksRequest`, `GetTaskRequest`, and `GetTaskRangeRequest`. `start` and `stop` follow Python slice semantics: `start` is inclusive, `stop` is exclusive, and both may be omitted.

These are HTTP-only. There are no WebSocket message types for task discovery — the `/ws` session protocol stays focused on `reset`, `step`, `state`, and `close`.

### `env_name`

`{env_name}` is the name passed to `create_app(..., env_name="latex_ocr_env")`. If you do not pass one, it defaults to the environment factory's class name (for example `LatexOCREnvironment`). Matching is case-insensitive, and any other name returns `404`. Pass `env_name` explicitly for any environment that implements the Task API — the generated URL is part of your public surface, and a class rename should not break a trainer.

```python
from openenv.core.env_server import create_app

app = create_app(
    LatexOCREnvironment,
    LatexOCRAction,
    LatexOCRObservation,
    env_name="latex_ocr_env",
)
```

### Split normalization

`list_splits()` may return plain strings, dicts, or Pydantic models. The server normalizes each entry to `{"name": ..., "type": ...}`:

- a dict or model is passed through as-is (after JSON conversion)
- a string becomes `{"name": s, "type": s}` when `s` is `train`, `validation`, or `test`
- any other string becomes `{"name": s, "type": "validation"}`

So returning `["train", "holdout"]` yields `[{"name": "train", "type": "train"}, {"name": "holdout", "type": "validation"}]`. Return dicts yourself if you need to control the `type` for a custom split name.

### Error semantics

| Condition | Status |
|-----------|--------|
| `{env_name}` does not match the server's environment | `404 Not Found` |
| Environment does not define the method, or it raises `NotImplementedError` | `501 Not Implemented` |
| Method raises `IndexError` (index out of range) | `400 Bad Request` |

Raise `IndexError` from `get_task()` for an out-of-range index so callers get a `400` rather than a `500`. Reserve `NotImplementedError` for capabilities the environment genuinely does not support — for example, random access on a streaming split.

## Implementing It

A dataset-backed environment reads its dataset in `__init__`, answers discovery from metadata, and loads the actual row in `reset()`:

```python
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State


class LatexOCREnvironment(Environment):
    SPLITS = ["train", "test"]

    def __init__(self, dataset_name: str = "unsloth/LaTeX_OCR") -> None:
        super().__init__()
        self.dataset_name = dataset_name

    # --- Task API: discovery only, no side effects ---

    def list_splits(self) -> list[str]:
        return self.SPLITS

    def num_tasks(self, split: str) -> int:
        return len(self._load_split(split))

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return [
            {"id": f"{split}-{i}", "index": i}
            for i in range(self.num_tasks(split))
        ]

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        n = self.num_tasks(split)
        if index < 0 or index >= n:
            raise IndexError(f"index {index} out of range for split {split} (n={n})")
        return {"id": f"{split}-{index}", "index": index, "split": split}

    def get_task_range(
        self, split: str, start: Optional[int] = None, stop: Optional[int] = None
    ) -> list[dict[str, Any]]:
        n = self.num_tasks(split)
        start = 0 if start is None else start
        stop = n if stop is None else min(stop, n)
        return [
            {"id": f"{split}-{i}", "index": i, "split": split}
            for i in range(start, stop)
        ]

    # --- Episode: this is where a task is actually loaded ---

    def reset(
        self,
        split: str = "test",
        index: Optional[int] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LatexOCRObservation:
        if split not in self.list_splits():
            raise ValueError(f"unknown split {split!r}; expected {self.list_splits()}")
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        row = self._load_row(split, index, seed)
        return LatexOCRObservation(image_base64=row["image"], done=False)
```

### Selecting a task in `reset()`

`ResetRequest` allows extra fields, and both the HTTP `/reset` route and the WebSocket `reset` message forward unknown keys to your `reset()` — after filtering them against its signature. So naming the parameters `split` and `index` in `reset()` is all it takes for this to work:

```python
result = env.reset(split="test", index=12)
```

The corresponding wire calls are `POST /reset` with `{"split": "test", "index": 12}`, or a WebSocket `{"type": "reset", "data": {"split": "test", "index": 12}}`. The server filters incoming keys against your `reset()` signature, so a key it does not declare is dropped silently unless the signature also has `**kwargs`. A misspelled parameter therefore shows up as "the environment ignored my selection" rather than an error — worth checking first when a task selection appears not to take effect.

When `index` is omitted, pick a row from `seed` so episodes stay reproducible.

### Streaming and very large splits

`list_tasks()` and `get_task_range()` can be asked for more rows than you want to materialize. Two defenses are worth building in:

- return positional stubs rather than real rows, and cap how many you generate (`list_tasks()` returning a bounded preview is fine — `num_tasks()` still reports the honest total)
- if a split is streamed sequentially and cannot be randomly accessed, raise `ValueError` or `NotImplementedError` from `reset(index=...)` and say which mode does support indexing

Report the true count from `num_tasks()` even when `list_tasks()` is truncated. A trainer that shards work by `num_tasks()` needs the real denominator.

## Consuming the Task API

The core clients (`EnvClient`, `SyncEnvClient`) do not ship Task API methods, because task specs are environment-specific. Environment clients add thin HTTP helpers alongside the inherited Gym-style `reset()`/`step()`:

```python
import requests

ENV_NAME = "latex_ocr_env"


class LatexOCREnv(EnvClient[LatexOCRAction, LatexOCRObservation, State]):
    def _task_url(self, path: str) -> str:
        # `base_url` may be ws(s)://, and the Task API is plain HTTP.
        base = self.base_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
        return f"{base}/{ENV_NAME}/{path}"

    def list_splits(self) -> list[str]:
        resp = requests.get(self._task_url("splits"), timeout=30)
        resp.raise_for_status()
        return [s["name"] for s in resp.json()]

    def num_tasks(self, split: str) -> int:
        resp = requests.post(
            self._task_url("num_tasks"),
            json={"split": split},
            timeout=60,
        )
        resp.raise_for_status()
        return int(resp.json()["num_tasks"])
```

Note that the Task API is served over HTTP while episodes typically run over the WebSocket `/ws` session. A client that was constructed from a `ws://` URL has to derive the HTTP base URL before calling these routes.

From a training or evaluation loop, discovery and episode control then compose naturally:

```python
with LatexOCREnv.from_docker_image("latex-ocr-env:latest").sync() as env:
    total = env.num_tasks("test")
    for index in range(total):
        result = env.reset(split="test", index=index)
        prediction = model(result.observation.image_base64)
        result = env.step(LatexOCRAction(latex=prediction))
        print(index, result.reward)
```

Or with `curl`, against a locally running server:

```bash
curl http://localhost:8000/list_environments
curl http://localhost:8000/latex_ocr_env/splits
curl -X POST http://localhost:8000/latex_ocr_env/num_tasks \
  -H 'Content-Type: application/json' -d '{"split": "test"}'
curl -X POST http://localhost:8000/latex_ocr_env/task_range \
  -H 'Content-Type: application/json' -d '{"split": "test", "start": 0, "stop": 4}'
```

## Imported Environments

`openenv import` generates wrappers that already implement the Task API by delegating to the source environment's own task and split methods, and whose `reset()` accepts `split` / `index` / `task_spec`. If you are bringing in an ORS/OpenReward or Prime Intellect Verifiers environment, you get task discovery without writing any of the above — see the [CLI reference](../reference/cli.md) for `openenv import`.

## Checklist

Before shipping a dataset-backed environment:

1. All five methods defined, or a deliberate `NotImplementedError` for the ones you cannot support.
2. Every method works on a fresh instance, with no reliance on `reset()` having run.
3. No side effects — no cursor advanced, no stream consumed, no episode state mutated.
4. `env_name` passed explicitly to `create_app()`.
5. `get_task()` raises `IndexError` for out-of-range indices.
6. `num_tasks()` reports the honest total, even if `list_tasks()` truncates.
7. `reset()` accepts `split` and `index` with the same names the trainer will use, and falls back to `seed`-driven selection when `index` is omitted.

## Related Reading

- [Concepts](concepts.md)
- [Training with OpenEnv](training.md)
- [Core API](../reference/core.md)
- [CLI reference](../reference/cli.md)
