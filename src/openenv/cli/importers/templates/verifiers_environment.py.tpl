from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import sys
import threading
from importlib import import_module
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
    ToolError,
    ToolErrorType,
)
from openenv.core.env_server.types import Observation, State


_VENDORED_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "${vendor_dir}"
_SOURCE_MODULE = "${source_import_module}"
_LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def _vendored_source_path():
    source_path = str(_VENDORED_SOURCE_ROOT)
    inserted = source_path not in sys.path
    if inserted:
        sys.path.insert(0, source_path)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(source_path)
            except ValueError:
                pass


with _vendored_source_path():
    _LOAD_ENVIRONMENT = getattr(import_module(_SOURCE_MODULE), "load_environment")


def _run_sync(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    if inspect.iscoroutine(value) and value.cr_frame is None:
        raise RuntimeError("Cannot await an already-consumed coroutine")

    async def await_value() -> Any:
        return await value

    return asyncio.run(await_value())


def _call_vendored(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is None or not running_loop.is_running():
        with _vendored_source_path():
            return _run_sync(func(*args, **kwargs))

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            with _vendored_source_path():
                result["value"] = _run_sync(func(*args, **kwargs))
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _dump(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, tuple):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _dump(value.model_dump())
    if hasattr(value, "__dict__"):
        return _dump(value.__dict__)
    return str(value)


def _coerce_prompt(task: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = task.get("prompt")
    if isinstance(prompt, list):
        return _dump(prompt)
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    question = task.get("question")
    if question is not None:
        return [{"role": "user", "content": str(question)}]
    return []


def _completion_messages(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    messages = arguments.get("messages")
    if isinstance(messages, list):
        return _dump(messages)
    completion = arguments.get("completion", arguments.get("answer", ""))
    return [{"role": "assistant", "content": str(completion)}]


def _load_environment() -> Any:
    sig = inspect.signature(_LOAD_ENVIRONMENT)
    kwargs: dict[str, Any] = {}
    config_param = sig.parameters.get("config")
    if config_param is not None and config_param.default is inspect.Parameter.empty:
        annotation = config_param.annotation
        if isinstance(annotation, type):
            kwargs["config"] = annotation()
    return _call_vendored(_LOAD_ENVIRONMENT, **kwargs)


def _dataset_to_tasks(dataset: Any) -> list[dict[str, Any]]:
    if dataset is None:
        return []
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        row = _dump(dataset[index])
        if isinstance(row, dict):
            row.setdefault("example_id", index)
            rows.append(row)
    return rows


class ${class_name_prefix}Environment(Environment):
    """OpenEnv wrapper around a vendored Prime Intellect Verifiers environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._vf_env: Any | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_spec: dict[str, Any] | None = None
        self._prompt: list[dict[str, Any]] = []
        self._last_reward: float | None = None
        self._done = False

    def _env(self) -> Any:
        if self._vf_env is None:
            self._vf_env = _load_environment()
        return self._vf_env

    def _rows_for_split(self, split: str) -> list[dict[str, Any]]:
        env = self._env()
        taskset = getattr(env, "taskset", None)
        if taskset is not None:
            if split in {"eval", "validation", "test"} and hasattr(taskset, "eval_rows"):
                rows = _dump(_call_vendored(taskset.eval_rows))
            elif hasattr(taskset, "rows"):
                rows = _dump(_call_vendored(taskset.rows))
            else:
                rows = []
            tasks = []
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                row_split = row.get("split")
                if row_split is not None and split not in {"eval", "validation", "test"} and row_split != split:
                    continue
                task = _call_vendored(taskset.task, row) if hasattr(taskset, "task") else row
                dumped = _dump(task)
                if isinstance(dumped, dict):
                    dumped.setdefault("example_id", index)
                    tasks.append(dumped)
            return tasks

        if split in {"eval", "validation", "test"} and hasattr(env, "get_eval_dataset"):
            return _dataset_to_tasks(_call_vendored(env.get_eval_dataset))
        if hasattr(env, "get_dataset"):
            return _dataset_to_tasks(_call_vendored(env.get_dataset))
        return []

    def list_splits(self) -> list[dict[str, Any]]:
        env = self._env()
        taskset = getattr(env, "taskset", None)
        names: list[str] = []
        if taskset is not None and hasattr(taskset, "rows"):
            for row in _dump(_call_vendored(taskset.rows)):
                if isinstance(row, dict) and row.get("split"):
                    names.append(str(row["split"]))
            if hasattr(taskset, "eval_rows"):
                try:
                    if len(_dump(_call_vendored(taskset.eval_rows))) > 0:
                        names.append("eval")
                except Exception:
                    pass
        else:
            if hasattr(env, "get_dataset"):
                names.append("train")
            if hasattr(env, "get_eval_dataset"):
                names.append("eval")
        if not names:
            names.append("train")

        seen = set()
        splits = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            split_type = name if name in {"train", "validation", "test"} else "validation"
            splits.append({"name": name, "type": split_type})
        return splits

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return self._rows_for_split(split)

    def num_tasks(self, split: str) -> int:
        return len(self.list_tasks(split))

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self.list_tasks(split)[index]

    def get_task_range(
        self,
        split: str,
        start: int | None = None,
        stop: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.list_tasks(split)[slice(start, stop)]

    def _first_task(self) -> tuple[str, int, dict[str, Any]]:
        splits = self.list_splits()
        if not splits:
            raise RuntimeError("Verifiers environment has no splits")
        split = splits[0]["name"]
        return split, 0, self.get_task(split, 0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_spec: dict[str, Any] | None = None,
        split: str | None = None,
        index: int | None = None,
        **kwargs: Any,
    ) -> Observation:
        if task_spec is None:
            if split is None and index is None:
                split, index, task_spec = self._first_task()
            elif split is None or index is None:
                raise ValueError("split and index must be provided together")
            else:
                task_spec = self.get_task(split, index)

        self._task_spec = _dump(task_spec)
        self._prompt = _coerce_prompt(self._task_spec)
        self._last_reward = None
        self._done = False
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            source_type="verifiers",
            task_spec=self._task_spec,
            split=split,
            index=index,
        )
        return Observation(
            done=False,
            reward=None,
            metadata={
                "source_type": "verifiers",
                "task_spec": self._task_spec,
                "prompt": self._prompt,
            },
        )

    def _ensure_session(self) -> None:
        if self._task_spec is None:
            raise RuntimeError("Call reset() before submitting Verifiers completions")

    def step(
        self,
        action: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        if isinstance(action, ListToolsAction):
            return ListToolsObservation(
                tools=[
                    Tool(
                        name="submit",
                        description="Submit a completion to score with the Verifiers environment.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "completion": {"type": "string"},
                                "messages": {"type": "array"},
                            },
                        },
                    )
                ]
            )
        if not isinstance(action, CallToolAction):
            raise TypeError(f"Unsupported action type: {type(action).__name__}")
        if action.tool_name != "submit":
            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(
                    error_type=ToolErrorType.TOOL_NOT_FOUND,
                    message=f"Unknown Verifiers wrapper tool: {action.tool_name}",
                ),
                done=False,
                reward=None,
            )

        self._ensure_session()
        assert self._task_spec is not None
        completion = _completion_messages(action.arguments)
        score = self._score_completion(completion)
        self._state.step_count += 1
        self._last_reward = score.get("reward")
        self._done = True
        return CallToolObservation(
            tool_name=action.tool_name,
            result=score,
            reward=score.get("reward"),
            done=True,
            metadata={"metrics": score.get("metrics", {})},
        )

    def _score_completion(self, completion: list[dict[str, Any]]) -> dict[str, Any]:
        env = self._env()
        assert self._task_spec is not None
        state: dict[str, Any] = {
            "input": dict(self._task_spec),
            "task": dict(self._task_spec),
            "prompt": self._prompt,
            "completion": completion,
            "answer": self._task_spec.get("answer", ""),
            "info": self._task_spec.get("info", {}),
            "trajectory": [],
            "reward": None,
            "metrics": None,
            "is_completed": True,
            "is_truncated": False,
        }

        taskset = getattr(env, "taskset", None)
        harness = getattr(env, "harness", None)
        if taskset is not None and harness is not None:
            task = _call_vendored(taskset.to_task, self._task_spec) if hasattr(taskset, "to_task") else self._task_spec
            state["task"] = _dump(task)
            with _vendored_source_path():
                maybe_state_cls = getattr(import_module("verifiers"), "State", None)
            if maybe_state_cls is not None and hasattr(maybe_state_cls, "for_task"):
                try:
                    vf_state = maybe_state_cls.for_task(task)
                    vf_state.update(state)
                    state = vf_state
                except Exception:
                    _LOGGER.exception("Failed to initialize Verifiers State for task")
            if hasattr(harness, "score_group"):
                _call_vendored(harness.score_group, [task], [state])

        elif hasattr(env, "rubric") and hasattr(env.rubric, "score_rollout"):
            _call_vendored(env.rubric.score_rollout, state)

        reward = state.get("reward")
        return {
            "completion": completion,
            "reward": reward,
            "metrics": _dump(state.get("metrics") or {}),
            "state": _dump(state),
        }

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        if self._vf_env is None:
            return
        try:
            teardown = getattr(self._vf_env, "_teardown", None)
            if callable(teardown):
                _call_vendored(teardown)
        finally:
            self._vf_env = None
