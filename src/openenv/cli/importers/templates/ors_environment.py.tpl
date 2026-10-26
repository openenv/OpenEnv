from __future__ import annotations

import asyncio
import contextlib
import inspect
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
    _ORIGINAL_ENV_CLASS = getattr(import_module(_SOURCE_MODULE), "${source_class}")


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


def _normalize_split(split: Any) -> dict[str, Any]:
    value = _dump(split)
    if isinstance(value, dict):
        return value
    split_name = str(value)
    split_type = split_name if split_name in {"train", "validation", "test"} else "validation"
    return {"name": split_name, "type": split_type}


def _tool_from_ors(tool: Any) -> Tool:
    value = _dump(tool)
    input_schema = value.get("input_schema") or value.get("inputSchema")
    if input_schema is None:
        input_schema = {"type": "object", "properties": {}}
    return Tool(
        name=value["name"],
        description=value.get("description") or "",
        input_schema=input_schema,
    )


class ${class_name_prefix}Environment(Environment):
    """OpenEnv wrapper around a vendored ORS/OpenReward environment."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._ors_cls = _ORIGINAL_ENV_CLASS
        self._ors_env: Any | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_spec: Any | None = None
        self._last_reward: float | None = None
        self._done = False

    def list_splits(self) -> list[dict[str, Any]]:
        splits = _call_vendored(self._ors_cls.list_splits)
        return [_normalize_split(split) for split in splits]

    def list_tasks(self, split: str) -> list[Any]:
        return _dump(_call_vendored(self._ors_cls.list_tasks, split))

    def num_tasks(self, split: str) -> int:
        return int(_call_vendored(self._ors_cls.num_tasks, split))

    def get_task(self, split: str, index: int) -> Any:
        return _dump(_call_vendored(self._ors_cls.get_task, split, index))

    def get_task_range(
        self,
        split: str,
        start: int | None = None,
        stop: int | None = None,
    ) -> list[Any]:
        return _dump(_call_vendored(self._ors_cls.get_task_range, split, start, stop))

    def _first_task(self) -> tuple[str, int, Any]:
        splits = self.list_splits()
        if not splits:
            raise RuntimeError("ORS environment has no splits")
        split = splits[0]["name"]
        return split, 0, self.get_task(split, 0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_spec: dict[str, Any] | None = None,
        split: str | None = None,
        index: int | None = None,
        secrets: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Observation:
        self.close()
        if task_spec is None:
            if split is None and index is None:
                split, index, task_spec = self._first_task()
            elif split is None or index is None:
                raise ValueError("split and index must be provided together")
            else:
                task_spec = self.get_task(split, index)

        self._task_spec = _dump(task_spec)
        self._ors_env = _call_vendored(
            self._ors_cls,
            task_spec=task_spec,
            secrets=secrets or {},
        )
        _call_vendored(self._ors_env.setup)
        prompt = _dump(_call_vendored(self._ors_env.get_prompt))
        self._last_reward = None
        self._done = False
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            source_type="ors",
            original_env_class="${source_class}",
            task_spec=self._task_spec,
            split=split,
            index=index,
        )
        return Observation(
            done=False,
            reward=None,
            metadata={
                "source_type": "ors",
                "original_env_class": "${source_class}",
                "task_spec": self._task_spec,
                "prompt": prompt,
            },
        )

    def _ensure_session(self) -> None:
        if self._ors_env is None:
            raise RuntimeError("Call reset() before invoking ORS tools")

    def _all_tools(self) -> list[Tool]:
        shared = _call_vendored(self._ors_cls.list_tools)
        tools = [_tool_from_ors(tool) for tool in getattr(shared, "tools", [])]
        if self._ors_env is not None:
            task_tools = _call_vendored(self._ors_env.list_task_tools)
            tools.extend(_tool_from_ors(tool) for tool in getattr(task_tools, "tools", []))
        return tools

    def step(
        self,
        action: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        if isinstance(action, ListToolsAction):
            return ListToolsObservation(tools=self._all_tools())
        if not isinstance(action, CallToolAction):
            raise TypeError(f"Unsupported action type: {type(action).__name__}")

        self._ensure_session()
        assert self._ors_env is not None
        result = _call_vendored(self._ors_env._call_tool, action.tool_name, action.arguments)
        root = getattr(result, "root", result)
        ok = getattr(root, "ok", False)
        self._state.step_count += 1

        if ok:
            output = root.output
            blocks = _dump(getattr(output, "blocks", []))
            metadata = _dump(getattr(output, "metadata", None)) or {}
            reward = getattr(output, "reward", None)
            done = bool(getattr(output, "finished", False))
            self._last_reward = reward
            self._done = done
            return CallToolObservation(
                tool_name=action.tool_name,
                result={"blocks": blocks, "metadata": metadata},
                reward=reward,
                done=done,
                metadata=metadata,
            )

        message = str(getattr(root, "error", "ORS tool call failed"))
        return CallToolObservation(
            tool_name=action.tool_name,
            result=None,
            error=ToolError(
                error_type=ToolErrorType.EXECUTION_ERROR,
                message=message,
            ),
            reward=None,
            done=False,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        if self._ors_env is None:
            return
        try:
            _call_vendored(self._ors_env.teardown)
        finally:
            self._ors_env = None
