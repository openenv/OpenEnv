"""
AWM Environment wraps 1,000 Agent World Model sub-environments into a single OpenEnv
environment. Each sub-environment is launched as a subprocess on demand
and accessed via MCP tool calls.
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction, Tool
from openenv.core.env_server.types import Action, State

from ..models import AWMListToolsObservation, AWMObservation
from .data_loader import AWMDataLoader, normalize_scenario_name
from .db_manager import cleanup_session_dir, create_database, save_snapshot
from .scenario_manager import ScenarioProcess
from .session_registry import registry as _registry
from .verifier import run_llm_judge, run_verifier
from .config import DEFAULT_REWARD_CONFIG

logger = logging.getLogger(__name__)

# Tools dispatched specially by step() (see _handle_done / _handle_verify /
# _handle_list_scenarios) rather than proxied to the sub-env subprocess.
# Not currently used to filter list_tools output — the sub-env subprocess
# never surfaces these names. Kept commented for documentation.
# HIDDEN_TOOLS = frozenset(["done", "verify", "__list_scenarios__"])

VALID_VERIFIER_MODES = {"sql", "code"}

# Reward types that map to format_error
FORMAT_ERROR_TYPES = {"tool_not_found", "invalid_args", "invalid_action"}

_TOOL_NOT_FOUND_KEYWORDS = ["not found", "unknown tool", "no tool"]
_INVALID_ARGS_KEYWORDS = [
    "invalid",
    "argument",
    "parameter",
    "required property",
    "validation error",
    "missing",
    "schema",
]
_TIMEOUT_KEYWORDS = ["timeout", "timed out"]


def _classify_tool_error(error_msg: str) -> str:
    """Classify a tool call error into a reward_type string."""
    lower = error_msg.lower()
    if any(kw in lower for kw in _TOOL_NOT_FOUND_KEYWORDS):
        return "tool_not_found"
    if any(kw in lower for kw in _INVALID_ARGS_KEYWORDS):
        return "invalid_args"
    if any(kw in lower for kw in _TIMEOUT_KEYWORDS):
        return "timeout"
    return "server_error"


def _run_async_oneshot(coro: Any) -> Any:
    """Run an async coroutine from sync context (one-shot, for LLM judge etc.)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class AWMEnvironment(Environment):
    """
    Lifecycle:
        1. reset(scenario="...", task_idx=...) -> starts a sub-env subprocess
        2. step(ListToolsAction()) -> lists tools from the sub-env
        3. step(CallToolAction(...)) -> proxies tool call to the sub-env
        4. step(CallToolAction(tool_name="verify", arguments={verifier_mode: "sql"|"code"})) -> runs verifier
        5. step(CallToolAction(tool_name="done")) -> ends episode, destroys environment
        6. close() -> kills subprocess, cleans up
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, data_loader: AWMDataLoader | None = None):
        super().__init__()

        self._data_loader = data_loader or AWMDataLoader()
        self._process = ScenarioProcess()

        self._state = State(episode_id=None, step_count=0)
        self._scenario: str | None = None
        self._task: str | None = None
        self._task_idx: int | None = None
        self._has_verifier: dict | None = None  # {sql: bool, code: bool}
        self._reset_ok: bool = False
        self._episode_done: bool = False

        self._session_dir: str | None = None
        self._db_path: str | None = None
        self._initial_db_path: str | None = None

        # LLM config for sql verifier mode
        self._llm_base_url: str | None = None
        self._llm_api_key: str | None = None
        self._llm_model: str | None = None

        self._tools_cache: list[dict] | None = None
        self._trajectory: list[dict] = []
        self._keep_session: bool = False

        # Reward config (customizable at reset)
        self._reward_config: dict = DEFAULT_REWARD_CONFIG.copy()

        # Session registry tracking
        self._registry_id: str | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario: str | None = None,
        task_idx: int | None = None,
        task: str | None = None,
        reward_config: dict | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        **kwargs: Any,
    ) -> AWMObservation:
        self._reset_ok = False
        self._episode_done = False

        if not scenario:
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="reset_error",
                error="Parameter 'scenario' is required",
            )

        scenario_key = normalize_scenario_name(scenario)

        if not self._data_loader.scenario_exists(scenario_key):
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="reset_error",
                error=f"Scenario '{scenario}' not found",
            )

        self._cleanup_session()

        self._scenario = scenario_key
        self._task_idx = task_idx
        self._tools_cache = None
        self._trajectory = []

        # Set custom reward config or use default
        self._reward_config = (
            reward_config.copy() if reward_config else DEFAULT_REWARD_CONFIG.copy()
        )

        self._llm_base_url = llm_base_url or os.environ.get("OPENENV_AWM_LLM_BASE_URL")
        self._llm_api_key = llm_api_key or os.environ.get("OPENENV_AWM_LLM_API_KEY")
        self._llm_model = llm_model or os.environ.get("OPENENV_AWM_LLM_MODEL")

        if task is not None:
            self._task = task
        elif task_idx is not None:
            tasks = self._data_loader.get_tasks(scenario_key)
            if 0 <= task_idx < len(tasks):
                self._task = tasks[task_idx]
            else:
                return AWMObservation(
                    done=False,
                    reward=None,
                    reward_type="reset_error",
                    error=f"task_idx {task_idx} out of range (0..{len(tasks) - 1})",
                )
        else:
            self._task = None

        # Check verifier support for both modes
        self._has_verifier = None
        if task_idx is not None:
            sql_verifier = self._data_loader.get_verifier(scenario_key, task_idx, "sql")
            code_verifier = self._data_loader.get_verifier(
                scenario_key, task_idx, "code"
            )

            sql_available = False
            code_available = False

            if sql_verifier:
                sql_code = sql_verifier.get("verification", {}).get("code", "")
                sql_available = bool(
                    sql_code and isinstance(sql_code, str) and len(sql_code.strip()) > 0
                )

            if code_verifier:
                code_code = code_verifier.get("verification", {}).get("code", "")
                code_available = bool(
                    code_code
                    and isinstance(code_code, str)
                    and len(code_code.strip()) > 0
                )

            if sql_available or code_available:
                self._has_verifier = {"sql": sql_available, "code": code_available}

        self._session_dir = tempfile.mkdtemp(prefix=f"openenv_awm_{scenario_key}_")
        self._db_path = f"{self._session_dir}/{scenario_key}.db"
        self._initial_db_path = f"{self._session_dir}/{scenario_key}_initial.db"

        logger.info(
            f"[reset] scenario={scenario_key} task_idx={task_idx} "
            f"session_dir={self._session_dir} "
            f"db={self._db_path} initial_db={self._initial_db_path}"
        )

        try:
            db_schema = self._data_loader.get_db_schema(scenario_key)
            sample_data = self._data_loader.get_sample_data(scenario_key)
            create_database(self._db_path, db_schema, sample_data)
            save_snapshot(self._db_path, self._initial_db_path)
        except Exception as e:
            logger.error(f"Failed to create database for {scenario_key}: {e}")
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="reset_error",
                error=f"Database creation failed: {e}",
            )

        try:
            full_code = self._data_loader.get_env_code(scenario_key)
            self._process.start(full_code, self._db_path, self._session_dir)
        except Exception as e:
            logger.error(f"Failed to start sub-env for {scenario_key}: {e}")
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="reset_error",
                error=f"Sub-environment start failed: {e}",
            )

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Register with session registry for idle tracking
        self._registry_id = self._state.episode_id
        _registry.register(self._registry_id, self, scenario=self._scenario)

        tools: list[dict] = []
        tool_error: str | None = None
        try:
            tools = self._process.list_tools()
            self._tools_cache = tools
        except Exception as e:
            tool_error = str(e)
            logger.warning(f"Failed to list tools on startup: {e}")

        if len(tools) == 0:
            self._reset_ok = True
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="reset_warning",
                scenario=scenario_key,
                task=self._task,
                task_idx=self._task_idx,
                has_verifier=self._has_verifier,
                num_tools=0,
                warning=f"Sub-env started but no tools discovered. {tool_error or ''}".strip(),
            )

        self._reset_ok = True
        return AWMObservation(
            done=False,
            reward=None,
            reward_type="reset_ok",
            scenario=scenario_key,
            task=self._task,
            task_idx=self._task_idx,
            has_verifier=self._has_verifier,
            num_tools=len(tools),
        )

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> AWMObservation | AWMListToolsObservation:
        if self._episode_done:
            return AWMObservation(
                done=True,
                reward=None,
                reward_type="episode_already_done",
                error="Episode has ended. Call reset() to start a new episode.",
            )

        self._state.step_count += 1

        # Update idle tracker
        if self._registry_id:
            _registry.touch(self._registry_id)

        if isinstance(action, ListToolsAction):
            return self._handle_list_tools()
        elif isinstance(action, CallToolAction):
            if action.tool_name == "done":
                return self._handle_done(action)
            elif action.tool_name == "verify":
                return self._handle_verify(action)
            elif action.tool_name == "__list_scenarios__":
                return self._handle_list_scenarios()
            else:
                return self._handle_call_tool(action, timeout_s)
        else:
            return AWMObservation(
                done=False,
                reward=self._get_reward("invalid_action"),
                reward_type="invalid_action",
                error=f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction.",
            )

    def _handle_list_tools(self) -> AWMListToolsObservation:
        """Return tools from the sub-environment (cached)."""
        if not self._process.is_running:
            obs = AWMListToolsObservation(
                tools=[],
                error="Sub-environment is not running. Call reset() first.",
            )
            self._trajectory.append(
                {
                    "action": "list_tools",
                    "success": False,
                    "error": obs.error,
                }
            )
            return obs

        if self._tools_cache is not None:
            tools = [
                Tool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in self._tools_cache
            ]
            tool_names = [t["name"] for t in self._tools_cache]
            self._trajectory.append(
                {
                    "action": "list_tools",
                    "success": True,
                    "num_tools": len(tools),
                    "tool_names": tool_names,
                }
            )
            return AWMListToolsObservation(tools=tools)

        try:
            raw_tools = self._process.list_tools()
            self._tools_cache = raw_tools
            tools = [
                Tool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in raw_tools
            ]
            tool_names = [t["name"] for t in raw_tools]
            self._trajectory.append(
                {
                    "action": "list_tools",
                    "success": True,
                    "num_tools": len(tools),
                    "tool_names": tool_names,
                }
            )
            return AWMListToolsObservation(tools=tools)
        except Exception as e:
            self._trajectory.append(
                {
                    "action": "list_tools",
                    "success": False,
                    "error": str(e),
                }
            )
            return AWMListToolsObservation(
                tools=[],
                error=f"Failed to list tools: {e}",
            )

    def _handle_call_tool(
        self, action: CallToolAction, timeout_s: float | None = None
    ) -> AWMObservation:
        """Proxy a tool call to the sub-environment subprocess."""
        if not self._process.is_running:
            return AWMObservation(
                done=False,
                reward=self._get_reward("server_error"),
                reward_type="server_error",
                tool_name=action.tool_name,
                error="Sub-environment is not running. Call reset() first.",
            )

        timeout = timeout_s if timeout_s is not None else 30.0

        try:
            result = self._process.call_tool(
                action.tool_name,
                action.arguments,
                timeout,
            )
        except Exception as e:
            return AWMObservation(
                done=False,
                reward=self._get_reward("server_error"),
                reward_type="server_error",
                tool_name=action.tool_name,
                error=str(e),
            )

        self._trajectory.append(
            {
                "action": "call_tool",
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "success": result["success"],
                "result": result.get("result"),
                "error": result.get("error"),
            }
        )

        if result["success"]:
            return AWMObservation(
                done=False,
                reward=self._get_reward("tool_call_ok"),
                reward_type="tool_call_ok",
                tool_name=action.tool_name,
                tool_result=result["result"],
            )

        error_msg = result.get("error", "Unknown error")
        error_type = _classify_tool_error(error_msg)
        return AWMObservation(
            done=False,
            reward=self._get_reward(error_type),
            reward_type=error_type,
            tool_name=action.tool_name,
            error=error_msg,
        )

    def _get_reward(self, reward_type: str) -> float:
        """Get reward value for a reward type using the configured reward config."""
        # Map format error types to format_error
        if reward_type in FORMAT_ERROR_TYPES:
            return self._reward_config.get("format_error", -1.0)
        # Return configured reward or 0.0 for unknown types
        return self._reward_config.get(reward_type, 0.0)

    def _handle_verify(self, action: CallToolAction) -> AWMObservation:
        """Handle the `verify` tool — run verifier with specified mode."""
        if not self._reset_ok or self._scenario is None:
            return AWMObservation(
                done=False,
                reward=self._get_reward("server_error"),
                reward_type="server_error",
                error="Cannot verify: environment not initialized "
                "(reset failed or not called)",
            )

        if self._task is None or self._task_idx is None:
            return AWMObservation(
                done=False,
                reward=self._get_reward("no_verifier"),
                reward_type="no_verifier",
                error="Cannot verify: no task specified at reset",
            )

        # Get verifier_mode from arguments
        args = action.arguments or {}
        verifier_mode = args.get("verifier_mode", "code")
        final_answer = args.get("final_answer")

        if verifier_mode not in VALID_VERIFIER_MODES:
            return AWMObservation(
                done=False,
                reward=self._get_reward("invalid_args"),
                reward_type="invalid_args",
                error=f"Invalid verifier_mode '{verifier_mode}'. "
                f"Must be one of: {', '.join(sorted(VALID_VERIFIER_MODES))}",
            )

        # Check if verifier is available for the requested mode
        if self._has_verifier is None or not self._has_verifier.get(
            verifier_mode, False
        ):
            return AWMObservation(
                done=False,
                reward=self._get_reward("no_verifier"),
                reward_type="no_verifier",
                scenario=self._scenario,
                task=self._task,
                task_idx=self._task_idx,
                error=f"No {verifier_mode} verifier available for this task",
            )

        verifier_entry = self._data_loader.get_verifier(
            self._scenario, self._task_idx, verifier_mode
        )

        if verifier_entry is None:
            return AWMObservation(
                done=False,
                reward=self._get_reward("no_verifier"),
                reward_type="no_verifier",
                scenario=self._scenario,
                task=self._task,
                task_idx=self._task_idx,
            )

        reward_type, verify_result = run_verifier(
            verifier_entry=verifier_entry,
            verifier_mode=verifier_mode,
            initial_db_path=self._initial_db_path,
            final_db_path=self._db_path,
            final_answer=final_answer,
        )

        # For SQL mode, run LLM judge
        if verifier_mode == "sql" and reward_type != "judge_error":
            raw_response_str = verifier_entry.get("verification", {}).get(
                "raw_response", "{}"
            )
            try:
                raw_response = json.loads(raw_response_str)
            except (json.JSONDecodeError, TypeError):
                raw_response = {}

            try:
                reward_type, judge_result = _run_async_oneshot(
                    run_llm_judge(
                        task=self._task,
                        verifier_result=verify_result,
                        llm_base_url=self._llm_base_url,
                        llm_api_key=self._llm_api_key,
                        llm_model=self._llm_model,
                        trajectory=self._trajectory,
                        verifier_reasoning=raw_response.get("reasoning", ""),
                        success_criteria=raw_response.get("success_criteria", ""),
                        failure_criteria=raw_response.get("failure_criteria", ""),
                    )
                )
                verify_result["llm_judge"] = judge_result
            except Exception as e:
                logger.error(f"LLM judge failed: {e}")
                reward_type = "judge_error"
                verify_result["llm_judge_error"] = str(e)

        self._trajectory.append(
            {
                "action": "verify",
                "arguments": args,
                "success": True,
                "reward_type": reward_type,
                "reward": self._get_reward(reward_type),
                "verify_result": verify_result,
            }
        )

        return AWMObservation(
            done=False,
            reward=self._get_reward(reward_type),
            reward_type=reward_type,
            verify_result=verify_result,
            scenario=self._scenario,
            task=self._task,
            task_idx=self._task_idx,
            steps_taken=self._state.step_count,
        )

    def _handle_done(self, action: CallToolAction) -> AWMObservation:
        """Handle the `done` tool — end episode and destroy environment (no verification).

        Accepts optional arguments:
            keep_session (bool): If True, keep the session tmp folder for debugging.
                                 Default False (folder is deleted on cleanup).
        """
        if not self._reset_ok or self._scenario is None:
            self._episode_done = True
            return AWMObservation(
                done=True,
                reward=self._get_reward("server_error"),
                reward_type="server_error",
                error="Cannot call done: environment not initialized "
                "(reset failed or not called)",
            )

        args = action.arguments or {}
        keep_session = bool(args.get("keep_session", False))

        # Save trajectory to JSON before stopping.
        # Capture locals for race safety — cleanup thread may null
        # self._session_dir between the check and the write.
        session_dir = self._session_dir
        trajectory = self._trajectory
        trajectory_path = None
        if session_dir and trajectory and os.path.isdir(session_dir):
            trajectory_path = f"{session_dir}/trajectory.json"
            try:
                with open(trajectory_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "scenario": self._scenario,
                            "task": self._task,
                            "task_idx": self._task_idx,
                            "steps": self._state.step_count,
                            "trajectory": trajectory,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                logger.info(f"[AWM done] trajectory saved: {trajectory_path}")
            except OSError:
                # Session dir may have been cleaned up concurrently
                trajectory_path = None
            except Exception as e:
                logger.warning(f"Failed to save trajectory: {e}")
                trajectory_path = None

        self._episode_done = True
        self._process.stop()

        if keep_session and self._session_dir:
            logger.info(f"[AWM done] keeping session dir: {self._session_dir}")
            self._keep_session = True
        else:
            self._keep_session = False

        return AWMObservation(
            done=True,
            reward=0.0,  # done itself doesn't give reward
            reward_type="episode_done",
            scenario=self._scenario,
            task=self._task,
            task_idx=self._task_idx,
            steps_taken=self._state.step_count,
            trajectory_path=trajectory_path,
            session_dir=self._session_dir if keep_session else None,
        )

    def _handle_list_scenarios(self) -> AWMObservation:
        """Handle the `__list_scenarios__` tool — return all scenario info."""
        try:
            all_scenarios = self._data_loader.list_scenarios()
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="tool_call_ok",
                scenarios=all_scenarios,
                total=len(all_scenarios),
            )
        except Exception as e:
            return AWMObservation(
                done=False,
                reward=None,
                reward_type="server_error",
                error=f"Failed to list scenarios: {e}",
            )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._cleanup_session()

    def _cleanup_session(self) -> None:
        """Stop subprocess and clean up session temp files.

        If ``_keep_session`` is True (set by ``done(keep_session=True)``),
        the session directory is preserved for manual inspection.
        """
        if self._registry_id:
            _registry.unregister(self._registry_id)
            self._registry_id = None
        self._process.stop()
        if self._session_dir:
            if getattr(self, "_keep_session", False):
                logger.info(f"Keeping session dir: {self._session_dir}")
            else:
                cleanup_session_dir(self._session_dir)
            self._session_dir = None
        self._db_path = None
        self._initial_db_path = None
        self._tools_cache = None
        self._trajectory = []
        self._reset_ok = False
        self._episode_done = False
        self._has_verifier = None
        self._keep_session = False
