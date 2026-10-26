"""Shared helpers for BrowserGym harness inference and evaluation examples."""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from browsergym_env import BrowserGymEnv
from browsergym_env.harness import (
    BrowserGymSessionFactory,
    build_browsergym_action_tool_call,
)
from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider
from openenv.core.env_server.mcp_types import JsonRpcErrorCode, JsonRpcResponse
from openenv.core.harness import (
    _resolve_env_reward,
    CLIHarnessAdapter,
    HarnessRolloutResult,
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStepResult,
    SessionMCPBridge,
    VerifyResult,
)
from openenv.core.llm_client import LLMResponse

DEFAULT_BROWSERGYM_IMAGE = "browsergym-env:latest"
DEFAULT_BENCHMARK = "miniwob"
DEFAULT_TASK_NAME = "click-test"
DEFAULT_MAX_STEPS = 8
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 200
FALLBACK_ACTION = "noop()"

SYSTEM_PROMPT = """You control a web browser through BrowserGym.

Reply with exactly one BrowserGym action string, such as:
- click('13')
- fill('42', 'hello world')
- send_keys('Enter')
- scroll('down')
- noop()

Use only one action. Do not explain your reasoning.
If you are unsure, reply with noop().
"""

ACTION_PREFIX_RE = re.compile(r"^(action|next action)\s*[:-]\s*", re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r"(click|fill|send_keys|scroll|noop)\s*\(.*?\)",
    re.IGNORECASE | re.DOTALL,
)


def _serialize_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def extract_browsergym_action(
    response_text: str,
    fallback_action: str = FALLBACK_ACTION,
) -> str:
    """Extract a BrowserGym action from free-form model output."""

    if not response_text:
        return fallback_action

    for raw_line in response_text.splitlines():
        line = raw_line.strip().strip("`")
        if not line:
            continue
        line = ACTION_PREFIX_RE.sub("", line)
        match = ACTION_PATTERN.search(line)
        if match:
            action = re.sub(r"\s+", " ", match.group(0).strip())
            open_paren = action.find("(")
            if open_paren > 0:
                action = f"{action[:open_paren].lower()}{action[open_paren:]}"
            return action.rstrip(";")

    match = ACTION_PATTERN.search(response_text)
    if match:
        action = re.sub(r"\s+", " ", match.group(0).strip())
        open_paren = action.find("(")
        if open_paren > 0:
            action = f"{action[:open_paren].lower()}{action[open_paren:]}"
        return action.rstrip(";")

    return fallback_action


def build_browsergym_tool_call_from_output(response_text: str):
    """Convert free-form model output into a BrowserGym tool call."""

    action = extract_browsergym_action(response_text)
    try:
        return build_browsergym_action_tool_call(action)
    except ValueError:
        return build_browsergym_action_tool_call(FALLBACK_ACTION)


def build_browsergym_chat_messages(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Convert harness messages into a plain chat-completions transcript."""

    chat_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""

        if role == "tool":
            tool_name = message.get("name", "tool")
            chat_messages.append(
                {
                    "role": "user",
                    "content": f"Observation from {tool_name}:\n{content}",
                }
            )
            continue

        if role not in {"user", "assistant", "system"}:
            role = "user"
        chat_messages.append({"role": role, "content": str(content)})

    return chat_messages


def build_openai_model_step(
    client: OpenAI,
    model: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """Build a white-box model step for BrowserGym inference/evaluation."""

    def model_step(messages, tools, sampling):
        del tools
        completion = client.chat.completions.create(
            model=model,
            messages=build_browsergym_chat_messages(
                messages,
                system_prompt=system_prompt,
            ),
            temperature=sampling.get("temperature", temperature),
            max_tokens=sampling.get("max_tokens", max_tokens),
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        tool_call = build_browsergym_tool_call_from_output(response_text)
        return ModelStepResult(
            response=LLMResponse(content=response_text, tool_calls=[tool_call]),
        )

    return model_step


def create_openai_client(
    *,
    api_base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAI:
    """Create a sync OpenAI-compatible client for the examples."""

    client_kwargs: dict[str, Any] = {}
    if api_base_url:
        client_kwargs["base_url"] = api_base_url

    resolved_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if resolved_key:
        client_kwargs["api_key"] = resolved_key

    return OpenAI(**client_kwargs)


@dataclass
class BrowserGymRuntime(AbstractContextManager["BrowserGymRuntime"]):
    """Runtime information for a BrowserGym server used by the examples."""

    base_url: str
    provider: LocalDockerProvider | None = None

    def close(self) -> None:
        if self.provider is not None:
            self.provider.stop_container()
            self.provider = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def start_browsergym_runtime(
    *,
    base_url: str | None = None,
    image: str = DEFAULT_BROWSERGYM_IMAGE,
    benchmark: str = DEFAULT_BENCHMARK,
    task_name: str = DEFAULT_TASK_NAME,
) -> BrowserGymRuntime:
    """Start or attach to a BrowserGym server for evaluation examples."""

    if base_url is not None:
        return BrowserGymRuntime(base_url=base_url)

    provider = LocalDockerProvider()
    env_vars = {
        "BROWSERGYM_BENCHMARK": benchmark,
        "BROWSERGYM_TASK_NAME": task_name,
    }
    runtime_base_url = provider.start_container(image=image, env_vars=env_vars)
    provider.wait_for_ready(runtime_base_url)
    return BrowserGymRuntime(base_url=runtime_base_url, provider=provider)


def build_browsergym_session_factory(
    *,
    base_url: str,
    task_name: str = DEFAULT_TASK_NAME,
) -> BrowserGymSessionFactory:
    """Create a BrowserGym session factory for a running environment server."""

    return BrowserGymSessionFactory(
        client_factory=lambda: BrowserGymEnv(base_url=base_url),
        default_task=task_name,
    )


def _rollout_final_state(rollout: HarnessRolloutResult) -> dict[str, Any]:
    return {
        "done": rollout.done,
        "metrics": dict(rollout.metrics),
        "events": [
            {
                "type": event.type,
                "payload": dict(event.payload),
            }
            for event in rollout.events
        ],
        "tool_trace": [
            {
                "tool_name": entry.tool_name,
                "arguments": dict(entry.arguments),
                "result": {
                    "data": entry.result.data,
                    "done": entry.result.done,
                    "metadata": dict(entry.result.metadata),
                    "error": entry.result.error,
                },
            }
            for entry in rollout.tool_trace
        ],
    }


@dataclass
class EpisodeEvaluation:
    """Normalized result from one inference/evaluation episode."""

    rollout: HarnessRolloutResult
    verify: VerifyResult
    reward: float
    done: bool
    success: bool
    step_count: int


def _finalize_episode(
    session,
    rollout: HarnessRolloutResult,
) -> EpisodeEvaluation:
    verify = session.verify(
        transcript=rollout.messages,
        final_state=_rollout_final_state(rollout),
    )
    reward = _resolve_env_reward(rollout, verify)
    done = bool(verify.done or rollout.done)
    step_count = int(verify.metrics.get("step_count", len(rollout.tool_trace)))
    success = reward > 0.0
    return EpisodeEvaluation(
        rollout=rollout,
        verify=verify,
        reward=reward,
        done=done,
        success=success,
        step_count=step_count,
    )


def run_white_box_episode(
    *,
    session_factory: BrowserGymSessionFactory,
    model_step,
    limits: HarnessRunLimits,
    task: Any = None,
    seed: int | None = None,
    episode_id: str | None = None,
) -> EpisodeEvaluation:
    """Run one BrowserGym episode through the white-box harness path."""

    session = session_factory.create(task=task, seed=seed, episode_id=episode_id)
    try:
        rollout = MCPHarnessAdapter().run_white_box(
            model_step=model_step,
            session=session,
            limits=limits,
        )
        return _finalize_episode(session, rollout)
    finally:
        session.close()


def run_black_box_episode(
    *,
    session_factory: BrowserGymSessionFactory,
    harness_adapter: CLIHarnessAdapter,
    limits: HarnessRunLimits,
    task: Any = None,
    seed: int | None = None,
    episode_id: str | None = None,
) -> EpisodeEvaluation:
    """Run one BrowserGym episode through the black-box harness path."""

    session = session_factory.create(task=task, seed=seed, episode_id=episode_id)
    try:
        rollout = harness_adapter.run_black_box(
            session=session,
            limits=limits,
        )
        return _finalize_episode(session, rollout)
    finally:
        session.close()


def summarize_episodes(episodes: list[EpisodeEvaluation]) -> dict[str, float]:
    """Aggregate reward and completion metrics across episodes."""

    if not episodes:
        return {
            "episodes": 0.0,
            "avg_reward": 0.0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
        }

    episode_count = len(episodes)
    total_reward = sum(episode.reward for episode in episodes)
    total_success = sum(1 for episode in episodes if episode.success)
    total_steps = sum(episode.step_count for episode in episodes)
    return {
        "episodes": float(episode_count),
        "avg_reward": total_reward / episode_count,
        "success_rate": total_success / episode_count,
        "avg_steps": total_steps / episode_count,
    }


def format_episode_summary(index: int, episode: EpisodeEvaluation) -> str:
    """Render one episode summary line for CLI output."""

    return (
        f"Episode {index}: reward={episode.reward:.2f} "
        f"success={episode.success} done={episode.done} steps={episode.step_count}"
    )


def _normalize_mcp_call_result(result: dict[str, Any]) -> dict[str, Any]:
    text_payload = result.get("error") or _serialize_json(result.get("data"))
    return {
        "content": [{"type": "text", "text": text_payload}],
        "structuredContent": {
            "data": result.get("data"),
            "done": result.get("done", False),
            "metadata": result.get("metadata", {}),
            "error": result.get("error"),
        },
        "data": result.get("data"),
        "isError": bool(result.get("error")),
    }


class SessionMCPHttpServer(AbstractContextManager["SessionMCPHttpServer"]):
    """Tiny HTTP wrapper that exposes SessionMCPBridge on /mcp."""

    def __init__(
        self,
        bridge: SessionMCPBridge,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self._bridge = bridge
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("HTTP bridge is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/mcp"

    def start(self) -> "SessionMCPHttpServer":
        if self._server is not None:
            return self

        bridge = self._bridge

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/mcp":
                    self.send_error(404)
                    return
                self._send_json({"status": "ok", "transport": "jsonrpc"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/mcp":
                    self.send_error(404)
                    return

                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                try:
                    request = json.loads(raw_body or b"{}")
                except json.JSONDecodeError:
                    self._send_json(
                        JsonRpcResponse.error_response(
                            JsonRpcErrorCode.PARSE_ERROR,
                            message="Parse error",
                        ).model_dump()
                    )
                    return

                try:
                    response = bridge.handle_request(request)
                except KeyError as exc:
                    response = JsonRpcResponse.error_response(
                        JsonRpcErrorCode.METHOD_NOT_FOUND,
                        message=str(exc.args[0]) if exc.args else "Method not found",
                        request_id=request.get("id"),
                    ).model_dump()
                except ValueError as exc:
                    response = JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INVALID_PARAMS,
                        message=str(exc),
                        request_id=request.get("id"),
                    ).model_dump()
                except Exception as exc:
                    response = JsonRpcResponse.error_response(
                        JsonRpcErrorCode.INTERNAL_ERROR,
                        message=str(exc),
                        request_id=request.get("id"),
                    ).model_dump()

                if (
                    request.get("method") == "tools/call"
                    and "result" in response
                    and isinstance(response["result"], dict)
                ):
                    response["result"] = _normalize_mcp_call_result(response["result"])

                self._send_json(response)

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="browsergym-session-mcp-http",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = [
    "BrowserGymRuntime",
    "DEFAULT_BENCHMARK",
    "DEFAULT_BROWSERGYM_IMAGE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MODEL",
    "DEFAULT_TASK_NAME",
    "EpisodeEvaluation",
    "FALLBACK_ACTION",
    "SYSTEM_PROMPT",
    "SessionMCPHttpServer",
    "build_browsergym_chat_messages",
    "build_browsergym_session_factory",
    "build_browsergym_tool_call_from_output",
    "build_openai_model_step",
    "create_openai_client",
    "extract_browsergym_action",
    "format_episode_summary",
    "run_white_box_episode",
    "run_black_box_episode",
    "start_browsergym_runtime",
    "summarize_episodes",
]
