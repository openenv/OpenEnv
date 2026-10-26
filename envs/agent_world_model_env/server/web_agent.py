"""LLM-driven agent loop for the AWM web UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from .prompts import DEFAULT_SYSTEM_PROMPT


def parse_tool_call(content: str) -> dict | None:
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def format_tools(tools: list[dict]) -> str:
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        name = t.get("name") or t.get("tool_name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema") or t.get("inputSchema") or {}
        lines.append(f"{i}. {name}")
        lines.append(f"   Description: {desc}")
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req} — "
                    f"{pinfo.get('description', '')}"
                )
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)


def _truncate(s: str, limit: int = 2000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, full length {len(s)} chars)"


@dataclass
class AgentEvent:
    kind: str  # "info" | "llm_response" | "tool_call" | "tool_result" | "verify" | "done" | "error"
    text: str = ""
    payload: dict = field(default_factory=dict)


class AwmAgent:
    def __init__(
        self,
        web_manager: Any,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 10,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ):
        self._web = web_manager
        self._client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._model = llm_model
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    async def _list_tools_dict(self) -> list[dict]:
        result = await self._web.step_environment({"type": "list_tools"})
        obs = result.get("observation", {}) or {}
        tools = obs.get("tools", []) or []
        out = []
        for t in tools:
            if isinstance(t, dict):
                out.append(t)
            else:
                out.append(
                    {
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", ""),
                        "input_schema": getattr(t, "input_schema", {}),
                    }
                )
        return out

    async def _call_tool(self, tool_name: str, args: dict) -> dict:
        return await self._web.step_environment(
            {"type": "call_tool", "tool_name": tool_name, "arguments": args}
        )

    async def run(
        self,
        task: str,
        verifier_mode: str | None = None,
        final_answer_fallback: str = "",
        auto_verify: bool = True,
        auto_done: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent on the already-reset env, yielding events as it goes."""
        try:
            tools = await self._list_tools_dict()
        except Exception as e:
            yield AgentEvent(kind="error", text=f"list_tools failed: {e}")
            return

        yield AgentEvent(
            kind="info",
            text=f"Discovered {len(tools)} tools.",
            payload={"tool_names": [t.get("name") for t in tools]},
        )
        tools_text = format_tools(tools)

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
            {
                "role": "user",
                "content": f"Available tools:\n{tools_text}",
            },
        ]

        last_assistant_content = ""

        for step in range(1, self._max_iterations + 1):
            if self._stop_requested:
                yield AgentEvent(kind="info", text="Stopped by user.")
                return

            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_completion_tokens=self._max_tokens,
                )
            except Exception as e:
                yield AgentEvent(
                    kind="error", text=f"LLM call failed at step {step}: {e}"
                )
                return

            content = resp.choices[0].message.content or ""
            last_assistant_content = content
            messages.append({"role": "assistant", "content": content})

            yield AgentEvent(
                kind="llm_response",
                text=content,
                payload={"step": step},
            )

            tc = parse_tool_call(content)
            if tc is None:
                yield AgentEvent(
                    kind="info",
                    text=f"No <tool_call> in step {step}; treating as final answer.",
                )
                break

            name = tc.get("name", "")
            arguments = tc.get("arguments") or {}

            yield AgentEvent(
                kind="tool_call",
                text=f"{name} {json.dumps(arguments, ensure_ascii=False)[:300]}",
                payload={"name": name, "arguments": arguments, "step": step},
            )

            tool_response = ""
            try:
                if name == "list_tools":
                    result = await self._web.step_environment({"type": "list_tools"})
                    obs = result.get("observation", {}) or {}
                    tools = await self._list_tools_dict()
                    tool_response = format_tools(tools)
                elif name == "call_tool":
                    inner_name = arguments.get("tool_name", "")
                    inner_args = arguments.get("arguments", "{}")
                    if isinstance(inner_args, str):
                        try:
                            inner_args = json.loads(inner_args)
                        except json.JSONDecodeError:
                            inner_args = {}
                    if not isinstance(inner_args, dict):
                        inner_args = {}
                    result = await self._call_tool(inner_name, inner_args)
                    obs = result.get("observation", {}) or {}
                    if obs.get("tool_result") is not None:
                        tr = obs["tool_result"]
                        tool_response = (
                            json.dumps(tr, ensure_ascii=False)
                            if not isinstance(tr, str)
                            else tr
                        )
                    elif obs.get("error"):
                        tool_response = f"Error: {obs['error']}"
                    else:
                        tool_response = json.dumps(obs, ensure_ascii=False)
                else:
                    tool_response = (
                        f"Error: Unknown function '{name}'. "
                        "Use 'list_tools' or 'call_tool'."
                    )
            except Exception as e:
                tool_response = f"Error during tool dispatch: {e}"

            yield AgentEvent(
                kind="tool_result",
                text=_truncate(tool_response),
                payload={"step": step},
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool response:\n{_truncate(tool_response, 6000)}",
                }
            )
        else:
            yield AgentEvent(
                kind="info",
                text=f"Max iterations ({self._max_iterations}) reached.",
            )

        if auto_verify and verifier_mode:
            verify_args: dict = {"verifier_mode": verifier_mode}
            if last_assistant_content:
                verify_args["final_answer"] = last_assistant_content
            elif final_answer_fallback:
                verify_args["final_answer"] = final_answer_fallback
            try:
                verify_result = await self._call_tool("verify", verify_args)
                obs = verify_result.get("observation", {}) or {}
                yield AgentEvent(
                    kind="verify",
                    text=(
                        f"reward_type={obs.get('reward_type')}  "
                        f"reward={verify_result.get('reward')}"
                    ),
                    payload={
                        "reward_type": obs.get("reward_type"),
                        "reward": verify_result.get("reward"),
                        "verify_result": obs.get("verify_result"),
                    },
                )
            except Exception as e:
                yield AgentEvent(kind="error", text=f"verify failed: {e}")

        if auto_done:
            try:
                done_result = await self._call_tool("done", {"keep_session": True})
                obs = done_result.get("observation", {}) or {}
                yield AgentEvent(
                    kind="done",
                    text=(
                        f"Episode done. trajectory_path="
                        f"{obs.get('trajectory_path') or '(none)'}"
                    ),
                    payload={
                        "trajectory_path": obs.get("trajectory_path"),
                        "session_dir": obs.get("session_dir"),
                    },
                )
            except Exception as e:
                yield AgentEvent(kind="error", text=f"done failed: {e}")
