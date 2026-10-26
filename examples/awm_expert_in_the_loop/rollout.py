"""The expert-in-the-loop rollout: one AWM task from reset to verify.

``run_task`` drives a single agent episode against a running AWM server using the
OpenEnv step API (``env.step(ListToolsAction()/CallToolAction(...))``). The agent
emits XML ``<tool_call>`` actions; ``list_tools`` and ``call_tool`` are forwarded to
the environment, while ``ask_expert`` is handled in the client by the optional
[`~expert.VerifierInformedExpert`] and never reaches the server.

The loop is shared by the benchmark, the basic runners, and any GRPO training
harness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from expert import VerifierInformedExpert
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from policies import AgentPolicy
from prompts import ADAPTIVE_PROMPT, BASELINE_PROMPT, NUDGE_ON_ERROR, NUDGE_ON_STALL

CONTENT_FILTER_MARKER = "content_filter"
RESET_ERROR = "reset_error"
SERVER_ERROR = "server_error"
_DEFAULT_MAX_TOOL_RESPONSE_CHARS = 3000
_OK_RESET = frozenset({"reset_ok", "reset_warning"})


@dataclass
class TaskResult:
    """Outcome of a single task episode.

    Attributes:
        scenario (`str`):
            Scenario name.
        task_idx (`int`):
            Task index within the scenario.
        task (`str`):
            Truncated task description.
        steps (`int`):
            Number of agent tool steps taken.
        errors (`int`):
            Number of tool/LLM errors encountered.
        expert_calls (`int`):
            Number of times the agent consulted the expert.
        reward_type (`str` or `None`):
            Environment reward label (`"complete"` on success, `CONTENT_FILTER_MARKER`
            when the run was dropped for a content filter).
        reward (`float` or `None`):
            Final scalar reward from the environment verifier.
        filtered (`bool`):
            Whether the run hit a content filter and was excluded.
    """

    scenario: str
    task_idx: int
    task: str
    steps: int
    errors: int
    expert_calls: int
    reward_type: str | None
    reward: float | None
    filtered: bool = False


def parse_tool_call(content: str) -> dict | None:
    """Extract the first ``<tool_call>`` JSON object from a model response."""
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def safe_parse_arguments(arguments: object) -> dict:
    """Coerce tool arguments (which a model may emit as a JSON string) to a dict."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def format_tools(tools, verbose: bool = False) -> str:
    """Render a tool list as text; ``verbose`` adds longer descriptions for the expert."""
    lines = [f"Available MCP tools ({len(tools)}):", "=" * 60]
    desc_len = 200 if verbose else 120
    for i, tool in enumerate(tools, 1):
        lines.append(f"{i}. {tool.name}")
        lines.append(f"   Description: {(tool.description or '')[:desc_len]}")
        schema = tool.input_schema or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if props:
            lines.append("   Parameters:")
            for name, info in props.items():
                req = " (required)" if name in required else ""
                desc = info.get("description", "")
                desc_part = f" - {desc[:80]}" if desc and verbose else ""
                lines.append(
                    f"     - {name}: {info.get('type', 'any')}{req}{desc_part}"
                )
        lines.append("")
    return "\n".join(lines)


def is_content_filter_error(exc: Exception) -> bool:
    """Heuristically detect content-filter rejections from the model provider."""
    text = str(exc).lower()
    return "content_filter" in text or "content management policy" in text


def _coerce_inner_args(value: object) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def has_tool_call_tags(content: str) -> bool:
    """Return `True` when a response attempted a tool call but failed parsing."""
    return "<tool_call" in content or "</tool_call>" in content


async def run_task(
    env,
    agent_policy: AgentPolicy,
    *,
    scenario: str,
    task_idx: int,
    expert: VerifierInformedExpert | None = None,
    max_iters: int = 15,
    stall_threshold: int = 5,
    max_tool_response_chars: int = _DEFAULT_MAX_TOOL_RESPONSE_CHARS,
    verbose: bool = False,
) -> TaskResult:
    """Run one AWM task and return its [`TaskResult`].

    Args:
        env:
            A connected [`~agent_world_model_env.AWMEnv`] client.
        agent_policy ([`~policies.AgentPolicy`]):
            Policy producing the agent's turns.
        scenario (`str`):
            Scenario to load.
        task_idx (`int`):
            Task index within the scenario.
        expert ([`~expert.VerifierInformedExpert`], *optional*):
            When provided, the agent may call ``ask_expert`` and is nudged toward it
            after errors or stalls. When `None`, the no-expert baseline runs.
        max_iters (`int`, *optional*, defaults to `15`):
            Maximum agent turns before the episode is verified.
        stall_threshold (`int`, *optional*, defaults to `5`):
            Steps without an expert call after which a stall nudge is injected.
        verbose (`bool`, *optional*, defaults to `False`):
            Print per-step progress.

    Returns:
        [`TaskResult`]: The episode outcome including final reward.
    """
    expert_available = expert is not None

    try:
        reset = await env.reset(scenario=scenario, task_idx=task_idx)
    except Exception as exc:
        if verbose:
            print(f"    reset failed: {str(exc)[:150]}")
        return TaskResult(scenario, task_idx, "", 0, 1, 0, RESET_ERROR, None)

    reset_type = getattr(reset.observation, "reward_type", None)
    task = reset.observation.task or ""
    if reset_type not in _OK_RESET or not task:
        if verbose:
            print(f"    unusable reset: reward_type={reset_type} task={task[:40]!r}")
        await _safe_done(env)
        return TaskResult(
            scenario, task_idx, task[:80], 0, 1, 0, reset_type or RESET_ERROR, None
        )

    steps = errors = expert_calls = 0
    final_answer = ""

    try:
        success_criteria = ""
        if expert_available:
            success_criteria = await expert.analyze(task, scenario, task_idx)
            if verbose and success_criteria:
                print(f"    verifier criteria: {success_criteria[:120]}...")

        system_prompt = ADAPTIVE_PROMPT if expert_available else BASELINE_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        cached_tool_schemas = ""

        for step in range(1, max_iters + 1):
            try:
                content = await agent_policy.complete(messages)
            except Exception as exc:
                if is_content_filter_error(exc):
                    return TaskResult(
                        scenario,
                        task_idx,
                        task[:80],
                        steps,
                        errors + 1,
                        expert_calls,
                        CONTENT_FILTER_MARKER,
                        None,
                        True,
                    )
                if verbose:
                    print(f"    step {step}: agent error {str(exc)[:150]}")
                errors += 1
                break

            messages.append({"role": "assistant", "content": content})

            call = parse_tool_call(content)
            if not call:
                if has_tool_call_tags(content):
                    steps += 1
                    errors += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool response:\nError: malformed <tool_call>. "
                                "Return exactly one valid JSON object inside the tags. "
                                'Example: <tool_call>{"name": "list_tools", '
                                '"arguments": {}}</tool_call>'
                            ),
                        }
                    )
                    continue
                # A plain (non-tool-call) message is the agent's final answer.
                final_answer = content
                break

            name = call["name"]
            steps += 1
            last_was_error = False

            if name == "list_tools":
                result = await env.step(ListToolsAction())
                tools = result.observation.tools
                cached_tool_schemas = format_tools(tools, verbose=True)
                tool_response = format_tools(tools)
                if verbose:
                    print(f"    step {step}: list_tools -> {len(tools)} tools")

            elif name == "ask_expert" and expert_available:
                args = safe_parse_arguments(call.get("arguments"))
                tool_response = await expert.advise(
                    args.get("task", task),
                    tool_schemas=cached_tool_schemas,
                    context=args.get("context", ""),
                    success_criteria=success_criteria,
                )
                expert_calls += 1
                if verbose:
                    print(f"    step {step}: ask_expert -> plan received")

            elif name == "call_tool":
                args = safe_parse_arguments(call.get("arguments"))
                tool_name = args.get("tool_name", "")
                inner_args = _coerce_inner_args(args.get("arguments", "{}"))
                result = await env.step(
                    CallToolAction(tool_name=tool_name, arguments=inner_args)
                )
                obs = result.observation
                if getattr(obs, "tool_result", None) is not None:
                    tool_response = (
                        obs.tool_result
                        if isinstance(obs.tool_result, str)
                        else json.dumps(obs.tool_result, ensure_ascii=False)
                    )
                elif getattr(obs, "error", None):
                    tool_response = f"Error: {obs.error}"
                    errors += 1
                    last_was_error = True
                else:
                    tool_response = json.dumps(obs.model_dump(), ensure_ascii=False)
                if verbose:
                    status = "ERR" if last_was_error else "OK"
                    print(f"    step {step}: call_tool({tool_name}) [{status}]")

            else:
                tool_response = "Error: unknown function. Use list_tools, call_tool" + (
                    ", or ask_expert." if expert_available else "."
                )
                errors += 1
                last_was_error = True

            if len(tool_response) > max_tool_response_chars:
                tool_response = (
                    tool_response[:max_tool_response_chars] + "... (truncated)"
                )

            nudge = ""
            if expert_available and name != "ask_expert":
                if last_was_error:
                    nudge = "\n\n" + NUDGE_ON_ERROR
                elif steps >= stall_threshold and expert_calls == 0:
                    nudge = "\n\n" + NUDGE_ON_STALL.format(steps=steps)

            messages.append(
                {"role": "user", "content": f"Tool response:\n{tool_response}{nudge}"}
            )

        verify = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": final_answer},
            )
        )
        reward_type = verify.observation.reward_type
        reward = verify.reward
        if verbose and reward_type != "complete":
            print(f"    verify: reward_type={reward_type} reward={reward}")

        return TaskResult(
            scenario,
            task_idx,
            task[:80],
            steps,
            errors,
            expert_calls,
            reward_type,
            reward,
        )
    except Exception as exc:
        if verbose:
            print(f"    episode error: {type(exc).__name__}: {str(exc)[:150]}")
        return TaskResult(
            scenario,
            task_idx,
            task[:80],
            steps,
            errors + 1,
            expert_calls,
            SERVER_ERROR,
            None,
        )
    finally:
        await _safe_done(env)


async def _safe_done(env) -> None:
    try:
        await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": False})
        )
    except Exception:
        pass
