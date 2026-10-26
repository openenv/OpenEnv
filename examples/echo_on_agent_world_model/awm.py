"""Adapt **Agent World Model (AWM)** episodes into ECHO role-masked trajectories.

``envs/agent_world_model_env`` (AgentWorldModel-1K, upstream) is a suite of **1,000
MCP tool-use environments / 10,000 tasks**. Each step returns an ``AWMObservation``
whose fields line up almost exactly with what ECHO needs per token:

    AWMObservation field   ->  ECHO role
    ─────────────────────      ─────────────────────────────────────────────
    task / scenario        ->  CONTEXT     (given; never a loss target)
    (the agent's tool call)->  ACTION      (GRPO / policy-gradient target)
    tool_result / error    ->  ENV_OUTPUT  (real world output — the ECHO target)
    verify_result          ->  ENV_OUTPUT  (real grader output)
    warning                ->  WARNING     (harness boilerplate — excluded by default)

That last row is the non-obvious win: AWM *already* separates real environment
output (``tool_result``/``error``) from harness ``warning`` text — precisely the
distinction ECHO's reference code carries via ``completion_warning_masks``. So an
AWM **tool-use** rollout maps directly onto ECHO's role masks; this module
serializes that common ``CallToolAction`` + ``verify`` subset. (List-tools
observations and ``done`` metadata are out of scope for this reference example.)

Two entry points:

* :func:`awm_episode_to_trajectory` — pure, offline: dict episode -> ``Trajectory``.
* :func:`live_capture` — opt-in: replay an episode's tool calls against a *running*
  ``agent_world_model_env`` server and build the trajectory from **real** observations.
"""

from __future__ import annotations

import json
from typing import Any

from echo import ACTION, CONTEXT, ENV_OUTPUT, Segment, Trajectory, WARNING

DEFAULT_SYSTEM = (
    "You are a tool-using agent operating inside an Agent World Model environment. "
    "Call tools to satisfy the task. Each tool returns an observation; read it and "
    "decide the next call. Finish by calling `verify`."
)


def _dumps(obj: Any) -> str:
    """Stable, compact-ish JSON so token roles are deterministic across runs."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(", ", ": "))


def serialize_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """An agent action, in a generic Hermes-style tool-call envelope (ACTION tokens)."""
    return f"<tool_call>\n{_dumps({'name': tool_name, 'arguments': arguments})}\n</tool_call>\n"


def serialize_env_output(payload: Any) -> str:
    """Real environment output, in a tool-response envelope (ENV_OUTPUT tokens)."""
    body = payload if isinstance(payload, str) else _dumps(payload)
    return f"<tool_response>\n{body}\n</tool_response>\n"


def _env_payload(obs: dict[str, Any]) -> Any:
    """Pick the real-world portion of an AWM observation, in priority order.

    ``tool_result`` is the common case; ``verify_result`` is the grader's output on
    the verify step (we fold ``reward_type`` in for realism); ``error`` is a real
    environment failure message. All three are genuine env output -> ENV_OUTPUT.
    """
    if obs.get("tool_result") is not None:
        return obs["tool_result"]
    if obs.get("verify_result") is not None:
        out = {"verify_result": obs["verify_result"]}
        if obs.get("reward_type") is not None:
            out["reward_type"] = obs["reward_type"]
        return out
    if obs.get("error") is not None:
        payload = {"error": obs["error"]}
        if obs.get("reward_type") is not None:
            payload["reward_type"] = obs["reward_type"]
        return payload
    return None


def awm_episode_to_trajectory(
    episode: dict[str, Any],
    *,
    system_preamble: str = DEFAULT_SYSTEM,
    include_tool_list: bool = True,
) -> Trajectory:
    """Turn an AWM episode dict into a role-segmented :class:`Trajectory`.

    Expected episode shape (matches :func:`live_capture` output and the bundled
    fixture, which is a real ``e_commerce_33`` capture)::

        {
          "scenario": "e_commerce_33",
          "task": "Search for 'wireless noise cancelling headphones', sort by ...",
          "task_idx": 0,
          "tools": ["search_products", "get_product_by_id", "add_item_to_cart", ...],
          "steps": [
            {"action": {"tool_name": "search_products", "arguments": {...}},
             "observation": {"tool_name": "search_products", "tool_result": "{...}",
                             "warning": null, "error": null, ...}},
            ...
          ],
          "reward": 0.0
        }

    Segment order per step is ACTION -> [WARNING] -> ENV_OUTPUT, so the model is
    always *conditioned on its action* before predicting the world's response.
    """
    task = episode.get("task", "")
    context_text = system_preamble + f"\n\nTask: {task}\n"
    if include_tool_list and episode.get("tools"):
        context_text += "Available tools: " + ", ".join(episode["tools"]) + "\n"

    segments: list[Segment] = [Segment(CONTEXT, context_text)]

    for step in episode.get("steps", []):
        action = step.get("action", {})
        obs = step.get("observation", {}) or {}

        segments.append(
            Segment(
                ACTION,
                serialize_tool_call(
                    action.get("tool_name", "unknown"),
                    action.get("arguments", {}) or {},
                ),
            )
        )

        # harness boilerplate is its own role so it never leaks into the world loss
        if obs.get("warning"):
            segments.append(Segment(WARNING, f"<warning>{obs['warning']}</warning>\n"))

        payload = _env_payload(obs)
        if payload is not None:
            segments.append(Segment(ENV_OUTPUT, serialize_env_output(payload)))

    return Trajectory(
        segments=segments,
        reward=float(episode.get("reward", 0.0) or 0.0),
        task_prompt=task,
        meta={
            "scenario": episode.get("scenario"),
            "task_idx": episode.get("task_idx"),
            "num_steps": len(episode.get("steps", [])),
        },
    )


def live_capture(base_url: str, episode: dict[str, Any]) -> dict[str, Any]:
    """Replay ``episode``'s tool calls against a *running* AWM server, capturing
    the **real** observations. Returns a new episode dict (same shape) you can feed
    to :func:`awm_episode_to_trajectory`.

    This proves the adapter on genuine environment output without needing a trained
    policy — the scripted actions stand in for what a policy would choose. Requires
    a running ``agent_world_model_env`` server; imports are lazy on purpose.
    """
    import asyncio

    from agent_world_model_env import AWMEnv  # type: ignore
    from openenv.core.env_server.mcp_types import CallToolAction  # type: ignore

    async def _run() -> dict[str, Any]:
        captured: list[dict[str, Any]] = []
        last_reward = 0.0
        async with AWMEnv(base_url=base_url) as env:
            # take the *real* task / scenario / tools from the server, not the fixture
            reset_res = await env.reset(
                scenario=episode["scenario"], task_idx=episode.get("task_idx", 0)
            )
            ro = reset_res.observation
            ro_dump = ro.model_dump() if hasattr(ro, "model_dump") else {}
            try:
                tools = [t.name for t in await env.list_tools()]
            except Exception:
                tools = episode.get("tools", [])

            ends_with_done = bool(episode["steps"]) and (
                episode["steps"][-1]["action"].get("tool_name") == "done"
            )
            try:
                for step in episode["steps"]:
                    action = step["action"]
                    result = await env.step(
                        CallToolAction(
                            tool_name=action["tool_name"],
                            arguments=action.get("arguments", {}),
                        )
                    )
                    obs = result.observation
                    obs_dict = (
                        obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
                    )
                    captured.append({"action": action, "observation": obs_dict})
                    if getattr(result, "reward", None) is not None:
                        last_reward = float(result.reward)
            finally:
                # `done` destroys the env subprocess — always release it
                if not ends_with_done:
                    try:
                        await env.step(
                            CallToolAction(
                                tool_name="done", arguments={"keep_session": False}
                            )
                        )
                    except Exception:
                        pass

        return {
            "scenario": ro_dump.get("scenario") or episode.get("scenario"),
            "task": ro_dump.get("task") or episode.get("task", ""),
            "task_idx": ro_dump.get("task_idx", episode.get("task_idx", 0)),
            "tools": tools,
            "steps": captured,
            "reward": last_reward,
        }

    return asyncio.run(_run())
