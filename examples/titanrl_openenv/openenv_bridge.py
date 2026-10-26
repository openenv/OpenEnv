# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Framework-agnostic bridge from an OpenEnv environment to message space.

This module contains the OpenEnv side of the TorchTitan-RL (``TitanRL``)
integration and depends **only** on ``openenv`` — it does not import
``torchtitan``. That keeps the translation logic (observation -> text,
assistant tool-call / text -> OpenEnv action) importable and unit-testable
without a training stack, and lets the thin :class:`MessageEnv` wrapper in
``titanrl_env.py`` stay a few lines long.

The bridge drives any OpenEnv server through
:class:`openenv.core.generic_client.GenericEnvClient`, so it works with every
environment that speaks the OpenEnv HTTP/WebSocket protocol without installing
that environment's package locally.

Example (standalone, no TorchTitan needed)::

    import asyncio
    from examples.titanrl_openenv.openenv_bridge import OpenEnvBridge

    async def main():
        bridge = OpenEnvBridge(base_url="http://localhost:8010", action_mode="text")
        turn = await bridge.start(seed=0)
        print(turn.text)                       # initial observation
        turn = await bridge.act_from_text("hello")
        print(turn.text, turn.reward, turn.done)
        await bridge.stop()

    asyncio.run(main())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.generic_client import GenericEnvClient

# Default OpenAI-style tool schema (a ``renderers.ToolSpec``-compatible dict) used
# when the environment is driven in ``"tool"`` mode. The assistant calls
# ``openenv_act`` with an ``action`` object that becomes the OpenEnv action dict.
DEFAULT_ACT_TOOL: dict[str, Any] = {
    "name": "openenv_act",
    "description": (
        "Take one action in the connected OpenEnv environment. Pass the action "
        "as a JSON object whose fields match the environment's action schema."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "object",
                "description": "The action payload for the environment.",
            },
        },
        "required": ["action"],
    },
}

# Observation dict keys tried, in order, when rendering an observation to text.
DEFAULT_OBSERVATION_KEYS: tuple[str, ...] = (
    "text",
    "message",
    "prompt",
    "observation",
    "output",
    "content",
)


@dataclass
class BridgeTurn:
    """One environment turn, translated out of OpenEnv into message primitives.

    Attributes:
        text: Human-readable rendering of the observation, suitable for use as a
            message ``content``.
        done: Whether the episode has terminated.
        reward: Scalar reward for this turn, or ``None`` if the environment
            reported none.
        raw_observation: The untouched observation returned by OpenEnv (a dict for
            :class:`GenericEnvClient`), for callers that want structured access.
    """

    text: str
    done: bool = False
    reward: Optional[float] = None
    raw_observation: Any = field(default=None)

    def as_message(self, role: str) -> dict[str, str]:
        """Return this turn as a chat message ``{"role": role, "content": text}``."""
        return {"role": role, "content": self.text}


def render_observation(
    observation: Any,
    *,
    observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
) -> str:
    """Best-effort rendering of an OpenEnv observation into a text string.

    Prefers the first non-empty string among ``observation_keys``, then a message
    inside ``metadata``, and finally a JSON dump of the observation with the
    ``reward``/``done`` control fields stripped.
    """
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        for key in observation_keys:
            value = observation.get(key)
            if isinstance(value, str) and value:
                return value
        metadata = observation.get("metadata")
        if isinstance(metadata, dict):
            for key in ("message", "text", "status", "output"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    return value
        payload = {k: v for k, v in observation.items() if k not in ("reward", "done")}
        return json.dumps(payload, default=str, sort_keys=True)
    return str(observation)


def normalize_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
    """Normalize a tool call into ``(name, arguments_dict)``.

    Accepts several shapes so the bridge stays decoupled from any particular
    renderer:

    * an object with ``.name`` and ``.arguments`` (e.g. a ``renderers``
      ``ParsedToolCall``),
    * a flat dict ``{"name": ..., "arguments": ...}``,
    * an OpenAI-style dict ``{"function": {"name": ..., "arguments": ...}}``.

    ``arguments`` may be a dict or a JSON-encoded string.
    """
    name: Any = None
    arguments: Any = None

    if hasattr(tool_call, "name") or hasattr(tool_call, "arguments"):
        name = getattr(tool_call, "name", None)
        arguments = getattr(tool_call, "arguments", None)
    elif isinstance(tool_call, dict):
        if isinstance(tool_call.get("function"), dict):
            fn = tool_call["function"]
            name = fn.get("name")
            arguments = fn.get("arguments")
        else:
            name = tool_call.get("name")
            arguments = tool_call.get("arguments")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    return (name if isinstance(name, str) else ""), arguments


class OpenEnvBridge:
    """Drive an OpenEnv environment and translate its turns to/from message space.

    Args:
        base_url: URL of the running OpenEnv server (``http://`` or ``ws://``).
        client: An explicit :class:`~openenv.core.env_client.EnvClient` to use.
            When omitted, a :class:`GenericEnvClient` is created from ``base_url``.
            Passing a client (or ``client_factory``) is how tests inject a stub.
        client_factory: Zero-arg callable returning an ``EnvClient``; overrides
            ``base_url``. Useful for tests and for typed clients.
        action_mode: ``"tool"`` (assistant emits tool calls) or ``"text"``
            (assistant's message text is the action).
        action_key: In ``"text"`` mode, the OpenEnv action field the text is
            placed under (default ``"message"``).
        tool_action_key: In ``"tool"`` mode, if the tool arguments contain this
            key its value is used as the action dict; otherwise the whole
            arguments dict is the action (default ``"action"``).
        reward_name: Label used when reporting rewards to callers.
        observation_keys: Keys tried when rendering observations to text.
        observation_renderer: Optional environment-specific renderer, used
            instead of :func:`render_observation`. Structured environments (e.g.
            chess, whose observation is a FEN plus a legal-move list) supply one
            via a :class:`~examples.titanrl_openenv.tasks.TaskProfile`.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8010",
        *,
        client: Optional[EnvClient] = None,
        client_factory: Optional[Callable[[], EnvClient]] = None,
        action_mode: str = "tool",
        action_key: str = "message",
        tool_action_key: Optional[str] = "action",
        reward_name: str = "openenv",
        observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
        observation_renderer: Optional[Callable[[Any], str]] = None,
    ) -> None:
        action_mode = action_mode.lower()
        if action_mode not in ("tool", "text"):
            raise ValueError(
                f"action_mode must be 'tool' or 'text', got {action_mode!r}"
            )
        self.base_url = base_url
        self.action_mode = action_mode
        self.action_key = action_key
        self.tool_action_key = tool_action_key
        self.reward_name = reward_name
        self.observation_keys = tuple(observation_keys)
        self.observation_renderer = observation_renderer

        self._client = client
        self._client_factory = client_factory
        self._started = False

    # -- lifecycle -----------------------------------------------------------

    def _ensure_client(self) -> EnvClient:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                self._client = GenericEnvClient(base_url=self.base_url)
        return self._client

    async def start(self, **reset_kwargs: Any) -> BridgeTurn:
        """Connect (if needed) and reset the environment; return the first turn."""
        client = self._ensure_client()
        result = await client.reset(**reset_kwargs)
        self._started = True
        return self._to_turn(result)

    async def stop(self) -> None:
        """Close the underlying client and release its resources. Idempotent."""
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._started = False

    # -- stepping ------------------------------------------------------------

    async def act(self, action: dict[str, Any]) -> BridgeTurn:
        """Send a raw OpenEnv action dict and return the resulting turn."""
        client = self._ensure_client()
        result = await client.step(action)
        return self._to_turn(result)

    async def act_from_text(self, text: str) -> BridgeTurn:
        """Wrap assistant text into an action (``text`` mode) and step."""
        return await self.act({self.action_key: text})

    async def act_from_tool_calls(self, tool_calls: Sequence[Any]) -> BridgeTurn:
        """Convert the first tool call into an OpenEnv action and step.

        A single OpenEnv step consumes one action, so only the first tool call is
        used; environments that need multi-action turns should expose that in
        their own action schema.
        """
        action = self.action_from_tool_calls(tool_calls)
        return await self.act(action)

    # -- translation helpers -------------------------------------------------

    def action_from_tool_calls(self, tool_calls: Sequence[Any]) -> dict[str, Any]:
        """Translate parsed tool calls into a single OpenEnv action dict."""
        if not tool_calls:
            return {}
        _name, arguments = normalize_tool_call(tool_calls[0])
        if self.tool_action_key is not None and isinstance(
            arguments.get(self.tool_action_key), dict
        ):
            return dict(arguments[self.tool_action_key])
        return arguments

    def render(self, observation: Any) -> str:
        """Render an observation to text using this bridge's renderer."""
        if self.observation_renderer is not None:
            return self.observation_renderer(observation)
        return render_observation(observation, observation_keys=self.observation_keys)

    def _to_turn(self, result: StepResult) -> BridgeTurn:
        observation = result.observation
        reward = result.reward
        done = bool(result.done)
        # StepResult carries reward/done, but envs may also embed them in the
        # observation dict; prefer StepResult, fall back to the observation.
        if isinstance(observation, dict):
            if reward is None and observation.get("reward") is not None:
                reward = observation.get("reward")
            done = done or bool(observation.get("done", False))
        reward = None if reward is None else float(reward)

        # The wire format strips ``reward``/``done`` out of the observation
        # payload and onto the StepResult, so put them back for the renderer:
        # environments that answer a bad action with a negative reward rather
        # than an error (chess, for one) need it to explain what happened.
        render_input = observation
        if isinstance(observation, dict):
            render_input = {**observation, "reward": reward, "done": done}

        return BridgeTurn(
            text=self.render(render_input),
            done=done,
            reward=reward,
            raw_observation=observation,
        )
