# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A TorchTitan-RL ``MessageEnv`` backed by an OpenEnv environment server.

:class:`OpenEnvMessageEnv` is the thin TorchTitan-RL (``TitanRL``) adapter: it
plugs an OpenEnv environment into the ``MessageEnv`` contract TitanRL's rollouter
expects (``init`` + ``step``, working purely in message space). All OpenEnv
protocol handling and observation/action translation lives in the
framework-agnostic :class:`~examples.titanrl_openenv.openenv_bridge.OpenEnvBridge`,
so this file only maps between message-space turns and bridge turns.

Because it subclasses ``MessageEnv``, importing this module requires
``torchtitan`` (and ``renderers``) to be installed. The bridge module can be used
and tested without them.
"""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message
from torchtitan.rl.rollout.environment import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

from .data import OpenEnvSample
from .openenv_bridge import DEFAULT_ACT_TOOL, OpenEnvBridge
from .tasks import get_task_profile


class OpenEnvMessageEnv(MessageEnv):
    """Multi-turn TitanRL env whose turns are served by an OpenEnv environment.

    Each rollout: ``init`` connects to the OpenEnv server, resets it, and returns
    the opening instruction plus the initial observation (and, in ``tool`` mode,
    the task's tool). Each ``step`` translates the assistant's turn into an
    OpenEnv action, applies it, and returns the resulting observation as an env
    message, ending the rollout when OpenEnv reports ``done`` (or, in ``tool``
    mode, when the assistant stops calling the tool).

    Everything environment-specific — instruction, tool schema, observation
    rendering — comes from the named :class:`~examples.titanrl_openenv.tasks.TaskProfile`
    in ``task``. The shipped ``"chess"`` profile drives OpenEnv's built-in
    ``envs/chess_env``; ``"generic"`` works against any OpenEnv server via the
    free-form ``openenv_act`` tool. Rewards OpenEnv reports are forwarded to the
    rubric via ``env_rewards``, together with any extra shaping rewards the
    profile derives from the observation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        base_url: str = "http://127.0.0.1:8010"
        """URL of the running OpenEnv environment server."""

        task: str = "generic"
        """Task profile name (``"chess"``, ``"generic"``, or one you register)."""

        action_mode: str = "tool"
        """``"tool"`` (assistant calls a tool) or ``"text"`` (message text is the action)."""

        action_key: str = "message"
        """In ``text`` mode, the OpenEnv action field the assistant text is placed under."""

        reward_name: str = "openenv"
        """Label under which OpenEnv step rewards are reported in ``env_rewards``."""

        env_role: str = "tool"
        """Chat role used for env reply messages (``"tool"`` or ``"user"``)."""

        include_initial_observation: bool = True
        """Append the reset observation to the opening prompt as context."""

    def __init__(self, config: Config, *, env_input: OpenEnvSample) -> None:
        self._config = config
        self._sample = env_input
        self._profile = get_task_profile(config.task)
        self._bridge = OpenEnvBridge(
            base_url=config.base_url,
            action_mode=config.action_mode,
            action_key=config.action_key,
            tool_action_key=self._profile.tool_action_key,
            reward_name=config.reward_name,
            observation_renderer=self._profile.render,
        )

    async def init(self) -> MessageEnvInitOutput:
        turn = await self._bridge.start(**dict(self._sample.reset_kwargs or {}))

        # The sample's prompt wins; otherwise the task profile's instruction.
        instruction = self._sample.prompt or self._profile.instruction

        messages: list[Message] = []
        if instruction:
            messages.append({"role": "user", "content": instruction})
        if self._config.include_initial_observation and turn.text:
            messages.append({"role": "user", "content": turn.text})
        if not messages:
            # A rollout needs at least one prompt message to render.
            messages.append({"role": "user", "content": turn.text})

        tools = (
            [self._profile.tool or DEFAULT_ACT_TOOL]
            if self._config.action_mode == "tool"
            else []
        )
        return MessageEnvInitOutput(init_prompt_messages=messages, tools=tools)

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        if self._config.action_mode == "tool":
            tool_calls = completion_message.get("tool_calls") or []
            if not tool_calls:
                # No tool call -> the assistant's reply is its final answer.
                return MessageEnvStepOutput(done=True)
            turn = await self._bridge.act_from_tool_calls(tool_calls)
        else:
            content = completion_message.get("content") or ""
            turn = await self._bridge.act_from_text(content)

        env_messages: list[Message] = []
        if turn.text:
            env_messages.append({"role": self._config.env_role, "content": turn.text})

        env_rewards: dict[str, float] = {}
        if turn.reward is not None:
            env_rewards[self._config.reward_name] = float(turn.reward)
        # Task-specific shaping read off the raw observation, reported under its
        # own keys so the rubric can weight it separately from the env's reward.
        if self._profile.shaping is not None:
            observation = turn.raw_observation
            if isinstance(observation, dict):
                # The wire format moves reward/done onto the StepResult; put them
                # back, as the renderer gets them, so shaping can tell a
                # finished game from an ongoing position.
                observation = {**observation, "reward": turn.reward, "done": turn.done}
            env_rewards.update(self._profile.shaping(observation))

        return MessageEnvStepOutput(
            env_messages=env_messages,
            done=turn.done,
            env_rewards=env_rewards,
        )

    async def close(self) -> None:
        await self._bridge.stop()
