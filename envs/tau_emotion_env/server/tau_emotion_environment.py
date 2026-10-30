# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""τ-emotion-bench as an OpenEnv environment.

Each episode is one τ-emotion-bench task: a customer-service conversation between
the agent and an LLM-simulated user who is in an emotional state (anxious,
frustrated, grieving, …) and leaves if the agent handles it badly. The domain has
a database, a policy and tools, and the agent acts through MCP tools:

- the domain's own tools (e.g. `get_user_details`, `book_reservation`), which
  read and write the task's database;
- `respond_to_user(message)`, which sends a message to the simulated user and
  returns their reply;
- `done()`, which ends the conversation from the agent's side.

The episode ends when the user or the agent stops, and τ-emotion-bench's own
evaluator turns the final database state into the reward: 1 when the task is
solved, 0 when it is not, and -1 when the user gave up on the agent.

The simulated user is an LLM called through litellm. It defaults to Hugging Face
Inference Providers.
"""

import asyncio
import os
import random
import uuid
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation
from tau_emotion.config import DEFAULT_MAX_STEPS
from tau_emotion.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
)
from tau_emotion.data_model.simulation import TerminationReason
from tau_emotion.registry import registry
from tau_emotion.utils.utils import DATA_DIR

from ..models import TauEmotionState
from .simulation import Simulation

HF_ROUTER = "https://router.huggingface.co/v1"

# The benchmark's reference user simulator is Qwen3-235B-A22B-Instruct-2507 at
# temperature 0 (τ-emotion-bench's `DEFAULT_LLM_USER`); the leaderboard's scores
# were produced with it.
REFERENCE_USER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_MODELS = {
    "hf": REFERENCE_USER_MODEL,
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-5",
}
# The domains the benchmark scores. `mock` is τ-emotion-bench's test fixture.
DOMAINS = [d for d in registry.get_domains() if d != "mock"]


def llm_config(
    provider: str,
    model: Optional[str] = None,
    hf_token: Optional[str] = None,
    api_base: Optional[str] = None,
) -> tuple[str, dict]:
    """
    The litellm model name and arguments for a provider.

    Args:
        provider (`str`):
            `"hf"` (Hugging Face Inference Providers, with `HF_TOKEN`), `"openai"`
            (with `OPENAI_API_KEY`; also any OpenAI-compatible server through
            `api_base`) or `"anthropic"` (with `ANTHROPIC_API_KEY`).
        model (`str`, *optional*):
            Model id. Defaults to the provider's entry in `DEFAULT_MODELS`.
        hf_token (`str`, *optional*):
            Token for `"hf"`. Defaults to `HF_TOKEN`.
        api_base (`str`, *optional*):
            For `"openai"`, the server to call instead of OpenAI, e.g. a vLLM
            server hosting the reference user model.

    Returns:
        `tuple[str, dict]`: the litellm model name and the call arguments.
    """
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"unknown provider {provider!r}; use one of {list(DEFAULT_MODELS)}"
        )
    model = model or DEFAULT_MODELS[provider]
    args: dict[str, Any] = {"temperature": 0.0, "num_retries": 5}
    if provider == "hf":
        # The router is OpenAI-compatible, so litellm talks to it as an OpenAI endpoint.
        args["api_base"] = HF_ROUTER
        args["api_key"] = hf_token or os.environ.get("HF_TOKEN")
        return f"openai/{model}", args
    if provider == "openai" and api_base:
        args["api_base"] = api_base
    return f"{provider}/{model}", args


# `respond_to_user` and `done` wait on the simulated customer, which is an LLM call
# with retries, so tool calls get longer than MCP's 30 s default.
TOOL_TIMEOUT_S = 300.0


# What FastMCP gives a function tool returning `str`, so clients get the text as `.data`.
TEXT_OUTPUT = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
    "x-fastmcp-wrap-result": True,
}


class TauEmotionTool(Tool):
    """An MCP tool that forwards its call to the running τ-emotion-bench simulation."""

    env: Any = None
    output_schema: dict[str, Any] = TEXT_OUTPUT

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        if self.name == "respond_to_user":
            action = AssistantMessage(role="assistant", content=arguments["message"])
        else:
            action = AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=uuid.uuid4().hex[:8], name=self.name, arguments=arguments
                    )
                ],
            )
        text = await asyncio.to_thread(self.env.act, action)
        return ToolResult(content=text, structured_content={"result": text})


RESPOND_TO_USER = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "What to say to the user."}
    },
    "required": ["message"],
}


def message_text(message: Optional[Message]) -> str:
    """What the agent reads of the message that came back: the user's reply, or
    the tool's result."""
    if message is None:
        return ""
    if isinstance(message, MultiToolMessage):
        return "\n".join(m.content or "" for m in message.tool_messages)
    return message.content or ""


class TauEmotionEnvironment(MCPEnvironment):
    """
    One τ-emotion-bench domain as an MCP environment.

    Args:
        domain (`str`, *optional*, defaults to `"airline"`):
            τ-emotion-bench domain: `airline`, `bank`, `basketball`, `movie`,
            `restaurant`, `retail`, `school`, `telecom`, `telehealth` or `travel`.
        split (`str`, *optional*, defaults to `"test"`):
            Task split, `test` (the benchmark) or `train`.
        user_provider (`str`, *optional*, defaults to `"hf"`):
            Provider for the simulated user. See [`llm_config`].
        user_model (`str`, *optional*):
            Model for the simulated user. Defaults to the provider's default.
        user_api_base (`str`, *optional*):
            For the `"openai"` provider, an OpenAI-compatible server to call.
        max_steps (`int`, *optional*, defaults to τ-emotion-bench's 200):
            Limit on messages per conversation.
        hf_token (`str`, *optional*):
            Token for the `"hf"` provider, e.g. a visitor's. Defaults to `HF_TOKEN`.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        domain: str = "airline",
        split: str = "test",
        user_provider: str = "hf",
        user_model: Optional[str] = None,
        user_api_base: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        hf_token: Optional[str] = None,
    ):
        if not (DATA_DIR / "tau_emotion" / "domains").is_dir():
            raise FileNotFoundError(
                f"τ-emotion-bench data not found in {DATA_DIR}. Point TAU_EMOTION_DATA_DIR "
                "at the `data` folder of a tau-emotion-bench checkout (see the README)."
            )
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}; use one of {DOMAINS}")
        if split not in ("test", "train"):
            raise ValueError(f"unknown split {split!r}; use 'test' or 'train'")
        self.domain = domain
        self.split = split
        self.max_steps = max_steps
        self.user_provider = user_provider
        self.user_llm, self.user_llm_args = llm_config(
            user_provider, user_model, hf_token, user_api_base
        )

        self.tasks = {t.id: t for t in registry.get_tasks_loader(domain)(split)}

        mcp = FastMCP(f"tau_emotion_{domain}")
        for tool in registry.get_env_constructor(domain)().get_tools():
            schema = tool.openai_schema["function"]
            mcp.add_tool(
                TauEmotionTool(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                    env=self,
                )
            )
        mcp.add_tool(
            TauEmotionTool(
                name="respond_to_user",
                description="Send a message to the user and get their reply.",
                parameters=RESPOND_TO_USER,
                env=self,
            )
        )
        mcp.add_tool(
            TauEmotionTool(
                name="done",
                description="End the conversation, once the user's request is handled.",
                parameters={"type": "object", "properties": {}},
                env=self,
            )
        )
        super().__init__(mcp)

        self._simulation: Optional[Simulation] = None
        self._state = TauEmotionState(domain=domain, split=split)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Start a τ-emotion-bench task: the simulated user opens the conversation.

        Args:
            seed (`int`, *optional*):
                Picks the task when `task_id` is not given.
            episode_id (`str`, *optional*):
                Episode identifier. A UUID is generated when `None`.
            task_id (`str`, *optional*):
                The τ-emotion-bench task to run, e.g. `airline_test_0003`. A random
                task of the split otherwise.
            hf_token (`str`, *optional*):
                Token for the `"hf"` provider, used from now on in this session. Lets a
                client bring its own token to a server without one. It is not stored in
                the state.

        Returns:
            `Observation` whose metadata has the user's first message, the
            domain policy the agent must follow, and the task id.
        """
        if hf_token and self.user_provider == "hf":
            self.user_llm_args = {**self.user_llm_args, "api_key": hf_token}
        # Only conversations need the token, so the server runs without one.
        if self.user_provider == "hf" and not self.user_llm_args["api_key"]:
            raise ValueError(
                "No Hugging Face token. The simulated user runs on Hugging Face Inference "
                "Providers by default: pass hf_token to reset(), set HF_TOKEN on the "
                "server, or set TAU_EMOTION_USER_PROVIDER=openai|anthropic."
            )
        if task_id is None:
            task_id = random.Random(seed).choice(sorted(self.tasks))
        elif task_id not in self.tasks:
            raise ValueError(
                f"unknown task {task_id!r} in the {self.split} split of {self.domain}"
            )
        simulation = Simulation(
            domain=self.domain,
            task=self.tasks[task_id],
            user_llm=self.user_llm,
            user_llm_args=self.user_llm_args,
            max_steps=self.max_steps,
        )
        first_message = simulation.start()
        if (
            first_message is None
        ):  # τ-emotion-bench only logs why, e.g. a rejected token
            raise RuntimeError(
                f"The simulated user could not open the conversation with {self.user_llm}. "
                "Check its credential and the server log."
            )
        # Only once the new conversation has started, so a failed reset keeps the current one.
        self._end_conversation()
        self._simulation = simulation
        self._state = TauEmotionState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            domain=self.domain,
            split=self.split,
            task_id=task_id,
        )
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "user_message": message_text(first_message),
                "policy": simulation.policy,
                "task_id": task_id,
            },
        )

    def act(self, action: AssistantMessage) -> str:
        """Advance the simulation with one agent message and return what came back."""
        if self._simulation is None:
            raise RuntimeError("call reset() before calling tools")
        if self._state.done:
            return "The conversation has ended."
        observation = self._simulation.step(action)
        if self._simulation.done:
            self._state.done = True
            run = self._simulation.run
            # τ-emotion-bench only logs why, e.g. the user's model failed. The episode
            # ends unscored: `reward` stays None, and the error reaches the client.
            if run is None or run.reward_info is None:
                self._state.termination_reason = (
                    TerminationReason.INFRASTRUCTURE_ERROR.value
                )
                raise RuntimeError(
                    "The τ-emotion-bench simulation failed (often the simulated user's "
                    f"model, {self.user_llm}), so the conversation ended without a score. "
                    "See the server log."
                )
            self._state.termination_reason = run.termination_reason.value
            self._state.reward = run.reward_info.reward
            self._state.reward_info = run.reward_info.model_dump(mode="json")
            # The user's last words, when they ended it.
            observation = self._simulation.agent.observation
        return message_text(observation)

    def _end_conversation(self) -> None:
        """Stop an unfinished simulation, whose thread otherwise waits for the agent forever."""
        if self._simulation is not None and not self._state.done:
            self._simulation.abort()
        self._simulation = None

    def close(self) -> None:
        self._end_conversation()
        super().close()

    def _finish(self, observation: Observation) -> Observation:
        if not self._state.done:
            return observation
        return observation.model_copy(
            update={
                "done": True,
                "reward": self._state.reward,
                "metadata": {
                    **observation.metadata,
                    "termination_reason": self._state.termination_reason,
                    "reward_info": self._state.reward_info,
                },
            }
        )

    def step(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        self._state.step_count += 1
        return self._finish(
            super().step(action, timeout_s=timeout_s or TOOL_TIMEOUT_S, **kwargs)
        )

    async def step_async(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        self._state.step_count += 1
        return self._finish(
            await super().step_async(
                action, timeout_s=timeout_s or TOOL_TIMEOUT_S, **kwargs
            )
        )

    def _step_impl(
        self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction."
            },
        )

    @property
    def state(self) -> TauEmotionState:
        return self._state
