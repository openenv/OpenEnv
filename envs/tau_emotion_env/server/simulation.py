# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A τ-emotion-bench conversation driven one agent action at a time.

τ-emotion-bench's orchestrator runs a whole conversation in one call: it asks the
agent for its next message, sends it to the simulated user or the tools, and
loops until someone stops. To step it from outside, the orchestrator runs in a
thread with an agent whose `generate_next_message` blocks until `set_action`
hands it the next message. This is τ²-bench's gym agent, written against
τ-emotion-bench's classes.
"""

import threading
from copy import deepcopy
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from tau_emotion.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau_emotion.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
)
from tau_emotion.data_model.simulation import SimulationRun
from tau_emotion.data_model.tasks import Task
from tau_emotion.environment.tool import Tool, as_tool
from tau_emotion.orchestrator.orchestrator import Orchestrator
from tau_emotion.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau_emotion.registry import registry
from tau_emotion.user.user_simulator import UserSimulator


class GymAgentState(BaseModel):
    """The agent's view of the conversation."""

    messages: list[APICompatibleMessage]


def done() -> str:
    """Call this function when you are done with the task."""
    return GymAgent.STOP_TOKEN


class GymAgent(HalfDuplexAgent[GymAgentState]):
    """
    An agent whose messages come from outside the simulation.

    The orchestrator calls `generate_next_message` from its own thread; it records
    what the agent received and blocks until `set_action` provides the reply.
    """

    STOP_FUNCTION_NAME = "done"
    STOP_TOKEN = "###STOP###"

    def __init__(self, tools: list[Tool], domain_policy: str):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._observation: Optional[Message] = None
        self._next_action: Optional[AssistantMessage] = None
        self._agent_turn_finished = threading.Event()
        self._agent_turn_finished.set()
        self._lock = threading.Lock()
        self.aborted = False
        self.tools.append(as_tool(done))

    @property
    def observation(self) -> Optional[Message]:
        """The last message the agent received."""
        return self._observation

    @property
    def is_agent_turn(self) -> bool:
        """Whether the agent is waiting for `set_action`."""
        return not self._agent_turn_finished.is_set()

    def set_action(self, action: AssistantMessage) -> None:
        """Give the agent its next message; `generate_next_message` returns it."""
        with self._lock:
            if self._agent_turn_finished.is_set():
                raise RuntimeError("It is not the agent's turn to act.")
            self._next_action = action
            self._agent_turn_finished.set()

    def abort(self) -> None:
        """Make the agent stop at its next turn, and now if it is waiting."""
        with self._lock:
            self.aborted = True
            if not self._agent_turn_finished.is_set():
                self._next_action = AssistantMessage(
                    role="assistant", content=self.STOP_TOKEN
                )
                self._agent_turn_finished.set()

    def set_seed(self, seed: int) -> None:
        pass  # The agent is not random: its messages come from outside.

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> GymAgentState:
        return GymAgentState(messages=list(message_history or []))

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GymAgentState
    ) -> tuple[AssistantMessage, GymAgentState]:
        with self._lock:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            self._observation = deepcopy(message)
            if self.aborted:
                self._next_action = AssistantMessage(
                    role="assistant", content=self.STOP_TOKEN
                )
            else:
                self._agent_turn_finished.clear()
        self._agent_turn_finished.wait()
        with self._lock:
            response = self._next_action
            self._next_action = None
        response = self._stop_tool_call_as_stop_message(response)
        state.messages.append(response)
        return response, state

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[GymAgentState] = None,
    ) -> None:
        super().stop(message, state)
        with self._lock:
            self._observation = deepcopy(message)
            self._agent_turn_finished.set()

    def _stop_tool_call_as_stop_message(
        self, message: AssistantMessage
    ) -> AssistantMessage:
        """A call to `done` becomes the stop message the orchestrator recognises."""
        if message.tool_calls and any(
            call.name == self.STOP_FUNCTION_NAME for call in message.tool_calls
        ):
            message.content = self.STOP_TOKEN
            message.tool_calls = None
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        if message.tool_calls and any(
            call.name == cls.STOP_FUNCTION_NAME for call in message.tool_calls
        ):
            return True
        return bool(message.content) and cls.STOP_TOKEN in message.content


class Simulation:
    """
    One τ-emotion-bench task, stepped by the agent's messages.

    Args:
        domain (`str`):
            τ-emotion-bench domain, e.g. `airline`.
        task (`Task`):
            The task to run. Train tasks run, and are scored, against the
            domain's train database.
        user_llm (`str`):
            litellm model name of the simulated user.
        user_llm_args (`dict`):
            litellm arguments for the simulated user.
        max_steps (`int`):
            τ-emotion-bench's limit on messages per conversation.
    """

    def __init__(
        self,
        domain: str,
        task: Task,
        user_llm: str,
        user_llm_args: dict,
        max_steps: int,
    ):
        self.domain = domain
        self.task = task
        # The train tasks reference records that only exist in the train database,
        # so they run and are scored against it (see τ-emotion-bench's runner/build.py).
        self.env_kwargs = {"db_split": "train"} if task.split == "train" else {}

        environment = registry.get_env_constructor(domain)(**self.env_kwargs)
        self.policy = environment.get_policy()
        self.agent = GymAgent(tools=environment.get_tools(), domain_policy=self.policy)
        # No τ-emotion-bench domain gives the user tools of their own.
        user = UserSimulator(
            tools=None,
            instructions=task.user_scenario.to_tau_bench_instruction(),
            llm=user_llm,
            llm_args=user_llm_args,
        )
        self.orchestrator = Orchestrator(
            domain=domain,
            agent=self.agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
        )
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._aborted = False
        self.run: Optional[SimulationRun] = None

    @property
    def tools(self) -> list[Tool]:
        """The agent's tools: the domain's, plus `done`."""
        return self.agent.tools

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self) -> Optional[Message]:
        """Start the conversation and return the user's opening message.

        `None` when the simulation ended before the agent's first turn, e.g.
        because the user's model failed; the reason is in the server log.
        """
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._wait_for_agent_turn()

    def step(self, action: AssistantMessage) -> Optional[Message]:
        """Send the agent's message and return what came back: the user's reply
        or the tool's result. `None` once the conversation has ended; check
        `done` and `run`."""
        if self._done.is_set():
            return None
        self.agent.set_action(action)
        return self._wait_for_agent_turn()

    def abort(self) -> None:
        """End an unfinished conversation without scoring it."""
        if self._done.is_set():
            return
        self._aborted = True
        self.agent.abort()

    def _wait_for_agent_turn(self) -> Optional[Message]:
        while not self._done.is_set() and not self.agent.is_agent_turn:
            self._done.wait(timeout=0.01)
        if self._done.is_set():
            return None
        return self.agent.observation

    def _run(self) -> None:
        try:
            if self._aborted:
                return
            simulation = self.orchestrator.run()
            if self._aborted:
                return
            # Scoring replays the task's gold actions on a fresh environment, as
            # `tau_emotion.runner.simulation.run_simulation` does.
            simulation.policy = self.policy
            simulation.reward_info = evaluate_simulation(
                simulation=simulation,
                task=self.task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=self.domain,
                env_kwargs=self.env_kwargs,
            )
            self.run = simulation
        except Exception as e:
            logger.error(f"[{self.task.id}] The simulation failed: {e!r}")
        finally:
            self._done.set()


__all__ = ["GymAgent", "GymAgentState", "Simulation"]
