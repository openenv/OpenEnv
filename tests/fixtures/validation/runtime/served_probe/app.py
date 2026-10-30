"""A deterministic subject, with deliberate faults confined to test assets."""

import asyncio
import json
import os
from typing import Any

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Rubric, WeightedSum
from pydantic import Field


class ProbeAction(Action):
    increment: int = Field(ge=1, le=2, strict=True)


class ProbeObservation(Observation):
    counter: int = Field(strict=True)


class CounterRubric(Rubric):
    def forward(self, action, observation):
        return float(observation.counter >= 2)

    def validation_config(self):
        return {"threshold": 2}


class ProbeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id="uninitialized", step_count=0)
        self.counter = 0
        self.rubric = WeightedSum([CounterRubric(), CounterRubric()], [0.5, 0.5])

    def reset(self, seed=None, episode_id=None, **kwargs):
        self.counter = 0
        self._state = State(episode_id=episode_id or "probe", step_count=0)
        return ProbeObservation(counter=0, reward=0.0, done=False)

    def step(self, action, timeout_s=None, **kwargs):
        self.counter += action.increment
        self._state.step_count += 1
        observation = ProbeObservation(
            counter=self.counter,
            reward=float(self.counter >= 2),
            done=self._state.step_count >= 2,
        )
        observation.reward = self.rubric(action, observation)
        return observation

    @property
    def state(self):
        return self._state


class WireFault:
    """Corrupt real server responses after serialization, preserving raw defects."""

    def __init__(self, application, mode):
        self.application = application
        self.mode = mode

    async def __call__(self, scope, receive, send):
        steps = 0

        async def fault_receive():
            nonlocal steps
            message = await receive()
            if (
                self.mode in {"hung_step", "slow_step"}
                and message["type"] == "websocket.receive"
                and message.get("text")
                and json.loads(message["text"]).get("type") == "step"
            ):
                steps += 1
                if steps == 2:
                    # Tests signal the CLI only after the collector has completed
                    # a real reset, first step and both corresponding state reads.
                    os.write(1, b"OPENENV_VALIDATION_STEP_BLOCKED\n")
                    if self.mode == "slow_step":
                        await asyncio.sleep(5.2)
                    else:
                        await asyncio.Event().wait()
            return message

        async def fault_send(message: dict[str, Any]):
            if message["type"] == "websocket.send" and message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "observation":
                    payload = data["data"]
                    if self.mode == "bad_reward":
                        payload["reward"] = True
                    elif self.mode == "bad_observation":
                        payload["observation"]["counter"] = "not-an-integer"
                    elif self.mode == "missing_done":
                        payload.pop("done", None)
                elif data.get("type") == "state" and self.mode == "bad_state":
                    data["data"]["episode_id"] = "wrong-episode"
                message = {**message, "text": json.dumps(data)}
            await send(message)

        await self.application(scope, fault_receive, fault_send)


def make_app(mode="good"):
    if mode == "startup_failure":
        raise RuntimeError(
            "Deliberate startup failure from the validation test fixture"
        )
    if mode not in {
        "good",
        "bad_reward",
        "bad_observation",
        "missing_done",
        "bad_state",
        "hung_step",
        "slow_step",
    }:
        raise ValueError(f"Unknown fixture mode: {mode}")
    return WireFault(
        create_app(
            ProbeEnvironment,
            ProbeAction,
            ProbeObservation,
            env_name="validation_probe",
            max_concurrent_envs=2,
        ),
        mode,
    )


def main():
    uvicorn.run(
        make_app(os.environ.get("VALIDATION_FAULT", "good")),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
