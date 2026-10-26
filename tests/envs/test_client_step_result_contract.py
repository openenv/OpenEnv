# SPDX-License-Identifier: BSD-3-Clause

"""Typed clients must read reward and done from where the server puts them.

`serialize_observation` deliberately excludes `reward` and `done` from the
nested observation dict and surfaces them on the envelope. A client that reads
them from the observation dict instead sees reward 0.0 and done False on every
step, which silently breaks any training loop or episode-termination check
built on it.

These tests drive each client's `_parse_result` with the output of the real
serializer, so they fail if a client and the wire format ever drift apart
again.
"""

from __future__ import annotations

from openenv.core.env_server.serialization import serialize_observation

# No `importorskip` here on purpose. These clients and their models depend only
# on openenv and pydantic, never on an engine package, so skipping when the
# engine is absent would quietly skip the very regression being guarded.


def test_serializer_keeps_reward_and_done_on_the_envelope():
    """Pin the wire contract the clients are being checked against."""
    from chess_env.models import ChessObservation

    payload = serialize_observation(ChessObservation(done=True, reward=1.0))

    assert payload["reward"] == 1.0
    assert payload["done"] is True
    assert "reward" not in payload["observation"]
    assert "done" not in payload["observation"]


class TestChessClient:
    def parse(self, observation):
        from chess_env.client import ChessEnv

        return ChessEnv._parse_result(None, serialize_observation(observation))

    def test_step_result_carries_reward_and_done(self):
        from chess_env.models import ChessObservation

        result = self.parse(
            ChessObservation(fen="8/8/8/8/8/8/8/K6k w - - 0 1", done=True, reward=1.0)
        )

        assert result.reward == 1.0
        assert result.done is True

    def test_observation_agrees_with_the_step_result(self):
        from chess_env.models import ChessObservation

        result = self.parse(ChessObservation(done=True, reward=-1.0))

        assert result.observation.reward == result.reward
        assert result.observation.done == result.done

    def test_other_observation_fields_still_arrive(self):
        from chess_env.models import ChessObservation

        result = self.parse(
            ChessObservation(
                fen="rn6/8/8/8/8/8/8/K6k b - - 0 1", is_check=True, result="1-0"
            )
        )

        assert result.observation.fen.startswith("rn6")
        assert result.observation.is_check is True
        assert result.observation.result == "1-0"


class TestSumoClient:
    def test_observation_agrees_with_the_step_result(self):
        from sumo_rl_env.client import SumoRLEnv
        from sumo_rl_env.models import SumoObservation

        result = SumoRLEnv._parse_result(
            None, serialize_observation(SumoObservation(done=True, reward=2.5))
        )

        assert result.reward == 2.5
        assert result.done is True
        assert result.observation.reward == 2.5
        assert result.observation.done is True


class TestSophistryClient:
    def test_observation_agrees_with_the_step_result(self):
        from sophistry_bench_sprint_env.client import SophistryBenchSprintEnv
        from sophistry_bench_sprint_env.models import AdvocacyObservation

        result = SophistryBenchSprintEnv._parse_result(
            None, serialize_observation(AdvocacyObservation(done=True, reward=0.75))
        )

        assert result.reward == 0.75
        assert result.done is True
        assert result.observation.reward == 0.75
        assert result.observation.done is True


class TestGridWorldClient:
    def test_step_result_carries_reward_and_done(self):
        from grid_world_env.client import GridWorldEnv
        from grid_world_env.models import GridWorldObservation

        payload = serialize_observation(
            GridWorldObservation(x=4, y=4, done=True, reward=1.0)
        )
        result = GridWorldEnv._parse_result(None, payload)

        assert result.reward == 1.0
        assert result.done is True
        assert (result.observation.x, result.observation.y) == (4, 4)

    def test_observation_and_metadata_agree_with_the_step_result(self):
        from grid_world_env.client import GridWorldEnv
        from grid_world_env.models import GridWorldObservation

        payload = serialize_observation(
            GridWorldObservation(
                x=4, y=4, done=True, reward=1.0, metadata={"goal": True}
            )
        )
        result = GridWorldEnv._parse_result(None, payload)

        assert result.metadata == {"goal": True}
        assert result.observation.reward == 1.0
        assert result.observation.done is True


class TestCalendarClient:
    def test_step_result_keeps_metadata(self):
        from calendar_env.client import CalendarEnv

        payload = {
            "observation": {"success": True, "tool_result": {"ok": True}},
            "reward": 0.5,
            "done": False,
            "metadata": {"tool": "create_event"},
        }
        result = CalendarEnv._parse_step_result(None, payload)

        assert result.reward == 0.5
        assert result.metadata == {"tool": "create_event"}
