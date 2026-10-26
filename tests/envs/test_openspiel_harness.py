# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OpenSpiel harness session adapter."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openspiel_env.harness import OpenSpielSessionFactory, render_tic_tac_toe_board
from openspiel_env.models import OpenSpielAction, OpenSpielObservation, OpenSpielState


class FakeOpenSpielClient:
    """In-memory OpenSpiel client that plays a scripted TTT game."""

    def __init__(self, winning: bool = True):
        self._winning = winning
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[OpenSpielAction] = []
        self.closed = False

    def reset(self, **kwargs: Any) -> StepResult[OpenSpielObservation]:
        self.reset_calls.append(kwargs)
        return StepResult(
            observation=OpenSpielObservation(
                info_state=[1.0] * 9 + [0.0] * 18,
                legal_actions=list(range(9)),
                game_phase="playing",
                current_player_id=0,
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: OpenSpielAction) -> StepResult[OpenSpielObservation]:
        self.step_calls.append(action)
        done = len(self.step_calls) >= 1
        reward = 1.0 if (done and self._winning) else 0.0
        info_state = (
            [0.0] * 9 + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 9
        )
        return StepResult(
            observation=OpenSpielObservation(
                info_state=info_state,
                legal_actions=[] if done else list(range(1, 9)),
                game_phase="terminal" if done else "playing",
                current_player_id=0,
            ),
            reward=reward,
            done=done,
        )

    def state(self) -> OpenSpielState:
        return OpenSpielState(
            step_count=len(self.step_calls),
            game_name="tic_tac_toe",
        )

    def close(self) -> None:
        self.closed = True


class TestRenderTicTacToeBoard:
    def test_renders_empty_board_with_action_ids(self):
        empty = [1.0] * 9 + [0.0] * 18
        board = render_tic_tac_toe_board(empty)
        assert "0" in board and "8" in board

    def test_renders_x_and_o_pieces(self):
        info_state = list([0.0] * 27)
        # cell 0 -> X
        info_state[9 + 0] = 1.0
        # cell 4 -> O
        info_state[18 + 4] = 1.0
        # remaining empties
        for idx in (1, 2, 3, 5, 6, 7, 8):
            info_state[idx] = 1.0
        board = render_tic_tac_toe_board(info_state)
        assert "X" in board
        assert "O" in board

    def test_empty_string_when_info_state_wrong_shape(self):
        assert render_tic_tac_toe_board([0.0] * 5) == ""


class TestOpenSpielSessionFactory:
    def test_initial_message_includes_legal_actions(self):
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client, game_name="tic_tac_toe")

        session = factory.create(episode_id="ttt-000000")
        try:
            initial = session.initial_messages()
        finally:
            session.close()

        assert len(initial) == 1
        content = initial[0]["content"]
        assert "tic_tac_toe" in content
        assert "Legal actions: [0, 1, 2, 3, 4, 5, 6, 7, 8]" in content
        assert "Board:" in content

    def test_play_move_tool_call_invokes_step_with_action_id(self):
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client)

        session = factory.create(episode_id="ttt-000001")
        try:
            tool_result = session.call_tool("play_move", {"action_id": 4})
        finally:
            session.close()

        assert client.step_calls[0].action_id == 4
        assert client.step_calls[0].game_name == "tic_tac_toe"
        assert tool_result.done is True
        assert tool_result.metadata["reward"] == 1.0
        assert "board" in tool_result.data

    def test_session_closes_underlying_client(self):
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client)

        session = factory.create()
        session.close()

        assert client.closed is True

    def test_reset_forwards_seed_and_episode_id(self):
        client = FakeOpenSpielClient()
        factory = OpenSpielSessionFactory(lambda: client)

        factory.create(seed=7, episode_id="ttt-000002")

        assert client.reset_calls[0]["seed"] == 7
        assert client.reset_calls[0]["episode_id"] == "ttt-000002"
