# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke test for the rollout collect pipeline on OpenSpiel TTT.

Wires together the real components that a teacher-rollout job would use:

- ``OpenSpielSessionFactory`` (the env-side adapter for ``tic_tac_toe``)
- ``MCPHarnessAdapter`` (the white-box ReAct harness from #471)
- ``CollectRunner`` + ``RolloutSerializer`` (this PR)

The only fake here is the OpenSpiel *client* itself — a deterministic
scripted game — so we can verify the full pipeline without Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.harness import HarnessRunLimits, MCPHarnessAdapter, ModelStepResult
from openenv.core.harness.collect import CollectRunner, RolloutSerializer
from openenv.core.llm_client import LLMResponse, ToolCall
from openspiel_env.harness import OpenSpielSessionFactory
from openspiel_env.models import OpenSpielAction, OpenSpielObservation, OpenSpielState


class ScriptedOpenSpielClient:
    """A tic_tac_toe client that wins after ``moves_until_win`` plays."""

    def __init__(self, moves_until_win: int = 1):
        self._moves_until_win = moves_until_win
        self._played = 0
        self.closed = False

    def reset(self, **kwargs: Any) -> StepResult[OpenSpielObservation]:
        self._played = 0
        empty_board = [1.0] * 9 + [0.0] * 18
        return StepResult(
            observation=OpenSpielObservation(
                info_state=empty_board,
                legal_actions=list(range(9)),
                current_player_id=0,
                game_phase="playing",
            ),
            reward=0.0,
            done=False,
        )

    def step(self, action: OpenSpielAction) -> StepResult[OpenSpielObservation]:
        self._played += 1
        done = self._played >= self._moves_until_win
        reward = 1.0 if done else 0.0
        info_state = [0.0] * 27
        # mark cell 0 as X just to have a non-empty board
        info_state[9] = 1.0
        for i in range(1, 9):
            info_state[i] = 1.0
        return StepResult(
            observation=OpenSpielObservation(
                info_state=info_state,
                legal_actions=[] if done else list(range(1, 9)),
                current_player_id=0,
                game_phase="terminal" if done else "playing",
            ),
            reward=reward,
            done=done,
        )

    def state(self) -> OpenSpielState:
        return OpenSpielState(step_count=self._played, game_name="tic_tac_toe")

    def close(self) -> None:
        self.closed = True


def _teacher_model_step(messages, tools, sampling):
    """Deterministic teacher: always call play_move with the first legal action.

    Stands in for a hosted LLM (e.g. gpt-5-mini). Real teachers will parse
    the prompt to pick an action_id.
    """
    del messages, tools, sampling
    return ModelStepResult(
        response=LLMResponse(
            content="Playing the first legal cell.",
            tool_calls=[
                ToolCall(
                    id="teacher-0",
                    name="play_move",
                    args={"action_id": 0},
                ),
            ],
        ),
    )


def test_collect_writes_trl_ready_jsonl_dataset(tmp_path: Path):
    factory = OpenSpielSessionFactory(
        lambda: ScriptedOpenSpielClient(moves_until_win=1),
        game_name="tic_tac_toe",
    )
    runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=RolloutSerializer(tmp_path),
        limits=HarnessRunLimits(max_turns=4),
    )

    result = runner.run(
        model_step=_teacher_model_step,
        num_episodes=5,
        episode_id_prefix="ttt",
    )

    assert result.num_collected == 5
    assert result.num_skipped == 0
    assert result.success_rate == 1.0

    lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])

    # Schema sanity — enough to feed TRL SFTTrainer directly.
    assert first["episode_id"] == "ttt-000000"
    assert first["reward"] == 1.0
    assert first["done"] is True
    assert any(m["role"] == "user" for m in first["messages"])
    assert any(m["role"] == "assistant" for m in first["messages"])
    assert any(m["role"] == "tool" for m in first["messages"])
    assert first["tool_trace"][0]["tool_name"] == "play_move"
    assert first["tool_trace"][0]["arguments"] == {"action_id": 0}


def test_collect_resume_skips_prior_episodes(tmp_path: Path):
    factory = OpenSpielSessionFactory(
        lambda: ScriptedOpenSpielClient(moves_until_win=1),
    )
    serializer = RolloutSerializer(tmp_path)

    first_runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=serializer,
        limits=HarnessRunLimits(max_turns=4),
    )
    first = first_runner.run(model_step=_teacher_model_step, num_episodes=2)
    assert first.num_collected == 2

    second_runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=RolloutSerializer(tmp_path),
        limits=HarnessRunLimits(max_turns=4),
    )
    second = second_runner.run(model_step=_teacher_model_step, num_episodes=4)

    assert second.num_collected == 2
    assert second.num_skipped == 2
    assert len((tmp_path / "results.jsonl").read_text().strip().splitlines()) == 4


def test_collect_should_keep_filters_losing_rollouts(tmp_path: Path):
    # Alternate winning/losing clients on each create().
    wins = [True, False, True, True]
    client_iter = iter(wins)

    def client_factory():
        winning = next(client_iter)
        return ScriptedOpenSpielClient(moves_until_win=1 if winning else 10)

    factory = OpenSpielSessionFactory(client_factory)
    runner = CollectRunner(
        session_factory=factory,
        harness_adapter=MCPHarnessAdapter(),
        serializer=RolloutSerializer(tmp_path),
        # 3 turns is enough for the winning cases but not the losing one.
        limits=HarnessRunLimits(max_turns=3),
    )

    result = runner.run(
        model_step=_teacher_model_step,
        num_episodes=4,
        should_keep=lambda record: record.reward > 0.0,
    )

    assert result.num_collected == 3
    assert result.num_dropped == 1
    lines = (tmp_path / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert json.loads(line)["reward"] == 1.0
