import pytest
from llm_capacity_env.server.capacity_environment import CapacityEnvironment
from llm_capacity_env.server.engine import EPISODE_DAYS, modeled_cost, MODELS
from openenv.core.env_server.mcp_types import CallToolAction


def test_mcp_reward_emitted_once_and_state_is_copied():
    env = CapacityEnvironment()
    try:
        env.reset(seed=11)
        invalid = env.step(
            CallToolAction(tool_name="reserve_capacity", arguments={"reservations": {}})
        )
        assert invalid.reward == 0 and env.state.step_count == 0
        # A previous-day plan uses only information already revealed to the agent.
        history = env.replay.observe()["history"]
        plan = history[-1]["tokens"]
        expected = env.replay.reserve(plan)["previous"]["reward"]
        env.reset(seed=11)
        good = env.step(
            CallToolAction(
                tool_name="reserve_capacity", arguments={"reservations": plan}
            )
        )
        assert good.error is None and good.reward == pytest.approx(expected)
        assert good.reward != 0 and env.state.step_count == 1
        view = env.step(
            CallToolAction(tool_name="observe_demand_history", arguments={})
        )
        assert view.reward == 0 and not view.done
        snapshot = env.state
        snapshot.step_count = 99
        assert env.state.step_count == 1
    finally:
        env.close()


def test_useful_causal_policy_outscores_not_committing_and_zero_reservations():
    env = CapacityEnvironment(split="test")
    try:
        view = env.reset(seed=17).metadata
        waiting_reward = sum(
            env.step(
                CallToolAction(tool_name="observe_demand_history", arguments={})
            ).reward
            for _ in range(50)
        )
        assert waiting_reward == 0 and env.replay.turn == 0 and not env.replay.done
        total_reward, unreserved_cost = 0.0, 0.0
        for _ in range(EPISODE_DAYS):
            # This baseline uses yesterday's demand; it cannot read future truth.
            result = env.step(
                CallToolAction(
                    tool_name="reserve_capacity",
                    arguments={"reservations": view["history"][-1]["tokens"]},
                )
            )
            assert result.error is None
            view = env.replay.observe()
            total_reward += result.reward
            unreserved_cost += modeled_cost(
                {m: [0] * 24 for m in MODELS}, view["previous"]["realized_tokens"]
            )
            assert (
                env.step(
                    CallToolAction(tool_name="observe_demand_history", arguments={})
                ).reward
                == 0
            )
        assert result.done and env.state.step_count == EPISODE_DAYS
        assert total_reward > waiting_reward
        assert total_reward == pytest.approx(
            (unreserved_cost - env.replay.total_cost) / 1000
        )
        env.reset(seed=17)
        zero_rewards = [
            env.step(
                CallToolAction(
                    tool_name="reserve_capacity",
                    arguments={"reservations": {m: [0] * 24 for m in MODELS}},
                )
            ).reward
            for _ in range(EPISODE_DAYS)
        ]
        assert zero_rewards == [0] * EPISODE_DAYS and env.replay.done
    finally:
        env.close()


def test_zero_reservations_complete_episode_with_zero_savings():
    env = CapacityEnvironment(split="test")
    try:
        env.reset(seed=17)
        for _ in range(EPISODE_DAYS):
            result = env.step(
                CallToolAction(
                    tool_name="reserve_capacity",
                    arguments={"reservations": {m: [0] * 24 for m in MODELS}},
                )
            )
            assert result.error is None and result.reward == 0
        assert result.done and env.replay.total_cost > 0
    finally:
        env.close()
