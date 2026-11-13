from llm_capacity_env.server.capacity_environment import CapacityEnvironment
from llm_capacity_env.server.engine import MODELS
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


def test_tools_controls_reward_and_error_accounting():
    env = CapacityEnvironment()
    try:
        env.reset(seed=11)
        result = env.step(ListToolsAction())
        assert {t.name for t in result.tools} == {
            "observe_demand_history",
            "reserve_capacity",
        }
        assert result.reward == 0
        assert env.state.step_count == 0
        bad = env.step(
            CallToolAction(tool_name="reserve_capacity", arguments={"reservations": {}})
        )
        assert bad.reward == 0 and env.state.step_count == 0
        # A successful decision emits its cost once; read-only calls cannot collect it again.
        good = env.step(
            CallToolAction(
                tool_name="reserve_capacity",
                arguments={"reservations": {m: [0] * 24 for m in MODELS}},
            )
        )
        assert -2 <= good.reward < 0
        assert env.state.step_count == 1
        view = env.step(
            CallToolAction(tool_name="observe_demand_history", arguments={})
        )
        assert view.reward == 0 and not view.done
        snapshot = env.state
        snapshot.step_count = 99
        assert env.state.step_count == 1
    finally:
        env.close()
