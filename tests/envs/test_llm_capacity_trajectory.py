import asyncio
import json

from llm_capacity_env.server.capacity_environment import CapacityEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


def action():
    return CallToolAction(
        tool_name="reserve_capacity",
        arguments={"reservations": {"ChatGPT": [0] * 24, "GPT-4": [0] * 24}},
    )


def test_server_emits_complete_trace_and_separate_reset_files(tmp_path):
    env = CapacityEnvironment(trace_dir=tmp_path)
    try:
        env.reset(seed=3, episode_id="../../outside")
        for _ in range(14):
            env.step(action())
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        rows = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(rows) == 15 and rows[0]["type"] == "reset"
        assert rows[0]["episode_id"] == "../../outside"
        assert all(row["action"]["tool_name"] == "reserve_capacity" for row in rows[1:])
        assert rows[-1]["terminal"] and rows[-1]["step_count"] == 14
        assert (
            abs(sum(row["reward"] for row in rows[1:]) + env.replay.total_cost / 1000)
            < 1e-10
        )
        assert all("observation" in row for row in rows)
        env.reset(seed=3)
        assert len(list(tmp_path.glob("*.jsonl"))) == 2
    finally:
        env.close()


def test_async_calls_are_recorded_without_reward_repetition(tmp_path):
    env = CapacityEnvironment(trace_dir=tmp_path)

    async def run():
        env.reset(seed=5)
        await env.step_async(action())
        await env.step_async(
            CallToolAction(tool_name="observe_demand_history", arguments={})
        )

    try:
        asyncio.run(run())
        rows = [
            json.loads(line)
            for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()
        ]
        assert len(rows) == 3
        assert rows[1]["reward"] < 0 and rows[2]["reward"] == 0
        assert rows[1]["step_count"] == rows[2]["step_count"] == 1
    finally:
        env.close()
