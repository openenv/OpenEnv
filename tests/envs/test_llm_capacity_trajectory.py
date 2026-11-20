import asyncio
import importlib
import json
import os

import pytest
from llm_capacity_env.server.capacity_environment import CapacityEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


def action():
    return CallToolAction(
        tool_name="reserve_capacity",
        arguments={"reservations": {"ChatGPT": [0] * 24, "GPT-4": [0] * 24}},
    )


def test_server_does_not_enable_recording_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CAPACITY_TRACE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    import llm_capacity_env.server.app as app_module

    importlib.reload(app_module)
    assert "CAPACITY_TRACE_DIR" not in os.environ
    env = CapacityEnvironment()
    try:
        for _ in range(20):
            env.reset(seed=3)
            env.step(action())
        assert env._recorder.directory is None
        assert list(tmp_path.iterdir()) == []
    finally:
        env.close()


def test_recording_can_be_enabled_through_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPACITY_TRACE_DIR", str(tmp_path))
    env = CapacityEnvironment()
    try:
        env.reset(seed=3)
        env.step(action())
        assert len(list(tmp_path.glob("*.jsonl"))) == 1
    finally:
        env.close()


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
        assert [row["reward"] for row in rows[1:]] == [0] * 14
        assert env.replay.total_cost == pytest.approx(
            sum(
                row["observation"]["result"]["data"]["previous"]["modeled_cost"]
                for row in rows[1:]
            )
        )
        assert all("observation" in row for row in rows)
        env.reset(seed=3)
        assert len(list(tmp_path.glob("*.jsonl"))) == 2
    finally:
        env.close()


def test_async_calls_are_recorded_without_reward_repetition(tmp_path):
    env = CapacityEnvironment(trace_dir=tmp_path)

    async def run():
        view = env.reset(seed=5).metadata
        await env.step_async(
            CallToolAction(
                tool_name="reserve_capacity",
                arguments={"reservations": view["history"][-1]["tokens"]},
            )
        )
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
        assert rows[1]["reward"] != 0 and rows[2]["reward"] == 0
        assert rows[1]["step_count"] == rows[2]["step_count"] == 1
    finally:
        env.close()
