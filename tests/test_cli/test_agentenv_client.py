# SPDX-License-Identifier: BSD-3-Clause

"""Generated AgentEnv client discovery over real HTTP and WebSocket transports."""

import asyncio
import importlib.util
import socket

import pytest
import uvicorn
from openenv.cli.importers.agentenv import AgentEnvImporter, detect_agentenv_bundles
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
)
from openenv.core.env_server.types import Observation, State


class ChangingToolsEpisode(Environment):
    """Minimal episode with task-local tools that can also change during a turn."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, *, index=0, **kwargs):
        self.tool = ("echo", "other")[index]
        return Observation()

    def step(self, action, **kwargs):
        if isinstance(action, ListToolsAction):
            return ListToolsObservation(
                tools=[
                    Tool(
                        name=name,
                        description="Episode tool",
                        input_schema={"type": "object"},
                    )
                    for name in (self.tool, "end_turn")
                ]
            )
        assert isinstance(action, CallToolAction)
        assert action.tool_name in (self.tool, "end_turn")
        self.tool = "next"
        return CallToolObservation(tool_name=action.tool_name, result=action.arguments)

    @property
    def state(self):
        return State()


@pytest.fixture
async def generated_client(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "tasks").mkdir(parents=True)
    (bundle / "tasks" / "task.json").write_text(
        '[{"id": "turn", "type": "prompt_agent"}]'
    )
    destination = tmp_path / "desk_env"
    AgentEnvImporter().generate(
        source=bundle,
        destination=destination,
        env_name="desk_env",
        detected=detect_agentenv_bundles(bundle)[0],
    )
    spec = importlib.util.spec_from_file_location(
        "generated_agentenv_client", destination / "client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = create_app(
        ChangingToolsEpisode,
        CallToolAction,
        CallToolObservation,
        reset_observation_cls=Observation,
        max_concurrent_envs=8,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", lifespan="off")
        )
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            async with module.DeskEnv(base_url=f"http://127.0.0.1:{port}") as client:
                yield client
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=10)


def names(tools):
    return [tool.name for tool in tools]


async def test_generated_client_discovers_new_task_after_reset(generated_client):
    client = generated_client
    await client.reset(split="all", index=0)
    assert names(await client.list_tools()) == ["echo", "end_turn"]
    await client.call_tool("end_turn", response="Done.")
    await client.reset(split="all", index=1)
    assert names(await client.list_tools()) == ["other", "end_turn"]
    assert await client.has_tool("other")
    assert not await client.has_tool("echo")


@pytest.mark.parametrize("transition", ["end_turn", "step", "tool"])
async def test_generated_client_discovers_changed_turn_tools(
    generated_client, transition
):
    client = generated_client
    await client.reset(split="all", index=0)
    assert names(await client.list_tools()) == ["echo", "end_turn"]
    if transition == "step":
        await client.step(
            CallToolAction(tool_name="end_turn", arguments={"response": "Done."})
        )
    else:
        name = "end_turn" if transition == "end_turn" else "echo"
        result = await client.call_tool(
            name, response="Done.", use_cache="tool argument"
        )
        assert result["result"] == {"response": "Done.", "use_cache": "tool argument"}
    # Explicit cache requests retain the inherited API, including positional use.
    assert names(await client.list_tools(use_cache=True)) == ["echo", "end_turn"]
    assert names(await client.list_tools(True)) == ["echo", "end_turn"]
    assert names(await client.list_tools()) == ["next", "end_turn"]
    assert names(await client.list_tools(use_cache=False)) == ["next", "end_turn"]
    assert (await client.get_tool("next")).name == "next"
