"""Real registered MCP tools and bounded task metadata use existing APIs."""

from starlette.testclient import TestClient
from test_served_probe import fixture


def test_real_tools_share_the_replayed_instance_and_tasks_allow_bounded_listing(
    monkeypatch,
):
    token = "fixture-tool-test-token-" + "0" * 32
    monkeypatch.setenv("OPENENV_VALIDATION_TOKEN", token)
    with TestClient(fixture.make_app()) as client:
        counts = {}
        for split in ("train", "test"):
            counts[split] = client.post(
                "/validation_probe/num_tasks", json={"split": split}
            ).json()["num_tasks"]
            preview = client.post(
                "/validation_probe/tasks", json={"split": split}
            ).json()["tasks"]
            assert len(preview) == 1 < counts[split]
        assert counts == {"train": 4, "test": 2}
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "validation_open",
                    "data": {"schema_version": 1, "token": token},
                }
            )
            capability = ws.receive_json()["data"]["capability"]

            def rpc(method, params):
                ws.send_json(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": method,
                            "params": params,
                        },
                    }
                )
                return ws.receive_json()["data"]

            tools = rpc("tools/list", {})["result"]["tools"]
            assert {tool["name"] for tool in tools} == {"increment", "read_counter"}
            assert (
                rpc("tools/call", {"name": "read_counter", "arguments": {}})["result"]
                == 0
            )
            result = rpc(
                "tools/call", {"name": "increment", "arguments": {"amount": 2}}
            )
            assert result["result"]["counter"] == 2
            assert (
                rpc("tools/call", {"name": "read_counter", "arguments": {}})["result"]
                == 2
            )
            ws.send_json(
                {"type": "reset", "data": {"seed": 1, "episode_id": "after-tools"}}
            )
            assert ws.receive_json()["data"]["observation"]["counter"] == 0
            ws.send_json({"type": "state"})
            assert ws.receive_json()["data"]["step_count"] == 0
            ws.send_json(
                {
                    "type": "validation_read",
                    "data": {"schema_version": 1, "capability": capability},
                }
            )
            telemetry = ws.receive_json()["data"]
            assert telemetry["seed"] == {
                "requested": True,
                "value": 1,
                "accepted": True,
            }
            assert [row["operation"] for row in telemetry["trajectory"]["records"]] == [
                "reset",
                "state",
            ]
            assert telemetry["trajectory"]["complete"] is True


def test_non_mcp_subject_explicitly_rejects_tool_discovery(monkeypatch):
    from test_session_telemetry_protocol import app_for, SessionEnv

    class NonMCPEnv(SessionEnv):
        def __init__(self):
            super().__init__()
            self.mcp_server = None

    with TestClient(app_for(monkeypatch, NonMCPEnv)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "mcp",
                    "data": {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {},
                    },
                }
            )
            reply = ws.receive_json()
            assert reply["type"] == "mcp"
            assert reply["data"]["error"]["code"] == -32601
