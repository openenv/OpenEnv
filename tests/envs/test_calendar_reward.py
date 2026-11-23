"""calendar_env must not reward a tool call that failed."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

CALENDAR_SERVER = (
    Path(__file__).resolve().parents[2] / "envs" / "calendar_env" / "server"
)
if str(CALENDAR_SERVER) not in sys.path:
    sys.path.insert(0, str(CALENDAR_SERVER))

from openenv_wrapper.mcp_env_environment import MCPEnvironment  # noqa: E402

MISSING_PARAM = "The parameter 'calendarId' is required and cannot be empty."


@pytest.mark.parametrize(
    ("result", "reward"),
    [
        ({"text": MISSING_PARAM, "isError": True}, -0.5),
        ({"text": "Calendar not found", "isError": True, "status_code": 404}, -0.5),
        ({"text": "{}", "isError": False, "status_code": 200}, 1.0),
        ({"text": "{}", "isError": False}, 0.5),
    ],
)
def test_reward_follows_the_tool_result(result, reward):
    assert MCPEnvironment()._compute_reward("get_calendar", result) == reward
