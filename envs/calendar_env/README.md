---
title: Calendar Environment Server
emoji: 📅
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8004
base_path: /docs
tags:
  - openenv
---
# Calendar Environment

A simulated Google Calendar with multiple users, calendars and events, exposed as 37 MCP tools through the OpenEnv `reset`/`step`/`state` interface. Each episode runs against a SQLite database that you seed with SQL, and task completion is checked with SQL queries on that database.

## Server Setup

### Docker (Recommended)

```bash
cd envs/calendar_env
docker build -t calendar-env:latest .
docker run --rm -p 8004:8004 calendar-env:latest
curl http://localhost:8004/health   # {"status":"healthy","service":"calendar-env"}
```

### Without Docker

```bash
cd envs/calendar_env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8004
```

The tools call the server's own REST API at `http://localhost:$API_PORT` (default `8004`). If you serve on another port, set `API_PORT` to match.

## Quick Start

`CalendarEnv` is an HTTP client. Every request carries the database ID (`x-database-id`) and the acting user's access token (`x-access-token`). The tokens are the `static_token` values of the seeded users.

```python
from calendar_env import CalendarEnv

# Alice's token from the seeded demo users (server/data/multi_user_sample.py), not a real credential.
ALICE_TOKEN = "ya29.A0ARrdaM-k9Vq7GzY2pL4mQf8sN1xT0bR3uHcJWv5yKzP6eF2.qwErTyUIopASDfGhJkLzXcVbNm12_34-56"

with CalendarEnv(base_url="http://localhost:8004", database_id="demo", access_token=ALICE_TOKEN) as env:
    # Reset the database and seed it with the built-in sample data (4 users)
    env.reset(sql_content=env.get_sample_sql())

    tools = env.list_tools()
    print(len(tools))  # 37

    obs = env.call_tool("create_event", {
        "calendarId": "alice-primary",
        "summary": "Team sync",
        "start": {"dateTime": "2026-10-09T10:00:00Z"},
        "end": {"dateTime": "2026-10-09T10:30:00Z"},
    })
    print(obs.success, obs.reward)  # True 1.0

    # Check the outcome with SQL
    state = env.state(verify_queries=["SELECT COUNT(*) AS n FROM events WHERE summary = 'Team sync'"])
    print(state["verification_results"][0]["result"])  # [{'n': 1}]
```

`reset(sql_content=...)` (or `reset_with_sql_file(path)`) resets the database to a clean state and runs the SQL. `step()` takes a `CalendarAction` with `action_type="ListToolsAction"` or `action_type="ToolCallAction"` plus `tool_name` and `arguments`. `list_tools()` and `call_tool()` wrap it.

## Tools

The tools follow the Google Calendar API v3:

| Group | Tools |
|-------|-------|
| Events | `list_events`, `get_event`, `create_event`, `update_event`, `patch_event`, `delete_event`, `move_event`, `quick_add_event`, `import_event`, `get_event_instances`, `watch_events` |
| Calendars | `create_calendar`, `get_calendar`, `update_calendar`, `patch_calendar`, `delete_calendar`, `clear_calendar` |
| Calendar list | `get_calendar_list`, `get_calendar_from_list`, `add_calendar_to_list`, `update_calendar_in_list`, `replace_calendar_in_list`, `remove_calendar_from_list`, `watch_calendar_list` |
| Access control | `list_acl_rules`, `get_acl_rule`, `insert_acl_rule`, `update_acl_rule`, `patch_acl_rule`, `delete_acl_rule`, `watch_acl` |
| Settings | `get_settings`, `list_settings`, `watch_settings` |
| Other | `get_colors`, `query_freebusy`, `get_user_by_email` |

`list_tools()` returns each tool's name, description and JSON input schema.

## Reward

Each step returns a shaped reward:

| Outcome | Reward |
|---------|--------|
| `ListToolsAction` | +0.1 |
| Tool call with a 2xx status | +1.0 |
| Tool call that succeeds without a status code | +0.5 |
| Tool result with an `error` key or a 4xx/5xx status | -0.5 |
| Unknown tool | -0.5 |
| Missing `tool_name`, invalid access token or tool execution failure | -1.0 |

Episodes never end on their own (`done` is always `False`). Task success is measured separately, by running SQL checks on the database through `state(verify_queries=[...])`.

## Scenario Benchmark

`client.py` also runs an LLM agent on a scenario and verifies the result. Set the model's API key (`llm_api_key` in the config, or `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`), then:

```bash
python client.py --scenario scenario_config.json
```

The results (tool calls, verifier outcomes) go to `response_output/`. The main fields of `scenario_config.json`:

- `llm_provider` (`openai`, `anthropic` or `google`), `llm_model`, `llm_api_key`
- `user_prompt`, `system_prompt`: the task and the agent's instructions
- `context`: request headers for the acting user, such as `x-access-token`
- `seed_database_file`: SQL file to seed the database
- `verifiers`: checks run after the agent finishes. `database_state` compares a SQL query result with an expected value (`equals`, `greater_than`, `less_than`, `contains`), `response_check` asks an LLM to judge a SQL result, and `tool_execution` checks that the expected tools were called.
- `expected_tools`: tools the agent should use

For interactive evaluation, see the [notebook](client_notebooks/OpenEnv_and_mcp_Single_Gym_Client_Meta_Turing.ipynb).
