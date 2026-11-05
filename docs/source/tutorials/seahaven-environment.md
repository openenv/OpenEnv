# Build a Tool-Use Environment with Seahaven

A tool-use agent needs something stateful to act on: a CRM it can add contacts to, a payments API it can refund through, an issue tracker it can close tickets in. A mock that returns canned responses doesn't keep the agent's writes, and a real system can't be reset between episodes. [Seahaven](https://github.com/Kiln-AI/Seahaven) is a Python framework for building *worlds*, working copies of a tool or API backed by SQLite, and `seahaven serve` runs a world as an OpenEnv environment:

- Every WebSocket session gets a private instance of the world.
- `reset` starts that instance from a frozen starting state, a *fixture*, with a seed for the ids and randomness the world generates.
- `state()` returns a log of every row the episode inserted, updated or deleted, which you grade the episode on.

You write the world's tables, as SQL, and its tools, as Python functions. This tutorial builds a small CRM world, serves it, drives an episode with OpenEnv's `GenericEnvClient`, and grades the episode from the change log.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.14 or later for the world, which is what Seahaven requires.
- `openenv` for the client. The world's server runs on the `openenv` 0.5.x that Seahaven's `serve` extra pins. The client code below was checked with `openenv` 0.5.0 and 0.8.0.

This tutorial was tested against Seahaven 0.5.0 and 0.6.0.

## 1. Scaffold a World

```bash
uvx seahaven new crm_world
cd crm_world && uv sync
```

`seahaven new` writes a complete project. The files this tutorial works with are:

```
crm_world/
  src/crm_world/
    world.py                 # the World every tool registers on
    errors.py                # the world's error classes
    schema/001_items.sql     # the tables, as SQLite DDL
    tools/__init__.py        # imports every tool module
    tools/items.py           # the tools
    openenv_app.py           # the OpenEnv app: seahaven.openenv.app(world)
  fixtures_src/generate.py   # how each fixture is built
  fixtures/                  # frozen fixtures
  tests/test_items.py        # tests for the scaffold's tools
```

The scaffold ships an `items` table and two tools for it. Remove them, since the CRM replaces them:

```bash
rm src/crm_world/schema/001_items.sql src/crm_world/tools/items.py tests/test_items.py
```

## 2. Define a Table

Save the table as `src/crm_world/schema/001_contacts.sql`:

```sql
CREATE TABLE contacts (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    stage TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;
```

Every instance of the world gets its own SQLite database built from these files. Every table needs a primary key, because the change log in step 7 identifies each row by it.

## 3. Write the Tools

A tool is a plain function whose first parameter is the instance context, `ctx`. Seahaven builds each tool's JSON schema from its signature, and its description from the docstring. Save these three tools in `src/crm_world/tools/contacts.py`:

```python
from typing import Any, Literal

import seahaven

from crm_world.errors import NotFound
from crm_world.world import world

Stage = Literal["lead", "qualified", "customer", "lost"]


@world.tool
def list_contacts(ctx: seahaven.Ctx, stage: Stage | None = None) -> list[dict[str, Any]]:
    """List contacts, optionally only those at one pipeline stage."""
    if stage is None:
        return ctx.db.rows("SELECT * FROM contacts ORDER BY email")
    return ctx.db.rows("SELECT * FROM contacts WHERE stage = ? ORDER BY email", stage)


@world.tool
def create_lead(ctx: seahaven.Ctx, email: str) -> dict[str, Any]:
    """Add a contact to the pipeline as a new lead."""
    lead = {"id": ctx.ids.uuid(), "email": email, "stage": "lead", "updated_at": ctx.clock.iso()}
    ctx.db.execute(
        "INSERT INTO contacts (id, email, stage, updated_at) VALUES (?, ?, ?, ?)",
        lead["id"],
        lead["email"],
        lead["stage"],
        lead["updated_at"],
    )
    return lead


@world.tool
def set_stage(ctx: seahaven.Ctx, contact_id: str, stage: Stage) -> dict[str, Any]:
    """Move a contact to another pipeline stage."""
    ctx.db.execute(
        "UPDATE contacts SET stage = ?, updated_at = ? WHERE id = ?",
        stage,
        ctx.clock.iso(),
        contact_id,
    )
    contact = ctx.db.one("SELECT * FROM contacts WHERE id = ?", contact_id)
    if contact is None:
        raise NotFound("contact", contact_id)
    return contact
```

`ctx.ids` and `ctx.clock` are the instance's seeded id stream and its clock. Using them instead of `uuid.uuid4()` and `datetime.now()` is what lets the same seed replay the same run. `NotFound` is one of the error classes the scaffold put in `errors.py`; the agent receives it as data on the observation, and the episode carries on.

Register the new module in `src/crm_world/tools/__init__.py`:

```python
from crm_world.tools import contacts

__all__ = ["contacts"]
```

Then check the world:

```bash
uv run seahaven check
```

`seahaven check` prints nothing when the world is clean, and names the fix for each problem it finds, such as a tool module nothing imports or a table without a primary key.

## 4. Freeze a Fixture

A fixture is the frozen database every episode starts from. `fixtures_src/generate.py` holds one builder function per fixture. Add a builder for three leads, after the scaffold's `empty` builder:

```python
def three_leads(inst: seahaven.Instance) -> None:
    """`three_leads`: three contacts, all at the `lead` stage."""
    for email in ["ada@example.com", "alan@example.com", "barbara@example.com"]:
        inst.call("create_lead", email=email)
```

and register it in the `BUILDERS` and `DESCRIPTIONS` tables below the builders:

```python
BUILDERS = {"empty": empty, "three_leads": three_leads}
DESCRIPTIONS = {
    "empty": "The schema with no rows. Start here to write a history.",
    "three_leads": "Three contacts, all new leads.",
}
```

Then build it:

```bash
uv run python -c "from fixtures_src.generate import build; build('three_leads')"
```

This writes `fixtures/three_leads/`. The fixture's rows are starting state, so they are not in an episode's change log.

## 5. Serve It

```bash
uv run --extra serve seahaven serve
```

The server listens on `http://127.0.0.1:8000`. Its WebSocket endpoint, `/ws`, is the OpenEnv session protocol, and every connection gets its own instance. A web console at `http://127.0.0.1:8000/console` lets you call tools and read the state by hand.

## 6. Run an Episode

In a second terminal, in any Python environment with `openenv` installed, save this as `episode.py` and run it. The `call` helper stands in for the agent: in a real rollout, a model picks each tool call from the tool list.

```python
import json

from openenv import GenericEnvClient
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


def call(env, tool_name, **arguments):
    """Call one tool, and return its result or raise with the world's error message."""
    observation = env.step(CallToolAction(tool_name=tool_name, arguments=arguments)).observation
    if observation["error"] is not None:
        raise RuntimeError(observation["error"]["message"])
    return observation["result"]


with GenericEnvClient(base_url="http://127.0.0.1:8000") as env:
    # A private copy of the fixture. The seed fixes the ids the world mints, and
    # the tick clock fixes its timestamps, so this episode replays exactly.
    env.reset(fixture="three_leads", seed=7, clock_mode="tick")

    # The tool list, with a JSON schema per tool, to hand to a model.
    tools = env.step(ListToolsAction()).observation["tools"]
    print([tool["name"] for tool in tools])

    # The agent's task: add grace@example.com as a lead and qualify her.
    print([contact["email"] for contact in call(env, "list_contacts", stage="lead")])
    grace = call(env, "create_lead", email="grace@example.com")
    call(env, "set_stage", contact_id=grace["id"], stage="qualified")

    document = env.state()

with open("state.json", "w") as f:
    json.dump(document, f, indent=2)
```

```
['list_contacts', 'create_lead', 'set_stage']
['ada@example.com', 'alan@example.com', 'barbara@example.com']
```

`reset` takes Seahaven's own arguments: `fixture`, `seed`, `clock_mode`, and others such as `now`, to start the instance's clock at a given instant. A tool that fails, for example `set_stage` with an unknown id, returns an observation whose `error` is set and whose `metadata["seahaven_error"]` carries the world's error code; the session stays open.

## 7. Grade From the Change Log

`state()` returns Seahaven's *state document*: the fixture, seed and clock the episode ran with, and under `state.db.log`, one record per row each call changed. This episode's log has two records:

```json
[
  {"i": 1, "world": "main", "table": "contacts", "op": "insert",
   "key": {"id": "2074607d-1513-4ef8-b170-40a04fb82cda"},
   "before": null,
   "after": {"id": "2074607d-1513-4ef8-b170-40a04fb82cda", "email": "grace@example.com",
             "stage": "lead", "updated_at": "2026-01-01T00:00:02.000Z"}},
  {"i": 2, "world": "main", "table": "contacts", "op": "update",
   "key": {"id": "2074607d-1513-4ef8-b170-40a04fb82cda"},
   "before": {"stage": "lead", "updated_at": "2026-01-01T00:00:02.000Z"},
   "after": {"stage": "qualified", "updated_at": "2026-01-01T00:00:03.000Z"}}
]
```

`i` is the call that made the change (`list_contacts` was call 0 and changed nothing). An insert carries the whole row, and an update carries only the columns it changed. To grade, replay the log into the final value of each row the episode touched, and check those rows against the task. Save this as `grade.py`:

```python
import json


def grade(document):
    """1.0 if the episode added grace@example.com as a qualified lead and changed nothing else."""
    rows = {}  # the final value of each row the episode changed, by world, table and key
    for record in document["state"]["db"]["log"]:
        key = (record["world"], record["table"], tuple(record["key"].values()))
        if record["op"] == "delete":
            rows[key] = None
        else:
            rows[key] = {**(rows.get(key) or {}), **record["after"]}

    if len(rows) != 1:
        return 0.0  # the agent changed nothing, or a contact it was not asked to
    (row,) = rows.values()
    if row and row.get("email") == "grace@example.com" and row.get("stage") == "qualified":
        return 1.0
    return 0.0


with open("state.json") as f:
    print("reward:", grade(json.load(f)))
```

```
reward: 1.0
```

The grader reads only what changed, so it doesn't depend on how the agent got there, and the same world and fixture can serve many tasks, each with its own grader. Seahaven's [state document reference](https://github.com/Kiln-AI/Seahaven/blob/v0.6.0/src/seahaven/docs/state.md) defines the log and how to fold it when a row changes more than once.

> [!NOTE]
> A Seahaven environment doesn't compute rewards: every observation has `reward=None` and `done=False`, because whether a final state is correct depends on the task, which the world doesn't know. This differs from the OpenEnv convention of computing the reward inside the environment ([Reward Design](../guides/rewards)). Grade from `state()` in the code that drives the episode, as above, and bound each episode by turns or tool calls. OpenEnv's harness rollout helpers (`CollectRunner` and `build_harness_rollout_func`) require a reward produced by the environment, so they raise `ValueError` on a Seahaven episode.

## A Real Example: Stripe World

[Stripe World](https://github.com/Kiln-AI/stripe_world) is a Seahaven world of Stripe's Billing and Payments API: 25 tables, 155 API operations, a billing engine for subscriptions and invoices, and eight of the ten tools on Stripe's MCP server. It is built and served the same way as the CRM above, and is also hosted as a [Hugging Face Space](https://huggingface.co/spaces/scosman/stripe_world). Its README has a script that runs 100 graded episodes against one server.

## Learn More

- Seahaven's [serving guide](https://github.com/Kiln-AI/Seahaven/blob/v0.6.0/src/seahaven/docs/serving_and_openenv.md): `reset` arguments, `SeahavenClient`, the wire protocol, and publishing a world to a hub as a Docker image or Space.
- Seahaven's [authoring guide](https://github.com/Kiln-AI/Seahaven/blob/v0.6.0/src/seahaven/docs/authoring.md), [fixtures guide](https://github.com/Kiln-AI/Seahaven/blob/v0.6.0/src/seahaven/docs/db_schema_and_fixtures.md) and [testing guide](https://github.com/Kiln-AI/Seahaven/blob/v0.6.0/src/seahaven/docs/testing.md), for tests that call a world's tools in process with pytest.
- [MCP Environments](mcp-environment) for the `CallToolAction` and `ListToolsAction` types used above.
- [Your First Environment](../guides/first-environment) to write an OpenEnv environment directly, with rewards computed on the server.
