"""Run an actual pinned environment with an explicitly labelled wire fault."""

import importlib
import json
import os
import time
from pathlib import Path


def record(event, **values):
    with Path(os.environ["BENCH_WITNESS"]).open("a") as stream:
        stream.write(
            json.dumps({"event": event, "monotonic": time.monotonic(), **values}) + "\n"
        )


module = importlib.import_module(os.environ["BENCH_APP"])
original = module.app
case = os.environ["BENCH_CASE"]
trigger = json.loads(os.environ["BENCH_TRIGGER"])


async def app(scope, receive, send):
    """Observe the real application; mutate only matching step responses."""
    if scope["type"] != "websocket":
        return await original(scope, receive, send)
    pending = None
    session = {"resets": 0, "steps": 0}

    async def observed_receive():
        nonlocal pending
        message = await receive()
        if message["type"] == "websocket.receive" and "text" in message:
            request = json.loads(message["text"])
            pending = request
            if request.get("type") == "reset":
                session["resets"] += 1
            if request.get("type") == "step":
                session["steps"] += 1
        return message

    async def observed_send(message):
        if (
            message["type"] != "websocket.send"
            or "text" not in message
            or pending is None
        ):
            return await send(message)
        response = json.loads(message["text"])
        operation = pending.get("type")
        matched = operation == "step" and pending.get("data") == trigger
        if operation in {"reset", "step", "state"}:
            record(
                "upstream_response",
                operation=operation,
                request=pending,
                response=response,
                session=dict(session),
            )
        if matched and response.get("type") == "observation":
            if case == "malformed_observation":
                key = os.environ["BENCH_OBSERVATION_KEY"]
                response["data"]["observation"][key] = {"exp1_invalid_type": True}
            elif case == "invalid_reward":
                response["data"]["reward"] = "exp1-invalid-reward"
            elif case == "hung_step":
                record(
                    "fault_activated",
                    case=case,
                    action=trigger,
                    upstream_completed=True,
                    session=dict(session),
                )
                return
            if case in {"malformed_observation", "invalid_reward"}:
                message = {**message, "text": json.dumps(response)}
                record(
                    "fault_activated",
                    case=case,
                    action=trigger,
                    downstream=response,
                    session=dict(session),
                )
        await send(message)

    await original(scope, observed_receive, observed_send)
