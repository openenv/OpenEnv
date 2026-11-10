"""Offline input and wire-fault checks; never import a subject environment."""

import asyncio
import gzip
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


HERE = Path(__file__).parent
TRIGGER = {"expression": "1 + 2"}
OBSERVATION = {
    "type": "observation",
    "data": {"observation": {"target": 3}, "reward": 1, "done": False},
}


def load_server(monkeypatch, tmp_path, original, case):
    source = ModuleType("exp1_fake_source")
    source.app = original
    monkeypatch.setitem(sys.modules, source.__name__, source)
    for key, value in {
        "BENCH_APP": source.__name__,
        "BENCH_CASE": case,
        "BENCH_TRIGGER": json.dumps(TRIGGER),
        "BENCH_OBSERVATION_KEY": "target",
        "BENCH_WITNESS": str(tmp_path / "witness.jsonl"),
    }.items():
        monkeypatch.setenv(key, value)
    return runpy.run_path(str(HERE / "server.py"))["app"]


@pytest.mark.parametrize("cleanup", ["disconnect", "cancel"])
@pytest.mark.asyncio
async def test_hung_step_withholds_response_and_keeps_receiving(
    monkeypatch, tmp_path, cleanup
):
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    completed, cleaned = asyncio.Event(), asyncio.Event()
    sleeps = []

    async def immediate_sleep(delay):
        # Exercise eventual forwarding on the old wrapper without waiting 30s.
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)

    async def original(scope, receive, send):
        try:
            while True:
                message = await receive()
                if message["type"] == "websocket.disconnect":
                    return
                await send({"type": "websocket.send", "text": json.dumps(OBSERVATION)})
                completed.set()
        finally:
            cleaned.set()

    app = load_server(monkeypatch, tmp_path, original, "hung_step")
    request = {"type": "step", "data": TRIGGER}
    await incoming.put({"type": "websocket.receive", "text": json.dumps(request)})
    task = asyncio.create_task(app({"type": "websocket"}, incoming.get, outgoing.put))
    try:
        await asyncio.wait_for(completed.wait(), timeout=1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(outgoing.get(), timeout=0.01)
        assert sleeps == []
        # A nonmatching step still completes on the same session.
        await incoming.put(
            {"type": "websocket.receive", "text": '{"type":"step","data":{}}'}
        )
        response = await asyncio.wait_for(outgoing.get(), timeout=1)
        assert json.loads(response["text"]) == OBSERVATION
        if cleanup == "disconnect":
            await incoming.put({"type": "websocket.disconnect"})
            await asyncio.wait_for(task, timeout=1)
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert cleaned.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    events = [
        json.loads(line)
        for line in (tmp_path / "witness.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "upstream_response",
        "fault_activated",
        "upstream_response",
    ]
    assert events[0]["request"] == request
    assert events[0]["response"] == OBSERVATION
    assert events[1]["upstream_completed"] is True
    assert events[1]["action"] == TRIGGER
    assert events[1]["session"] == {"resets": 0, "steps": 1}


@pytest.mark.parametrize(
    "case", ["healthy", "hung_step", "invalid_reward", "malformed_observation"]
)
@pytest.mark.asyncio
async def test_wire_faults_preserve_other_responses_and_original_witness(
    monkeypatch, tmp_path, case
):
    requests = [
        {"type": "reset"},
        {"type": "step", "data": {}},
        {"type": "state"},
        {"type": "step", "data": TRIGGER},
        {"type": "step", "data": TRIGGER},
    ]
    responses = [
        OBSERVATION,
        OBSERVATION,
        {"type": "state", "data": {}},
        {"type": "error"},
        OBSERVATION,
    ]
    incoming = iter(
        {"type": "websocket.receive", "text": json.dumps(r)} for r in requests
    )
    sent = []

    async def receive():
        return next(incoming)

    async def send(message):
        sent.append(message)

    async def original(scope, receive, send):
        await send({"type": "websocket.accept"})
        for response in responses:
            await receive()
            await send({"type": "websocket.send", "text": json.dumps(response)})
        await send({"type": "websocket.send", "bytes": b"untouched"})
        await send({"type": "websocket.close"})

    app = load_server(monkeypatch, tmp_path, original, case)
    await app({"type": "websocket"}, receive, send)
    expected = json.loads(json.dumps(responses))
    if case == "hung_step":
        expected.pop()
    elif case == "invalid_reward":
        expected[-1]["data"]["reward"] = "exp1-invalid-reward"
    elif case == "malformed_observation":
        expected[-1]["data"]["observation"]["target"] = {"exp1_invalid_type": True}
    assert sent == [
        {"type": "websocket.accept"},
        *({"type": "websocket.send", "text": json.dumps(r)} for r in expected),
        {"type": "websocket.send", "bytes": b"untouched"},
        {"type": "websocket.close"},
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "witness.jsonl").read_text().splitlines()
    ]
    upstream = [e for e in events if e["event"] == "upstream_response"]
    assert [e["request"] for e in upstream] == requests
    assert [e["response"] for e in upstream] == responses
    assert sum(e["event"] == "fault_activated" for e in events) == (case != "healthy")


@pytest.mark.parametrize("corruption", [None, "compressed", "uncompressed"])
def test_prepare_gzip_pack_layout_and_hash_rejection(monkeypatch, tmp_path, corruption):
    body = b'{"task_id":"fake-train-task","split":"train"}\n'
    raw = gzip.compress(body, mtime=0)
    manifest = json.dumps(
        {"sha256": hashlib.sha256(body).hexdigest(), "tasks": 1}
    ).encode()
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_bytes(manifest)
    bucket = tmp_path / "bucket.gz"
    bucket.write_bytes(b"corrupt download" if corruption == "compressed" else raw)
    pins = {
        "environments": [
            {
                "config": {"package": "fake_portsim"},
                "files": [
                    {
                        "path": "tasks/dock-v1-train/manifest.json",
                        "url": source_manifest.as_uri(),
                        "sha256": hashlib.sha256(manifest).hexdigest(),
                    }
                ],
                "bucket": {
                    "url": bucket.as_uri(),
                    "compressed_sha256": hashlib.sha256(raw).hexdigest(),
                    "uncompressed_sha256": "0" * 64
                    if corruption == "uncompressed"
                    else hashlib.sha256(body).hexdigest(),
                },
            }
        ]
    }
    script = tmp_path / "prepare.py"
    script.write_bytes((HERE / "prepare.py").read_bytes())
    (tmp_path / "source-pins.json").write_text(json.dumps(pins))
    output = tmp_path / "output"
    pack = output / "portsim-train"
    pack.mkdir(parents=True)
    (pack / "tasks.jsonl.gz").write_bytes(b"corrupt cache")
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])

    if corruption:
        with pytest.raises(ValueError, match="hash mismatch|content does not match"):
            runpy.run_path(str(script), run_name="__main__")
        assert not (pack / "manifest.json").exists()
        assert not (output / "config.json").exists()
        return

    runpy.run_path(str(script), run_name="__main__")
    assert {p.name for p in pack.iterdir()} == {"tasks.jsonl.gz", "manifest.json"}
    assert (pack / "tasks.jsonl.gz").read_bytes() == raw
    assert (pack / "manifest.json").read_bytes() == manifest
    config = json.loads((output / "config.json").read_text())[0]
    assert config["root"] == str(output / "sources" / "fake_portsim")
    assert config["environment"] == {"BERTH_TASKS_DIR": str(pack)}
    # Fully verified inputs work offline without the original file URLs.
    source_manifest.unlink()
    bucket.unlink()
    runpy.run_path(str(script), run_name="__main__")
