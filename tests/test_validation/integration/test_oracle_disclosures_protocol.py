"""Complete oracle disclosures through real HTTP and WebSocket responses."""

import importlib.util
import json
import shutil
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.validation.runner import run_validation
from openenv.validation.types import Level
from support.runtime import FakeRuntimeProvider

CHECKOUT = Path(__file__).resolve().parents[3]
FIXTURE = CHECKOUT / "tests/fixtures/validation/runtime/served_probe"
spec = importlib.util.spec_from_file_location(
    "oracle_review_fixture", FIXTURE / "app.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
CANARY = "oracle-review-private-answer-9a6b4920"
SCRIPT = '#!/bin/sh\necho "' + CANARY + '"\n'
COMMON = "from pathlib import Path"


class AlterWire:
    def __init__(self, app, mode, artifact):
        self.app, self.mode, self.artifact = app, mode, artifact

    async def __call__(self, scope, receive, send):
        async def output(message):
            if (
                scope["type"] == "http"
                and scope["path"] == "/schema"
                and self.mode
                in {
                    "action_schema",
                    "state_schema",
                    "reset_schema",
                    "bom_schema",
                    "bom_json_schema",
                }
            ):
                if message["type"] == "http.response.start":
                    message = {
                        **message,
                        "headers": [
                            (k, v)
                            for k, v in message["headers"]
                            if k != b"content-length"
                        ],
                    }
                elif message["type"] == "http.response.body" and message.get("body"):
                    data = json.loads(message["body"])
                    field, attribute = {
                        "action_schema": ("action", "description"),
                        "state_schema": ("state", "default"),
                        "reset_schema": ("reset", "default"),
                        "bom_schema": ("action", "description"),
                        "bom_json_schema": ("action", "default"),
                    }[self.mode]
                    data.setdefault(field, {})[attribute] = (
                        json.loads(self.artifact)
                        if self.mode == "bom_json_schema"
                        else self.artifact
                    )
                    raw = json.dumps(data)
                    if self.mode in {"bom_schema", "bom_json_schema"}:
                        raw = "\ufeff" + raw.replace("oracle", r"\u006fracle")
                    message = {**message, "body": raw.encode()}
            if (
                message["type"] == "websocket.send"
                and message.get("text")
                and self.mode
                in {"unicode_ws", "json_state", "common_line", "state_only"}
            ):
                data = json.loads(message["text"])
                if data.get("type") == "observation" and self.mode != "state_only":
                    obs = data["data"]["observation"]
                    obs.setdefault("metadata", {})["sample"] = (
                        json.loads(self.artifact)
                        if self.mode == "json_state"
                        else COMMON
                        if self.mode == "common_line"
                        else self.artifact
                    )
                if data.get("type") == "state" and self.mode == "state_only":
                    data["data"]["oracle_for_orchestration"] = self.artifact
                raw = json.dumps(data, separators=(",", ":"))
                if self.mode == "unicode_ws":
                    encoded = (
                        '"' + "".join("\\u%04x" % ord(c) for c in self.artifact) + '"'
                    )
                    raw = raw.replace(json.dumps(self.artifact), encoded)
                    assert json.loads(raw) == data
                message = {**message, "text": raw}
            await send(message)

        await self.app(scope, receive, output)


@contextmanager
def serve(app):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = "http://127.0.0.1:%d" % listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", lifespan="on")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while (
                not server.started and thread.is_alive() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert server.started
            yield url
        finally:
            server.should_exit = True
            thread.join(5)
            assert not thread.is_alive()


@pytest.mark.parametrize(
    "mode",
    [
        "http403",
        "http206",
        "redirect",
        "redirect_body",
        "directory",
        "action_schema",
        "state_schema",
        "reset_schema",
        "bom_schema",
        "bom_json_schema",
        "unicode_ws",
        "json_state",
        "common_line",
        "state_only",
    ],
)
def test_real_runner(mode, tmp_path):
    root = tmp_path / "package"
    shutil.copytree(FIXTURE, root)
    artifact = SCRIPT
    if mode in {"json_state", "bom_json_schema"}:
        artifact = json.dumps({"answer": CANARY}, indent=2) + "\n"
        manifest = (
            (root / "openenv.yaml")
            .read_text()
            .replace("    oracle:\n", "    set_state: true\n    oracle:\n")
            .replace("form: script", "form: injected_state")
        )
        (root / "openenv.yaml").write_text(manifest)
    if mode == "common_line":
        artifact = "#!/usr/bin/env python3\n" + COMMON + '\nprint("' + CANARY + '")\n'
    (root / "oracle/solve.sh").write_text(artifact)
    if mode == "directory":
        (root / "openenv.yaml").write_text(
            (root / "openenv.yaml")
            .read_text()
            .replace("location: oracle/solve.sh", "location: oracle")
        )
    wire_app = fixture.make_app("good")
    app = wire_app.application
    if mode in {"http403", "http206"}:
        app.add_api_route(
            "/oracle/solve.sh",
            lambda: PlainTextResponse(
                artifact, status_code=403 if mode == "http403" else 206
            ),
        )
    if mode == "redirect":
        app.add_api_route(
            "/oracle/solve.sh", lambda: RedirectResponse("/download/solve.sh")
        )
        app.add_api_route("/download/solve.sh", lambda: PlainTextResponse(artifact))
    if mode == "redirect_body":
        app.add_api_route(
            "/oracle/solve.sh",
            lambda: PlainTextResponse(
                artifact, status_code=302, headers={"Location": "/landing"}
            ),
        )
        app.add_api_route(
            "/landing", lambda: PlainTextResponse("not found", status_code=404)
        )
    if mode == "directory":
        app.mount("/oracle", StaticFiles(directory=root / "oracle"), name="oracle")
    with serve(AlterWire(wire_app, mode, artifact)) as url:
        provider = FakeRuntimeProvider()
        provider.subject.base_url = url
        bundle = tmp_path / "evidence"
        report = run_validation(
            root, max_level=Level.RUNTIME, provider=provider, artifacts_dir=bundle
        )
        results = {r.check_id: r for r in report.results}
        assert results["runtime.startup"].status.value == "pass", results[
            "runtime.startup"
        ]
        assert results["runtime.observation_schema"].status.value == "pass", results[
            "runtime.observation_schema"
        ]
        if mode in {"http403", "http206", "redirect", "directory"}:
            accessible = httpx.get(
                url + "/oracle/solve.sh", follow_redirects=True, trust_env=False
            )
            assert accessible.text == artifact
        if mode == "directory":
            assert httpx.get(url + "/oracle/", trust_env=False).status_code == 404
        if mode == "action_schema":
            assert (
                httpx.get(url + "/schema", trust_env=False).json()["action"][
                    "description"
                ]
                == artifact
            )
        if mode in {"action_schema", "state_schema", "reset_schema"}:
            saved = json.loads((bundle / "collector-evidence.json").read_text())
            field, attribute = {
                "action_schema": ("action", "description"),
                "state_schema": ("state", "default"),
                "reset_schema": ("reset", "default"),
            }[mode]
            assert saved["schema_response"][field][attribute] == artifact
        if mode in {"bom_schema", "bom_json_schema"}:
            from openenv.validation.graders import Subject
            from openenv.validation.graders.runtime.oracle import (
                OracleContainmentGrader,
            )
            from openenv.validation.runtime.collector import collect_runtime_evidence
            from openenv.validation.runtime.contracts import RuntimePlan
            from test_runtime_artifacts import rebuild

            raw = httpx.get(url + "/schema", trust_env=False).content
            assert raw.startswith(b"\xef\xbb\xbf")
            plan = RuntimePlan.model_validate_json(
                (root / "validation/runtime.json").read_text()
            )
            collected = collect_runtime_evidence(url, plan, episode_timeout_s=5)
            assert collected.failure_reason is None
            assert collected.schema_response_json.encode("utf-8") == raw
            original = Subject(root, report.manifest, None, None, bundle, collected)
            assert OracleContainmentGrader().run(original).status.value == "fail"
            saved = json.loads((bundle / "collector-evidence.json").read_text())
            assert saved["schema_response_available"] is True
            assert saved["schema_response_parse_failed"] is False
            assert saved["redacted"] is False
            field = "default" if mode == "bom_json_schema" else "description"
            expected = json.loads(artifact) if mode == "bom_json_schema" else artifact
            assert saved["schema_response"]["action"][field] == expected
            replay = Subject(root, report.manifest, None, None, bundle, rebuild(bundle))
            assert OracleContainmentGrader().run(replay).status.value == "fail"
        if mode == "redirect_body":
            probe = json.loads((bundle / "oracle-http.json").read_text())[0]
            assert probe["redirect_bodies"][0]["body"] == artifact
        outcome = results["runtime.oracle_containment"]
        expected = "pass" if mode == "common_line" else "fail"
        assert outcome.status.value == expected, outcome
        print(
            json.dumps(
                {
                    "case": mode,
                    "oracle_status": outcome.status.value,
                    "evidence": outcome.evidence,
                    "measured": outcome.measured,
                }
            )
        )
