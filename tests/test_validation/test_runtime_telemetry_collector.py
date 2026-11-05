"""Telemetry cannot expand its byte budget or retain replay credentials."""

import json

import httpx
import pytest
from openenv.validation.runtime import collector
from openenv.validation.runtime.contracts import RuntimePlan
from websockets.exceptions import ConnectionClosedError

TOKEN = "run-authorization-" + "x" * 32
CAPABILITY = "socket-capability-" + "y" * 32


def collect(
    monkeypatch,
    *,
    snapshot=None,
    leaked_response=None,
    leak_operation="reset",
    escaped=False,
    malformed=False,
    opening_reply=None,
    opening_error=None,
    send_error=None,
):
    original = httpx.Client
    monkeypatch.setattr(
        collector.httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"observation": {}})
            ),
            **kwargs,
        ),
    )

    class Connection:
        def send(self, raw):
            self.request = json.loads(raw)
            if self.request["type"] == "validation_open" and send_error is not None:
                raise send_error

        def recv(self, timeout):
            operation = self.request["type"]
            if operation == "validation_open":
                if opening_error is not None:
                    raise opening_error
                if opening_reply is not None:
                    return opening_reply
                response = {
                    "type": operation,
                    "data": {"schema_version": 1, "capability": CAPABILITY},
                }
            elif operation == "validation_read":
                response = {
                    "type": "validation",
                    "data": snapshot or {"schema_version": 1},
                }
            elif operation == "state":
                response = {
                    "type": "state",
                    "data": {"episode_id": "test", "step_count": 0},
                }
            else:
                response = {
                    "type": "observation",
                    "data": {
                        "observation": {
                            "message": leaked_response
                            if operation == leak_operation
                            else None
                        },
                        "done": False,
                        "reward": 0.0,
                    },
                }
            raw = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            if escaped and leaked_response:
                raw = raw.replace(
                    leaked_response,
                    "".join(f"\\u{ord(c):04x}" for c in leaked_response),
                )
            return raw[:-1] if malformed and operation == leak_operation else raw

        def close(self):
            pass

    monkeypatch.setattr(collector, "connect", lambda *args, **kwargs: Connection())
    plan = RuntimePlan.model_validate(
        {
            "plan_schema_version": "1",
            "reset": {"episode_id": "test", "seed": 1},
            "actions": [{"increment": 1}],
        }
    )
    return collector.collect_runtime_evidence(
        "http://127.0.0.1:8000", plan, episode_timeout_s=2, validation_token=TOKEN
    )


def test_utf8_telemetry_stays_inside_received_budget(monkeypatch):
    monkeypatch.setattr(collector, "MAX_TRACE_BYTES", 2048)
    result = collect(monkeypatch, snapshot={"schema_version": 1, "text": "😀" * 400})
    assert result.telemetry_error is None
    assert len(result.telemetry_json.encode()) <= 2048
    assert "😀" in result.telemetry_json


@pytest.mark.parametrize("credential", [TOKEN, CAPABILITY])
def test_telemetry_cannot_persist_credentials_under_arbitrary_keys(
    monkeypatch, credential
):
    result = collect(monkeypatch, snapshot={"schema_version": 1, "debug": [credential]})
    assert result.telemetry_json is None
    assert result.telemetry_error == "session telemetry failed (ValueError)"
    assert result.failure_reason is None
    assert len(result.exchanges) == 4
    assert credential not in repr(result)


@pytest.mark.parametrize(
    "credential,escaped",
    [(TOKEN, False), (CAPABILITY, False), (TOKEN, True), (CAPABILITY, True)],
)
def test_normal_wire_cannot_persist_plain_or_escaped_credentials(
    monkeypatch, credential, escaped
):
    result = collect(monkeypatch, leaked_response=credential, escaped=escaped)
    assert result.exchanges == ()
    assert result.failure_reason == "reset failed (ValueError)"
    assert credential not in repr(result)


@pytest.mark.parametrize(
    "reply,error_name",
    [
        (TOKEN.encode(), "ValueError"),
        ("not-json-" + TOKEN, "JSONDecodeError"),
        (json.dumps([TOKEN]), "ValueError"),
    ],
    ids=["binary", "non-json", "list"],
)
def test_malformed_opening_reply_keeps_the_ordinary_session(
    monkeypatch, reply, error_name
):
    result = collect(monkeypatch, opening_reply=reply)
    assert result.telemetry_error == f"session telemetry failed ({error_name})"
    assert result.telemetry_json is None
    assert result.failure_reason is None
    assert [row.operation for row in result.exchanges] == [
        "reset",
        "state",
        "step",
        "state",
    ]
    assert TOKEN not in repr(result)


@pytest.mark.parametrize(
    "error", [ConnectionClosedError(None, None), TimeoutError(TOKEN)]
)
def test_opening_transport_failure_still_aborts_collection(monkeypatch, error):
    result = collect(monkeypatch, opening_error=error)
    assert result.failure_phase == "validation_open"
    assert result.failure_reason == f"validation_open failed ({type(error).__name__})"
    assert result.telemetry_json is None
    assert not result.exchanges
    assert TOKEN not in repr(result)


def test_opening_cancellation_is_not_downgraded_to_optional_telemetry(monkeypatch):
    with pytest.raises(collector.RuntimeCollectionInterrupted) as error:
        collect(monkeypatch, opening_error=KeyboardInterrupt(TOKEN))
    assert error.value.evidence.failure_phase == "validation_open"
    assert not error.value.evidence.exchanges
    assert TOKEN not in repr(error.value.evidence)


@pytest.mark.parametrize("code", ["FACTORY_ERROR", "CAPACITY_REACHED", "SESSION_ERROR"])
@pytest.mark.parametrize("closed_before_send", [False, True])
def test_terminal_opening_error_preserves_only_its_safe_code(
    monkeypatch, code, closed_before_send
):
    result = collect(
        monkeypatch,
        opening_reply=json.dumps(
            {"type": "error", "data": {"code": code, "message": TOKEN}}
        ),
        send_error=ConnectionClosedError(None, None) if closed_before_send else None,
    )
    assert result.failure_reason == f"validation_open failed ({code})"
    assert result.exchanges == ()
    assert result.telemetry_json is None
    assert TOKEN not in repr(result)


@pytest.mark.parametrize("code", ["UNKNOWN_TYPE", "VALIDATION_ERROR", TOKEN])
def test_optional_opening_refusal_does_not_fail_the_episode(monkeypatch, code):
    result = collect(
        monkeypatch,
        opening_reply=json.dumps(
            {"type": "error", "data": {"code": code, "message": TOKEN}}
        ),
    )
    assert result.failure_reason is None
    assert result.telemetry_json is None
    assert (
        result.telemetry_error
        == "session telemetry unavailable or authorization refused"
    )
    assert [row.operation for row in result.exchanges] == [
        "reset",
        "state",
        "step",
        "state",
    ]
    assert TOKEN not in repr(result)


@pytest.mark.parametrize(
    "reply",
    [
        "not-json-" + TOKEN,
        json.dumps(
            {
                "type": "validation_open",
                "data": {"schema_version": 1, "capability": CAPABILITY},
            }
        ),
    ],
)
def test_closed_send_cannot_become_an_optional_or_successful_handshake(
    monkeypatch, reply
):
    result = collect(
        monkeypatch,
        opening_reply=reply,
        send_error=ConnectionClosedError(None, None),
    )
    assert result.failure_reason == "validation_open failed (ConnectionClosedError)"
    assert result.exchanges == ()
    assert result.telemetry_json is None
    assert TOKEN not in repr(result)


@pytest.mark.parametrize("credential", [TOKEN, CAPABILITY])
@pytest.mark.parametrize("operation", ["reset", "step"])
def test_malformed_wire_cannot_retain_escaped_credentials(
    monkeypatch, credential, operation
):
    result = collect(
        monkeypatch,
        leaked_response=credential,
        leak_operation=operation,
        escaped=True,
        malformed=True,
    )
    assert [row.operation for row in result.exchanges] == (
        [] if operation == "reset" else ["reset", "state"]
    )
    assert result.failure_reason == f"{operation} failed (JSONDecodeError)"
