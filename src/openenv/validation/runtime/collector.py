"""Bounded raw protocol collection in one OpenEnv orchestration session."""

import json
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from ...core.env_server.types import WSErrorCode
from .contracts import RuntimeEvidence, RuntimePlan, WireExchange
from .transport import abort_socket, http_deadline

MAX_MESSAGE_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 8 * 1024 * 1024


def _server_error_code(response):
    if isinstance(response, dict) and response.get("type") == "error":
        data = response.get("data")
        code = data.get("code") if isinstance(data, dict) else None
        # Only protocol constants are safe diagnostics, never subject error text.
        if isinstance(code, str) and code in {item.value for item in WSErrorCode}:
            return code
    return None


def _bounded_call(connection, operation, timeout_s):
    if timeout_s <= 0:
        raise TimeoutError("episode deadline exceeded")
    expired = threading.Event()
    deadline = time.monotonic() + timeout_s

    def abort():
        expired.set()
        abort_socket(connection.socket)

    watchdog = threading.Timer(timeout_s, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        operation()
    except KeyboardInterrupt:
        abort_socket(connection.socket)
        raise
    except Exception:
        if expired.is_set():
            raise TimeoutError("transport deadline exceeded") from None
        raise
    finally:
        watchdog.cancel()
        watchdog.join()
    if expired.is_set() or time.monotonic() >= deadline:
        abort_socket(connection.socket)
        raise TimeoutError("transport deadline exceeded")


class RuntimeCollectionInterrupted(KeyboardInterrupt):
    """Cancellation carrying the completed, immutable episode prefix."""

    def __init__(self, evidence: RuntimeEvidence):
        super().__init__("runtime collection interrupted")
        self.evidence = evidence


def collect_runtime_evidence(
    base_url: str,
    plan: RuntimePlan,
    *,
    episode_timeout_s: float,
    request_timeout_s: float | None = None,
    validation_token: str | None = None,
) -> RuntimeEvidence:
    """
    Preserve schema and reset/step/state responses without model coercion.

    The collector never imports submitted code or reconnects midway through an
    episode. Transport failures retain the completed prefix and a bounded reason.
    Graders inspect the original envelopes rather than convenience-client defaults.

    Args:
        base_url (`str`):
            Provider-owned control endpoint.
        plan ([`~openenv.validation.runtime.contracts.RuntimePlan`]):
            Validated, bounded reset and action inputs.
        episode_timeout_s (`float`):
            Deadline for the complete collection, including schema retrieval.
        request_timeout_s (`float`, *optional*):
            Optional per-operation cap. By default, each operation may use the
            remaining declared episode budget.
        validation_token (`str`, *optional*):
            Run-scoped telemetry authorization; never retained in evidence.

    Returns:
        [`~openenv.validation.runtime.contracts.RuntimeEvidence`]: raw evidence.
    """
    deadline = time.monotonic() + episode_timeout_s
    exchanges = []
    schema_json = reset_schema_json = None
    phase = "schema"
    trace_bytes = 0
    telemetry_json = None
    telemetry_error = None
    server_code = None

    def remaining() -> float:
        value = deadline - time.monotonic()
        if request_timeout_s is not None:
            value = min(request_timeout_s, value)
        if value <= 0:
            raise TimeoutError("episode deadline exceeded")
        return value

    try:
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            with (
                http_deadline(remaining()) as extensions,
                client.stream(
                    "GET",
                    base_url.rstrip("/") + "/schema",
                    timeout=remaining(),
                    headers={"Accept-Encoding": "identity"},
                    extensions=extensions,
                ) as response,
            ):
                response.raise_for_status()
                # iter_bytes() transparently decompresses. Reject compressed
                # bodies before touching the stream so the byte budget also
                # bounds allocation, even when a subject ignores our header.
                if (
                    response.headers.get("Content-Encoding", "identity").strip().lower()
                    != "identity"
                ):
                    raise ValueError("compressed schema responses are not supported")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    remaining()
                    if len(payload) + len(chunk) > MAX_MESSAGE_BYTES:
                        raise ValueError("schema exceeds size bound")
                    payload.extend(chunk)
            schema = json.loads(payload)
            if not isinstance(schema, dict) or "observation" not in schema:
                raise ValueError("missing observation schema")
            schema_payload = json.dumps(
                schema["observation"],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if validation_token and validation_token in schema_payload:
                raise ValueError("schema contains validation credentials")
            schema_json = schema_payload

            if "reset_observation" in schema:
                reset_schema_payload = json.dumps(
                    schema["reset_observation"],
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if validation_token and validation_token in reset_schema_payload:
                    raise ValueError("schema contains validation credentials")
                reset_schema_json = reset_schema_payload

        endpoint = urlsplit(base_url)
        ws_url = urlunsplit(
            (
                "wss" if endpoint.scheme == "https" else "ws",
                endpoint.netloc,
                endpoint.path.rstrip("/") + "/ws",
                "",
                "",
            )
        )
        phase = "connect"
        connection = connect(
            ws_url,
            proxy=None,
            open_timeout=remaining(),
            close_timeout=1,
            max_size=MAX_TRACE_BYTES if validation_token else MAX_MESSAGE_BYTES,
            max_queue=1,
            compression=None,
        )
        complete = False
        try:

            def receive_response(request):
                nonlocal server_code
                try:
                    _bounded_call(
                        connection, lambda: connection.send(request), remaining()
                    )
                except ConnectionClosed:
                    # Session creation can fail before the first send. recv() still
                    # delivers a queued error, but the failed send cannot succeed.
                    try:
                        raw = connection.recv(timeout=remaining())
                        if (
                            isinstance(raw, str)
                            and len(raw.encode()) <= MAX_TRACE_BYTES
                        ):
                            server_code = _server_error_code(json.loads(raw))
                    except Exception:
                        # Preserve the original ConnectionClosed if diagnostics fail.
                        pass
                    raise
                return connection.recv(timeout=remaining())

            def telemetry_request(operation, data):
                request = json.dumps({"type": operation, "data": data})
                raw = receive_response(request)
                if not isinstance(raw, str) or len(raw.encode()) > MAX_TRACE_BYTES:
                    raise ValueError("invalid telemetry response")
                response = json.loads(raw)
                if not isinstance(response, dict):
                    raise ValueError("invalid telemetry envelope")
                return response

            capability = None
            if validation_token:
                phase = "validation_open"
                try:
                    response = telemetry_request(
                        phase, {"schema_version": 1, "token": validation_token}
                    )
                except (ValueError, RecursionError) as exc:
                    # A consumed malformed reply only invalidates optional telemetry.
                    # Transport failure still aborts this same-session collection.
                    telemetry_error = f"session telemetry failed ({type(exc).__name__})"
                else:
                    code = _server_error_code(response)
                    if code in {
                        WSErrorCode.FACTORY_ERROR,
                        WSErrorCode.CAPACITY_REACHED,
                        WSErrorCode.SESSION_ERROR,
                    }:
                        server_code = code
                        raise ValueError("server returned a terminal error")
                    data = response.get("data")
                    if (
                        response.get("type") == "validation_open"
                        and isinstance(data, dict)
                        and data.get("schema_version") == 1
                        and isinstance(data.get("capability"), str)
                        and 16 <= len(data["capability"]) <= 256
                    ):
                        capability = data["capability"]
                    else:
                        telemetry_error = (
                            "session telemetry unavailable or authorization refused"
                        )

            def contains_credential(value):
                if isinstance(value, str):
                    return any(
                        secret and secret in value
                        for secret in (validation_token, capability)
                    )
                if isinstance(value, dict):
                    return any(
                        contains_credential(key) or contains_credential(child)
                        for key, child in value.items()
                    )
                if isinstance(value, list):
                    return any(contains_credential(child) for child in value)
                return False

            def exchange(operation: str, data: dict | None = None) -> dict:
                nonlocal phase, trace_bytes, server_code
                phase = operation
                request = {"type": operation}
                if data is not None:
                    request["data"] = data
                request_json = json.dumps(request, allow_nan=False)
                raw = receive_response(request_json)
                if not isinstance(raw, str):
                    raise ValueError("binary response is not the JSON protocol")
                if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                    raise ValueError("response exceeds size bound")
                trace_bytes += len(raw.encode("utf-8")) + len(
                    request_json.encode("utf-8")
                )
                if trace_bytes > MAX_TRACE_BYTES:
                    raise ValueError("trace exceeds total size bound")
                # Subjects can echo the authorization into arbitrary JSON fields.
                # Check decoded values as well as raw text (which may be malformed).
                if contains_credential(raw):
                    raise ValueError("response contains validation credentials")
                try:
                    parsed = json.loads(raw)
                except (ValueError, RecursionError):
                    # Malformed JSON cannot be checked for escaped credentials.
                    if validation_token:
                        raise
                    parsed = None
                if contains_credential(parsed):
                    raise ValueError("response contains validation credentials")
                exchanges.append(
                    WireExchange(
                        operation=operation,
                        request_json=request_json,
                        response_json=raw,
                    )
                )
                response = json.loads(raw)
                if isinstance(response, dict) and response.get("type") == "error":
                    server_code = _server_error_code(response)
                    raise ValueError("server returned an error")
                expected = "state" if operation == "state" else "observation"
                if (
                    not isinstance(response, dict)
                    or response.get("type") != expected
                    or not isinstance(response.get("data"), dict)
                ):
                    raise ValueError("unexpected response envelope")
                return response["data"]

            reset = dict(plan.reset.options)
            reset.update(seed=plan.reset.seed, episode_id=plan.reset.episode_id)
            observation = exchange("reset", reset)
            exchange("state")
            for action in plan.actions:
                if observation.get("done") is True:
                    break
                observation = exchange("step", action)
                exchange("state")
            if capability:
                phase = "validation_read"
                try:
                    response = telemetry_request(
                        phase, {"schema_version": 1, "capability": capability}
                    )
                    if response.get("type") != "validation" or not isinstance(
                        response.get("data"), dict
                    ):
                        raise ValueError("invalid telemetry envelope")
                    if contains_credential(response["data"]):
                        raise ValueError("telemetry contains validation credentials")
                    snapshot = json.dumps(
                        response["data"],
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if len(snapshot.encode("utf-8")) > MAX_TRACE_BYTES:
                        raise ValueError("telemetry exceeds size bound")
                    telemetry_json = snapshot
                except Exception as exc:
                    telemetry_error = f"session telemetry failed ({type(exc).__name__})"
            complete = True
        finally:
            # Teardown is best effort and cannot replace an in-episode failure
            # or invalidate an otherwise completely measured episode.
            if complete:
                try:
                    _bounded_call(
                        connection,
                        lambda: connection.send(json.dumps({"type": "close"})),
                        min(remaining(), 1.0),
                    )
                except (Exception, KeyboardInterrupt):
                    pass  # A server may close immediately after its final response.
            try:
                _bounded_call(connection, connection.close, 1.0)
            except (Exception, KeyboardInterrupt):
                abort_socket(connection.socket)
        return RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=schema_json,
            telemetry_json=telemetry_json,
            telemetry_error=telemetry_error,
            reset_observation_schema_json=reset_schema_json,
        )
    except KeyboardInterrupt:
        raise RuntimeCollectionInterrupted(
            RuntimeEvidence(
                exchanges=tuple(exchanges),
                observation_schema_json=schema_json,
                reset_observation_schema_json=reset_schema_json,
                failure_phase=phase,
                failure_reason=f"{phase} failed (KeyboardInterrupt)",
                telemetry_json=telemetry_json,
                telemetry_error=telemetry_error,
            )
        ) from None
    except Exception as exc:
        # Exception text may include submitted payloads or URL credentials.
        return RuntimeEvidence(
            exchanges=tuple(exchanges),
            observation_schema_json=schema_json,
            reset_observation_schema_json=reset_schema_json,
            failure_phase=phase,
            failure_reason=f"{phase} failed ({server_code or type(exc).__name__})",
            telemetry_json=telemetry_json,
            telemetry_error=telemetry_error,
        )
