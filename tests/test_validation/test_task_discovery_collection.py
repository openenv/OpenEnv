import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from openenv.validation.runtime import discovery


def collect(monkeypatch, handler, *, advertised=None, **kwargs):
    original = httpx.Client

    def respond(request):
        if request.url.path == "/list_environments":
            return (
                advertised
                if advertised is not None
                else httpx.Response(200, json=["name?secret"])
            )
        return handler(request)

    def client(**options):
        assert options["limits"].max_keepalive_connections == 0
        assert {key: value for key, value in options.items() if key != "limits"} == {
            "trust_env": False,
            "follow_redirects": False,
        }
        return original(transport=httpx.MockTransport(respond), **options)

    monkeypatch.setattr(discovery.httpx, "Client", client)
    return discovery.collect_task_evidence(
        "http://127.0.0.1:8000", deadline=time.monotonic() + 1, **kwargs
    )


def test_task_sampling_uses_true_counts_and_two_bounded_specs(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.query == b""
        assert request.url.raw_path.startswith(b"/name%3Fsecret/")
        if request.url.path.endswith("/splits"):
            return httpx.Response(200, json=[{"name": "train"}, {"name": "empty"}])
        data = json.loads(request.content)
        if request.url.path.endswith("/num_tasks"):
            return httpx.Response(
                200, json={"num_tasks": 10**9 if data["split"] == "train" else 0}
            )
        assert request.url.path.endswith("/task") and data["index"] < 2
        return httpx.Response(200, json={"task": [data["split"], data["index"]]})

    payload, error = collect(monkeypatch, handler)
    assert error is None
    assert json.loads(payload) == {
        "environments": ["name?secret"],
        "splits": [{"name": "train"}, {"name": "empty"}],
        "counts": {"train": 10**9, "empty": 0},
        "previews": {"train": [["train", 0], ["train", 1]], "empty": []},
    }
    assert len(seen) == 5


@pytest.mark.parametrize(
    "names", [[], {}, "probe", [None], [""], ["a", "b"], ["a", "a"], [".."], ["a/b"]]
)
def test_invalid_environment_inventory_is_not_guessed(monkeypatch, names):
    def handler(request):
        pytest.fail("invalid environment inventory must not request task routes")

    payload, error = collect(
        monkeypatch, handler, advertised=httpx.Response(200, json=names)
    )
    assert payload is None and error == "task discovery failed (ValueError)"


@pytest.mark.parametrize(
    "fault", ["unsupported", "compressed", "oversized", "long_namespace"]
)
def test_environment_inventory_uses_discovery_response_bounds(monkeypatch, fault):
    def handler(request):
        pytest.fail("failed environment discovery must not request task routes")

    if fault == "unsupported":
        response = httpx.Response(501, text="private-error")
    elif fault == "compressed":
        response = httpx.Response(
            200, headers={"Content-Encoding": "br"}, stream=httpx.ByteStream(b"invalid")
        )
    elif fault == "long_namespace":
        response = httpx.Response(200, json=["x" * 65537])
    else:
        response = httpx.Response(
            200, content=b"x" * (discovery.MAX_RESPONSE_BYTES + 1)
        )
    payload, error = collect(monkeypatch, handler, advertised=response)
    assert payload is None and error.startswith("task discovery failed (")
    assert "private" not in error


@pytest.mark.parametrize(
    "fault",
    [
        "unsupported",
        "redirect",
        "too_large",
        "duplicate",
        "boolean_count",
        "too_many_splits",
        "compressed",
        "missing_task",
    ],
)
def test_bad_discovery_is_explicit_failure_without_private_values(monkeypatch, fault):
    def handler(request):
        if request.url.path.endswith("/splits"):
            if fault == "unsupported":
                return httpx.Response(501, text="private-error")
            if fault == "redirect":
                return httpx.Response(
                    302, headers={"Location": "http://secret.invalid"}
                )
            if fault == "too_large":
                return httpx.Response(
                    200, content=b"x" * (discovery.MAX_RESPONSE_BYTES + 1)
                )
            if fault == "duplicate":
                return httpx.Response(200, json=[{"name": "train"}] * 2)
            if fault == "too_many_splits":
                return httpx.Response(200, json=[{}] * (discovery.MAX_TASK_SPLITS + 1))
            if fault == "compressed":
                return httpx.Response(
                    200, headers={"Content-Encoding": "br"}, content=b"invalid"
                )
            return httpx.Response(200, json=[{"name": "train"}])
        if request.url.path.endswith("/num_tasks"):
            return httpx.Response(
                200, json={"num_tasks": True if fault == "boolean_count" else 1}
            )
        return httpx.Response(
            200,
            json={"wrong": "private-error"}
            if fault == "missing_task"
            else {"task": "valid"},
        )

    payload, error = collect(monkeypatch, handler)
    assert payload is None
    assert error.startswith("task discovery failed (")
    assert "private" not in error and "secret" not in error


def test_expired_deadline_makes_no_network_request(monkeypatch):
    def handler(request):
        pytest.fail("expired discovery budget must not contact the subject")

    payload, error = collect(monkeypatch, handler, request_timeout_s=0)
    assert payload is None and error == "task discovery failed (TimeoutError)"


@pytest.mark.parametrize("request_timeout_s,expected", [(None, 12.0), (3.0, 3.0)])
def test_request_timeout_uses_episode_deadline_unless_explicitly_capped(
    monkeypatch, request_timeout_s, expected
):
    original = httpx.Client
    timeouts = []

    def respond(request):
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json=["probe"] if request.url.path == "/list_environments" else [],
        )

    monkeypatch.setattr(discovery.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        discovery.httpx,
        "Client",
        lambda **options: original(transport=httpx.MockTransport(respond), **options),
    )
    payload, error = discovery.collect_task_evidence(
        "http://127.0.0.1:8000",
        deadline=112.0,
        request_timeout_s=request_timeout_s,
    )
    assert error is None and json.loads(payload)["splits"] == []
    assert len(timeouts) == 2
    assert all(set(timeout.values()) == {expected} for timeout in timeouts)


@pytest.mark.parametrize("phase", ["headers", "chunk_header"])
@pytest.mark.parametrize("request_timeout_s", [None, 0.2])
def test_dripping_task_response_cannot_extend_the_http_deadline(
    phase, request_timeout_s
):
    stopped = threading.Event()
    paths = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            paths.append(self.path)
            if self.path == "/list_environments":
                body = b'["probe"]'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            assert self.path == "/probe/splits"
            self.close_connection = True
            prefix = (
                b"HTTP/1.1 200 OK\r\nX-Slow: "
                if phase == "headers"
                else b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2;"
            )
            suffix = (
                b"\r\nContent-Length: 2\r\n\r\n[]"
                if phase == "headers"
                else b"\r\n[]\r\n0\r\n\r\n"
            )
            try:
                self.connection.sendall(prefix)
                # Each byte arrives before the inactivity timeout. The full
                # header or chunk extension still exceeds the request budget.
                for _ in range(100):
                    if stopped.wait(0.01):
                        return
                    self.connection.sendall(b"x")
                self.connection.sendall(suffix)
            except OSError:
                pass  # The deadline closes the peer's transport mid-response.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        payload, error = discovery.collect_task_evidence(
            f"http://127.0.0.1:{server.server_port}",
            deadline=started + (0.2 if request_timeout_s is None else 5),
            request_timeout_s=request_timeout_s,
        )
        elapsed = time.monotonic() - started
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert paths == ["/list_environments", "/probe/splits"]
    assert payload is None and error == "task discovery failed (TimeoutError)"
    assert elapsed < 0.75


@pytest.mark.parametrize("split,task", [("train", "😀" * 40), ("s" * 200, None)])
def test_retained_task_evidence_respects_byte_bound(monkeypatch, split, task):
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_BYTES", 400)

    def handler(request):
        if request.url.path.endswith("/splits"):
            return httpx.Response(200, json=[{"name": split}])
        if request.url.path.endswith("/num_tasks"):
            return httpx.Response(200, json={"num_tasks": 1 if task else 0})
        return httpx.Response(200, json={"task": task})

    payload, error = collect(monkeypatch, handler)
    if task:
        assert error is None
        assert "😀" in payload and len(payload.encode()) <= 400
    else:
        # Split names occur in three sections; received bytes alone do not bound
        # the normalized evidence document.
        assert payload is None and error == "task discovery failed (ValueError)"
