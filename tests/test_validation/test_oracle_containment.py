"""`runtime.oracle_containment` traces complete oracle files through bounded probes."""

import gzip
import socket
import threading
import time
import tracemalloc
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openenv.validation.runner import _get, MAX_REDIRECTS
from openenv.validation.runtime.contracts import HttpProbe

ANSWER = "openenv-oracle-private-answer-9a6b4920"
SCRIPT = f'#!/usr/bin/env python3\nfrom pathlib import Path\nprint("{ANSWER}")\n'


@contextmanager
def raw_server(respond):
    """Accept one connection, read the request, then let `respond` write to it."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)

        def serve():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    connection.recv(65536)
                    respond(connection)
            except OSError:
                pass  # The probe's deadline closes its end of the connection.

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            thread.join(timeout=10)
        assert not thread.is_alive()


@pytest.mark.parametrize("slow_phase", ["headers", "body"])
def test_probe_deadline_bounds_a_trickling_server(slow_phase):
    body = SCRIPT.encode()
    headers = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body)

    def respond(connection):
        if slow_phase == "body":
            connection.sendall(headers)
        for byte in headers + body if slow_phase == "headers" else body:
            connection.sendall(bytes([byte]))
            time.sleep(0.025)

    with raw_server(respond) as url:
        started = time.monotonic()
        probe = _get(url, "oracle/solve.py", started + 0.3)
        elapsed = time.monotonic() - started
    assert probe.failure == "TimeoutError"
    assert elapsed < 1.0, f"{slow_phase} trickle escaped the deadline: {elapsed:.2f}s"
    if slow_phase == "body":
        assert probe.status == 200
        assert 0 < len(probe.body) < len(body)


def test_compressed_probe_response_is_rejected_before_decoding():
    payload = gzip.compress(b"x" * (12 * 1024 * 1024))
    headers = b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: %d\r\n\r\n"
    with raw_server(lambda c: c.sendall(headers % len(payload) + payload)) as url:
        tracemalloc.start()
        try:
            probe = _get(url, "oracle/solve.py", time.monotonic() + 5)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert probe == HttpProbe("/oracle/solve.py", 200, failure="compressed response")
    assert peak < 4 * 1024 * 1024, f"decoded allocation peaked at {peak} bytes"


def test_oversized_probe_body_is_kept_to_the_bound_and_marked():
    body = b"y" * 4096
    headers = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body)
    with raw_server(lambda c: c.sendall(headers + body)) as url:
        probe = _get(url, "oracle/solve.py", time.monotonic() + 5, max_bytes=1000)
    assert probe == HttpProbe(
        "/oracle/solve.py", 200, b"y" * 1000, "body exceeds the byte bound"
    )


@contextmanager
def http_server(routes):
    """Serve fixed `(status, headers, body)` responses by path; record requests."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, headers, body = routes.get(self.path, (404, {}, b"Not Found"))
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_probe_follows_only_bounded_same_origin_redirects():
    with http_server({"/download/solve.py": (200, {}, SCRIPT.encode())}) as (
        elsewhere,
        elsewhere_requests,
    ):
        routes = {
            "/oracle/solve.py": (302, {"Location": "/download/solve.py"}, b""),
            "/download/solve.py": (403, {}, SCRIPT.encode()),
            "/oracle/away.py": (
                307,
                {"Location": elsewhere + "/download/solve.py"},
                b"",
            ),
            "/oracle/loop.py": (302, {"Location": "/oracle/loop.py"}, b""),
        }
        with http_server(routes) as (url, requests):
            deadline = time.monotonic() + 5
            followed = _get(url, "oracle/solve.py", deadline)
            away = _get(url, "oracle/away.py", deadline)
            loop = _get(url, "oracle/loop.py", deadline)
    assert followed == HttpProbe(
        "/oracle/solve.py", 403, SCRIPT.encode(), redirect_bodies=(b"",)
    )
    assert away == HttpProbe("/oracle/away.py", 307, failure="cross-origin redirect")
    assert elsewhere_requests == []
    assert loop == HttpProbe(
        "/oracle/loop.py",
        302,
        failure="too many redirects",
        redirect_bodies=(b"",) * MAX_REDIRECTS,
    )
    assert requests.count("/oracle/loop.py") == MAX_REDIRECTS + 1


def oracle_subject(tmp_path, *, responses=(), probes=None, location="oracle/solve.py"):
    from types import SimpleNamespace

    from openenv.validation.runtime.contracts import RuntimeEvidence, WireExchange

    (tmp_path / "oracle").mkdir(exist_ok=True)
    (tmp_path / "oracle/solve.py").write_text(SCRIPT)
    return SimpleNamespace(
        root=tmp_path,
        manifest=SimpleNamespace(
            capabilities=SimpleNamespace(oracle=SimpleNamespace(location=location))
        ),
        runtime_evidence=RuntimeEvidence(
            exchanges=tuple(WireExchange("step", "{}", text) for text in responses)
            or (WireExchange("reset", "{}", "{}"),),
            schema_response_json='{"observation":{}}',
        ),
        isolation_evidence=None,
        oracle_http=tuple(probes)
        if probes is not None
        else (HttpProbe("/oracle/solve.py", 404),),
    )


def grade(subject):
    from openenv.validation.graders.runtime.oracle import OracleContainmentGrader

    return OracleContainmentGrader().run(subject)


@pytest.mark.parametrize("status", [200, 206, 302, 403, 500])
def test_complete_disclosure_fails_for_every_status_even_when_incomplete(
    tmp_path, status
):
    result = grade(
        oracle_subject(
            tmp_path,
            probes=[
                HttpProbe("/oracle/solve.py", status, SCRIPT.encode(), "incomplete")
            ],
        )
    )
    assert result.status.value == "fail"


@pytest.mark.parametrize(
    "probes",
    [
        (),
        (HttpProbe("/oracle/solve.py", None),),
        (HttpProbe("/other.py", 404),),
        (HttpProbe("/oracle/solve.py", 200, b"", "TimeoutError"),),
        (HttpProbe("/oracle/solve.py", 200, b"", "body exceeds the byte bound"),),
        (HttpProbe("/oracle/solve.py", 302),),
        (HttpProbe("/oracle/solve.py", 206),),
        (HttpProbe("/oracle/solve.py", 500),),
    ],
)
def test_missing_failed_partial_and_redirect_probes_cannot_pass(tmp_path, probes):
    result = grade(oracle_subject(tmp_path, probes=probes))
    assert result.status.value == "skip"
    assert "required oracle HTTP probes" in result.evidence[-1]


@pytest.mark.parametrize(
    "overlap",
    [
        "from pathlib import Path",
        "from collections.abc import Callable, Iterator, Sequence",
        'print("' + ANSWER + '")',
    ],
)
def test_isolated_lines_do_not_establish_disclosure(tmp_path, overlap):
    import json

    subject = oracle_subject(tmp_path, responses=[json.dumps({"output": overlap})])
    (tmp_path / "oracle/solve.py").write_text(
        SCRIPT + "from collections.abc import Callable, Iterator, Sequence\n"
    )
    result = grade(subject)
    assert result.status.value == "pass", result.evidence
    assert (
        "MCP tool access and reward-artifact writes were not tested"
        in result.evidence[0]
    )
    assert result.measured["mcp_tool_access_tested"] is False


@pytest.mark.parametrize("source", ["raw", "unicode", "malformed"])
def test_complete_script_matches_raw_unicode_and_malformed_json(tmp_path, source):

    escaped = '"' + "".join("\\u%04x" % ord(c) for c in SCRIPT) + '"'
    response = {
        "raw": SCRIPT,
        "unicode": '{"output":' + escaped + "}",
        "malformed": SCRIPT + '{"unfinished":',
    }[source]
    assert grade(oracle_subject(tmp_path, responses=[response])).status.value == "fail"


def test_structured_artifact_matches_nested_json_without_key_order_or_formatting(
    tmp_path,
):
    import json

    subject = oracle_subject(
        tmp_path, responses=['{"data":{"answer":{"b":[2,3],"a":"' + ANSWER + '"}}}']
    )
    (tmp_path / "oracle/solve.py").write_text(
        json.dumps({"a": ANSWER, "b": [2, 3]}, indent=4)
    )
    assert grade(subject).status.value == "fail"


@pytest.mark.parametrize("missing", ["schema", "transcript", "failed"])
def test_incomplete_runtime_sources_skip_unless_they_disclose_the_artifact(
    tmp_path, missing
):
    from dataclasses import replace

    subject = oracle_subject(tmp_path)
    changes = {
        "schema": {"schema_response_json": None},
        "transcript": {"exchanges": ()},
        "failed": {"failure_reason": "step failed (TimeoutError)"},
    }[missing]
    subject.runtime_evidence = replace(subject.runtime_evidence, **changes)
    assert grade(subject).status.value == "skip"
    subject.oracle_http = (HttpProbe("/oracle/solve.py", 403, SCRIPT.encode()),)
    assert grade(subject).status.value == "fail"


@pytest.mark.parametrize("limit", ["files", "bytes"])
def test_inventory_omissions_are_measured_and_never_pass(tmp_path, limit):
    from openenv.validation.graders.runtime.oracle import (
        MAX_ORACLE_BYTES,
        MAX_ORACLE_FILES,
    )

    subject = oracle_subject(tmp_path, location="oracle")
    (tmp_path / "oracle/solve.py").unlink()
    if limit == "files":
        for i in range(MAX_ORACLE_FILES):
            (tmp_path / "oracle" / f"{i:02d}.txt").write_text(
                f"private-oracle-artifact-number-{i:02d}"
            )
        (tmp_path / "oracle/z_secret.py").write_text(SCRIPT)
        subject.oracle_http = tuple(
            HttpProbe(f"/oracle/{i:02d}.txt", 404) for i in range(MAX_ORACLE_FILES)
        )
        expected_bytes = len(SCRIPT.encode())
    else:
        (tmp_path / "oracle/solve.py").write_bytes(
            b"x" * MAX_ORACLE_BYTES + SCRIPT.encode()
        )
        expected_bytes = len(SCRIPT.encode())
    result = grade(subject)
    assert result.status.value == "skip"
    assert result.measured["omitted_files"] == 1
    assert result.measured["omitted_bytes"] == expected_bytes
    if limit == "files":
        subject.oracle_http += (
            HttpProbe("/oracle/00.txt", 403, b"private-oracle-artifact-number-00"),
        )
        assert grade(subject).status.value == "fail"


def test_directory_probes_use_file_urls_and_share_the_body_budget(
    tmp_path, monkeypatch
):
    from openenv.validation import runner

    subject = oracle_subject(tmp_path, location="oracle")
    (tmp_path / "oracle/other.py").write_text("second-private-artifact")
    monkeypatch.setattr(runner, "MAX_ORACLE_PROBE_BYTES", 10)
    routes = {
        "/oracle/other.py": (404, {}, b"x" * 8),
        "/oracle/solve.py": (200, {}, SCRIPT.encode()),
    }
    with http_server(routes) as (url, requests):
        subject.oracle_http = runner._oracle_probes(url, tmp_path, "oracle")
    assert requests == ["/oracle/other.py", "/oracle/solve.py"]
    assert sum(len(probe.body) for probe in subject.oracle_http) == 10
    assert subject.oracle_http[1].failure == "body exceeds the byte bound"
    assert grade(subject).status.value == "skip"


def test_directory_probes_share_a_deadline(tmp_path, monkeypatch):
    from openenv.validation import runner

    oracle_subject(tmp_path)
    (tmp_path / "oracle/z_later.py").write_text("another-private-artifact")
    monkeypatch.setattr(runner, "ORACLE_PROBE_TIMEOUT_S", 0.1)
    calls = []

    def get(base_url, path, deadline, max_bytes):
        calls.append(path)
        time.sleep(0.15)
        return HttpProbe("/" + path, None, failure="TimeoutError")

    monkeypatch.setattr(runner, "_get", get)
    probes = runner._oracle_probes("http://unused", tmp_path, "oracle")
    assert calls == ["oracle/solve.py"]
    assert probes[-1].failure == "oracle probe budget exhausted"


def test_redirect_bodies_are_scanned_and_share_the_byte_limit(tmp_path):
    subject = oracle_subject(tmp_path)
    routes = {
        "/oracle/solve.py": (302, {"Location": "/landing"}, SCRIPT.encode()),
        "/landing": (404, {}, b"nothing"),
    }
    with http_server(routes) as (url, _):
        probe = _get(url, "oracle/solve.py", time.monotonic() + 5)
        bounded = _get(
            url, "oracle/solve.py", time.monotonic() + 5, len(SCRIPT.encode()) + 3
        )
    subject.oracle_http = (probe,)
    assert grade(subject).status.value == "fail"
    assert (
        len(bounded.body) + sum(map(len, bounded.redirect_bodies))
        == len(SCRIPT.encode()) + 3
    )
    assert bounded.failure == "body exceeds the byte bound"


def test_closed_transport_cannot_pass(tmp_path):
    subject = oracle_subject(tmp_path)
    with raw_server(lambda connection: None) as url:
        probe = _get(url, "oracle/solve.py", time.monotonic() + 2)
    assert probe.status is None
    assert probe.failure == "RemoteProtocolError"
    subject.oracle_http = (probe,)
    assert grade(subject).status.value == "skip"


def test_non_ascii_json_strings_and_structured_values_match(tmp_path):
    import json

    subject = oracle_subject(tmp_path)
    artifact = SCRIPT + "# oracle α秘密🦄\n"
    (tmp_path / "oracle/solve.py").write_text(artifact)
    from dataclasses import replace

    from openenv.validation.runtime.contracts import WireExchange

    subject.runtime_evidence = replace(
        subject.runtime_evidence,
        exchanges=(
            WireExchange(
                "step", "{}", json.dumps({"data": artifact}, ensure_ascii=True)
            ),
        ),
    )
    assert grade(subject).status.value == "fail"


@pytest.mark.parametrize("structured", [False, True])
def test_encoded_redirect_disclosure_is_scanned_as_a_separate_response(
    tmp_path, structured
):
    import json

    subject = oracle_subject(tmp_path)
    artifact = {"answer": ANSWER} if structured else SCRIPT
    if structured:
        (tmp_path / "oracle/solve.py").write_text(json.dumps(artifact, indent=2))
    encoded = json.dumps({"output": artifact})
    if not structured:
        encoded = '{"output":"' + "".join("\\u%04x" % ord(c) for c in SCRIPT) + '"}'
    routes = {
        "/oracle/solve.py": (302, {"Location": "/landing"}, encoded.encode()),
        "/landing": (404, {}, b"nothing"),
    }
    with http_server(routes) as (url, _):
        subject.oracle_http = (_get(url, "oracle/solve.py", time.monotonic() + 5),)
    assert grade(subject).status.value == "fail"
    assert subject.oracle_http[0].redirect_bodies == (encoded.encode(),)


def test_string_json_artifacts_match_equivalent_escaping(tmp_path):
    import json

    subject = oracle_subject(
        tmp_path, responses=[json.dumps({"output": ANSWER + " α秘密🦄"})]
    )
    (tmp_path / "oracle/solve.py").write_text(
        json.dumps(ANSWER + " α秘密🦄", ensure_ascii=False)
    )
    assert grade(subject).status.value == "fail"


@pytest.mark.parametrize("unreadable", ["binary", "symlink", "permission"])
def test_unreadable_inventory_siblings_do_not_erase_a_confirmed_leak(
    tmp_path, monkeypatch, unreadable
):
    from pathlib import Path

    subject = oracle_subject(tmp_path, location="oracle", probes=[])
    sibling = tmp_path / "oracle/zz_unreadable"
    if unreadable == "binary":
        sibling.write_bytes(b"\xff\xfe")
    elif unreadable == "symlink":
        sibling.symlink_to(tmp_path / "elsewhere")
    else:
        sibling.write_text("inaccessible")
        original = Path.open

        def denied(path, *args, **kwargs):
            if path == sibling:
                raise PermissionError("test fixture")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    result = grade(subject)
    assert result.status.value == "skip"
    assert result.measured["inventory_errors"] == 1
    subject.oracle_http = (HttpProbe("/oracle/solve.py", 200, SCRIPT.encode()),)
    assert grade(subject).status.value == "fail"


@pytest.mark.parametrize("boundary", ["files", "bytes"])
def test_complete_inventory_at_the_exact_bound_can_pass(tmp_path, boundary):
    from openenv.validation.graders.runtime.oracle import (
        MAX_ORACLE_BYTES,
        MAX_ORACLE_FILES,
    )

    subject = oracle_subject(tmp_path, location="oracle")
    if boundary == "files":
        for i in range(MAX_ORACLE_FILES - 1):
            (tmp_path / "oracle" / f"{i:02d}.txt").write_text(
                f"private-artifact-number-{i:02d}"
            )
    else:
        (tmp_path / "oracle/solve.py").write_bytes(b"x" * MAX_ORACLE_BYTES)
    subject.oracle_http = tuple(
        HttpProbe("/oracle/" + path.name, 404)
        for path in sorted((tmp_path / "oracle").iterdir())
    )
    result = grade(subject)
    assert result.status.value == "pass", result.evidence
    assert result.measured["omitted_files"] == result.measured["omitted_bytes"] == 0
