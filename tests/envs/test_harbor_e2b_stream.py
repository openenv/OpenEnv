import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor.environments.e2b")

from openenv.harbor.e2b_stream import E2BStreamingEnvironment
from tenacity import wait_none


def environment(files):
    env = object.__new__(E2BStreamingEnvironment)
    env._sandbox = SimpleNamespace(files=files)
    return env


def test_directory_upload_preserves_bytes_paths_and_closes_streams(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/binary").write_bytes(bytes(range(256)) * 100)
    (tmp_path / "text").write_text("hello\n")
    observed, handles = {}, []

    async def write_files(entries, **kwargs):
        assert kwargs == {"gzip": True, "use_octet_stream": True, "request_timeout": 30}
        for entry in entries:
            handles.append(entry["data"])
            observed[entry["path"]] = entry["data"].read()

    env = environment(SimpleNamespace(write_files=write_files))
    asyncio.run(env.upload_dir(tmp_path, "/logs/agent"))
    assert observed == {
        "/logs/agent/nested/binary": bytes(range(256)) * 100,
        "/logs/agent/text": b"hello\n",
    }
    assert all(handle.closed for handle in handles)


def test_retry_reopens_source_and_replays_identical_file_bytes(tmp_path, monkeypatch):
    source = tmp_path / "log"
    source.write_bytes(b"exact evidence")
    seen, handles = [], []

    async def write(path, stream, **kwargs):
        handles.append(stream)
        seen.append((path, stream.read()))
        if len(seen) == 1:
            raise TimeoutError("transport stalled")

    monkeypatch.setattr(E2BStreamingEnvironment.upload_file.retry, "wait", wait_none())
    asyncio.run(
        environment(SimpleNamespace(write=write)).upload_file(source, "/logs/log")
    )
    assert seen == [("/logs/log", b"exact evidence")] * 2
    assert all(handle.closed for handle in handles)


def test_hung_upload_has_total_deadline_and_bounded_retries(tmp_path, monkeypatch):
    (tmp_path / "log").write_text("data")
    deadlines, calls = [], []
    real_wait_for = asyncio.wait_for

    async def bounded(coro, timeout):
        deadlines.append(timeout)
        return await real_wait_for(coro, timeout=0.005)

    async def write_files(entries, **kwargs):
        calls.extend(entry["data"] for entry in entries)
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "wait_for", bounded)
    monkeypatch.setattr(E2BStreamingEnvironment.upload_dir.retry, "wait", wait_none())
    with pytest.raises(TimeoutError):
        asyncio.run(
            environment(SimpleNamespace(write_files=write_files)).upload_dir(
                tmp_path, "/logs"
            )
        )
    assert deadlines == [120, 120]
    assert len(calls) == 2 and all(handle.closed for handle in calls)


def test_cancellation_is_propagated_without_retry(tmp_path):
    source = tmp_path / "log"
    source.write_bytes(b"data")
    calls = []

    async def write(path, stream, **kwargs):
        calls.append(stream)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            environment(SimpleNamespace(write=write)).upload_file(source, "/logs/log")
        )
    assert len(calls) == 1 and calls[0].closed
