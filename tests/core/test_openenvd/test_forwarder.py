# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os
import shutil
import stat
import tempfile

import pytest
from openenv.core.openenvd.forwarder import (
    serve_tcp_to_unix,
    serve_unix_to_tcp,
    serve_unix_to_unix,
)


@pytest.fixture
def sock_dir():
    # Unix socket paths are limited to ~104 bytes; pytest's tmp_path can exceed it.
    path = tempfile.mkdtemp(prefix="oe-", dir="/tmp" if os.path.isdir("/tmp") else None)
    yield path
    shutil.rmtree(path, ignore_errors=True)


async def _echo_until_eof(reader, writer):
    # Reads everything, then replies, so it only answers after a half-close.
    data = await reader.read()
    writer.write(b"echo:" + data)
    await writer.drain()
    writer.close()


async def _roundtrip(reader, writer, payload=b"hello"):
    writer.write(payload)
    await writer.drain()
    writer.write_eof()
    reply = await asyncio.wait_for(reader.read(), 2)
    writer.close()
    return reply


def _port(server):
    return server.sockets[0].getsockname()[1]


def test_tcp_to_unix(sock_dir):
    async def main():
        target = os.path.join(sock_dir, "up.sock")
        upstream = await asyncio.start_unix_server(_echo_until_eof, path=target)
        fwd = await serve_tcp_to_unix("127.0.0.1", 0, target)
        reader, writer = await asyncio.open_connection("127.0.0.1", _port(fwd))
        reply = await _roundtrip(reader, writer)
        fwd.close()
        upstream.close()
        return reply

    assert asyncio.run(main()) == b"echo:hello"


def test_unix_to_tcp_replaces_stale_socket(sock_dir):
    async def main():
        upstream = await asyncio.start_server(_echo_until_eof, "127.0.0.1", 0)
        listen = os.path.join(sock_dir, "svc.sock")
        stale = await asyncio.start_unix_server(_echo_until_eof, path=listen)
        stale.close()
        await stale.wait_closed()
        fwd = await serve_unix_to_tcp(listen, "127.0.0.1", _port(upstream))
        assert stat.S_IMODE(os.stat(listen).st_mode) == 0o666
        reader, writer = await asyncio.open_unix_connection(listen)
        reply = await _roundtrip(reader, writer, b"x" * 200_000)
        fwd.close()
        upstream.close()
        return reply

    assert asyncio.run(main()) == b"echo:" + b"x" * 200_000


def test_unix_to_unix(sock_dir):
    async def main():
        target = os.path.join(sock_dir, "t.sock")
        listen = os.path.join(sock_dir, "l.sock")
        upstream = await asyncio.start_unix_server(_echo_until_eof, path=target)
        fwd = await serve_unix_to_unix(listen, target)
        reader, writer = await asyncio.open_unix_connection(listen)
        reply = await _roundtrip(reader, writer)
        fwd.close()
        upstream.close()
        return reply

    assert asyncio.run(main()) == b"echo:hello"


def test_unreachable_target_closes_client(sock_dir):
    async def main():
        fwd = await serve_tcp_to_unix(
            "127.0.0.1", 0, os.path.join(sock_dir, "missing.sock")
        )
        reader, writer = await asyncio.open_connection("127.0.0.1", _port(fwd))
        data = await asyncio.wait_for(reader.read(), 2)
        writer.close()
        fwd.close()
        return data

    assert asyncio.run(main()) == b""
