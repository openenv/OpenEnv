# SPDX-License-Identifier: BSD-3-Clause

"""Byte forwarders between TCP and Unix sockets, on asyncio.

Stdlib only, since the container shim imports it. A container's only way out
is the directory of Unix sockets mounted at `/run/openenvd`; these forwarders
bridge that directory to the loopback TCP ports its principal expects.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import stat
from asyncio.base_events import Server
from typing import Awaitable, Callable

_CHUNK = 64 * 1024

Connect = Callable[[], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]]


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while True:
        data = await reader.read(_CHUNK)
        if not data:
            break
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        with contextlib.suppress(OSError):
            writer.write_eof()


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError, asyncio.CancelledError):
        await writer.wait_closed()


async def splice(
    reader_a: asyncio.StreamReader,
    writer_a: asyncio.StreamWriter,
    reader_b: asyncio.StreamReader,
    writer_b: asyncio.StreamWriter,
) -> None:
    """Copy bytes both ways until both directions finish, then close both ends.

    EOF on one side is forwarded as a half-close to the other, so a peer that
    shuts down writing still receives the reply.

    Args:
        reader_a (`asyncio.StreamReader`):
            Reader of the first connection.
        writer_a (`asyncio.StreamWriter`):
            Writer of the first connection.
        reader_b (`asyncio.StreamReader`):
            Reader of the second connection.
        writer_b (`asyncio.StreamWriter`):
            Writer of the second connection.
    """
    pumps = [
        asyncio.ensure_future(_pump(reader_a, writer_b)),
        asyncio.ensure_future(_pump(reader_b, writer_a)),
    ]
    try:
        await asyncio.wait(pumps, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for pump in pumps:
            pump.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        await asyncio.gather(_close(writer_a), _close(writer_b))


def _handler(connect: Connect):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            upstream_reader, upstream_writer = await connect()
        except OSError:
            await _close(writer)
            return
        await splice(reader, writer, upstream_reader, upstream_writer)

    return handle


async def _start_unix(listen_path: str, connect: Connect) -> Server:
    with contextlib.suppress(FileNotFoundError):
        if stat.S_ISSOCK(os.lstat(listen_path).st_mode):
            os.unlink(listen_path)
    server = await asyncio.start_unix_server(_handler(connect), path=listen_path)
    os.chmod(listen_path, 0o666)
    return server


async def serve_tcp_to_unix(host: str, port: int, unix_path: str) -> Server:
    """Listen on TCP and forward each connection to a Unix socket.

    Args:
        host (`str`):
            Address to listen on, usually `127.0.0.1`.
        port (`int`):
            Port to listen on. `0` picks a free port.
        unix_path (`str`):
            Unix socket each connection is forwarded to.

    Returns:
        `asyncio.base_events.Server`: The listening server.
    """
    return await asyncio.start_server(
        _handler(lambda: asyncio.open_unix_connection(unix_path)), host, port
    )


async def serve_unix_to_tcp(listen_path: str, host: str, port: int) -> Server:
    """Listen on a Unix socket and forward each connection to TCP.

    Args:
        listen_path (`str`):
            Unix socket path to listen on. A stale socket there is removed.
        host (`str`):
            TCP host to connect to.
        port (`int`):
            TCP port to connect to.

    Returns:
        `asyncio.base_events.Server`: The listening server.
    """
    return await _start_unix(listen_path, lambda: asyncio.open_connection(host, port))


async def serve_unix_to_unix(listen_path: str, target_path: str) -> Server:
    """Listen on a Unix socket and forward each connection to another.

    Args:
        listen_path (`str`):
            Unix socket path to listen on. A stale socket there is removed.
        target_path (`str`):
            Unix socket each connection is forwarded to.

    Returns:
        `asyncio.base_events.Server`: The listening server.
    """
    return await _start_unix(
        listen_path, lambda: asyncio.open_unix_connection(target_path)
    )
