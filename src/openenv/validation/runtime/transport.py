"""Socket cancellation for bounded validation transport operations."""

import socket
import threading
import time
from contextlib import contextmanager


def abort_socket(transport_socket):
    """Interrupt pending reads/writes before closing the transport."""
    try:
        transport_socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # The peer or another cleanup path may already have closed it.
    try:
        transport_socket.close()
    except OSError:
        pass  # Cleanup must not replace the original operation error.


@contextmanager
def http_deadline(timeout_s):
    """Yield HTTPX trace extensions enforcing one request/body wall deadline.

    The request must establish a fresh connection. Reused clients must disable
    keep-alive with `httpx.Limits(max_keepalive_connections=0)` so the trace exposes
    each request's transport before headers are read.
    """
    if timeout_s <= 0:
        raise TimeoutError("HTTP deadline elapsed")
    deadline = time.monotonic() + timeout_s
    expired = threading.Event()
    watched_socket = None

    def abort():
        expired.set()
        if watched_socket is not None:
            abort_socket(watched_socket)

    def trace(event, info):
        nonlocal watched_socket
        if event == "connection.connect_tcp.complete":
            # Retain a duplicate descriptor: TLS wrapping detaches the original
            # socket object, but shutdown on this handle still aborts the shared
            # connection, including a handshake or a slow header/body read.
            watched_socket = info["return_value"].get_extra_info("socket").dup()
            if expired.is_set() or time.monotonic() >= deadline:
                abort()
                raise TimeoutError("HTTP deadline elapsed")

    watchdog = threading.Timer(timeout_s, abort)
    watchdog.daemon = True
    watchdog.start()
    try:
        yield {"trace": trace}
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("HTTP deadline elapsed")
    except Exception:
        if expired.is_set():
            raise TimeoutError("HTTP deadline elapsed") from None
        raise
    finally:
        watchdog.cancel()
        watchdog.join()
        if watched_socket is not None:
            watched_socket.close()
