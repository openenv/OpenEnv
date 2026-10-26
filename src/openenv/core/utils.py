# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for OpenEnv core."""

import asyncio
import concurrent.futures
import re


def run_async_safely(coro):
    """
    Run an async coroutine safely from any context.

    This handles the case where we may already be inside an async event loop
    (e.g., when called from an async framework). In that case, asyncio.run()
    would fail, so we use a ThreadPoolExecutor to run in a separate thread.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in async context - run in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        # No async context - use asyncio.run() directly
        return asyncio.run(coro)


def convert_to_ws_url(url: str) -> str:
    """
    Convert an HTTP/HTTPS URL to a WS/WSS URL.

    Args:
        url: The URL to convert.

    Returns:
        The converted WebSocket URL.
    """
    ws_url = url.rstrip("/")
    if ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[7:]
    elif ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[8:]
    elif not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
        ws_url = "ws://" + ws_url
    return ws_url


def hf_space_subdomain(repo_id: str) -> str:
    """
    Convert a Hugging Face Space repo ID to its subdomain / registry name.

    The Hub lowercases the repo ID and replaces every character outside
    ``[a-z0-9]`` with ``-`` (e.g. ``openenv/echo_env`` is served at
    ``openenv-echo-env.hf.space`` and ``registry.hf.space/openenv-echo-env``).

    Args:
        repo_id: Space repo ID in ``owner/name`` form.

    Returns:
        The Space subdomain.
    """
    return re.sub(r"[^a-z0-9]", "-", repo_id.lower())
