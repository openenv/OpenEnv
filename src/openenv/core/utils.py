# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for OpenEnv core."""

import asyncio
import concurrent.futures
import json
import os
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


_SECRET_PATTERN = re.compile(
    r"hf_[A-Za-z0-9]{30,}"
    r"|sk-[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_SECRET_ENV_NAME = re.compile(r"(^|_)(TOKEN|KEY|SECRET|PASSWORD)(_|$)", re.IGNORECASE)


def redact_secrets(text: str) -> tuple[str, int]:
    """
    Replace credentials in `text` with `<REDACTED>`.

    Covers Hugging Face, OpenAI/Anthropic, GitHub and AWS keys, private key blocks, and the value of
    every environment variable of this process with `TOKEN`, `KEY`, `SECRET` or `PASSWORD` as a part
    of its name (`HF_TOKEN`, `OPENAI_API_KEY`), also in its JSON-escaped form. Agent traces can hold
    any of them: a model that runs `env` or reads a config file puts the value in its own output.

    Args:
        text (`str`):
            Text to scan.

    Returns:
        `tuple[str, int]`: The redacted text and the number of secrets replaced.
    """
    count = 0
    for name, value in os.environ.items():
        if not _SECRET_ENV_NAME.search(name) or len(value) < 8:
            continue
        # Traces are JSON, where quotes, backslashes and non-ASCII characters are escaped. The
        # escaped form goes first: it can contain the raw one (a value ending in a backslash).
        for form in (json.dumps(value)[1:-1], value):
            count += text.count(form)
            text = text.replace(form, "<REDACTED>")
    text, matches = _SECRET_PATTERN.subn("<REDACTED>", text)
    return text, count + matches
