"""Bounded task metadata sampling through the production task API."""

import json
import time
from urllib.parse import quote

import httpx

from .transport import http_deadline

MAX_TASK_SPLITS = 64
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_DISCOVERY_BYTES = 8 * 1024 * 1024


def collect_task_evidence(base_url, *, deadline, request_timeout_s=None):
    """Return raw counts and at most two task specs per split, never a full listing.

    ``deadline`` is the collector's absolute monotonic episode deadline. Optional
    ``request_timeout_s`` may further cap each request. Errors are sanitized;
    a failed/unsupported endpoint never becomes an empty success.
    Task metadata routes intentionally use independent environment instances.
    """
    total_bytes = 0

    def remaining():
        budget = deadline - time.monotonic()
        if request_timeout_s is not None:
            budget = min(request_timeout_s, budget)
        if budget <= 0:
            raise TimeoutError("task discovery deadline exceeded")
        return budget

    try:
        prefix = base_url.rstrip("/")
        with httpx.Client(
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:

            def request(method, route, payload=None):
                nonlocal total_bytes
                # Each request opens a transport for its deadline watchdog;
                # pooled connections do not emit the connection trace event.
                with (
                    http_deadline(remaining()) as extensions,
                    client.stream(
                        method,
                        prefix + route,
                        json=payload,
                        timeout=remaining(),
                        headers={"Accept-Encoding": "identity"},
                        extensions=extensions,
                    ) as response,
                ):
                    response.raise_for_status()
                    if (
                        response.headers.get("Content-Encoding", "identity")
                        .strip()
                        .lower()
                        != "identity"
                    ):
                        raise ValueError("compressed discovery response")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        remaining()
                        total_bytes += len(chunk)
                        if (
                            len(body) + len(chunk) > MAX_RESPONSE_BYTES
                            or total_bytes > MAX_DISCOVERY_BYTES
                        ):
                            raise ValueError("task discovery exceeds byte budget")
                        body.extend(chunk)
                return json.loads(body)

            environments = request("GET", "/list_environments")
            if (
                not isinstance(environments, list)
                or len(environments) != 1
                or not isinstance(environments[0], str)
                or not environments[0]
                or environments[0] in {".", ".."}
                or "/" in environments[0]
                or "\\" in environments[0]
            ):
                raise ValueError("expected one valid task environment namespace")
            prefix += "/" + quote(environments[0], safe="")
            splits = request("GET", "/splits")
            if not isinstance(splits, list) or len(splits) > MAX_TASK_SPLITS:
                raise ValueError("invalid or excessive splits")
            counts, previews = {}, {}
            for split in splits:
                name = split["name"]
                if not isinstance(name, str) or not name or name in counts:
                    raise ValueError("invalid or duplicate split name")
                count = request("POST", "/num_tasks", {"split": name})["num_tasks"]
                if type(count) is not int or count < 0:
                    raise ValueError("invalid task count")
                counts[name] = count
                previews[name] = [
                    request("POST", "/task", {"split": name, "index": index})["task"]
                    for index in range(min(2, count))
                ]
            discovered = json.dumps(
                {
                    "environments": environments,
                    "splits": splits,
                    "counts": counts,
                    "previews": previews,
                },
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(discovered.encode()) > MAX_DISCOVERY_BYTES:
                raise ValueError("task evidence exceeds byte budget")
            return discovered, None
    except (
        httpx.HTTPError,
        httpx.InvalidURL,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
    ) as error:
        return None, f"task discovery failed ({type(error).__name__})"
