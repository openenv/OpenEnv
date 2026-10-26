#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, Sequence

import requests
from huggingface_hub import HfApi, get_token

DEFAULT_SPACE_SUFFIX = "-0.2.3"
GRADIO_HTML_MARKERS = (
    "<gradio-app",
    "window.gradio_config",
    "window.__gradio_config__",
    "gradio_api_info",
)


def collect_space_ids(
    api: HfApi, namespace: str, suffixes: Sequence[str], explicit: Sequence[str]
) -> list[str]:
    suffixes = tuple(suffixes)
    spaces = list(api.list_spaces(author=namespace, limit=500))
    selected: list[str] = []

    for space in spaces:
        if any(space.id.endswith(suffix) for suffix in suffixes):
            selected.append(space.id)

    for space_id in explicit:
        if space_id not in selected:
            selected.append(space_id)

    return selected


def pick_domain_candidates(raw_domains: Iterable[dict]) -> list[dict]:
    ready = []
    fallback = []
    for domain in raw_domains:
        name = domain.get("domain")
        if not name:
            continue
        if domain.get("stage") == "READY":
            ready.append(domain)
        else:
            fallback.append(domain)
    return ready or fallback


def endpoint_ok(path: str, status_code: int | None) -> bool:
    if status_code is None:
        return False
    if path == "/mcp":
        return status_code in {200, 404}
    return status_code == 200


REPL_RESET_PAYLOAD = {
    "context": "alpha beta gamma",
    "task_prompt": "Count the words",
}
REPL_STEP_PAYLOAD = {"action": {"code": "count = len(context.split())"}}


def response_is_gradio_html(response: requests.Response, details) -> bool:
    """Recognize a successful Gradio page after redirects."""
    if response.status_code != 200:
        return False
    if "text/html" not in response.headers.get("Content-Type", ""):
        return False
    if not isinstance(details, str):
        return False

    normalized = details.lower()
    return any(marker in normalized for marker in GRADIO_HTML_MARKERS)


def extract_response_details(response: requests.Response):
    """Return JSON when possible, otherwise trimmed text."""
    if "application/json" in response.headers.get("Content-Type", ""):
        try:
            return response.json()
        except ValueError:
            pass
    return response.text.strip()


def make_probe_result(
    method: str,
    path: str,
    status: int | None,
    ok: bool,
    details,
    payload=None,
) -> dict:
    result = {
        "path": path,
        "method": method,
        "status": status,
        "ok": ok,
        "details": details,
    }
    if payload is not None:
        result["payload"] = payload
    return result


def run_probe_request(
    session: requests.Session,
    base_url: str,
    headers: dict,
    method: str,
    path: str,
    timeout: float,
    payload=None,
    ok_fn=None,
) -> dict:
    full_url = base_url.rstrip("/") + path
    try:
        response = session.request(
            method,
            full_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        details = extract_response_details(response)
        ok = ok_fn(response, details) if ok_fn is not None else endpoint_ok(path, response.status_code)
        return make_probe_result(method, path, response.status_code, ok, details, payload)
    except requests.RequestException as exc:
        return make_probe_result(
            method,
            path,
            None,
            False,
            f"{exc.__class__.__name__}: {exc}",
            payload,
        )


def probe_generic_space(
    session: requests.Session,
    base_url: str,
    headers: dict,
    timeout: float,
) -> list[dict]:
    endpoints = [
        ("GET", "/health", None, None),
        ("GET", "/metadata", None, None),
        ("GET", "/schema", None, None),
        ("POST", "/mcp", None, None),
    ]
    return [
        run_probe_request(
            session,
            base_url,
            headers,
            method,
            path,
            timeout,
            payload=payload,
            ok_fn=ok_fn,
        )
        for method, path, payload, ok_fn in endpoints
    ]


def gradio_web_ok_html(response: requests.Response, details) -> bool:
    return response_is_gradio_html(response, details)


def gradio_web_ok_reset(response: requests.Response, details) -> bool:
    return response.status_code == 200 and isinstance(details, dict) and "observation" in details


def probe_gradio_web_space(
    session: requests.Session,
    base_url: str,
    headers: dict,
    timeout: float,
) -> list[dict]:
    endpoints = [
        ("GET", "/", None, gradio_web_ok_html),
        ("GET", "/web", None, gradio_web_ok_html),
        ("GET", "/web/", None, gradio_web_ok_html),
        ("GET", "/health", None, None),
        ("GET", "/metadata", None, None),
        ("GET", "/schema", None, None),
        ("POST", "/reset", None, gradio_web_ok_reset),
    ]
    return [
        run_probe_request(
            session,
            base_url,
            headers,
            method,
            path,
            timeout,
            payload=payload,
            ok_fn=ok_fn,
        )
        for method, path, payload, ok_fn in endpoints
    ]


def repl_web_ok_reset(response: requests.Response, details) -> bool:
    if response.status_code != 200 or not isinstance(details, dict):
        return False
    observation = details.get("observation", {})
    available_variables = observation.get("available_variables") or []
    return (
        observation.get("context_preview") == REPL_RESET_PAYLOAD["context"]
        and "context" in available_variables
    )


def repl_web_ok_step(response: requests.Response, details) -> bool:
    if response.status_code != 200 or not isinstance(details, dict):
        return False
    result = details.get("observation", {}).get("result", {})
    locals_snapshot = result.get("locals_snapshot") or {}
    return result.get("success") is True and str(locals_snapshot.get("count")) == "3"


def repl_web_ok_state(response: requests.Response, details) -> bool:
    if response.status_code != 200 or not isinstance(details, dict):
        return False
    namespace_keys = details.get("namespace_keys") or []
    return (
        details.get("context") == REPL_RESET_PAYLOAD["context"]
        and details.get("task_prompt") == REPL_RESET_PAYLOAD["task_prompt"]
        and "count" in namespace_keys
    )


def probe_repl_web_space(
    session: requests.Session,
    base_url: str,
    headers: dict,
    timeout: float,
) -> list[dict]:
    endpoints = [
        ("GET", "/", None, gradio_web_ok_html),
        ("GET", "/web", None, gradio_web_ok_html),
        ("GET", "/web/", None, gradio_web_ok_html),
        ("GET", "/health", None, None),
        ("GET", "/metadata", None, None),
        ("GET", "/schema", None, None),
        ("POST", "/mcp", None, None),
        ("POST", "/web/reset", REPL_RESET_PAYLOAD, repl_web_ok_reset),
        ("POST", "/web/step", REPL_STEP_PAYLOAD, repl_web_ok_step),
        ("GET", "/web/state", None, repl_web_ok_state),
    ]
    return [
        run_probe_request(
            session,
            base_url,
            headers,
            method,
            path,
            timeout,
            payload=payload,
            ok_fn=ok_fn,
        )
        for method, path, payload, ok_fn in endpoints
    ]


def probe_space(
    base_url: str,
    headers: dict,
    timeout: float = 20.0,
    probe_profile: str = "generic",
) -> list[dict]:
    session = requests.Session()
    if probe_profile == "repl_web":
        return probe_repl_web_space(session, base_url, headers, timeout)
    if probe_profile == "gradio_web":
        return probe_gradio_web_space(session, base_url, headers, timeout)
    return probe_generic_space(session, base_url, headers, timeout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify private OpenEnv spaces by suffix/explicit IDs."
    )
    parser.add_argument(
        "--hf-namespace",
        default="openenv",
        help="Namespace owning the Spaces (default: openenv).",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=[],
        help=(
            "Suffix to match Spaces names "
            f"(repeatable; defaults to {DEFAULT_SPACE_SUFFIX} when omitted)."
        ),
    )
    parser.add_argument(
        "--space-id",
        action="append",
        default=[],
        help="Explicit Space id to include (repeatable).",
    )
    parser.add_argument(
        "--probe-profile",
        choices=["generic", "gradio_web", "repl_web"],
        default="generic",
        help="Probe contract to validate (default: generic).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout in seconds (default: 20).",
    )
    args = parser.parse_args()
    suffixes = args.suffix or ([] if args.space_id else [DEFAULT_SPACE_SUFFIX])

    token = get_token()
    if not token:
        print("error: no HuggingFace token found; run `hf auth login`.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    space_ids = collect_space_ids(api, args.hf_namespace, suffixes, args.space_id)
    if not space_ids:
        print("error: no spaces found with the provided suffix/ids.", file=sys.stderr)
        sys.exit(1)

    verification: list[dict] = []
    headers = {"Authorization": f"Bearer {token}"}
    overall_success = True

    for sid in space_ids:
        info = api.space_info(sid, expand=["runtime"])
        runtime = getattr(info, "runtime", None)
        stage = getattr(runtime, "stage", None) if runtime else None
        domains = getattr(runtime, "raw", {}).get("domains", []) if runtime else []
        domain_candidates = pick_domain_candidates(domains) if domains else []
        error_message = getattr(runtime, "raw", {}).get("errorMessage") if runtime else None

        domain_attempts: list[dict] = []
        chosen_domain: str | None = None
        space_success = False

        if domain_candidates:
            for candidate in domain_candidates:
                domain_name = candidate["domain"]
                base_url = f"https://{domain_name}"
                checks = probe_space(
                    base_url,
                    headers,
                    timeout=args.timeout,
                    probe_profile=args.probe_profile,
                )
                success = all(check["ok"] for check in checks)
                domain_attempts.append(
                    {
                        "domain": domain_name,
                        "stage": candidate.get("stage"),
                        "success": success,
                        "checks": checks,
                    }
                )
                if success:
                    chosen_domain = domain_name
                    space_success = True
                    break

        overall_success = overall_success and space_success

        verification.append(
            {
                "space": sid,
                "stage": stage,
                "domain": chosen_domain
                or (domain_candidates[0]["domain"] if domain_candidates else None),
                "runtime_error": error_message,
                "probe_profile": args.probe_profile,
                "checks": domain_attempts[-1]["checks"] if domain_attempts else [],
                "success": space_success,
                "domain_attempts": domain_attempts,
            }
        )

    print(json.dumps(verification, indent=2))
    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
