"""Everything that must be true before the server accepts a request.

Four checks, in the order that fails cheapest first:

1. **LLM** — is it reachable, and what can it return? A vLLM without
   `--return-tokens-as-token-ids --logprobs-mode processed_logprobs` answers every request perfectly
   well and returns no ids, so every rebuilt training row would be empty with nothing reporting an
   error. That failure has no loud edge, so the endpoint is probed first and the answer — the capture
   level — is attached to everything this server later produces. Only an *unreachable* endpoint is
   fatal: one that cannot return token ids is an eval backend, and saying so loudly beats refusing to
   start, since every hosted provider lands in that category.
2. **sandbox credentials** — via Harbor's own `preflight()`, so the message names the exact missing
   variable rather than us guessing at one.
3. **datasets** — resolved and downloaded up front. A 2000-task repo takes real time to fetch, and a
   mistyped dataset name should fail here rather than on the first rollout.
4. **report** — print what is usable and, more usefully, what is not and why.

A caller gets the same `Capabilities` object the server serves over the wire, so what is printed at
startup and what a client can query are the same data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openenv.core.harness.capture.validate_llm import ENGINE_HINT

from .capabilities import Capabilities, capabilities

# Load once, at import, so that Harbor's per-backend preflight sees the keys. Harbor reads
# credentials from the process environment and never from a file, so a `.env` that is not exported
# is invisible to it.
_ENV_LOADED = False


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Export `KEY=value` pairs from a dotenv file into the process environment.

    Existing variables win: an operator who exported something deliberately should not have it
    silently replaced by a checked-in file.

    Args:
        path (`str` or `Path`, *optional*):
            The file to read. Defaults to `$OPENENV_ENV_FILE`, then `./.env`.

    Returns:
        `list[str]`: Names of the variables that were set (values are never returned or logged).
    """
    global _ENV_LOADED
    candidate = Path(path or os.environ.get("OPENENV_ENV_FILE") or ".env").expanduser()
    if not candidate.is_file():
        return []

    applied: list[str] = []
    for raw in candidate.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    _ENV_LOADED = True
    return applied


def prepare(
    *,
    llm_url: str | None = None,
    model: str | None = None,
    datasets: list[str] | None = None,
    sandboxes: tuple[str, ...] | None = None,
    env_file: str | Path | None = None,
    require_llm: bool = True,
    quiet: bool = False,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    provider: str = "openai",
) -> Capabilities:
    """Run every startup check and return what this server can do.

    Args:
        llm_url (`str`, *optional*):
            OpenAI-spec inference endpoint. No default: callers pass it explicitly.
        model (`str`, *optional*):
            Served model id. Defaults to `$OPENENV_MODEL`, else the LLM's only served model.
        datasets (`list[str]`, *optional*):
            Dataset specs to serve. Defaults to `$OPENENV_DATASETS` (comma-separated).
        sandboxes (`tuple[str, ...]`, *optional*):
            Backends to check. Defaults to `$OPENENV_SANDBOXES`, else the known set.
        env_file (`str` or `Path`, *optional*):
            Dotenv file to load before checking credentials.
        require_llm (`bool`, *optional*, defaults to `True`):
            Require a reachable endpoint. This no longer means "must be trainable": an endpoint that
            cannot return token ids is an eval backend, and refusing it would rule out every hosted
            provider. It must still answer.
        quiet (`bool`, *optional*, defaults to `False`):
            Suppress the printed report.
        api_key (`str`, *optional*):
            Upstream credential. Defaults to `$OPENENV_LLM_API_KEY`.
        auth_header (`str`, *optional*, defaults to `"Authorization"`):
            Header to send `api_key` under.

    Returns:
        [`Capabilities`]: Harnesses, sandboxes, datasets and LLM status, including `capture_level`.

    Raises:
        RuntimeError: If `require_llm` and no URL was given, or the endpoint is unreachable.
    """
    load_env_file(env_file)

    model = model or os.environ.get("OPENENV_MODEL", "")
    # Read after `load_env_file`, so a key in the dotenv is picked up like every other credential.
    api_key = api_key or os.environ.get("OPENENV_LLM_API_KEY", "") or None
    if datasets is None:
        raw = os.environ.get("OPENENV_DATASETS", "")
        datasets = [d.strip() for d in raw.split(",") if d.strip()]
    if sandboxes is None:
        raw = os.environ.get("OPENENV_SANDBOXES", "")
        sandboxes = tuple(s.strip() for s in raw.split(",") if s.strip()) or None

    if require_llm and not llm_url:
        # A serving deployment no longer needs one. The hazard this guarded against — an unset
        # endpoint yielding rollouts that look fine and carry no token ids — is now handled where it
        # belongs: every rollout names its engine, that engine is probed when the session is created,
        # and the measured tier travels with the result. So an engineless server is a server waiting
        # to be told which engine to use, not a misconfigured one.
        #
        # Callers that genuinely need an engine up front (`harbor rollout`, which runs a batch itself
        # and has nowhere else to get one) still pass `require_llm=True` and still get this.
        raise RuntimeError(
            "no LLM URL given. Pass --llm-url (or llm_url=) explicitly; there is no default, "
            "because this entry point runs rollouts itself and has no session to take an engine "
            "from. A served deployment (`harbor serve`) does not need one: rollouts name their own."
        )

    llm: dict[str, Any] = {}
    if llm_url:
        # Imported from the module rather than the package: the package re-exports a *function*
        # named `validate_llm`, which shadows the same-named submodule.
        from openenv.core.harness.capture.validate_llm import list_models, validate_llm

        # With no model given, ask the LLM what it serves. Convenient, and it also removes the
        # commonest startup mistake: guessing a short alias for a server that publishes its full
        # repo id.
        if not model:
            served = list_models(llm_url, api_key=api_key, auth_header=auth_header)
            model = served[0] if len(served) == 1 else ""

        report = (
            validate_llm(
                llm_url,
                model,
                api_key=api_key,
                auth_header=auth_header,
                provider=provider,
            )
            if model
            else None
        )
        if report is None:
            llm = {
                "url": llm_url,
                "model": "",
                "ok": False,
                "findings": [
                    "no model given and the endpoint does not serve exactly one; pass --model"
                ],
                "authenticated": bool(api_key),
            }
        else:
            llm = {
                "url": llm_url,
                "model": report.model,
                "ok": report.ok,
                "findings": report.findings,
                "served_models": report.served_models,
                "capture_level": report.capture_level,
                "rollout_type": report.rollout_type,
                "trainable": report.trainable,
                "reachable": report.reachable,
                "param_fixes": report.param_fixes,
                "logprobs_mode": report.logprobs_mode,
                "tool_support": report.tool_support,
                # A boolean, never the key: `Capabilities` is rendered to stdout and served over the
                # wire by `/metadata`.
                "authenticated": bool(api_key),
            }

    kwargs: dict[str, Any] = {"datasets": datasets, "llm": llm}
    if sandboxes:
        kwargs["sandboxes"] = sandboxes
    caps = capabilities(**kwargs)

    if not quiet:
        print(caps.render())

    # Only unreachability is fatal now. An endpoint that answers but cannot return token ids is an
    # eval backend, which is a supported way to run this server — the tier is stamped on `caps`, on
    # `/health` and on every result, and no training contract is ever built from it. Refusing here
    # instead would mean OpenAI, Anthropic and HF Inference Providers could not be used at all.
    if require_llm and llm and not llm.get("reachable"):
        raise RuntimeError(
            "inference endpoint is not usable:\n  "
            + "\n  ".join(llm.get("findings") or ["unreachable"])
            + "\n\n"
            + ENGINE_HINT
        )
    return caps
