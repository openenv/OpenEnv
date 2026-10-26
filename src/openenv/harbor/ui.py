"""Human-facing UI for a Harbor env server.

Three tabs. **Tasks**: pick a dataset and a task, read it (instruction, files, settings), and run an
agent on it. **Runs**: every rollout this server has run, live or finished, one at a time or two to
four side by side. **Setup**: what this machine can run, and why anything it cannot.

It is a Gradio app, like every OpenEnv UI, with custom HTML components where Gradio has no widget
for the job: the task list (thousands of rows, filtered as you type), the task viewer (a file tree
read on demand), the run card, the run list and the trajectory. Gradio supplies the tabs, the state
and the event wiring.

A rollout uses the endpoint the server was started with, or one the visitor connects in the page: a
Hugging Face token and a model on Inference Providers, or any OpenAI-compatible URL such as vLLM.
Connecting is a gate, not a hint: an endpoint without token-id capture answers every request
normally and returns nothing trainable, so a rollout looks perfect and is worthless for training.
What visitors may do (use the server's endpoint, bring their own, see each other's runs) is set per
deployment; see `ui_settings`.

Every argument that names a dataset is checked against the datasets this server serves or that were
added from the Hub in this process. The UI's handlers are callable by anyone who can load the page,
and a dataset spec that is a local path would otherwise let a browser list and read the server's
own files.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
from collections import OrderedDict
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr

from . import ui_data, ui_icons, ui_pages, ui_runs, ui_settings
from .ui_icons import js_prelude

# Hub datasets added from the page in this process, on top of the ones the server was started with.
_ADDED: list[str] = []
_ADDED_LOCK = threading.RLock()
_REMOVING: set[str] = set()
_HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _asset(name: str) -> str:
    return resources.files("openenv.harbor").joinpath("ui_assets", name).read_text()


def _can_add_datasets() -> bool:
    """Whether the page may download datasets from the Hub. Off by default on a Space: a public page
    that downloads any dataset a visitor names is a disk-filling button."""
    return ui_settings.load().add_datasets


def _allowed(spec: str, served: list[str]) -> bool:
    with _ADDED_LOCK:
        return bool(spec) and (spec in served or spec in _ADDED)


_MAX_ADDED = 20
_MAX_INSPECTED = 128
_INSPECT_TTL = 300
_INSPECTED: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_INSPECTED_LOCK = threading.Lock()


def _added_file(settings: ui_settings.UISettings) -> Path | None:
    """Where datasets added from the page are listed, so they come back after a restart: next to
    the run history, whose runs may refer to them. `None` when history is off."""
    if settings.bucket and settings.bucket_mount:
        return None  # the bucket's own folders are the list (`ui_data.added_in_bucket`)
    return settings.runs_dir / ".added-datasets.json" if settings.runs_dir else None


def _save_added(settings: ui_settings.UISettings) -> None:
    path = _added_file(settings)
    if path is None:
        return
    with _ADDED_LOCK:
        saved = sorted(set(_ADDED))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=0))
    except OSError:
        pass


def _load_added(settings: ui_settings.UISettings, served: list[str]) -> None:
    path = _added_file(settings)
    if path is None or not settings.add_datasets:
        return
    try:
        listed = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    with _ADDED_LOCK:
        for spec in listed if isinstance(listed, list) else []:
            # Checked again without the network: the file is ours, but a path must never slip in.
            spec = str(spec)
            parts = spec.split("/")
            if (
                _HUB_ID.match(spec)
                and not any(p in (".", "..") or p.startswith(".") for p in parts)
                and spec not in served
                and spec not in _ADDED
                and len(_ADDED) < _MAX_ADDED
            ):
                _ADDED.append(spec)


def _inspect_hub(spec: str) -> dict[str, Any]:
    """Inspect one Hub dataset, caching only successful answers for a bounded time."""
    now = time.monotonic()
    with _INSPECTED_LOCK:
        cached = _INSPECTED.get(spec)
        if cached is not None and now - cached[0] < _INSPECT_TTL:
            _INSPECTED.move_to_end(spec)
            return cached[1]
        _INSPECTED.pop(spec, None)
    summary = ui_data.hub_summary(spec)
    with _INSPECTED_LOCK:
        _INSPECTED[spec] = (time.monotonic(), summary)
        _INSPECTED.move_to_end(spec)
        while len(_INSPECTED) > _MAX_INSPECTED:
            _INSPECTED.popitem(last=False)
    return summary


def _remove_added(
    spec: str, served: list[str], settings: ui_settings.UISettings
) -> dict[str, Any]:
    """Serialise removal so two requests cannot delete or mutate the same dataset."""
    with _ADDED_LOCK:
        if spec in served or spec not in _ADDED or spec in _REMOVING:
            return {"error": "Only datasets added from this page can be removed."}
        _REMOVING.add(spec)
    try:
        ui_data.remove_added(spec, settings)
    except Exception as exc:  # noqa: BLE001
        try:
            return {
                "error": f"Could not remove it: {type(exc).__name__}: {str(exc)[:200]}"
            }
        finally:
            with _ADDED_LOCK:
                _REMOVING.discard(spec)
    with _ADDED_LOCK:
        _ADDED.remove(spec)
        _REMOVING.discard(spec)
        _save_added(settings)
        return {"ok": True, "spec": spec}


def _hub_problem(spec: str) -> str | None:
    """Why a dataset typed into the page may not be added, or `None`.

    A public Hub dataset only. Not a local path (the loader tries one first, so `src/..` would serve
    this server's own files), and not a private or gated one, which the server's token might open
    for a visitor who has no access of their own.
    """
    parts = spec.split("/")
    if (
        not _HUB_ID.match(spec)
        or any(p in (".", "..") or p.startswith(".") for p in parts)
        or Path(spec).expanduser().exists()
    ):
        return "Enter a Hugging Face dataset id, like org/name."
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=False).dataset_info(spec)
    except Exception:  # noqa: BLE001 - private, missing, or the Hub is down: all the same answer
        return f"{spec} is not a public dataset on the Hub."
    if info.private or info.gated:
        return f"{spec} is private or gated. Only public datasets can be added from the page."
    return None


from .ui_trace import (  # noqa: F401 - re-exported: tests and callers read them from `ui`
    _conversation_html,
    _findings_html,
    _markdown,
    _result_html,
    _transcript_html,
    _turns_html,
    count_actions,
)


def _contract(r: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the legacy audit export, including excluded turns and task rewards.

    Eval rollouts return None; malformed captures raise ValueError.
    """
    turns = r.get("turns") or []
    if not turns or r.get("rollout_type", "train") == "eval":
        return None
    from .contract import export_training_contract
    from .models import HarborRolloutResult

    return export_training_contract(HarborRolloutResult.model_validate(r))


def _training_trace(r: dict[str, Any]) -> dict[str, Any] | None:
    """Download the same validated token objects returned by HarborSession."""
    if not r.get("turns") or r.get("rollout_type", "train") == "eval":
        return None
    from .contract import to_training_trace
    from .models import HarborRolloutResult

    return to_training_trace(HarborRolloutResult.model_validate(r)).model_dump(
        mode="json"
    )


# ── the page's own pieces ─────────────────────────────────────────────────────────────────────────


# Capabilities are credential checks and SDK imports. Worth doing once, not per page load, and
# worth redoing on request, because adding a key is the usual fix for an unusable sandbox.
_CAPS: dict[str, Any] = {"value": None}
_CAPS_LOCK = threading.Lock()


def _capabilities(datasets: list[str], *, refresh: bool = False) -> Any:
    from .capabilities import capabilities
    from .serving import HarborService

    with _CAPS_LOCK:
        if _CAPS["value"] is None or refresh:
            service = HarborService.current()
            llm = (
                {
                    "url": service.llm_url,
                    "model": service.model,
                    "capture_level": service.capture_level,
                    "reachable": True,
                    "ok": service.capture_level == "tokens",
                }
                if service is not None and service.llm_url
                else {}
            )
            _CAPS["value"] = capabilities(datasets=datasets or None, llm=llm)
        return _CAPS["value"]


def _agent_choices(
    caps: Any, *, purpose: str, provider: str, include_experimental: bool
) -> tuple[list[tuple[str, str]], dict[str, str], int]:
    """Which agents to offer, per the qualification evidence, and the profile each was qualified with.

    Stable agents are offered by default and experimental ones only on request, as the provider
    qualification guide specifies. Without a report every agent is experimental, which is why the
    count of hidden ones is returned: an empty list with no reason reads as a broken page.

    Returns:
        `tuple` of `(label, name)` choices, `{harness: profile}`, and how many experimental agents
        the filter hides.
    """
    from .qualification import harness_maturity_rows
    from .seams import get as get_seam

    report_path = os.environ.get("OPENENV_HARBOR_QUALIFICATION_REPORT", "")
    try:
        evidence = json.loads(Path(report_path).read_text()) if report_path else None
        tiers = {
            name: tier
            for name, tier, _ in harness_maturity_rows(
                [h.name for h in caps.harnesses], evidence
            )
        }
    except (OSError, ValueError, TypeError):
        tiers = {h.name: "experimental" for h in caps.harnesses}
        evidence = None
    profile_provider = "vllm" if purpose == "train" else provider
    profiles: dict[str, str] = {}
    unavailable: set[str] = set()
    for cell in (evidence or {}).get("cells", []):
        if cell.get("provider") != profile_provider:
            continue
        config = cell.get("configuration") or {}
        profile = config.get("acp_profile") or config.get("nemo_profile")
        if profile:
            name = cell["harness"]
            profiles[name] = profile
            try:
                get_seam(name, profile=profile)
            except (ValueError, KeyError):
                unavailable.add(name)
    # Agents that passed a development run first, so the likely choices lead the list.
    ordered = sorted(
        (h for h in caps.harnesses if h.name not in unavailable),
        key=lambda h: (h.status != "validated", h.name),
    )
    choices, hidden = [], 0
    for h in ordered:
        tier = tiers.get(h.name, "experimental")
        if tier == "stable" or (include_experimental and tier == "experimental"):
            label = f"{h.name} · {h.dialect} · {tier}"
            if h.name in profiles:
                label += f" · profile: {profiles[h.name]}"
            choices.append((label, h.name))
        elif tier == "experimental":
            hidden += 1
    return choices, profiles, hidden


def _server_engine(
    caps: Any,
    include_experimental: bool = False,
    settings: ui_settings.UISettings | None = None,
) -> dict[str, Any]:
    """The endpoint the server was started with, as the page's default engine.

    It carries no key: the capture proxy already holds the server's credential and uses it for any
    rollout that does not name a different endpoint. A deployment that does not share it with
    visitors (`OPENENV_HARBOR_UI_SERVER_ENDPOINT=0`) starts every visitor with no engine.
    """

    settings = settings or ui_settings.load()
    service = ui_settings.shared_endpoint(settings)
    if service is None:
        return {
            "ok": False,
            "reason": "Connect a model to run rollouts.",
            "include_experimental": include_experimental,
        }
    level = service.capture_level
    purpose = "train" if level == "tokens" else "eval"
    provider = str(service.provider or "openai")
    choices, profiles, hidden = _agent_choices(
        caps,
        purpose=purpose,
        provider=provider,
        include_experimental=include_experimental,
    )
    return {
        "ok": True,
        "server_default": True,
        "source": "server",
        "url": "",
        "model": service.model,
        "host": urlparse(service.llm_url).netloc or service.llm_url,
        "capture_level": level,
        "trainable": level == "tokens",
        "purpose": purpose,
        "provider": provider,
        "allowed_harnesses": [v for _, v in choices],
        "harness_profiles": profiles,
        "choices": choices,
        "hidden_agents": hidden,
        "include_experimental": include_experimental,
        "notes": [],
    }


def validate_endpoint(
    url: str,
    model: str,
    api_key: str,
    provider: str = "openai",
    purpose: str = "eval",
    include_experimental: bool = False,
    datasets: list[str] | None = None,
    private_urls: bool = True,
    kind: str = "custom URL",
) -> dict[str, Any]:
    """Probe an endpoint typed into the page and describe it as an engine for this browser session.

    A rollout reaches it through the capture proxy's pool, so the tier comes from a real probe of
    this endpoint rather than from what the server booted with.

    Args:
        url (`str`):
            OpenAI-spec endpoint.
        model (`str`):
            Served model id; read from the endpoint when it serves exactly one.
        api_key (`str`):
            Credential for a hosted endpoint, or `""`.
        provider (`str`, *optional*, defaults to `"openai"`):
            Upstream API family.
        purpose (`str`, *optional*, defaults to `"eval"`):
            `"eval"`, or `"train"` to require exact token capture.
        include_experimental (`bool`, *optional*, defaults to `False`):
            Offer agents the qualification evidence marks experimental.
        datasets (`list[str]`, *optional*):
            Served datasets, for the capability report.
        private_urls (`bool`, *optional*, defaults to `True`):
            Whether `url` may resolve to a loopback or private address. The server makes the call, so
            on a public deployment this is off (see `ui_settings.url_problem`).
        kind (`str`, *optional*, defaults to `"custom URL"`):
            How a run from this engine is labelled in the run list.

    Returns:
        `dict`: the engine, with `ok` and either `reason` or the agent choices. It holds `api_key`
        for the server-side session state; the page is only ever sent `_card` of it.
    """
    from openenv.core.harness.capture.validate_llm import list_models, validate_llm

    from .capabilities import capabilities
    from .seams import agent_facing_model
    from .serving import HarborService

    url = (url or "").strip().rstrip("/")
    api_key = (api_key or "").strip() or None
    base = {
        "include_experimental": include_experimental,
        "custom": {
            "url": url,
            "model": model or "",
            "provider": provider,
            "purpose": purpose,
        },
    }
    if not url:
        return {
            **base,
            "ok": False,
            "reason": "Enter the endpoint's URL.",
        }
    problem = ui_settings.url_problem(url, private_ok=private_urls)
    if problem:
        return {**base, "ok": False, "reason": problem}
    if not model:
        served_models = list_models(url, timeout=15, api_key=api_key)
        if len(served_models) != 1:
            return {
                **base,
                "ok": False,
                "reason": "Pick a model: this endpoint serves "
                + ", ".join(served_models[:12])
                if served_models
                else "Nothing reachable at that URL. Check it, and the API key if it needs one.",
            }
        model = served_models[0]
    # Bounded: the page waits on this, and an endpoint that never answers must not hold it for minutes.
    report = validate_llm(url, model, api_key=api_key, provider=provider, timeout=45)
    if not report.reachable or (purpose == "train" and not report.trainable):
        why = "; ".join(report.findings) or (
            "exact engine tokens are required for training"
            if report.reachable
            else "unreachable"
        )
        return {
            **base,
            "ok": False,
            "reason": f"Not usable: {why}. Training needs vLLM with --return-tokens-as-token-ids "
            "--logprobs-mode processed_logprobs, or SGLang from git main; eval works with any "
            "reachable endpoint.",
        }
    caps = capabilities(
        datasets=list(datasets or []) or None,
        llm={
            "url": url,
            "model": model,
            "ok": report.ok,
            "capture_level": report.capture_level,
            "reachable": True,
            "authenticated": bool(api_key),
        },
    )
    choices, profiles, hidden = _agent_choices(
        caps,
        purpose=purpose,
        provider=provider,
        include_experimental=include_experimental,
    )
    notes = []
    leaf = agent_facing_model(model)
    if leaf != model:
        notes.append(f"Sent to agents as {leaf}, rewritten back on the way out.")
    notes += [f"Upstream compatibility: {fix}" for fix in report.param_fixes]
    notes += [
        f.split(": ", 2)[-1]
        for f in report.findings
        if "behaviour_changed" in f or "tool_call" in f
    ]
    service = HarborService.current()
    server_url = service.llm_url if service is not None else ""
    if server_url and server_url.rstrip("/") != url:
        notes.append(
            "Rollouts from this page use this endpoint, not the server's default."
        )
    return {
        **base,
        "ok": True,
        "kind": kind,
        "url": url,
        "model": model,
        "host": urlparse(url).netloc or url,
        "capture_level": report.capture_level,
        "trainable": report.trainable,
        # Held in server-side session state so Run can reach a token-gated endpoint. Never sent
        # back to the page: `_card` leaves it out.
        "api_key": api_key or "",
        "provider": provider,
        "purpose": purpose,
        "allowed_harnesses": [v for _, v in choices],
        "harness_profiles": profiles,
        "choices": choices,
        "hidden_agents": hidden,
        "sandboxes": list(caps.available_sandboxes),
        "notes": notes,
    }


def _local_token() -> str:
    try:
        from huggingface_hub import get_token

        return get_token() or ""
    except Exception:  # noqa: BLE001 - no hub client, or an unreadable token file: just none
        return ""


def connect_endpoint(
    form: dict[str, Any],
    *,
    include_experimental: bool = False,
    datasets: list[str] | None = None,
    settings: ui_settings.UISettings | None = None,
    account_token: str | None = None,
) -> dict[str, Any]:
    """The run card's endpoint form, validated into an engine.

    Two sources. `hf`: a model on Hugging Face Inference Providers, reached through the router with
    the visitor's token (or, run locally, this machine's), optionally pinned to one provider or to the
    router's `fastest`/`cheapest` policy. `url`: any OpenAI-compatible or Anthropic endpoint, such as a
    vLLM the visitor runs. Both end in `validate_endpoint`, which probes the endpoint for real.

    Args:
        form (`dict`):
            `source`, and for `hf`: `model`, `route`, `api_key`, `local_token`, `use_account`; for
            `url`: `url`, `model`, `api_key`, `api` (`openai` or `anthropic`), `purpose`.
        account_token (`str`, *optional*):
            The signed-in visitor's Hugging Face token (`inference-api` scope), used when the form
            asks for `use_account`. Gradio reads it from the session; it is never in the form.

    Returns:
        `dict`: the engine, as `validate_endpoint` returns it, with the form (minus the key) under
        `custom` so the card can show what was connected.
    """
    settings = settings or ui_settings.load()
    source = "hf" if form.get("source") == "hf" else "url"
    key = str(form.get("api_key") or "").strip()
    remembered = {
        k: form.get(k)
        for k in (
            "source",
            "model",
            "route",
            "url",
            "api",
            "purpose",
            "local_token",
            "use_account",
        )
        if k in form
    }
    remembered["source"] = source
    if not settings.visitor_endpoints:
        return {
            "ok": False,
            "custom": remembered,
            "reason": "This server runs rollouts on its own endpoint only.",
        }
    common = {"include_experimental": include_experimental, "datasets": datasets}
    if source == "hf":
        model = str(form.get("model") or "").strip()
        route = str(form.get("route") or "").strip()
        if not model:
            return {"ok": False, "custom": remembered, "reason": "Pick a model."}
        if not key and form.get("use_account") and settings.hf_login:
            key = account_token or ""
            if not key:
                return {
                    "ok": False,
                    "custom": remembered,
                    "reason": "Your Hugging Face sign-in has expired. Sign in again, or paste a token.",
                }
        if not key and form.get("local_token") and settings.local_token:
            key = _local_token()
        if not key:
            return {
                "ok": False,
                "custom": remembered,
                "reason": "Enter a Hugging Face token with the Inference Providers permission.",
            }
        engine = validate_endpoint(
            ui_data.HF_ROUTER,
            f"{model}:{route}" if route else model,
            key,
            provider="hf",
            purpose="eval",  # the router returns no token ids, so nothing from it is trainable
            kind="Hugging Face",
            **common,
        )
    else:
        engine = validate_endpoint(
            str(form.get("url") or ""),
            str(form.get("model") or ""),
            key,
            provider="anthropic" if form.get("api") == "anthropic" else "openai",
            purpose="train" if form.get("purpose") == "train" else "eval",
            private_urls=settings.private_urls,
            kind="custom URL",
            **common,
        )
    engine["source"] = source
    engine["custom"] = remembered
    return engine


def _card(
    engine: dict[str, Any],
    selection: dict[str, Any] | None,
    caps: Any,
    message: tuple[str, str] | None = None,
    settings: ui_settings.UISettings | None = None,
    profile: Any = None,
) -> dict[str, Any]:
    """What the run card shows. Everything the browser receives about the engine is chosen here, so
    the API key held in the engine never reaches the page. `profile` is the signed-in visitor's
    Hugging Face profile, where sign-in is set up and they used it."""
    from .serving import HarborService

    settings = settings or ui_settings.load()
    service = HarborService.current()
    shared = ui_settings.shared_endpoint(settings)
    server = (
        {
            "model": shared.model,
            "host": urlparse(shared.llm_url).netloc or shared.llm_url,
            "level_text": ui_pages.LEVEL_TEXT.get(shared.capture_level, ""),
            "train": shared.capture_level == "tokens",
        }
        if shared is not None
        else None
    )
    sources = (["server"] if server else []) + (
        ["hf", "url"] if settings.visitor_endpoints else []
    )
    public = str(service.public_url or "") if service is not None else ""
    host = urlparse(public).hostname or ""
    harnesses = {h.name: h for h in caps.harnesses}
    agents = []
    for label, name in engine.get("choices") or []:
        h = harnesses.get(name)
        agents.append(
            {
                "value": name,
                "label": label,
                "host_side": h is not None and h.kind == "base",
            }
        )
    values = [a["value"] for a in agents]
    hidden = engine.get("hidden_agents") or 0
    sandboxes = [
        {"name": s.name, "available": bool(s.available), "detail": s.detail}
        for s in caps.sandboxes
    ]
    available = [s["name"] for s in sandboxes if s["available"]]
    empty = (
        "No agent is qualified as stable for this model yet."
        if hidden
        else "No agent is available for this endpoint."
    )
    return {
        "stamp": time.time(),
        "task": (
            {k: selection.get(k) for k in ("dataset", "index", "title")}
            if selection and selection.get("spec")
            else None
        ),
        "rollouts": settings.rollouts,
        "sources": sources,
        "server": server,
        "local_token": settings.local_token and bool(_local_token()),
        "hf_login": {
            "on": settings.hf_login,
            "user": (profile.username or profile.name) if profile is not None else None,
        },
        "private_urls": settings.private_urls,
        "engine": {
            "ok": bool(engine.get("ok")),
            "model": engine.get("model"),
            "host": engine.get("host"),
            "source": engine.get("source")
            or ("server" if engine.get("server_default") else ""),
            "train": engine.get("purpose") == "train",
            "level_text": ui_pages.LEVEL_TEXT.get(
                engine.get("capture_level") or "", ""
            ),
            "notes": engine.get("notes") or [],
            "reason": engine.get("reason"),
        },
        "custom": engine.get("custom"),
        "agents": agents,
        "agent": "opencode"
        if "opencode" in values
        else (values[0] if values else None),
        "agents_empty": empty,
        "hidden_agents": hidden,
        "include_experimental": bool(engine.get("include_experimental")),
        "sandboxes": sandboxes,
        "sandbox": "e2b"
        if "e2b" in available
        else (available[0] if available else None),
        # A capture proxy on loopback cannot be reached from a remote sandbox, only by host-side agents.
        "proxy_local": host in ("127.0.0.1", "localhost", "0.0.0.0", "::1"),
        "message": {"tone": message[0], "text": message[1]} if message else None,
    }


def _payload(evt: Any) -> dict[str, Any]:
    data = getattr(evt, "_data", None)
    return data if isinstance(data, dict) else {}


def _signature(listing: list[dict[str, Any]]) -> str:
    """What changes when a run changes state; ticks that change nothing re-render nothing."""
    return json.dumps([(r.get("id"), r.get("status")) for r in listing])


def _js(name: str) -> str:
    """A component's script, with the shared icon set in front of it."""
    return js_prelude() + _asset(name)


def harbor_gradio_builder(
    *,
    datasets: list[str] | None = None,
    title: str | None = None,
) -> gr.Blocks:
    """Build the Harbor UI.

    Args:
        datasets (`list[str]`, *optional*):
            Dataset specs served by this server. Each is browsable; Hub datasets can be added from the
            page when `OPENENV_HARBOR_UI_ADD_DATASETS` allows it.
        title (`str`, *optional*):
            Page title. Defaults to `"OpenEnv × Harbor"`.

    Returns:
        `gr.Blocks`: The interface.
    """
    from .serving import HarborService

    served = list(datasets or [])
    title = title or "OpenEnv × Harbor"
    settings = ui_settings.load()
    runs = ui_runs.manager()
    if not settings.private_urls:
        ui_settings.guard_redirects()
    _load_added(settings, served)
    with _ADDED_LOCK:
        for spec in ui_data.added_in_bucket(settings, served):
            if spec not in _ADDED:
                _ADDED.append(spec)

    def viewer(visitor: str | None) -> str | None:
        """Whose runs this visitor may see: everyone's (`None`), or their own."""
        return (
            None if settings.run_visibility == "all" else ui_settings.owner_of(visitor)
        )

    def listing(visitor: str | None) -> list[dict[str, Any]]:
        return runs.list(viewer(visitor))

    # History is any `*.json` in the runs folder, so a record can be old or hand-edited. One that
    # cannot be drawn is reported as that, rather than breaking the tab it is on.
    def runs_page(
        visitor: str | None, selected: str = "", compare: list[str] | None = None
    ) -> str:
        try:
            return ui_pages.runs_html(
                listing(visitor),
                selected,
                compare,
                history=runs.store is not None,
                own=settings.run_visibility == "own",
            )
        except Exception as exc:  # noqa: BLE001
            return ui_pages.unreadable("Runs", exc)

    def run_page(run_id: str, visitor: str | None) -> str:
        owner = viewer(visitor)
        try:
            return ui_pages.run_html(
                runs.get(run_id, owner),
                runs.live(run_id, owner),
                HarborService.current(),
            )
        except Exception as exc:  # noqa: BLE001
            return ui_pages.unreadable("Runs", exc, run=True)

    # ── server functions: called from the components' JavaScript ──────────────────────────────
    # Gradio passes a server function one value: nothing becomes `[]`, one argument arrives as is,
    # several arrive as a list. So each takes exactly one parameter.
    def hb_datasets(_: Any = None) -> dict[str, Any]:
        from .tasks import resolve_task_dirs

        out = []
        with _ADDED_LOCK:
            added = list(_ADDED)
        for spec in served + [s for s in added if s not in served]:
            row: dict[str, Any] = {
                "spec": spec,
                "label": ui_data.hub_id(spec),
                "added": spec not in served,
                # Removing is adding's undo, so it takes the same permission.
                "removable": spec in added
                and spec not in served
                and _can_add_datasets(),
            }
            try:
                row["num_tasks"] = len(resolve_task_dirs(spec))
            except Exception as exc:  # noqa: BLE001 - one broken dataset must not hide the others
                row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            out.append(row)
        return {"datasets": out, "can_add": _can_add_datasets()}

    def hb_tasks(spec: str) -> dict[str, Any]:
        if not _allowed(spec, served):
            return {"error": "That dataset is not served here."}
        try:
            return {"rows": ui_data.task_rows(spec)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    def hb_hub(query: str) -> list[dict[str, Any]]:
        if not _can_add_datasets():
            return []
        try:
            return ui_data.search_hub(query)
        except Exception:  # noqa: BLE001 - the Hub being unreachable just means no suggestions
            return []

    def hb_add(spec: str) -> dict[str, Any]:
        """Start adding a Hub dataset; the page follows it with `hb_add_status`."""
        spec = (spec or "").strip()
        if not _can_add_datasets():
            return {
                "state": "error",
                "error": "Adding datasets is turned off on this server.",
            }
        target = ui_data.added_spec(spec, settings)
        with _ADDED_LOCK:
            if target in served or target in _ADDED or spec in served:
                return {"spec": spec, "state": "done", "target": target}
            full = len(_ADDED) >= _MAX_ADDED
        problem = _hub_problem(spec)
        if problem:
            return {"spec": spec, "state": "error", "error": problem}
        if full:
            return {
                "spec": spec,
                "state": "error",
                "error": f"This server already holds {_MAX_ADDED} added datasets.",
            }
        return ui_data.start_add(spec, _remember_added, settings)

    def hb_remove(spec: str) -> dict[str, Any]:
        """Remove a dataset added from the page, files and all: its bucket folder, or its download.
        One the server was started with is not the page's to remove."""
        spec = str(spec or "")
        if not _can_add_datasets():
            return {
                "error": "Adding and removing datasets is turned off on this server."
            }
        return _remove_added(spec, served, settings)

    def hb_add_status(spec: str) -> dict[str, Any]:
        return ui_data.add_status(str(spec or "")) or {"spec": spec, "state": "unknown"}

    def hb_inspect(spec: str) -> dict[str, Any]:
        """A Hub dataset's task count (null when not in Harbor's layout) and size, before adding it."""
        spec = str(spec or "").strip()
        if not _can_add_datasets() or not _HUB_ID.match(spec):
            return {"spec": spec, "tasks": None, "bytes": None}
        try:
            return {"spec": spec, **_inspect_hub(spec)}
        except Exception as exc:  # noqa: BLE001 - distinguish Hub failure from an empty dataset
            return {
                "spec": spec,
                "tasks": None,
                "bytes": None,
                "error": f"Could not inspect it: {type(exc).__name__}: {str(exc)[:200]}",
            }

    def _remember_added(spec: str) -> None:
        with _ADDED_LOCK:
            if spec not in _ADDED:
                _ADDED.append(spec)
            _save_added(settings)

    def hb_file(args: list[Any]) -> dict[str, Any]:
        spec, index, path = (list(args or []) + ["", 0, ""])[:3]
        if not _allowed(str(spec), served):
            return {"error": "That dataset is not served here."}
        try:
            return ui_data.read_task_file(spec, int(index), str(path))
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def hb_models(_: Any = None) -> dict[str, Any]:
        if not settings.visitor_endpoints:
            return {"models": []}
        try:
            return {"models": ui_data.hf_models()}
        except Exception as exc:  # noqa: BLE001 - no list means the visitor types a model id instead
            return {"models": [], "error": f"Could not list models: {str(exc)[:160]}"}

    def hb_served(args: list[Any]) -> dict[str, Any]:
        """The models an endpoint serves, for the card's model field. Guarded like a connect."""
        from openenv.core.harness.capture.validate_llm import list_models

        url, key = (list(args or []) + ["", ""])[:2]
        url = str(url or "").strip().rstrip("/")
        if not settings.visitor_endpoints:
            return {"error": "This server runs rollouts on its own endpoint only."}
        problem = ui_settings.url_problem(url, private_ok=settings.private_urls)
        if problem:
            return {"error": problem}
        models = list_models(url, timeout=10, api_key=str(key or "").strip() or None)
        if not models:
            return {
                "error": "Nothing listed at that URL. Check it, and the key if it needs one."
            }
        return {"models": models[:200]}

    def hb_download(args: list[Any]) -> dict[str, Any]:
        token, kind = (list(args or []) + ["", ""])[:2]
        run_id = ui_pages.granted(str(token))
        rec = runs.get(run_id) if run_id else None
        result = (rec or {}).get("result")
        if not result:
            return {"error": "This link has expired. Open the run again."}
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rec.get("id") or "rollout"))
        if kind in {"contract", "training_trace"}:
            try:
                contract = (
                    _training_trace(result)
                    if kind == "training_trace"
                    else _contract(result)
                )
            except (
                ValueError
            ) as exc:  # the exporter refuses a rollout with FATAL findings
                return {"error": str(exc)[:200]}
            if contract is None:
                return {
                    "error": "An eval rollout has nothing to train on, so it has no contract."
                }
            return {
                "name": f"{name}.{kind}.json",
                "text": json.dumps(contract, indent=2),
            }
        return {
            "name": f"{name}.json",
            "text": json.dumps(result, indent=2, default=str),
        }

    # ── handlers ─────────────────────────────────────────────────────────────────────────────────
    # Each tab shows a list or one item, never both. Which one is decided by the stylesheet from what
    # is rendered (a task head means a task is open; a run page means a run is), not by toggling
    # visibility, so the list keeps its filters and scroll, and no late update can show both.

    def on_load(visitor: str | None):
        # Not the run card or its engine: those belong to the task page and are written when a task
        # opens. A `#task=` link opens one while this is still running, and writing the card here too
        # would replace that task's card with an empty one whenever this finished second.
        visitor = visitor or ui_settings.new_visitor()
        caps = _capabilities(served)
        return (
            ui_pages.header_html(title, served, caps, settings),
            ui_pages.setup_html(caps, served, settings),
            runs_page(visitor),
            _signature(listing(visitor)),
            visitor,
        )

    def open_task(
        engine: dict, visitor: str | None, spec: str, index: int, profile: Any = None
    ):
        """The task page for one task: (selection, head, body, card, engine)."""
        caps = _capabilities(served)
        # The first task a page opens starts from the server's endpoint.
        engine = engine or _server_engine(caps, settings=settings)
        if not _allowed(spec, served):
            return (gr.skip(),) * 5
        try:
            detail = ui_data.task_detail(spec, index)
        except Exception as exc:  # noqa: BLE001
            return (
                {},
                ui_pages.task_error_head(spec, index),
                f'<div class="hb-panel">{ui_pages.empty("alert", "Could not open this task", html.escape(str(exc)[:300]))}</div>',
                _card(engine, None, caps, settings=settings, profile=profile),
                engine,
            )
        selection = {
            "spec": spec,
            "dataset": spec,
            "index": index,
            "name": detail["name"],
            "title": detail["title"],
        }
        mine = [
            r
            for r in listing(visitor)
            if r.get("dataset") == spec and r.get("task_index") == index
        ]
        return (
            selection,
            ui_pages.task_head_html(detail, ui_data.common_tags(spec)),
            ui_pages.task_html(detail, mine),
            _card(engine, selection, caps, settings=settings, profile=profile),
            engine,
        )

    def on_open_task(
        engine: dict,
        visitor: str | None,
        evt: gr.EventData,
        profile: gr.OAuthProfile | None = None,
    ):
        data = _payload(evt)
        return open_task(
            engine,
            visitor,
            str(data.get("spec") or ""),
            int(data.get("index") or 0),
            profile,
        )

    def on_back_to_tasks():
        return ""

    def on_tab_again(sel: dict, visitor: str | None, evt: gr.EventData):
        """Tasks or Runs clicked while already showing: back to that tab's list."""
        skip = gr.skip()
        if _payload(evt).get("back") == "tasks":
            return "", skip, skip, skip
        state = {**sel, "id": ""}
        return skip, state, "", runs_page(visitor, "", sel.get("compare", []))

    def show_run(run_id: str, visitor: str | None, sel: dict):
        return {"id": run_id, "compare": sel.get("compare", [])}, run_page(
            run_id, visitor
        )

    def on_open_run(sel: dict, visitor: str | None, evt: gr.EventData):
        return show_run(str(_payload(evt).get("id") or ""), visitor, sel)

    def on_pick_runs(sel: dict, evt: gr.EventData):
        ids = [str(i) for i in (_payload(evt).get("ids") or [])][:4]
        return {**sel, "compare": ids}

    def on_compare(sel: dict, visitor: str | None, evt: gr.EventData):
        ids = [str(i) for i in (_payload(evt).get("ids") or [])][:4]
        owner = viewer(visitor)
        return {"id": "", "compare": ids}, ui_pages.compare_html(
            [runs.get(i, owner) for i in ids]
        )

    def on_task_page(sel: dict, visitor: str | None, evt: gr.EventData):
        """A run listed on a task page was clicked: show it on the Runs tab."""
        run_id = str(_payload(evt).get("run") or "")
        if not run_id:
            return (gr.skip(),) * 4
        state, view = show_run(run_id, visitor, sel)
        return gr.Tabs(selected="runs"), state, view, runs_page(visitor, run_id)

    def on_run_page(
        engine: dict,
        sel: dict,
        visitor: str | None,
        evt: gr.EventData,
        profile: gr.OAuthProfile | None = None,
    ):
        """The run page's own links: back to the list, another run, or the task it ran."""
        data = _payload(evt)
        skip = gr.skip()
        if data.get("run"):
            return (*show_run(str(data["run"]), visitor, sel), *(skip,) * 7)
        if data.get("task") is not None:
            task = open_task(
                engine,
                visitor,
                str(data.get("task") or ""),
                int(data.get("index") or 0),
                profile,
            )
            return (skip, skip, gr.Tabs(selected="tasks"), *task, skip)
        listing_now = runs_page(visitor, "", sel.get("compare", []))
        return ({**sel, "id": ""}, "", skip, *(skip,) * 5, listing_now)

    def on_experimental(
        engine: dict,
        selection: dict,
        evt: gr.EventData,
        profile: gr.OAuthProfile | None = None,
    ):
        include = bool(_payload(evt).get("include_experimental"))
        caps = _capabilities(served)
        if engine.get("server_default") or not engine:
            engine = _server_engine(
                caps, include_experimental=include, settings=settings
            )
        elif not engine.get("ok"):
            engine = {**engine, "include_experimental": include}
        else:
            choices, profiles, hidden_n = _agent_choices(
                caps,
                purpose=engine.get("purpose", "eval"),
                provider=engine.get("provider", "openai"),
                include_experimental=include,
            )
            engine = {
                **engine,
                "choices": choices,
                "allowed_harnesses": [v for _, v in choices],
                "harness_profiles": profiles,
                "hidden_agents": hidden_n,
                "include_experimental": include,
            }
        return _card(
            engine, selection, caps, settings=settings, profile=profile
        ), engine

    def on_connect(
        engine: dict,
        selection: dict,
        evt: gr.EventData,
        profile: gr.OAuthProfile | None = None,
        token: gr.OAuthToken | None = None,
    ):
        new = connect_endpoint(
            _payload(evt),
            include_experimental=bool(engine.get("include_experimental")),
            datasets=served,
            settings=settings,
            account_token=token.token if token is not None else None,
        )
        caps = _capabilities(served)
        if not new.get("ok"):
            # A failed check leaves the working engine exactly as it was, `custom` included: the card
            # compares its form with `custom` to know whether Run still means what it shows.
            return _card(
                engine,
                selection,
                caps,
                ("bad", str(new.get("reason"))),
                settings,
                profile=profile,
            ), engine
        return _card(
            new,
            selection,
            caps,
            ("ok", f"Connected to {new['model']}."),
            settings,
            profile=profile,
        ), new

    def on_server_default(
        engine: dict, selection: dict, profile: gr.OAuthProfile | None = None
    ):
        caps = _capabilities(served)
        new = _server_engine(
            caps,
            include_experimental=bool(engine.get("include_experimental")),
            settings=settings,
        )
        return _card(new, selection, caps, settings=settings, profile=profile), new

    def on_run(
        engine: dict,
        selection: dict,
        visitor: str | None,
        evt: gr.EventData,
        profile: gr.OAuthProfile | None = None,
    ):
        """Start a rollout and show it on the Runs tab. It keeps running if the page is closed."""
        data = _payload(evt)
        harness, sandbox = str(data.get("agent") or ""), str(data.get("sandbox") or "")
        caps = _capabilities(served)
        keep = (gr.skip(),) * 5

        def say(text: str):
            return (
                _card(
                    engine, selection, caps, ("bad", text), settings, profile=profile
                ),
                *keep,
            )

        if not settings.rollouts:
            return say("Rollouts are turned off on this server.")
        if not engine.get("ok"):
            return say("Connect a model first.")
        if engine.get("server_default") and not settings.server_endpoint:
            return say("This server's endpoint is not shared. Connect your own model.")
        if not engine.get("server_default") and not settings.visitor_endpoints:
            return say("This server runs rollouts on its own endpoint only.")
        if harness not in engine.get("allowed_harnesses", []):
            return say(
                "That agent is outside the qualified set for this endpoint. Pick another."
            )
        spec = (selection or {}).get("spec", "")
        if not _allowed(spec, served):
            return say("Pick a task first.")
        # The card only offers these, but the event is the browser's to send.
        if sandbox not in caps.available_sandboxes:
            return say("That sandbox is not available on this server.")
        # Harbor fills `${VAR}` in a task's settings from this server's environment, where its keys
        # are. Even a served task cannot receive them when a visitor controls the model: its trace is
        # visible to that visitor, so the model can print a secret it finds in the sandbox.
        try:
            reads_env = ui_data.reads_environment(spec, int(selection.get("index", 0)))
        except Exception as exc:  # noqa: BLE001 - a task that cannot be read cannot be run
            return say(
                f"Could not read this task: {type(exc).__name__}: {str(exc)[:200]}"
            )
        if reads_env and spec not in served:
            return say(
                "This task reads environment variables or files from the server, which only "
                "datasets the server was started with may do."
            )
        if reads_env and not engine.get("server_default"):
            return say(
                "This task passes the server's environment variables or files into the sandbox, so "
                "it runs only on the server's own endpoint, never on a model you connect."
            )
        service = HarborService.current()
        if service is None:
            return say(
                "This server has no capture proxy running, so it cannot run rollouts."
            )
        visitor = visitor or ui_settings.new_visitor()
        try:
            run_id = runs.start(
                engine=engine,
                spec=spec,
                index=int(selection.get("index", 0)),
                harness=harness,
                sandbox=sandbox,
                service=service,
                owner=ui_settings.owner_of(visitor),
                title=str(selection.get("title") or ""),
                per_owner=settings.max_runs_per_visitor,
                private_urls=settings.private_urls,
                # a browser id is free to replace, so a signed-in visitor's cap follows the account
                quota=f"hf:{profile.username}"
                if profile is not None and profile.username
                else "",
            )
        except (RuntimeError, IndexError, ValueError) as exc:
            return say(str(exc))
        except Exception as exc:  # noqa: BLE001 - the card must never be left on "Starting…"
            return say(f"Could not start: {type(exc).__name__}: {str(exc)[:200]}")
        return (
            _card(
                engine,
                selection,
                caps,
                ("ok", f"Started {harness} on {sandbox}."),
                settings,
                profile=profile,
            ),
            gr.Tabs(selected="runs"),
            {"id": run_id, "compare": []},
            runs_page(visitor, run_id),
            run_page(run_id, visitor),
            _signature(listing(visitor)),
        )

    def on_tick(sel: dict, sig: str, visitor: str | None):
        """Refresh the run list when a run changes state, and the open run while it is live."""
        items = listing(visitor)
        new_sig = _signature(items)
        run_id = sel.get("id") or ""
        live = runs.live(run_id, viewer(visitor)) if run_id else None
        changed = new_sig != sig
        list_out = (
            runs_page(visitor, run_id, sel.get("compare", [])) if changed else gr.skip()
        )
        # live: redraw as it grows; just finished: draw the result once
        view = (
            run_page(run_id, visitor)
            if run_id and (live is not None or changed)
            else gr.skip()
        )
        return list_out, view, new_sig

    def on_refresh_setup():
        caps = _capabilities(served, refresh=True)
        return ui_pages.setup_html(caps, served, settings), ui_pages.header_html(
            title, served, caps, settings
        )

    # ── layout ───────────────────────────────────────────────────────────────────────────────────
    html_opts = {"padding": False, "apply_default_css": False, "elem_classes": "hb"}
    with gr.Blocks(title=title, fill_width=True) as app:
        # apply_default_css=False on every HTML block: Gradio's default wraps it in `.prose`, and the
        # OpenEnv theme strips the border and background from anything directly inside `.prose`.
        gr.HTML(
            f"<style>{_asset('harbor.css')}</style>{ui_icons.json_tag()}",
            padding=False,
            apply_default_css=False,
        )
        header = gr.HTML(
            ui_pages.header_html(title, served, settings=settings),
            js_on_load=_js("header.js"),
            **html_opts,
        )
        engine_state = gr.State({})
        selection = gr.State({})
        run_sel = gr.State({"id": "", "compare": []})
        runs_sig = gr.State("")
        # Which runs are this browser's: a random id kept in the browser, stored on each run as a digest.
        visitor = gr.BrowserState(
            None,
            storage_key="openenv-harbor-visitor",
            secret=ui_settings.visitor_secret(settings),
        )

        with gr.Tabs(selected="tasks", elem_classes="hb-tabs") as tabs:
            with gr.Tab("Tasks", id="tasks"):
                with gr.Column(elem_classes="hb-page hb-browse"):
                    browser = gr.HTML(
                        html_template=_asset("task_browser.html"),
                        js_on_load=_js("task_browser.js"),
                        server_functions=[
                            hb_datasets,
                            hb_tasks,
                            hb_hub,
                            hb_add,
                            hb_add_status,
                            hb_inspect,
                            hb_remove,
                        ],
                        **html_opts,
                    )
                with gr.Column(elem_classes="hb-page hb-taskpage"):
                    task_head = gr.HTML("", js_on_load=_js("task_head.js"), **html_opts)
                    with gr.Row(elem_classes="hb-row", equal_height=False):
                        with gr.Column(elem_classes="hb-main", min_width=0):
                            task_view = gr.HTML(
                                "",
                                js_on_load=_js("task_view.js"),
                                server_functions=[hb_file],
                                **html_opts,
                            )
                        with gr.Column(elem_classes="hb-side", min_width=0):
                            card = gr.HTML(
                                {},
                                html_template="",
                                js_on_load=_js("run_card.js"),
                                server_functions=[hb_models, hb_served],
                                **html_opts,
                            )

            with gr.Tab("Runs", id="runs"):
                with gr.Column(elem_classes="hb-page hb-runlist"):
                    runs_list = gr.HTML(
                        runs_page(None), js_on_load=_js("run_list.js"), **html_opts
                    )
                with gr.Column(elem_classes="hb-page hb-runpage"):
                    run_view = gr.HTML(
                        "",
                        js_on_load=_js("run_view.js"),
                        server_functions=[hb_download],
                        **html_opts,
                    )

            with gr.Tab("Setup", id="setup"):
                setup_view = gr.HTML("", **html_opts)
                with gr.Row(elem_classes="hb-actions"):
                    refresh_btn = gr.Button("Check sandboxes again", size="sm")
                _qualification_panel()

        timer = gr.Timer(2.0)

        # ── wiring ───────────────────────────────────────────────────────────────────────────────
        task_out = [selection, task_head, task_view, card, engine_state]
        run_out = [run_sel, run_view]
        app.load(
            on_load,
            [visitor],
            [header, setup_view, runs_list, runs_sig, visitor],
        )
        header.select(
            on_tab_again, [run_sel, visitor], [task_head, run_sel, run_view, runs_list]
        )
        browser.select(on_open_task, [engine_state, visitor], task_out)
        task_head.select(on_back_to_tasks, None, [task_head])
        task_view.select(on_task_page, [run_sel, visitor], [tabs, *run_out, runs_list])
        # Not `.change`: Gradio fires that on every value update from Python too, which would loop.
        card.select(on_experimental, [engine_state, selection], [card, engine_state])
        # Gradio runs one call of each event at a time by default. A connect waits on someone's
        # endpoint and a tick fires every two seconds per open page, so neither may queue the rest.
        card.input(
            on_connect,
            [engine_state, selection],
            [card, engine_state],
            concurrency_limit=8,
        )
        card.clear(on_server_default, [engine_state, selection], [card, engine_state])
        card.submit(
            on_run,
            [engine_state, selection, visitor],
            [card, tabs, run_sel, runs_list, run_view, runs_sig],
            concurrency_limit=8,
        )
        runs_list.select(on_open_run, [run_sel, visitor], run_out)
        runs_list.input(on_pick_runs, [run_sel], [run_sel])
        runs_list.submit(on_compare, [run_sel, visitor], run_out)
        run_view.select(
            on_run_page,
            [engine_state, run_sel, visitor],
            [*run_out, tabs, *task_out, runs_list],
        )
        timer.tick(
            on_tick,
            [run_sel, runs_sig, visitor],
            [runs_list, run_view, runs_sig],
            show_progress="hidden",
            concurrency_limit=None,
        )
        refresh_btn.click(on_refresh_setup, None, [setup_view, header])
    return app


def _qualification_panel() -> None:
    """The recorded harness/provider evidence: what was qualified, where, and with what."""
    from .qualification import (
        harness_maturity_rows,
        qualification_details,
        qualification_rows,
    )
    from .seams import SEAMS

    report_path = os.environ.get("OPENENV_HARBOR_QUALIFICATION_REPORT", "")

    def read_evidence():
        try:
            evidence = (
                json.loads(Path(report_path).read_text()) if report_path else None
            )
            return (
                qualification_rows(list(SEAMS), evidence),
                qualification_details(evidence),
                harness_maturity_rows(list(SEAMS), evidence),
                "Loaded recorded evidence."
                if evidence
                else "No qualification report configured (OPENENV_HARBOR_QUALIFICATION_REPORT).",
            )
        except (OSError, ValueError, TypeError) as exc:
            return (
                qualification_rows(list(SEAMS)),
                [],
                harness_maturity_rows(list(SEAMS)),
                f"Invalid qualification report: {exc}",
            )

    rows, details, maturity, status = read_evidence()
    with gr.Accordion("Harness and provider qualification evidence", open=False):
        gr.Markdown(
            "Recorded results apply to the listed model, harness version and captures. They do not "
            "certify the endpoint selected on the Tasks tab. Stable means all four recorded profiles "
            "passed, including optimizer replay; experimental adapters have partial or pending "
            "support; unstable ones have no passing profile."
        )
        status_md = gr.Markdown(status)
        maturity_table = gr.Dataframe(
            headers=["Harness", "Maturity", "Qualification scope"],
            value=maturity,
            interactive=False,
        )
        evidence_table = gr.Dataframe(
            headers=[
                "Harness",
                "OpenAI eval",
                "Anthropic eval",
                "HF eval",
                "vLLM training",
            ],
            value=rows,
            interactive=False,
        )
        detail_table = gr.Dataframe(
            headers=[
                "Harness",
                "Provider",
                "Status",
                "Model",
                "Harness version",
                "Tasks",
                "Workflow profile",
                "Optimizer scope",
                "Optimizer model revision",
                "Capture evidence",
                "Reason",
            ],
            value=details,
            interactive=False,
        )
        refresh = gr.Button("Refresh recorded evidence", size="sm")
        refresh.click(
            read_evidence, [], [evidence_table, detail_table, maturity_table, status_md]
        )
