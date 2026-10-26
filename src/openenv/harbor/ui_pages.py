"""The Harbor UI's pages, rendered as HTML: the header, a task, the run list, a run, a comparison,
and the setup view. The task browser and the run card render themselves in the browser (see
`ui_assets/`); everything else is built here, from data the handlers in `ui.py` hand over.

Everything shown comes from a dataset, a model or a sandbox, so every value is escaped.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlparse

from . import ui_runs, ui_settings
from .ui_icons import icon
from .ui_trace import (
    _conversation_html,
    _duration,
    _e,
    _markdown,
    _plural,
    _result_html,
    _transcript_html,
    _turns_html,
    count_actions,
)

LEVEL_TEXT = {
    "tokens": "token ids and logprobs",
    "logprobs": "logprobs, no token ids",
    "text": "text only",
}


def _short(spec: Any) -> str:
    return str(spec or "").rstrip("/").split("/")[-1]


def reward_badge(reward: Any) -> str:
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return '<span class="hb-reward none">–</span>'
    tone = "full" if value >= 0.999 else "part" if value > 0 else "zero"
    return f'<span class="hb-reward {tone}">{value:.2f}</span>'


_STATUS = {
    "starting": ("live", "Starting"),
    "running": ("live", "Running"),
    "done": ("ok", "Done"),
    "failed": ("bad", "Failed"),
}


def status_badge(status: str | None) -> str:
    tone, label = _STATUS.get(status or "", ("", status or "unknown"))
    mark = (
        '<span class="hb-pulse"></span>'
        if tone == "live"
        else icon("check" if tone == "ok" else "x", 12)
        if tone
        else ""
    )
    return f'<span class="hb-status {tone}">{mark}{_e(label)}</span>'


def ago(ts: Any) -> str:
    if not ts:
        return ""
    s = time.time() - float(ts)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return time.strftime("%b %-d", time.localtime(float(ts)))


def size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.0f} KB" if n < 1048576 else f"{n / 1048576:.1f} MB"


def empty(ic: str, title: str, text: str = "") -> str:
    return f'<div class="hb-empty">{icon(ic, 20)}<h3>{_e(title)}</h3>{f"<p>{text}</p>" if text else ""}</div>'


# ── downloads ────────────────────────────────────────────────────────────────────────────────────
# A run page's download buttons call a server function, which cannot see who is asking. So the page
# carries a grant: an unguessable token, issued when the run was rendered for someone allowed to see
# it, that names that one run. With per-visitor runs, only a visitor who could open the run holds one.

_GRANTS: dict[str, tuple[str, float]] = {}
_BY_RUN: dict[str, str] = {}
_GRANTS_LOCK = threading.Lock()
_GRANT_TTL = 6 * 3600


def grant(run_id: str) -> str:
    """A token for `run_id`'s downloads. A live run page is redrawn every two seconds, so an
    existing token for the run is reused rather than a new one issued each time."""
    now = time.time()
    with _GRANTS_LOCK:
        for k in [k for k, (_, t) in _GRANTS.items() if now - t > _GRANT_TTL]:
            run = _GRANTS.pop(k)[0]
            if _BY_RUN.get(run) == k:
                del _BY_RUN[run]
        token = _BY_RUN.get(run_id)
        if token is None or now - _GRANTS[token][1] > _GRANT_TTL / 2:
            token = secrets.token_urlsafe(18)
            _GRANTS[token] = (run_id, now)
            _BY_RUN[run_id] = token
    return token


def granted(token: str) -> str | None:
    with _GRANTS_LOCK:
        hit = _GRANTS.get(str(token or ""))
    return hit[0] if hit and time.time() - hit[1] <= _GRANT_TTL else None


# ── header ───────────────────────────────────────────────────────────────────────────────────────

DOCS_URL = "https://huggingface.co/docs/openenv/environments/harbor"


def header_html(
    title: str,
    datasets: list[str],
    caps: Any = None,
    settings: ui_settings.UISettings | None = None,
) -> str:
    from .serving import HarborService

    settings = settings or ui_settings.load()
    service = HarborService.current()
    facts = []
    if not settings.rollouts:
        facts.append(f"<span>{icon('file', 14)}read-only task browser</span>")
    elif ui_settings.shared_endpoint(settings) is not None:
        train = service.capture_level == "tokens"
        facts.append(
            f"<span>{icon('cpu', 14)}<b>{_e(service.model)}</b>{'training capture' if train else 'eval only'}</span>"
        )
    elif settings.visitor_endpoints:
        facts.append(f"<span>{icon('key', 14)}bring your own model</span>")
    if caps is not None and settings.rollouts:
        ok = len(caps.available_sandboxes)
        facts.append(
            f'<span><span class="hb-dot {"ok" if ok else "bad"}"></span><b>{ok}</b> of {len(caps.sandboxes)} sandboxes ready</span>'
        )
    return (
        f'<div class="hb-top" data-docs="{DOCS_URL}"><div class="hb-brand"><b>{_e(title)}</b></div>'
        f'<div class="hb-facts">{"".join(facts)}</div></div>'
    )


# ── a task ───────────────────────────────────────────────────────────────────────────────────────


def _config_value(detail: dict[str, Any], section: str, key: str) -> str:
    for s in detail.get("config") or []:
        if s["label"] == section:
            for r in s["rows"]:
                if r["key"] == key:
                    return str(r["value"])
    return ""


def task_head_html(detail: dict[str, Any], common: set[str] | None = None) -> str:
    """Above the task page: where it is, what it is called, and the few facts that decide a run.

    `common` are the tags every task of the dataset carries, which say nothing about this one.
    """
    common = common or set()
    kick = [
        f'<span class="hb-chip">{_e(c)}</span>'
        for c in (detail.get("category"), detail.get("difficulty"))
        if c
    ]
    tags = [
        k
        for k in detail.get("keywords") or []
        if k not in common and k != detail.get("category")
    ]
    kick += [f'<span class="hb-chip outline">{_e(k)}</span>' for k in tags[:4]]
    files = detail.get("files") or []
    timeout = _config_value(detail, "Agent", "Timeout")
    image = _config_value(detail, "Environment", "Image")
    facts = [
        f"<span>{icon('file', 14)}<b>{len(files):,}</b> file{'s' if len(files) != 1 else ''}</span>"
    ]
    if timeout:
        facts.append(
            f"<span>{icon('clock', 14)}agent timeout <b>{_e(timeout)}</b></span>"
        )
    if image:
        facts.append(
            f'<span title="{_e(image)}">{icon("box", 14)}<code>{_e(image if len(image) < 60 else image[:57] + "…")}</code></span>'
        )
    facts.append(
        f'<button type="button" class="hb-link" data-copy>{icon("link", 14)}Copy link</button>'
    )
    return (
        f'<div class="tp-head" data-spec="{_e(detail["dataset"])}" data-index="{int(detail["index"])}">'
        f'<nav class="hb-crumbs"><button type="button" class="back" data-back>{icon("arrowLeft", 14)}Tasks</button>{icon("chevronRight", 13)}'
        f'<button type="button" data-back title="{_e(detail["dataset"])}">{_e(_short(detail["dataset"]))}</button>{icon("chevronRight", 13)}'
        f"<span>#{int(detail['index'])} · {_e(detail['name'])}</span></nav>"
        + (f'<div class="kick">{"".join(kick)}</div>' if kick else "")
        + f'<h1>{_e(detail["title"])}</h1><div class="hb-facts">{"".join(facts)}</div></div>'
    )


def _panel(sec: str, ic: str, title: str, aside: str, body: str) -> str:
    return (
        f'<section class="tp-sec hb-panel" data-sec="{sec}"><div class="hb-panel-h"><h2>{icon(ic, 15)}{_e(title)}</h2>'
        f'<span class="aside">{aside}</span></div><div class="hb-panel-b">{body}</div></section>'
    )


def _run_rows(runs: list[dict[str, Any]]) -> str:
    return "".join(
        f'<button type="button" class="tp-run" data-run="{_e(r.get("id"))}">{status_badge(r.get("status"))}{reward_badge(r.get("reward"))}'
        f'<span class="m"><code>{_e(r.get("harness"))}</code> on {_e(r.get("sandbox"))} · {_e(r.get("model"))}</span>'
        f'<span class="w">{_e(ago(r.get("created")))}</span></button>'
        for r in runs[:20]
    )


def task_html(detail: dict[str, Any], runs: list[dict[str, Any]] | None = None) -> str:
    """The task page: what the agent is asked, what the task holds, what it runs with, and its runs."""
    runs = runs or []
    files = detail.get("files") or []
    first = next(
        (f["path"] for f in files if f["path"] == "instruction.md"),
        files[0]["path"] if files else "",
    )
    # By folder, files before subfolders, so a folder's header appears once, like an editor's tree.
    files = sorted(
        files,
        key=lambda f: (
            f["path"].count("/") > 0,
            f["path"].split("/")[:-1],
            f["path"].rsplit("/", 1)[-1],
        ),
    )
    tree, last_dir = [], None
    for f in files:
        parts = f["path"].split("/")
        folder = "/".join(parts[:-1])
        if folder != last_dir:
            if folder:
                tree.append(f'<div class="dir">{icon("folder", 13)}{_e(folder)}/</div>')
            last_dir = folder
        tree.append(
            f'<button type="button" data-file="{_e(f["path"])}" aria-current="{str(f["path"] == first).lower()}">'
            f"{icon('file', 13)}<span>{_e(parts[-1])}</span><em>{size(f['size'])}</em></button>"
        )
    if detail.get("files_truncated"):
        tree.append(f'<div class="dir">first {len(files):,} files</div>')

    instruction = (
        _markdown(detail.get("instruction") or "")
        or '<p class="faint">This task has no instruction.md.</p>'
    )
    meta_rows = []
    desc = detail.get("description") or ""
    if desc and desc != detail["title"]:
        meta_rows.append(("Description", _e(desc)))
    shown = {
        "title",
        "category",
        "difficulty",
        "keywords",
        "tags",
    }  # already in the head
    meta_rows += [
        (_e(m["key"]), _e(m["value"]))
        for m in detail.get("metadata") or []
        if m["key"] not in shown
    ]
    withheld = detail.get("withheld") or []
    if withheld:
        meta_rows.append(
            (
                "Withheld",
                ", ".join(f"<code>{_e(k)}</code>" for k in withheld)
                + " hold the answer, so they are not summarised here. They are in "
                '<button type="button" class="hb-link" data-file-open="task.toml">task.toml</button>.',
            )
        )
    meta = (
        '<div class="tp-meta"><dl class="hb-kv">'
        + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in meta_rows)
        + "</dl></div>"
        if meta_rows
        else ""
    )
    groups = (
        "".join(
            f'<div><h4>{_e(s["label"])}</h4><dl class="hb-kv">'
            + "".join(
                f'<dt>{_e(r["key"])}</dt><dd title="{_e(r.get("full") or r["value"])}">{_e(r["value"]) or "<span class=faint>set</span>"}</dd>'
                for r in s["rows"]
            )
            + "</dl></div>"
            for s in detail.get("config") or []
        )
        or '<p class="faint">task.toml sets nothing beyond the defaults.</p>'
    )
    tools = (
        f'<button type="button" class="hb-btn sm ghost" data-wrap aria-pressed="false" title="Wrap long lines">{icon("wrap", 14)}</button>'
        f'<button type="button" class="hb-btn sm ghost" data-copy-file title="Copy this file">{icon("copy", 14)}</button>'
        f'<button type="button" class="hb-btn sm ghost" data-full title="Full view (Esc to close)">{icon("maximize", 14)}</button>'
    )
    files_body = (
        f'<div class="tp-files"><div class="tp-tree">{"".join(tree)}</div>'
        f'<div class="tp-view"><div class="tp-view-h"><span class="p">{_e(first)}</span><em></em><span class="tools">{tools}</span></div>'
        f'<pre class="tp-code" data-path="{_e(first)}"></pre></div></div>'
        if files
        else '<p class="faint">No files.</p>'
    )
    runs_body = (
        f'<div class="tp-runs">{_run_rows(runs)}</div>'
        if runs
        else '<p class="faint">No runs of this task yet. Pick an agent and a sandbox on the right, then press Run.</p>'
    )
    toc = (
        '<nav class="tp-toc"><p>On this page</p>'
        '<a data-to="task" class="on">The task</a>'
        f'<a data-to="files">Files<em>{len(files):,}</em></a>'
        '<a data-to="env">Environment</a>'
        f'<a data-to="runs">Runs<em>{len(runs):,}</em></a></nav>'
    )
    return (
        f'<div class="tp" data-spec="{_e(detail["dataset"])}" data-index="{int(detail["index"])}"><div class="tp-grid">{toc}<div class="tp-body">'
        + _panel(
            "task",
            "message",
            "The task",
            "exactly what the agent receives",
            f'<div class="hb-prose">{instruction}</div>{meta}',
        )
        + _panel(
            "files",
            "folder",
            "Files",
            f'{_plural(len(files), "file")}<button type="button" class="hb-link" data-full style="margin-left:12px">{icon("maximize", 13)}Full view</button>'
            if files
            else "",
            files_body,
        )
        + _panel(
            "env",
            "box",
            "Environment",
            "from task.toml",
            f'<div class="tp-groups">{groups}</div>',
        )
        + _panel(
            "runs", "play", "Runs of this task", _plural(len(runs), "run"), runs_body
        )
        + "</div></div></div>"
    )


def task_error_head(spec: str, index: int) -> str:
    """A head with just the way back, for a task that would not open."""
    return (
        f'<div class="tp-head"><nav class="hb-crumbs"><button type="button" class="back" data-back>{icon("arrowLeft", 14)}Tasks</button>{icon("chevronRight", 13)}'
        f"<span>{_e(_short(spec))} · #{int(index)}</span></nav></div>"
    )


# ── the run list ─────────────────────────────────────────────────────────────────────────────────


def _kind(status: Any) -> str:
    return (
        "live" if status in ui_runs.LIVE else "failed" if status == "failed" else "done"
    )


def runs_html(
    runs: list[dict[str, Any]],
    selected: str = "",
    compare: list[str] | None = None,
    *,
    history: bool = True,
    own: bool = False,
) -> str:
    """Every run this visitor may see, newest first, as a filterable table."""
    compare = compare or []
    live = sum(1 for r in runs if r.get("status") in ui_runs.LIVE)
    graded = [r for r in runs if r.get("reward") is not None]
    solved = sum(1 for r in graded if float(r["reward"]) > 0)
    scope = (
        "Rollouts started from this browser."
        if own
        else "Every rollout this server has run."
    )
    keep = "" if history else " Finished runs are kept until the server restarts."
    stats = (
        f'<div class="tb-stats"><div><b>{len(runs):,}</b><span>runs</span></div><div><b>{live}</b><span>running</span></div>'
        + (
            f"<div><b>{solved}/{len(graded)}</b><span>solved</span></div>"
            if graded
            else ""
        )
        + "</div>"
    )
    head = f'<div class="rs-head"><div><h1>Runs</h1><p>{scope}{keep}</p></div>{stats}</div>'
    if not runs:
        return f'<div class="rs">{head}<div class="hb-panel">{empty("play", "No runs yet", "Open a task on the Tasks tab, pick an agent and a sandbox, and press Run.")}</div></div>'
    counts = {
        k: sum(1 for r in runs if _kind(r.get("status")) == k)
        for k in ("live", "done", "failed")
    }
    seg = (
        '<div class="hb-seg" data-rs-seg>'
        f'<button type="button" data-s="" aria-pressed="true">All <span>{len(runs)}</span></button>'
        f'<button type="button" data-s="live" aria-pressed="false">Running <span>{counts["live"]}</span></button>'
        f'<button type="button" data-s="done" aria-pressed="false">Finished <span>{counts["done"]}</span></button>'
        f'<button type="button" data-s="failed" aria-pressed="false">Failed <span>{counts["failed"]}</span></button></div>'
    )
    bar = (
        f'<div class="rs-bar">{seg}<div class="tb-search">{icon("search", 15)}'
        '<input class="hb-input rs-q" type="search" placeholder="Filter by task, agent or model" aria-label="Filter runs"></div>'
        f'<div class="right"><span class="rs-n">Tick two to four to compare</span><button type="button" class="hb-btn sm" data-rs-compare disabled>{icon("columns", 14)}Compare</button></div></div>'
    )
    rows = []
    for r in runs[:500]:
        rid = _e(r.get("id"))
        is_live = r.get("status") in ui_runs.LIVE
        title = r.get("task_title") or r.get("task_name")
        rows.append(
            f'<div class="rs-row" data-rs-id="{rid}" data-s="{_kind(r.get("status"))}" aria-current="{str(r.get("id") == selected).lower()}">'
            f'<span><input type="checkbox" data-rs-pick="{rid}" aria-label="Compare this run" {"checked" if r.get("id") in compare else ""}></span>'
            f"<span>{status_badge(r.get('status'))}</span>"
            f'<span class="t"><b title="{_e(title)}">{_e(title)}</b><em>{_e(_short(r.get("dataset")))} · #{_e(r.get("task_index"))}</em></span>'
            f'<span class="m"><code>{_e(r.get("harness"))}</code> · {_e(r.get("sandbox"))}</span>'
            f'<span class="m" title="{_e(r.get("endpoint"))}">{_e(r.get("model"))}</span>'
            f'<span class="rw">{"" if is_live else reward_badge(r.get("reward"))}</span>'
            f'<span class="d">{_e(_duration(r.get("wall_s")))}</span><span class="w">{_e(ago(r.get("created")))}</span></div>'
        )
    table = (
        '<div class="rs-table"><div class="rs-row head"><span></span><span>Status</span><span>Task</span><span>Agent</span><span>Model</span>'
        f'<span class="rw">Reward</span><span class="d">Duration</span><span class="w">Started</span></div>{"".join(rows)}</div>'
    )
    return f'<div class="rs">{head}{bar}{table}</div>'


# ── a run ────────────────────────────────────────────────────────────────────────────────────────


def _stepper(rec: dict[str, Any], live: Any, turns: int) -> str:
    result = rec.get("result") or {}
    if live is not None:
        states = (
            [("on", "Setup", "sandbox and agent")]
            if not turns
            else [("ok", "Setup", "ready")]
        )
        states.append(
            ("on", "Agent", _plural(turns, "model call"))
            if turns
            else ("off", "Agent", "waiting")
        )
        states.append(("off", "Result", "verifier runs last"))
    elif result:
        n = int(result.get("n_turns") or 0)
        reward = result.get("reward")
        several = len(result.get("rewards") or {}) > 1
        if result.get("ok") and reward is not None:
            graded = ("ok", "Result", f"reward {float(reward):.2f}")
        elif result.get("ok") and several:
            graded = ("ok", "Result", _plural(len(result["rewards"]), "reward"))
        elif result.get("ok"):
            graded = ("bad", "Result", "not graded")
        else:
            graded = ("bad", "Result", _e(result.get("exception_type") or "failed"))
        states = [
            ("ok", "Setup", "ready"),
            ("ok" if n else "bad", "Agent", _plural(n, "model call")),
            graded,
        ]
    else:
        states = [
            ("bad", "Setup", "did not start"),
            ("off", "Agent", ""),
            ("off", "Result", ""),
        ]
    cells = []
    for tone, label, sub in states:
        mark = (
            icon("check", 12)
            if tone == "ok"
            else icon("x", 12)
            if tone == "bad"
            else ""
        )
        cells.append(
            f'<div class="rp-step {tone}"><span class="mk">{mark}</span><b>{label}</b><span class="t"></span><em>{sub}</em></div>'
        )
    return f'<div class="rp-stepper" style="--n:3">{"".join(cells)}</div>'


def _run_head(rec: dict[str, Any], token: str) -> str:
    started = float(rec.get("created") or time.time())
    elapsed = rec.get("wall_s") if rec.get("finished") else time.time() - started
    result = rec.get("result") or {}
    # Only where the exporter will produce one: it refuses eval rollouts and FATAL findings.
    trainable = (
        result
        and result.get("rollout_type") != "eval"
        and any(t.get("completion_token_ids") for t in result.get("turns") or [])
        and not any(str(f).startswith("[FATAL") for f in result.get("findings") or [])
    )
    actions = [
        f'<button type="button" class="hb-btn sm" data-task="{_e(rec.get("dataset"))}" data-index="{_e(rec.get("task_index"))}">{icon("file", 14)}View task</button>'
    ]
    if result:
        actions.append(
            f'<button type="button" class="hb-btn sm" data-dl="result" data-grant="{_e(token)}">{icon("download", 14)}Result JSON</button>'
        )
    if trainable:
        actions.append(
            f'<button type="button" class="hb-btn sm" data-dl="contract" data-grant="{_e(token)}">{icon("download", 14)}Capture audit</button>'
        )
        actions.append(
            f'<button type="button" class="hb-btn sm" data-dl="training_trace" data-grant="{_e(token)}">{icon("download", 14)}Training trace</button>'
        )
    title = rec.get("task_title") or rec.get("task_name")
    return (
        '<div class="rp-head">'
        f'<nav class="hb-crumbs"><button type="button" class="back" data-back>{icon("arrowLeft", 14)}Runs</button>{icon("chevronRight", 13)}<span>{_e(rec.get("id"))}</span></nav>'
        f'<div class="kick">{status_badge(rec.get("status"))}<span class="when">{icon("clock", 13)}{_e(_duration(elapsed))} · started {_e(ago(started))}</span></div>'
        f"<h1>{_e(title)}</h1>"
        '<div class="rp-bar"><div class="hb-facts">'
        f"<span>{icon('cpu', 14)}<b>{_e(rec.get('model'))}</b>{_e(rec.get('endpoint'))}</span>"
        f"<span>{icon('terminal', 14)}<b>{_e(rec.get('harness'))}</b></span>"
        f"<span>{icon('box', 14)}{_e(rec.get('sandbox'))}</span>"
        f"<span>{icon('database', 14)}{_e(_short(rec.get('dataset')))} · #{_e(rec.get('task_index'))}</span></div>"
        f'<div class="rp-actions">{"".join(actions)}</div></div></div>'
    )


def _details_html(rec: dict[str, Any]) -> str:
    result = rec.get("result") or {}
    level = result.get("capture_level")
    rows = [
        ("Agent", f"<code>{_e(rec.get('harness'))}</code>"),
        ("Sandbox", _e(rec.get("sandbox"))),
        ("Model", _e(rec.get("model"))),
        ("Endpoint", _e(rec.get("endpoint"))),
    ]
    if level:
        rows.append(
            (
                "Capture",
                f'<span title="Training needs token ids: vLLM with --return-tokens-as-token-ids">{_e(LEVEL_TEXT.get(level, level))}'
                f" · {'training' if result.get('rollout_type') != 'eval' else 'eval only'}</span>",
            )
        )
    rows += [
        ("Dataset", _e(rec.get("dataset"))),
        (
            "Task",
            f"#{_e(rec.get('task_index'))} · <code>{_e(rec.get('task_name'))}</code>",
        ),
        ("Run id", f"<code>{_e(rec.get('id'))}</code>"),
    ]
    return (
        f'<section class="hb-panel"><div class="hb-panel-h"><h3>{icon("info", 15)}Run details</h3></div>'
        '<div class="hb-panel-b"><dl class="hb-kv">'
        + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
        + "</dl></div></section>"
    )


def _timeline(body: str, n_calls: int, actions: int | None, live: bool) -> str:
    counts = _plural(n_calls, "model call") + (
        f" · {_plural(actions, 'action')}" if actions is not None else ""
    )
    return (
        f'<div class="tl-bar"><h2>What the agent did</h2><span class="n">{counts}{" so far" if live else ""}</span>'
        f'<div class="right"><button type="button" class="hb-btn sm ghost" data-expand>Expand all</button></div></div>{body}'
    )


def live_view(live: Any, service: Any) -> tuple[str, str, int]:
    """A rollout in flight, as (timeline, side panel, model calls so far)."""
    turns, transcript, rows = 0, "", []
    session = (
        service.capture.registry.get(live.session_id)
        if (service is not None and live.session_id)
        else None
    )
    elapsed = time.time() - live.created
    rows.append(("Elapsed", _duration(elapsed)))
    if session is not None:
        st = session.graph.stats()
        turns = st.get("n_turns", 0)
        rows += [
            ("Model calls", f"{turns:,}"),
            ("Conversations", f"{st.get('n_roots', 0):,}"),
            (
                "Sampled tokens",
                f"{sum(len(n.sampled_ids or []) for n in session.graph.nodes()):,}",
            ),
        ]
        if st.get("n_discarded"):
            rows.append(("Discarded calls", f"{st['n_discarded']:,}"))
        if turns:
            rows.append(("Since last call", _duration(session.idle_seconds)))
        transcript = _transcript_html(session)
    phase = (
        "The agent is working."
        if turns
        else "Preparing the sandbox and the agent, or waiting for the first reply."
    )
    side = (
        f'<section class="hb-panel"><div class="hb-panel-h"><h3>{icon("gauge", 15)}Progress</h3>'
        f'<span class="aside"><span class="hb-status live"><span class="hb-pulse"></span>Running</span></span></div>'
        f'<div class="hb-panel-b" style="display:grid;gap:12px"><p class="faint" style="font-size:13px">{phase}</p>'
        '<dl class="hb-kv">'
        + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
        + "</dl></div></section>"
    )
    body = (
        transcript
        or '<div class="tl-empty">The trajectory appears here once the first model call completes.</div>'
    )
    return _timeline(body, turns, None, True), side, turns


def run_html(rec: dict[str, Any] | None, live: Any = None, service: Any = None) -> str:
    """A run: its trajectory on the left; its result, checks and details on the right."""
    if not rec:
        return f'<div class="rp">{_back("Runs")}<div class="hb-panel">{empty("search", "Run not found", "It may be from before the server restarted with history off, or started from another browser.")}</div></div>'
    token = grant(str(rec.get("id")))
    result = rec.get("result") or {}
    if live is not None:
        main, side, turns = live_view(live, service)
    elif result:
        turns = int(result.get("n_turns") or 0)
        main = _timeline(
            _conversation_html(result), turns, count_actions(result), False
        )
        if any(t.get("completion_token_ids") for t in result.get("turns") or []):
            main += (
                f'<details class="hb-panel" style="margin-top:16px"><summary class="hb-panel-h">{icon("chevronRight", 13, "chev")}'
                f'<h3>Tokens per model call</h3></summary><div class="hb-panel-b">{_turns_html(result)}</div></details>'
            )
        side = _result_html(result)
    else:
        turns = 0
        main = f'<div class="hb-panel"><div class="hb-panel-b"><pre class="rp-err">{_e(rec.get("error") or "The rollout ended without a result.")}</pre></div></div>'
        side = ""
    side += _details_html(rec)
    return (
        f'<div class="rp" data-run="{_e(rec.get("id"))}">{_run_head(rec, token)}{_stepper(rec, live, turns)}'
        f'<div class="rp-grid"><div class="rp-main">{main}</div><aside class="rp-side">{side}</aside></div></div>'
    )


def unreadable(label: str, exc: Exception, *, run: bool = False) -> str:
    """A page for a record this version cannot draw, with the way back."""
    body = empty(
        "alert",
        "This could not be shown",
        _e(f"{type(exc).__name__}: {str(exc)[:200]}"),
    )
    return f'<div class="{"rp" if run else "rs"}">{_back(label) if run else ""}<div class="hb-panel">{body}</div></div>'


def _back(label: str) -> str:
    return f'<div class="rp-head"><nav class="hb-crumbs"><button type="button" class="back" data-back>{icon("arrowLeft", 14)}{_e(label)}</button></nav></div>'


# ── a comparison ─────────────────────────────────────────────────────────────────────────────────


def compare_html(records: list[dict[str, Any] | None]) -> str:
    """Two to four rollouts side by side: outcome, cost in calls and time, and what each agent did."""
    recs = [r for r in records if r]
    if len(recs) < 2:
        return f'<div class="rp">{_back("Runs")}<div class="hb-panel">{empty("columns", "Pick two to four runs", "Tick them in the run list, then press Compare.")}</div></div>'
    sums = [ui_runs.summarize(r) for r in recs]
    res = [r.get("result") or {} for r in recs]

    def tools(r: dict[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in r.get("turns") or []:
            for c in t.get("tool_calls") or []:
                key = str(c.get("name", "?"))
                out[key] = out.get(key, 0) + 1
        if not out and count_actions(r):
            out["terminal"] = count_actions(
                r
            )  # keystrokes, for agents that act by typing
        return out

    def best(values: list[Any], pick: str) -> list[bool]:
        nums = [v for v in values if isinstance(v, (int, float))]
        if len(nums) < 2 or len(set(nums)) == 1:
            return [False] * len(values)
        target = max(nums) if pick == "max" else min(nums)
        return [v == target for v in values]

    code = lambda v: f"<code>{_e(v)}</code>"  # noqa: E731
    rows = [
        ("Reward", [s.get("reward") for s in sums], "max", reward_badge),
        ("Status", [s.get("status") for s in sums], None, status_badge),
        ("Agent", [s.get("harness") for s in sums], None, code),
        ("Model", [s.get("model") for s in sums], None, _e),
        ("Sandbox", [s.get("sandbox") for s in sums], None, _e),
        ("Endpoint", [s.get("endpoint") for s in sums], None, _e),
        (
            "Capture",
            [
                f"{s.get('rollout_type') or '?'} · {s.get('capture_level') or '?'}"
                for s in sums
            ],
            None,
            _e,
        ),
        ("Model calls", [s.get("n_turns") for s in sums], "min", _e),
        # Actions, not tool calls: an agent that types into a terminal (terminus-2) makes none.
        ("Actions", [count_actions(r) for r in res], None, _e),
        (
            "Generated tokens",
            [s.get("generated") for s in sums],
            None,
            lambda v: f"{int(v or 0):,}",
        ),
        ("Trace check", [s.get("atif") for s in sums], None, _e),
        ("Duration", [s.get("wall_s") for s in sums], "min", _duration),
        ("Findings", [len(r.get("findings") or []) for r in res], None, _e),
    ]
    heads = "".join(
        f'<th><span style="display:flex;align-items:center;gap:8px"><span class="cmp-tag">{chr(65 + i)}</span>'
        f'<button type="button" class="hb-link" data-run="{_e(s.get("id"))}" title="Open this run">{_e(s.get("harness"))} · {_e(s.get("model"))}</button></span></th>'
        for i, s in enumerate(sums)
    )
    body = []
    for label, values, pick, fmt in rows:
        marks = best(values, pick) if pick else [False] * len(values)
        same = len({json.dumps(v, default=str) for v in values}) == 1
        cells = "".join(
            f'<td class="{"best" if m else ""}">{fmt(v)}</td>'
            for v, m in zip(values, marks)
        )
        body.append(
            f'<tr style="{"color:var(--faint)" if same else ""}"><td>{_e(label)}</td>{cells}</tr>'
        )
    used = [tools(r) for r in res]
    names = sorted(
        {n for u in used for n in u}, key=lambda n: -sum(u.get(n, 0) for u in used)
    )
    tool_rows = "".join(
        f"<tr><td><code>{_e(n)}</code></td>"
        + "".join(f'<td class="num">{u.get(n, "–")}</td>' for u in used)
        + "</tr>"
        for n in names[:30]
    )
    tasks = {(s.get("dataset"), s.get("task_index")) for s in sums}
    title = (
        recs[0].get("task_title") or recs[0].get("task_name")
        if len(tasks) == 1
        else "Different tasks"
    )
    note = (
        ""
        if len(tasks) == 1
        else f'<div class="hb-note warn">{icon("alert", 15)}These rollouts are of different tasks, so their rewards do not compare directly.</div>'
    )
    tools_panel = (
        f'<section class="hb-panel"><div class="hb-panel-h"><h3>{icon("tool", 15)}Tools used</h3><span class="aside">calls per tool</span></div>'
        f'<div class="hb-tbl" style="border:0"><table><thead><tr><th>Tool</th>{heads}</tr></thead><tbody>{tool_rows}</tbody></table></div></section>'
        if tool_rows
        else ""
    )
    return (
        '<div class="rp cmp">'
        f'<div class="rp-head" style="margin:0"><nav class="hb-crumbs"><button type="button" class="back" data-back>{icon("arrowLeft", 14)}Runs</button>{icon("chevronRight", 13)}<span>compare</span></nav>'
        f'<h1>{_e(title)}</h1><div class="hb-facts"><span>{icon("columns", 14)}Comparing <b>{len(recs)}</b> runs</span>'
        "<span>Faint rows are the same in every run; bold marks the best.</span></div></div>"
        f"{note}"
        f'<section class="hb-panel"><div class="hb-tbl" style="border:0"><table><thead><tr><th></th>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>'
        f"{tools_panel}</div>"
    )


# ── setup ────────────────────────────────────────────────────────────────────────────────────────


def _on(value: Any) -> str:
    if value is None or value == "":
        return '<span class="faint">not set</span>'
    if isinstance(value, bool):
        return f'<span class="hb-status {"ok" if value else ""}">{"on" if value else "off"}</span>'
    return f"<b>{_e(value)}</b>"


def _ticks(text: str) -> str:
    """Escape, then show `backticked` words as code."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(text))


def _ready(ok: bool) -> str:
    return (
        '<span class="hb-status ok">ready</span>'
        if ok
        else '<span class="hb-status bad">unavailable</span>'
    )


def setup_html(
    caps: Any, datasets: list[str], settings: ui_settings.UISettings | None = None
) -> str:
    from .serving import HarborService

    settings = settings or ui_settings.load()
    service = HarborService.current()
    if service is not None and service.llm_url:
        proxy = str(service.public_url or "not started")
        endpoint = (
            '<dl class="hb-kv">'
            f"<dt>Model</dt><dd>{_e(service.model)}</dd>"
            f"<dt>Endpoint</dt><dd>{_e(urlparse(service.llm_url).netloc or service.llm_url)}</dd>"
            f"<dt>Capture</dt><dd>{_e(LEVEL_TEXT.get(service.capture_level, service.capture_level))}</dd>"
            f"<dt>Capture proxy</dt><dd><code>{_e(proxy)}</code>{' (mounted on this server)' if service.mounted else ''}</dd>"
            f"<dt>Visitors may use it</dt><dd>{_on(settings.server_endpoint)}</dd>"
            "</dl>"
        )
    else:
        endpoint = '<p class="faint">None. Visitors connect their own model on the run card.</p>'
    where = (
        "a Hugging Face Space"
        if settings.on_space
        else "a network address (visitors are treated as the public)"
        if settings.exposed
        else "this machine only"
    )
    rows = []
    for row in ui_settings.ROWS:
        value = getattr(settings, row.attr)
        if row.attr == "run_history" and settings.runs_dir is not None:
            value = f"on · {settings.runs_dir}"
        change = f"<code>{_e(row.env)}</code>" + (
            f"<br><code>{_e(row.flag)}</code>" if row.flag else ""
        )
        rows.append(
            f"<tr><td>{_e(row.label)}</td><td>{_on(value)}</td>"
            f"<td>{change}</td><td>{_ticks(row.help)}</td></tr>"
        )
    sandboxes = "".join(
        f"<tr><td><code>{_e(s.name)}</code></td><td>{_ready(bool(s.available))}</td>"
        f'<td class="faint">{_e(s.detail)}</td></tr>'
        for s in caps.sandboxes
    )
    agents = "".join(
        f"<tr><td><code>{_e(h.name)}</code></td><td>{_e(h.dialect)}</td>"
        f'<td>{"on this server" if h.kind == "base" else "in the sandbox"}</td><td class="faint">{_e(h.status)}</td></tr>'
        for h in sorted(caps.harnesses, key=lambda h: (h.status != "validated", h.name))
    )
    sets = (
        "".join(
            f'<tr><td><code>{_e(d.get("name"))}</code></td><td class="num">{int(d.get("num_tasks") or 0):,}</td>'
            f'<td class="faint">{_e(d.get("error") or "")}</td></tr>'
            for d in caps.datasets or []
        )
        or '<tr><td colspan="3" class="faint">none</td></tr>'
    )

    def panel(
        ic: str,
        title: str,
        aside: str,
        body: str,
        wide: bool = False,
        table: bool = False,
    ) -> str:
        inner = body if table else f'<div class="hb-panel-b">{body}</div>'
        return (
            f'<section class="hb-panel{" wide" if wide else ""}"><div class="hb-panel-h"><h3>{icon(ic, 15)}{_e(title)}</h3>'
            f'<span class="aside">{aside}</span></div>{inner}</section>'
        )

    return (
        '<div class="su">'
        + panel("cpu", "Server endpoint", "", endpoint)
        + panel(
            "database",
            "Datasets",
            _plural(len(datasets), "dataset"),
            f'<div class="hb-tbl"><table><thead><tr><th>Dataset</th><th class="num">Tasks</th><th></th></tr></thead><tbody>{sets}</tbody></table></div>',
            table=True,
        )
        + panel(
            "box",
            "Sandboxes",
            f"{len(caps.available_sandboxes)} of {len(caps.sandboxes)} ready",
            f'<div class="hb-tbl"><table><thead><tr><th>Backend</th><th>Status</th><th>Why</th></tr></thead><tbody>{sandboxes}</tbody></table></div>',
            wide=True,
            table=True,
        )
        + panel(
            "shield",
            "Deployment settings",
            f"running on {where}",
            '<div class="hb-tbl su-set"><table><thead><tr><th>Setting</th><th>Value</th><th>Change with</th><th>What it does</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>",
            wide=True,
            table=True,
        )
        + panel(
            "terminal",
            "Agents",
            _plural(len(caps.harnesses), "agent"),
            '<div class="hb-tbl"><table><thead><tr><th>Agent</th><th>Dialect</th><th>Runs</th><th>Status</th></tr></thead>'
            f"<tbody>{agents}</tbody></table></div>",
            wide=True,
            table=True,
        )
        + "</div>"
    )
