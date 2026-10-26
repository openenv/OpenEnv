"""What a Harbor UI deployment lets its visitors do.

The same page runs on a laptop and as a public Space, and the right answer to "may a visitor do
this?" differs between the two. On a laptop the visitor is the operator: they want every run they
have made, their vLLM on localhost, their own Hugging Face token. On a public Space the visitor is
anyone with the URL: they should see their own runs rather than everyone's, never make the server
call an address on its private network, and spend the operator's key only if the operator says so
(`server_endpoint`: off on a Space unless turned on, so each visitor brings their own model).

Only a server that listens on loopback alone gets the laptop defaults. `openenv harbor serve` binds
0.0.0.0 unless told otherwise, and then everyone on the network is a visitor too, so it is treated
like a public deployment (`serve` records its host in `OPENENV_HARBOR_UI_HOST`; a server started
some other way is assumed reachable).

Every setting is an environment variable, so a Space sets them as variables and `openenv harbor
serve` sets them from its flags. Unset, each takes the default for where the server runs.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import secrets
import socket
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
VISIBILITY = ("all", "own")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _flag(env: str, default: bool) -> bool:
    raw = (os.environ.get(env) or "").strip().lower()
    return True if raw in _TRUE else False if raw in _FALSE else default


@dataclass(frozen=True)
class UISettings:
    """The resolved settings. [`load`] reads the environment each time it is called; the UI calls it
    once, when the page is built, so a change takes effect on the next server start."""

    on_space: bool
    exposed: bool
    rollouts: bool
    server_endpoint: bool
    visitor_endpoints: bool
    private_urls: bool
    local_token: bool
    add_datasets: bool
    run_history: bool
    runs_dir: Path | None
    run_visibility: str
    max_runs: int
    max_runs_per_visitor: int
    bucket: str | None = None
    bucket_mount: Path | None = None
    hf_login: bool = False
    # Which of a task's rewards is the headline one when it has several and none is named `reward`
    # (`serve --reward-key`, the same choice `rollout --reward-key` makes). Without it the run shows
    # every reward and none as the headline, rather than failing a task the agent may have solved.
    reward_key: str = ""


def _count(env: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(env) or default))
    except ValueError:
        return default


def load() -> UISettings:
    space = bool(os.environ.get("SPACE_ID"))
    host = (os.environ.get("OPENENV_HARBOR_UI_HOST") or "").strip().lower()
    exposed = space or host not in LOOPBACK
    history = _flag("OPENENV_HARBOR_RUN_HISTORY", not space)
    runs_dir = None
    if history:
        if os.environ.get("OPENENV_HARBOR_RUNS_DIR"):
            runs_dir = Path(os.environ["OPENENV_HARBOR_RUNS_DIR"]).expanduser()
        elif space and Path("/data").is_dir():
            runs_dir = Path("/data/harbor-runs")
        else:
            runs_dir = Path.home() / ".cache" / "openenv" / "harbor" / "runs"
    visibility = (os.environ.get("OPENENV_HARBOR_RUN_VISIBILITY") or "").strip().lower()
    max_runs = _count("OPENENV_HARBOR_UI_MAX_RUNS", 4)
    return UISettings(
        on_space=space,
        exposed=exposed,
        rollouts=_flag("OPENENV_HARBOR_UI_ROLLOUTS", True),
        # On a Space anyone with the URL is a visitor, so the operator's key is theirs only on request.
        server_endpoint=_flag("OPENENV_HARBOR_UI_SERVER_ENDPOINT", not space),
        visitor_endpoints=_flag("OPENENV_HARBOR_UI_VISITOR_ENDPOINTS", True),
        private_urls=_flag("OPENENV_HARBOR_UI_PRIVATE_URLS", not exposed),
        # Never on a Space: the token there is the operator's secret, and whether visitors may spend
        # it is what `server_endpoint` decides, through the capture proxy that already holds it.
        local_token=not space and _flag("OPENENV_HARBOR_UI_LOCAL_TOKEN", not exposed),
        add_datasets=_flag("OPENENV_HARBOR_UI_ADD_DATASETS", not exposed),
        run_history=history,
        runs_dir=runs_dir,
        run_visibility=visibility
        if visibility in VISIBILITY
        else ("own" if space else "all"),
        max_runs=max_runs,
        max_runs_per_visitor=min(
            max_runs,
            _count("OPENENV_HARBOR_UI_MAX_RUNS_PER_VISITOR", 2 if space else max_runs),
        ),
        # `openenv harbor push` names the Space's bucket and where it is mounted. Datasets added from
        # the page then go into the bucket (a server-side copy) instead of the container's disk.
        bucket=os.environ.get("OPENENV_HARBOR_BUCKET") or None,
        bucket_mount=_mount(),
        # Sign-in exists only where the Hub has set up OAuth for the Space (`hf_oauth: true`).
        hf_login=space and bool(os.environ.get("OAUTH_CLIENT_ID")),
        reward_key=(os.environ.get("OPENENV_HARBOR_REWARD_KEY") or "").strip(),
    )


def _mount() -> Path | None:
    raw = os.environ.get("OPENENV_HARBOR_BUCKET_MOUNT")
    if not raw or not os.environ.get("OPENENV_HARBOR_BUCKET"):
        return None
    path = Path(raw)
    return path if path.is_dir() else None


@dataclass(frozen=True)
class Row:
    """One setting as the Setup tab lists it."""

    attr: str
    label: str
    env: str
    flag: str
    help: str


ROWS = (
    Row(
        "rollouts",
        "Rollouts",
        "OPENENV_HARBOR_UI_ROLLOUTS",
        "--rollouts",
        "Off makes the page a read-only task browser.",
    ),
    Row(
        "server_endpoint",
        "Use this server's endpoint",
        "OPENENV_HARBOR_UI_SERVER_ENDPOINT",
        "--share-endpoint",
        "Visitors may run on the endpoint and key the server started with. Default on, except on a Space. Off, each visitor brings their own.",
    ),
    Row(
        "visitor_endpoints",
        "Visitors' own endpoints",
        "OPENENV_HARBOR_UI_VISITOR_ENDPOINTS",
        "--visitor-endpoints",
        "A Hugging Face token and model, or any OpenAI-compatible URL such as vLLM. Keys stay in server memory for that page, never on disk.",
    ),
    Row(
        "private_urls",
        "Private and local URLs",
        "OPENENV_HARBOR_UI_PRIVATE_URLS",
        "--private-urls",
        "Whether a visitor's URL may point at localhost or a private network. Default on only when the server listens on this machine alone (`--host 127.0.0.1`). A `serve` flag only: on a Space those addresses are the Space's own network.",
    ),
    Row(
        "local_token",
        "This machine's HF token",
        "OPENENV_HARBOR_UI_LOCAL_TOKEN",
        "",
        "Offer the token from `hf auth login`. Default on only when the server listens on this machine alone; never on a Space.",
    ),
    Row(
        "run_visibility",
        "Who sees runs",
        "OPENENV_HARBOR_RUN_VISIBILITY",
        "--run-visibility",
        "`all`: every visitor sees every run. `own`: each browser sees the runs it started. Default all locally, own on a Space.",
    ),
    Row(
        "run_history",
        "Keep finished runs",
        "OPENENV_HARBOR_RUN_HISTORY",
        "--run-history",
        "Save results to disk so they survive a restart. Default on locally, off on a Space. OPENENV_HARBOR_RUNS_DIR picks the folder.",
    ),
    Row(
        "add_datasets",
        "Add Hub datasets from the page",
        "OPENENV_HARBOR_UI_ADD_DATASETS",
        "--add-datasets",
        "Public Hub datasets only, copied into the configured Space bucket or otherwise downloaded to this server. Their tasks may not read the server's environment variables. Default on only when the server listens on this machine alone.",
    ),
    Row(
        "reward_key",
        "Headline reward",
        "OPENENV_HARBOR_REWARD_KEY",
        "--reward-key",
        "Which reward a run reports, for tasks with several and none named `reward` (a comma-separated preference order is accepted). Unset, such a run shows each of them.",
    ),
    Row(
        "max_runs",
        "Rollouts at once",
        "OPENENV_HARBOR_UI_MAX_RUNS",
        "",
        "Each holds a sandbox and spends endpoint credit.",
    ),
    Row(
        "bucket",
        "Bucket for added datasets",
        "OPENENV_HARBOR_BUCKET",
        "",
        "Set by `push`. Datasets added from the page are copied into it, server side, and read through its mount at OPENENV_HARBOR_BUCKET_MOUNT; removing one deletes it there. Without a bucket they download to this machine.",
    ),
    Row(
        "hf_login",
        "Sign in with Hugging Face",
        "OAUTH_CLIENT_ID",
        "--hf-login (push)",
        "Visitors may use their own Hugging Face account for Inference Providers instead of pasting a token. Only on a Space whose README has `hf_oauth: true`, which `push` writes.",
    ),
    Row(
        "max_runs_per_visitor",
        "Rollouts at once per visitor",
        "OPENENV_HARBOR_UI_MAX_RUNS_PER_VISITOR",
        "",
        "So one visitor cannot hold every slot. Default 2 on a Space.",
    ),
)


_PRIVATE = (
    "This server does not call private or local addresses for visitors. Use a public URL, or run "
    "the UI yourself to reach an endpoint on your own network (`openenv harbor serve --host "
    "127.0.0.1`, or `--private-urls`)."
)
# IPv6 forms that carry an IPv4 address inside, which `is_global` judges by the IPv6 range alone.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_COMPAT = ipaddress.ip_network("::/96")


def _public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address):
        inner = address.ipv4_mapped or address.sixtofour
        if inner is None and (address in _NAT64 or address in _COMPAT):
            inner = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        if address.teredo:
            inner = address.teredo[1]
        if inner is not None and not inner.is_global:
            return False
    return address.is_global


def url_problem(url: str, *, private_ok: bool) -> str | None:
    """Why this server should not call `url` for a visitor, or `None` when it may.

    A visitor's endpoint is called by this server, so on a public deployment a URL is a request to
    reach whatever that address is from inside the server's network: the Space's own services, a
    cloud metadata endpoint. Unless private addresses are allowed, every address the host resolves to
    must be public, and so must every redirect (see [`guard_redirects`]). The check resolves once and
    the request resolves again, so a host that answers differently the second time is not stopped
    here; rollouts check again before they start, which narrows that window without closing it.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "Enter an http(s) URL, like https://host/v1."
    if private_ok:
        return None
    try:
        infos = socket.getaddrinfo(
            parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP
        )
    except (socket.gaierror, UnicodeError, ValueError):
        return f"{parsed.hostname} does not resolve."
    for info in infos:
        if not _public(ipaddress.ip_address(str(info[4][0]).split("%")[0])):
            return _PRIVATE
    return None


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to a public address, so a public URL cannot bounce the probe inward."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - urllib's hook
        if url_problem(newurl, private_ok=False):
            raise urllib.error.HTTPError(newurl, code, _PRIVATE, headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_GUARDED = False


def guard_redirects() -> None:
    """Refuse redirects to private addresses in every `urllib` request this process makes.

    The endpoint checks (`validate_llm`, `list_models`) use `urllib`, which follows redirects, so
    checking only the URL a visitor typed would let `https://public.example` answer 302 with a
    link-local address. Installed once, and only where visitors' private URLs are refused: the
    process then holds a single policy, whoever makes the request.
    """
    global _GUARDED
    if not _GUARDED:
        urllib.request.install_opener(urllib.request.build_opener(_PublicRedirects))
        _GUARDED = True


def visitor_secret(settings: UISettings) -> str:
    """The key Gradio encrypts each browser's visitor id with in its local storage.

    Gradio encrypts in the browser, so the key is sent to the page and hides the id from nothing but
    casual reading; what makes the id a credential is that it is random and never shown. The key must
    still be stable: a new one per process would make every stored id unreadable after a restart,
    and with it every visitor's runs. With run history on, it lives next to the history.
    """
    if os.environ.get("OPENENV_HARBOR_UI_SECRET"):
        return os.environ["OPENENV_HARBOR_UI_SECRET"]
    if settings.runs_dir is None:
        return secrets.token_urlsafe(32)
    path = settings.runs_dir / ".visitor-secret"
    try:
        saved = path.read_text().strip()
        if saved:
            return saved
    except OSError:
        pass
    value = secrets.token_urlsafe(32)
    try:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(value)
        temporary.chmod(0o600)
        temporary.replace(path)
    except OSError as exc:
        warnings.warn(
            f"Could not persist the Harbor visitor secret; run ownership will reset after restart: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return value


def shared_endpoint(settings: UISettings) -> object | None:
    """The running service when it has an endpoint of its own that visitors may use, else `None`."""
    from .serving import HarborService

    service = HarborService.current()
    if service is None or not service.llm_url or not settings.server_endpoint:
        return None
    return service


def new_visitor() -> str:
    return secrets.token_urlsafe(24)


def owner_of(visitor: str | None) -> str:
    """What a run records about who started it: a digest, so a saved run never holds a live id."""
    return hashlib.sha256(visitor.encode()).hexdigest()[:24] if visitor else ""
