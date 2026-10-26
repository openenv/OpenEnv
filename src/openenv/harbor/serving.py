"""Serve Harbor tasks: Task API, MCP rollouts, and a human UI.

Locally this is two ports:

    :port          env server    Task API + MCP + UI   (faces the trainer / a browser)
    :capture_port  capture proxy                       (faces the sandbox, published)

Two ports on purpose. The sandbox is off-cluster and must reach the capture proxy over a public URL;
the env server has no business being publicly reachable, and sharing one port would expose it as
soon as the capture proxy became reachable.

On a hosted platform that inverts. A Space gets exactly one public URL and exposes one port, so
there is no second port to publish and nothing to forward. The capture app is mounted onto the env
server's own app at `CAPTURE_MOUNT` instead, and the sandbox reaches it at `<space-url>/capture`.
The proxy still refuses unregistered callers, which is what keeps a public mount from becoming an
open relay.
"""

from __future__ import annotations

import os
import secrets
import threading
from typing import Any

from openenv.core.harness.capture import CaptureServer

# Where the capture app is mounted when the env server hosts it directly.
CAPTURE_MOUNT = "/capture"


def space_public_url() -> str:
    """The public URL of the Space this process is running in, or `""` when it is not on one.

    Returns:
        `str`: e.g. `https://owner-name.hf.space`, with no trailing slash.
    """
    host = os.environ.get("SPACE_HOST", "").strip()
    if host:
        return "https://" + host.rstrip("/").removeprefix("https://").removeprefix(
            "http://"
        )
    # SPACE_HOST is the direct answer, but SPACE_ID is the variable that is always set, so derive
    # the hostname the same way `auto.auto_env` does.
    space_id = os.environ.get("SPACE_ID", "").strip()
    if space_id and "/" in space_id:
        slug = space_id.replace("/", "-").replace("_", "-").replace(".", "-").lower()
        return f"https://{slug}.hf.space"
    return ""


class HarborService:
    """Long-lived state for a serving process: capture proxy, forwarding, datasets.

    Held at module scope by `serve_harbor` so that the environment instances OpenEnv builds
    per-request can reach it. They must not own it: `/metadata` and `/schema` construct a throwaway
    environment on every call, so anything expensive on `__init__` would be paid per docs hit.
    """

    _instance: "HarborService | None" = None

    def __init__(
        self,
        *,
        llm_url: str = "",
        model: str = "",
        datasets: list[str],
        capture_port: int = 8100,
        expose: str = "gradio",
        api_key: str | None = None,
        auth_header: str = "Authorization",
        provider: str = "openai",
        capture_level: str = "tokens",
        max_output_tokens: int | None = 8192,
    ) -> None:
        # The management routes are the trainer's control plane; the proxy route is the agent's data
        # plane. Published or mounted, both are reachable by anyone who has the URL, so the control
        # plane gets its own key — minted here rather than configured, because nothing outside this
        # process needs to know it and an operator who has to invent one will skip it.
        # Honours $OPENENV_CAPTURE_ADMIN_KEY so an operator who wants to call the management routes
        # can choose the key; otherwise a random one, which locks the routes without putting a secret
        # nobody asked for into the logs.
        self.admin_key = os.environ.get(
            "OPENENV_CAPTURE_ADMIN_KEY"
        ) or secrets.token_urlsafe(24)
        self.llm_url = llm_url
        self.model = model
        self.datasets = datasets
        self.provider = provider
        self.capture_level = "text" if provider == "anthropic" else capture_level
        self.capture = CaptureServer(
            llm_url=llm_url,
            model=model,
            port=capture_port,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            auth_header=auth_header,
            provider=provider,
            capture_level=capture_level,
            admin_key=self.admin_key,
        )
        self._expose_kind = expose
        self.public_url = ""
        self.mounted = False
        self._forwarder: Any = None
        self._lock = threading.Lock()

    def start(self) -> str:
        """Make the capture proxy reachable from the sandbox and return its public URL.

        Two different situations, and conflating them is what makes the hosted case awkward:

        - **Hosted** (a Space). The platform already gives this process one public URL and exposes
          exactly one port. So the capture app is mounted onto the env server's own app under
          `CAPTURE_MOUNT` and reached at `<space>/capture`. No second port, no forwarding, nothing
          for the platform to object to.
        - **Local.** The sandbox runs off-cluster and cannot reach `127.0.0.1`, so the capture port
          is published by whichever forwarder was selected.
        """
        public = space_public_url()
        if public:
            # The env server's app serves it; `build_app` performs the mount.
            self.mounted = True
            self.public_url = f"{public}{CAPTURE_MOUNT}"
            return self.public_url

        from openenv.core.harness.capture.forwarding import make_forwarder

        self.capture.start()
        # A half-started service is worse than a failed one: the capture server owns a port and a
        # background thread, so leaving it up after the forwarder fails makes the next attempt fail
        # too, on a port conflict that has nothing to do with the real error.
        try:
            self._forwarder = make_forwarder(self._expose_kind)
            self.public_url = self._forwarder.start(self.capture.port)
        except BaseException:
            self._forwarder = None
            self.capture.stop()
            raise
        return self.public_url

    def stop(self) -> None:
        """Tear both halves down, even if the first half refuses to go.

        The capture port is released in a `finally` for the same reason `start()` unwinds on failure:
        a forwarder that raises on shutdown (a wedged tunnel process, a dead subprocess) would
        otherwise leave the proxy holding its port, and the next `start()` fails on a port conflict
        that says nothing about what actually went wrong.
        """
        try:
            if self._forwarder is not None:
                self._forwarder.stop()
        finally:
            self._forwarder = None
            self.capture.stop()

    @classmethod
    def current(cls) -> "HarborService | None":
        return cls._instance

    @classmethod
    def set_current(cls, service: "HarborService") -> None:
        cls._instance = service


def serve_harbor(
    *,
    llm_url: str = "",
    datasets: list[str],
    model: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    capture_port: int = 8100,
    expose: str = "gradio",
    env_file: str | None = None,
    api_key: str | None = None,
    auth_header: str = "Authorization",
    provider: str = "openai",
    max_output_tokens: int | None = 8192,
) -> None:
    """Boot the capture proxy, then serve the env server with the UI mounted.

    Args:
        llm_url (`str`):
            OpenAI-spec inference endpoint.
        datasets (`list[str]`):
            Dataset specs to serve as splits.
        port (`int`, *optional*, defaults to `8000`):
            Env server port.
        capture_port (`int`, *optional*, defaults to `8100`):
            Capture proxy port. This is the one published to the sandbox. Ignored on a hosted
            platform, where the proxy is mounted on the env server's own app instead.
        expose (`str`, *optional*, defaults to `"gradio"`):
            How the sandbox reaches the capture proxy locally: `gradio`, `cloudflare` or `direct`.
        api_key (`str`, *optional*):
            Credential for the inference endpoint. Defaults to `$OPENENV_LLM_API_KEY`. Never reaches
            the sandbox: the agent's key is a capture session id.
        auth_header (`str`, *optional*, defaults to `"Authorization"`):
            Header to send `api_key` under.
    """
    import uvicorn

    from .startup import prepare

    caps = prepare(
        llm_url=llm_url,
        model=model,
        datasets=datasets,
        env_file=env_file,
        # A served deployment does not need an engine to be useful: rollouts name their own, and it
        # is probed per engine when the session is created. Demanding one here coupled a server whose
        # real cost is its dataset tree to the boot order of a vLLM that restarts every run.
        require_llm=bool(llm_url),
        quiet=False,
        api_key=api_key,
        auth_header=auth_header,
        provider=provider,
    )
    model = caps.llm.get("model") or model or ""
    # `tokens` only when an engine was actually measured at it. With no default engine the default
    # level must be the weakest, so a rollout that somehow reaches the default is never mistaken for
    # a trainable one.
    capture_level = caps.llm.get("capture_level") or ("tokens" if llm_url else "text")
    # `prepare` has loaded the dotenv by now, so a key that lives only in --env-file is visible.
    api_key = api_key or os.environ.get("OPENENV_LLM_API_KEY") or None

    service = HarborService(
        llm_url=llm_url,
        model=model,
        datasets=datasets,
        capture_port=capture_port,
        expose=expose,
        api_key=api_key,
        auth_header=auth_header,
        provider=provider,
        capture_level=capture_level,
        max_output_tokens=max_output_tokens,
    )
    public = service.start()
    HarborService.set_current(service)

    where = "mounted on this app" if service.mounted else f":{capture_port}"
    print(f"\ncapture   {where} -> {public}")
    print(
        "          session routes are gated; set OPENENV_CAPTURE_ADMIN_KEY to call them yourself"
    )
    if capture_level != "tokens":
        # Repeated after the capabilities report, because this is the last line before the server
        # starts serving and it changes what every rollout from it is worth.
        print(
            f"mode      EVAL ONLY (capture_level={capture_level}) — rollouts carry reward and "
            "trace, nothing trainable"
        )
    print(
        f"server    http://{host}:{port}    (UI at /web, Task API at /{{env}}/splits)"
    )
    print("Ctrl-C to stop\n")

    # The UI is the whole point of this entry point, so turn it on rather than making the operator
    # discover an env var.
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

    app = build_app(datasets=datasets, llm_url=llm_url, model=model, llm=caps.llm)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        service.stop()


def build_app(
    *,
    datasets: list[str],
    llm_url: str = "",
    model: str = "",
    llm: dict[str, Any] | None = None,
) -> Any:
    """The FastAPI app: Task API + MCP + the Gradio UI."""
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_environment import (
        CallToolAction,
        CallToolObservation,
    )

    from .environment import HarborEnvironment
    from .models import HarborState
    from .ui import harbor_gradio_builder

    HarborEnvironment.configure(
        datasets=datasets, llm_url=llm_url, model=model, llm=llm
    )

    def gradio_builder(
        _web_manager: Any = None,
        _action_fields: Any = None,
        _metadata: Any = None,
        _is_chat: Any = None,
        display_title: str = "",
        _quick_start: Any = None,
    ) -> Any:
        """OpenEnv calls this positionally with six web-interface arguments.

        None of them is used: the Harbor UI drives rollouts through its own handlers rather than the
        generic action-field form, because a rollout is one long tool call, not a step. Even the
        title is its own, since OpenEnv's ("OpenEnv Agentic Environment: harbor_env") names the
        env class rather than what the page is.
        """
        return harbor_gradio_builder(datasets=datasets, title="OpenEnv × Harbor")

    app = create_app(
        HarborEnvironment,
        CallToolAction,
        CallToolObservation,
        state_cls=HarborState,
        env_name="harbor_env",
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
        gradio_builder=gradio_builder,
        custom_tab_name="Harbor",
        custom_tab_primary=True,
        show_default_tab=False,
    )

    # When the platform gives us a single port, the capture proxy rides on this app instead of
    # being published separately. Mounting strips the prefix, so the proxy's own catch-all still
    # sees `/v1/chat/completions` and every dialect keeps working unchanged.
    service = HarborService.current()
    if service is not None and service.mounted:
        app.mount(CAPTURE_MOUNT, service.capture.app)

    _attach_hf_login(app)
    # Every UI handler that changes something (a rollout, an added dataset) is a POST under /web,
    # and Gradio accepts those from any origin. Without this, any page a visitor opens could make
    # their browser start rollouts on this server's endpoint, or reach a server on their own machine
    # that the page itself cannot. Nothing a real visitor does is cross-site.
    app.add_middleware(SameOrigin)
    return app


def _attach_hf_login(app: Any) -> bool:
    """ "Sign in with Hugging Face", where the Hub has set it up for the Space (`hf_oauth: true`).

    Attached to this app rather than to the Gradio UI mounted at `/web`: Gradio's OAuth routes and
    the callback URL it registers assume the site root. Gradio then hands the signed-in visitor's
    token to any UI handler that asks for a `gr.OAuthToken`, through the session cookie set here.

    A cookie that identifies the visitor makes another website's request count as theirs; the
    same-origin check `build_app` installs for the UI covers that too.
    """
    from .ui_settings import load

    if not load().hf_login:
        return False
    # Gradio tells a Space from a laptop by `SYSTEM=spaces`, which Docker Spaces do not set. Without
    # it, `attach_oauth` installs its local stand-in, which signs every visitor in as the account of
    # the token the server holds (the operator's), with a token that calls nothing. `hf_login` is
    # only true on a Space that has an OAuth app, so this is that Space. It stays process-wide on
    # purpose: `attach_oauth` reads it once (its routes then use `SPACE_HOST`), and whatever else in
    # Gradio asks `gradio.utils.get_space()` afterwards should get the answer the login got.
    os.environ.setdefault("SYSTEM", "spaces")
    try:
        from gradio.oauth import attach_oauth
        from gradio.utils import get_space

        if get_space() is None:
            print("hf login  off: Gradio does not see a Space here")
            return False
        attach_oauth(app)
    except (ImportError, ValueError) as exc:
        print(f"hf login  off: {exc}")
        return False
    print("hf login  on (Inference Providers with the visitor's own account)")
    return True


class SameOrigin:
    """Refuse a state-changing request to the UI that a browser made from another site.

    Browsers label every request they make (`Sec-Fetch-Site`, `Origin`); servers calling this app,
    such as a sandbox reaching the capture proxy or a trainer on the Task API, send neither and pass.
    Only the UI (`/web`) is guarded: the Task API and MCP carry no visitor state, and browser tools
    such as the MCP Inspector call them from another origin on purpose.
    """

    def __init__(self, app: Any, prefix: str = "/web") -> None:
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        path = scope.get("path") or ""
        if (
            scope.get("type") == "http"
            and scope.get("method") not in ("GET", "HEAD", "OPTIONS")
            and (path == self.prefix or path.startswith(self.prefix + "/"))
        ):
            headers = {
                k.decode().lower(): v.decode() for k, v in scope.get("headers") or []
            }
            site = headers.get("sec-fetch-site", "")
            origin = headers.get("origin", "")
            if site == "cross-site" or (
                origin and not _same_host(origin, headers.get("host", ""))
            ):
                from starlette.responses import PlainTextResponse

                await PlainTextResponse("cross-site request refused", status_code=403)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def _same_host(origin: str, host: str) -> bool:
    """Whether a page at `origin` is this server's own.

    Its own hosts are the `Host` it was reached at, a Space's `SPACE_HOST`, and any listed in
    `OPENENV_HARBOR_UI_HOSTS` (for a proxy that rewrites `Host`). Never `X-Forwarded-Host`: a
    client sets that itself, so it would let any origin vouch for itself.
    """
    from urllib.parse import urlparse

    listed = ",".join(
        os.environ.get(k, "") for k in ("SPACE_HOST", "OPENENV_HARBOR_UI_HOSTS")
    )
    allowed = {host, *[h.strip() for h in listed.split(",") if h.strip()]}
    return urlparse(origin).netloc in allowed
