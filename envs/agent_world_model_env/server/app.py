"""
Each environment in Agent World Model is a self-contained FastAPI application
with SQLAlchemy/SQLite backend and MCP tool interface.

Usage:
    PYTHONPATH=src:envs uvicorn envs.agent_world_model_env.server.app:app \\
        --host 0.0.0.0 --port 8000

HTTP /reset and /step are disabled because AWM requires stateful WebSocket
connections — each HTTP request would create a fresh environment, dropping
the subprocess and tool cache.
"""

import os
import uvicorn

import gradio as gr
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from openenv.core.env_server.http_server import create_app

from ..models import AWMAction, AWMObservation
from .awm_environment import AWMEnvironment
from .config import MAX_CONCURRENT_ENVS
from .data_loader import AWMDataLoader
from .session_registry import registry as _registry
from .web_ui import build_awm_gradio_app


_shared_data_loader = AWMDataLoader()


def _env_factory():
    return AWMEnvironment(data_loader=_shared_data_loader)


app = create_app(
    _env_factory,
    AWMAction,
    AWMObservation,
    env_name="agent_world_model_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


def _swap_in_custom_gradio_ui() -> None:
    """Replace openenv's default web UI with the AWM Web Console.

    The framework's ``gradio_builder`` parameter wraps our blocks inside a
    ``Playground | Custom`` TabbedInterface, which we don't want. Instead we
    let the framework set up its default UI, then drop the default Mount +
    legacy ``/web/*`` HTTP endpoints and mount our own blocks at ``/web``.
    Pulls ``WebInterfaceManager`` out of the existing route closures.
    """
    if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() not in (
        "true",
        "1",
        "yes",
    ):
        return

    web_manager = None
    metadata = None
    for r in app.routes:
        for cell in getattr(getattr(r, "endpoint", None), "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if web_manager is None and v.__class__.__name__ == "WebInterfaceManager":
                web_manager = v
            if metadata is None and v.__class__.__name__ == "EnvironmentMetadata":
                metadata = v
        if web_manager is not None and metadata is not None:
            break
    if web_manager is None:
        return

    # /web in 0.2.1 is the legacy "HumanAgent Interface" HTMLResponse, not a
    # redirect — drop it together with the rest of the default UI's HTTP API.
    legacy_paths = {
        "/web",
        "/web/reset",
        "/web/step",
        "/web/state",
        "/web/metadata",
        "/ws/ui",
    }
    app.routes[:] = [
        r
        for r in app.routes
        if not (
            (getattr(r, "path", None) == "/web" and r.__class__.__name__ == "Mount")
            or getattr(r, "path", None) in legacy_paths
        )
    ]

    blocks = build_awm_gradio_app(
        web_manager,
        action_fields=None,
        metadata=metadata,
        is_chat_env=False,
        title="agent_world_model_env",
        quick_start_md=None,
    )
    gr.mount_gradio_app(app, blocks, path="/web")


_swap_in_custom_gradio_ui()


_HTTP_NOT_SUPPORTED_RESPONSE = {
    "error": "HTTP mode not supported for AWM environment",
    "reason": "AWM launches subprocesses on reset() that must persist across step() calls. "
    "HTTP is stateless - each request creates a new environment instance, "
    "losing the subprocess and all loaded tools.",
    "solution": "Use WebSocket endpoint instead",
    "examples": [
        "Python: AWMEnv(base_url='http://host:port')  # uses /ws internally",
        "Direct: connect to ws://host:port/ws",
    ],
}

app.routes[:] = [
    r for r in app.routes if getattr(r, "path", None) not in ("/reset", "/step")
]


@app.post("/reset", tags=["disabled"])
async def reset_not_supported():
    return JSONResponse(status_code=400, content=_HTTP_NOT_SUPPORTED_RESPONSE)


@app.post("/step", tags=["disabled"])
async def step_not_supported():
    return JSONResponse(status_code=400, content=_HTTP_NOT_SUPPORTED_RESPONSE)


@app.get("/stats", tags=["monitoring"])
async def stats():
    return JSONResponse(content=_registry.get_stats())


def _has_route(path: str) -> bool:
    return any(getattr(r, "path", None) == path for r in app.routes)


def _https_aware_redirect(request: Request, path: str) -> RedirectResponse:
    # HF's reverse proxy rewrites relative redirects into absolute URLs and
    # picks the scheme from the upstream request — which is HTTP. Build an
    # explicit absolute URL with the original scheme so the iframe doesn't
    # get blocked as mixed content.
    host = request.headers.get("x-forwarded-host") or request.headers.get(
        "host", request.url.netloc
    )
    proto = request.headers.get("x-forwarded-proto") or (
        "https" if host.endswith(".hf.space") else request.url.scheme
    )
    return RedirectResponse(url=f"{proto}://{host}{path}")


# 0.2.1 doesn't auto-redirect / and /web to /web/. HF Spaces hits both.
if not _has_route("/"):

    @app.get("/", include_in_schema=False)
    async def _root_redirect(request: Request):
        return _https_aware_redirect(request, "/web/")


if not _has_route("/web"):

    @app.get("/web", include_in_schema=False)
    async def _web_redirect(request: Request):
        return _https_aware_redirect(request, "/web/")


@app.middleware("http")
async def _force_https_redirects(request: Request, call_next):
    # HF Spaces' reverse proxy strips the original https scheme; any
    # absolute Location header we emit goes out as http:// which gets
    # blocked as mixed content inside the HF iframe. Force https for
    # *.hf.space hosts.
    response = await call_next(request)
    loc = response.headers.get("location")
    if loc and loc.startswith("http://") and ".hf.space" in loc:
        response.headers["location"] = "https://" + loc[len("http://") :]
    return response


def main():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
