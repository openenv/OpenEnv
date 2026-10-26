"""Create the stateful FastAPI server for the ThinkingBox adapter.

OpenEnv owns the WebSocket lifecycle while each environment instance delegates
tool execution to an externally managed ThinkingBox Session Proxy.
"""

import logging
from collections.abc import Callable

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core import Environment, HTTPEnvServer, ServerMode
from thinkingbox.common.config_types import SessionProxyConfig

from thinkingbox_env import benchmark_data
from thinkingbox_env.models import (
    ThinkingBoxAction,
    ThinkingBoxObservation,
    ThinkingBoxState,
)
from thinkingbox_env.server import config
from thinkingbox_env.server.config import load_runtime_settings
from thinkingbox_env.server.thinkingbox_environment import (
    _redacted_exc_info,
    ThinkingBoxEnvironment,
)


logger = logging.getLogger(__name__)


def create_thinkingbox_app(
    environment_factory: Callable[[], Environment] = ThinkingBoxEnvironment,
    *,
    max_concurrent_envs: int = config.MAX_CONCURRENT_ENVS,
) -> FastAPI:
    """Create the stateful OpenEnv WebSocket application.

    Args:
        environment_factory (`collections.abc.Callable`, *optional*):
            Factory for independent [`ThinkingBoxEnvironment`] instances.
        max_concurrent_envs (`int`, *optional*, defaults to configured value):
            Maximum simultaneously active environment instances.

    Returns:
        `fastapi.FastAPI`:
            Configured OpenEnv application.
    """
    app = FastAPI(
        title="ThinkingBox OpenEnv API",
        description=(
            "ThinkingBox episodes are stateful and supported only through the "
            "OpenEnv /ws WebSocket client."
        ),
    )
    server = HTTPEnvServer(
        environment_factory,
        ThinkingBoxAction,
        ThinkingBoxObservation,
        env_name="thinkingbox_env",
        max_concurrent_envs=max_concurrent_envs,
        state_cls=ThinkingBoxState,
    )
    server.register_routes(app, mode=ServerMode.PRODUCTION)
    app.state.openenv_server = server
    return app


app = create_thinkingbox_app()


@app.get("/ready", include_in_schema=False)
async def ready() -> JSONResponse:
    """Report observable canonical benchmark readiness components.

    Returns:
        `fastapi.responses.JSONResponse`:
            Component readiness with HTTP 200 only when all observable
            requirements are ready.
    """
    data_ok = benchmark_data.data_ready()
    configured = bool(config.THINKINGBOX_CONFIG)
    config_ok = False
    user_model_ok = False
    judge_model_ok = False
    default_proxy = SessionProxyConfig(
        endpoint_url=config.SESSION_PROXY_URL,
        timeout=config.PROXY_TIMEOUT,
    )
    try:
        settings = load_runtime_settings(
            config.THINKINGBOX_CONFIG or None,
            default_proxy,
        )
        config_ok = configured
        user_model_ok = config_ok and settings.user_model is not None
        judge_model_ok = config_ok and settings.judge_model is not None
    except Exception as exc:
        logger.warning(
            "ThinkingBox readiness configuration check failed",
            extra={
                "event_name": "thinkingbox.readiness_failure",
                "tb_component": "runtime_config",
                "exception_type": type(exc).__name__,
            },
            exc_info=_redacted_exc_info(exc),
        )
        settings = None

    proxy_ok = False
    proxy = settings.proxy if settings is not None else default_proxy
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{proxy.endpoint_url.rstrip('/')}/health")
            proxy_ok = response.is_success
    except Exception as exc:
        logger.warning(
            "ThinkingBox readiness proxy check failed",
            extra={
                "event_name": "thinkingbox.readiness_failure",
                "tb_component": "session_proxy",
                "exception_type": type(exc).__name__,
            },
            exc_info=_redacted_exc_info(exc),
        )

    ready_now = data_ok and config_ok and user_model_ok and judge_model_ok and proxy_ok
    return JSONResponse(
        {
            "ready": ready_now,
            "data": data_ok,
            "runtime_config": config_ok,
            "user_model": user_model_ok,
            "judge_model": judge_model_ok,
            "session_proxy": proxy_ok,
            "typesense": {
                "observable": False,
                "ready": None,
                "limitation": (
                    "Scenario-specific Typesense readiness is not observable "
                    "from the OpenEnv process."
                ),
            },
        },
        status_code=200 if ready_now else 503,
    )


def main() -> None:
    """Run the ThinkingBox OpenEnv application on its configured public port."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
