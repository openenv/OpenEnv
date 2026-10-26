# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the Jupyter environment."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

try:
    from ..models import JupyterState
    from .gradio_ui import jupyter_ui_builder
    from .jupyter_environment import JupyterEnvironment
except ImportError:  # pragma: no cover
    from models import JupyterState  # type: ignore
    from server.gradio_ui import jupyter_ui_builder  # type: ignore
    from server.jupyter_environment import JupyterEnvironment  # type: ignore


def _load_env_file() -> None:
    """Load env-local .env values for local development only."""

    candidate = Path(__file__).resolve().parents[1] / ".env"
    if not candidate.exists():
        return
    for raw in candidate.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# Older openenv-core releases lack these model parameters.
_create_app_kwargs = (
    {"reset_observation_cls": Observation}
    if "reset_observation_cls" in inspect.signature(create_app).parameters
    else {}
)
if "state_cls" in inspect.signature(create_app).parameters:
    _create_app_kwargs["state_cls"] = JupyterState

app = create_app(
    JupyterEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="jupyter_env",
    **_create_app_kwargs,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    gradio_builder=jupyter_ui_builder,
)


def main() -> None:
    """Entrypoint for direct local serving."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
