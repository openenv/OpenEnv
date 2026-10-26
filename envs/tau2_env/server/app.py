# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the τ²-bench environment.

Environment variables:
    TAU2_DOMAIN: τ²-bench domain (default: airline)
    TAU2_SPLIT: task split (default: test)
    TAU2_USER_PROVIDER: hf, openai or anthropic (default: hf)
    TAU2_USER_MODEL: model for the simulated user and judge (default: the provider's)
    HF_TOKEN / OPENAI_API_KEY / ANTHROPIC_API_KEY: credential for the provider
    TAU2_DATA_DIR: τ²-bench's `data` folder (set in the Docker image)
    MAX_CONCURRENT_ENVS: API sessions at once (default: 8)
"""

import json
import os
from typing import Any, Dict

from fastapi import Request
from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from pydantic import field_validator

from ..models import Tau2State
from .gradio_ui import build_ui
from .tau2_environment import Tau2Environment


def _env_factory(**overrides) -> Tau2Environment:
    """The server's environment. The web UI overrides `domain`, `split`, `user_model` or `hf_token`."""
    settings = {
        "domain": os.environ.get("TAU2_DOMAIN", "airline"),
        "split": os.environ.get("TAU2_SPLIT", "test"),
        "user_provider": os.environ.get("TAU2_USER_PROVIDER", "hf"),
        "user_model": os.environ.get("TAU2_USER_MODEL") or None,
    }
    return Tau2Environment(**{**settings, **overrides})


class Tau2CallToolAction(CallToolAction):
    """CallToolAction that accepts JSON strings for arguments (the web UI sends strings)."""

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            return json.loads(v)
        return v


# Gradio signs visitors in with Hugging Face only where `SYSTEM=spaces`, which Docker
# Spaces do not set (elsewhere it mocks the login).
if os.environ.get("SPACE_ID"):
    os.environ.setdefault("SYSTEM", "spaces")

app = create_app(
    _env_factory,
    Tau2CallToolAction,
    CallToolObservation,
    env_name="tau2_env",
    state_cls=Tau2State,
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "8")),
    gradio_builder=build_ui(_env_factory),
    custom_tab_name="τ²-bench",
    custom_tab_primary=True,
    title_override="τ²-bench",
)


# The sign-in button and Hugging Face's callback use these paths at the root, while the
# UI (and so Gradio's OAuth routes) lives under /web.
@app.get("/login/huggingface", include_in_schema=False)
@app.get("/login/callback", include_in_schema=False)
@app.get("/logout", include_in_schema=False)
def _oauth_under_web(request: Request) -> RedirectResponse:
    query = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(f"/web{request.url.path}{query}")


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
