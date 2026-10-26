# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app for the Terminus environment."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

try:
    from ..models import TerminusState
    from .gradio_ui import terminus_ui_builder
    from .terminus_env_environment import TerminusEnvironment
except ImportError:  # pragma: no cover
    from models import TerminusState  # type: ignore
    from server.gradio_ui import terminus_ui_builder  # type: ignore
    from server.terminus_env_environment import TerminusEnvironment  # type: ignore


def _load_env_file() -> None:
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
    _create_app_kwargs["state_cls"] = TerminusState

app = create_app(
    TerminusEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="terminus_env",
    **_create_app_kwargs,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    gradio_builder=terminus_ui_builder,
)


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
