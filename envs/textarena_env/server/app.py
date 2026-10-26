# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application entrypoint for the TextArena environment."""

from __future__ import annotations

import inspect
import logging
import os

from openenv.core.env_server.http_server import create_app

try:
    # When running as installed package
    from textarena_env.models import (
        TextArenaAction,
        TextArenaObservation,
        TextArenaState,
    )
    from textarena_env.server.environment import TextArenaEnvironment
    from textarena_env.server.gradio_ui import build_textarena_gradio_app
except ImportError:
    # When running uvicorn directly from textarena_env/
    from models import TextArenaAction, TextArenaObservation, TextArenaState

    from .environment import TextArenaEnvironment
    from .gradio_ui import build_textarena_gradio_app


def _parse_env_kwargs(prefix: str = "TEXTARENA_KW_") -> dict[str, str]:
    """Collect arbitrary environment kwargs from the process environment."""

    env_kwargs: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_key = key[len(prefix) :].lower()
            env_kwargs[env_key] = value
    return env_kwargs


env_id = os.getenv("TEXTARENA_ENV_ID", "Wordle-v0")
num_players = int(os.getenv("TEXTARENA_NUM_PLAYERS", "1"))
max_turns_env = os.getenv("TEXTARENA_MAX_TURNS")
max_turns = int(max_turns_env) if max_turns_env is not None else None
download_nltk = os.getenv("TEXTARENA_DOWNLOAD_NLTK", "1") in {"1", "true", "True"}

extra_kwargs = _parse_env_kwargs()
max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))


# Factory function to create TextArenaEnvironment instances
def create_textarena_environment():
    """Factory function that creates TextArenaEnvironment with config."""
    return TextArenaEnvironment(
        env_id=env_id,
        num_players=num_players,
        max_turns=max_turns,
        download_nltk=download_nltk,
        env_kwargs=extra_kwargs,
    )


# Create the FastAPI app
# Pass the factory function instead of an instance for WebSocket session support.
# When ENABLE_WEB_INTERFACE=true and openenv supports it, the Gradio UI has
# two tabs: Playground (default) and Custom (Wordle HTML block). See server/gradio_ui.py.
_logger = logging.getLogger(__name__)
_sig = inspect.signature(create_app)
if "gradio_builder" in _sig.parameters:
    # Each kwarg is guarded by inspect.signature so older openenv
    # releases that predate the param still boot this env.
    create_app_kwargs: dict = {
        "env_name": "textarena_env",
        "max_concurrent_envs": max_concurrent,
        "gradio_builder": build_textarena_gradio_app,
    }
    if "state_cls" in _sig.parameters:
        create_app_kwargs["state_cls"] = TextArenaState
    app = create_app(
        create_textarena_environment,
        TextArenaAction,
        TextArenaObservation,
        **create_app_kwargs,
    )
else:
    _logger.warning(
        "Installed openenv does not support gradio_builder; "
        "custom Gradio tab will not be available."
    )
    app = create_app(
        create_textarena_environment,
        TextArenaAction,
        TextArenaObservation,
        env_name="textarena_env",
        max_concurrent_envs=max_concurrent,
    )


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m textarena_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn textarena_env.server.app:app --workers 4
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
