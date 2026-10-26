# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the REPL Environment.

This module creates an HTTP server that exposes the REPLEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

The server includes llm_query and llm_query_batched support via HuggingFace Inference API,
enabling the Recursive Language Model (RLM) paradigm.

LLM Token Configuration:
    1. Client can pass `hf_token` in reset() - RECOMMENDED
    2. Server fallback: HF_TOKEN environment variable

LLM functions are created dynamically in REPLEnvironment.reset() based on the
available token (client or server).

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server

Environment Variables:
    HF_TOKEN: Fallback HuggingFace API token (client token takes priority)
    LLM_MODEL: Model to use for llm_query/llm_query_batched (default: Qwen/Qwen3.5-9B)
    MAX_CONCURRENT_ENVS: Maximum concurrent WebSocket sessions (default: 8)
"""

import inspect
import logging
import os

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import REPLAction, REPLObservation, REPLState
    from .gradio_ui import build_repl_gradio_app
    from .repl_environment import REPLEnvironment
except ImportError:
    from models import REPLAction, REPLObservation, REPLState
    from openenv.core.env_server.http_server import create_app
    from server.gradio_ui import build_repl_gradio_app
    from server.repl_environment import REPLEnvironment


# ============== CONFIGURATION ==============
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-9B")
HF_TOKEN = os.environ.get("HF_TOKEN")
REPL_MAX_ITERATIONS = int(os.environ.get("REPL_MAX_ITERATIONS", "30"))
REPL_MAX_OUTPUT_LENGTH = int(os.environ.get("REPL_MAX_OUTPUT_LENGTH", "8192"))
REPL_CONTEXT_PREVIEW_LENGTH = int(os.environ.get("REPL_CONTEXT_PREVIEW_LENGTH", "500"))
REPL_RLM_MAX_DEPTH = int(os.environ.get("REPL_RLM_MAX_DEPTH", "2"))
REPL_RLM_MAX_ITERATIONS = int(os.environ.get("REPL_RLM_MAX_ITERATIONS", "30"))
MAX_CONCURRENT_ENVS = int(os.environ.get("MAX_CONCURRENT_ENVS", "8"))
# ==========================================

_logger = logging.getLogger(__name__)

# Log LLM configuration
if HF_TOKEN:
    print("[REPL Server] LLM support ENABLED (server token configured)")
    print(f"[REPL Server] Default model: {LLM_MODEL}")
else:
    print("[REPL Server] No server HF_TOKEN configured")
    print(
        "[REPL Server] LLM functions will be enabled if client passes hf_token in reset()"
    )


def create_repl_environment() -> REPLEnvironment:
    """Factory function that creates REPLEnvironment with server config.

    LLM functions are created dynamically during `reset()` when a client
    passes `hf_token`. Rewards are computed via the default `REPLRubric`;
    pass `expected_answer` at reset time for outcome-based scoring.
    """
    return REPLEnvironment(
        max_iterations=REPL_MAX_ITERATIONS,
        max_output_length=REPL_MAX_OUTPUT_LENGTH,
        context_preview_length=REPL_CONTEXT_PREVIEW_LENGTH,
        rlm_max_depth=REPL_RLM_MAX_DEPTH,
        rlm_max_iterations=REPL_RLM_MAX_ITERATIONS,
    )


# Create the app with web interface and README integration.
_sig = inspect.signature(create_app)
if "gradio_builder" in _sig.parameters:
    # Each kwarg is guarded by inspect.signature so older openenv
    # releases that predate the param still boot this env.
    create_app_kwargs: dict = {
        "env_name": "repl_env",
        "max_concurrent_envs": MAX_CONCURRENT_ENVS,
        "gradio_builder": build_repl_gradio_app,
    }
    if "custom_tab_name" in _sig.parameters:
        create_app_kwargs["custom_tab_name"] = "REPL"
    if "custom_tab_primary" in _sig.parameters:
        create_app_kwargs["custom_tab_primary"] = True
    if "show_default_tab" in _sig.parameters:
        create_app_kwargs["show_default_tab"] = False
    if "title_override" in _sig.parameters:
        create_app_kwargs["title_override"] = (
            "OpenEnv REPL — Recursive Language Model playground"
        )
    if "state_cls" in _sig.parameters:
        create_app_kwargs["state_cls"] = REPLState
    app = create_app(
        create_repl_environment,
        REPLAction,
        REPLObservation,
        **create_app_kwargs,
    )
else:
    _logger.warning(
        "Installed openenv does not support gradio_builder; "
        "custom REPL Gradio tab will not be available."
    )
    app = create_app(
        create_repl_environment,
        REPLAction,
        REPLObservation,
        env_name="repl_env",
        max_concurrent_envs=MAX_CONCURRENT_ENVS,
    )


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.repl_env.server.app
        openenv serve repl_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
