# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the τ-emotion-bench environment.

Environment variables:
    TAU_EMOTION_DOMAIN: τ-emotion-bench domain (default: airline)
    TAU_EMOTION_SPLIT: task split, test or train (default: test)
    TAU_EMOTION_USER_PROVIDER: hf, openai or anthropic (default: hf)
    TAU_EMOTION_USER_MODEL: model for the simulated user (default: the provider's)
    TAU_EMOTION_USER_API_BASE: with the openai provider, an OpenAI-compatible server
    HF_TOKEN / OPENAI_API_KEY / ANTHROPIC_API_KEY: credential for the provider
    TAU_EMOTION_DATA_DIR: τ-emotion-bench's `data` folder (set in the Docker image)
    MAX_CONCURRENT_ENVS: API sessions at once (default: 8)
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from ..models import TauEmotionState
from .tau_emotion_environment import TauEmotionEnvironment


def _env_factory(**overrides) -> TauEmotionEnvironment:
    """The server's environment."""
    settings = {
        "domain": os.environ.get("TAU_EMOTION_DOMAIN", "airline"),
        "split": os.environ.get("TAU_EMOTION_SPLIT", "test"),
        "user_provider": os.environ.get("TAU_EMOTION_USER_PROVIDER", "hf"),
        "user_model": os.environ.get("TAU_EMOTION_USER_MODEL") or None,
        "user_api_base": os.environ.get("TAU_EMOTION_USER_API_BASE") or None,
    }
    return TauEmotionEnvironment(**{**settings, **overrides})


app = create_app(
    _env_factory,
    CallToolAction,
    CallToolObservation,
    env_name="tau_emotion_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "8")),
    state_cls=TauEmotionState,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
