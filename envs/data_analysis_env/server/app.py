# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the data-analysis environment.

Run it as root inside its container (the Docker image does) so each session's code runs as its
own unprivileged user. See the README's "Sandbox" section for what changes otherwise.

    uvicorn data_analysis_env.server.app:app --host 0.0.0.0 --port 8000

Environment variables:
    DATA_ANALYSIS_SPLIT: default split for reset() (`eval`).
    DATA_ANALYSIS_TASKS_ROOT: where task files are downloaded (a directory under the temp dir).
    DATA_ANALYSIS_MAX_FILE_MB: largest data file a task may have (10).
    DATA_ANALYSIS_EXEC_TIMEOUT_S: hard limit per run_python call (30).
    DATA_ANALYSIS_MEM_LIMIT_MB: address-space limit of a session (4096).
    DATA_ANALYSIS_SANDBOX_UID_BASE / _UID_COUNT: uids for sessions when root (61000, 1024).
    DATA_ANALYSIS_ISOLATE_NETWORK: `0` to skip the workers' network namespace (on by default).
    MAX_CONCURRENT_ENVS: concurrent WebSocket sessions (64).
"""

import inspect
import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from ..models import DataAnalysisState
from .data_analysis_environment import DataAnalysisEnvironment
from .sandbox import set_non_dumpable

# Same uid as the workers when the server is not root: don't let a worker that got past the
# audit hook ptrace this process or read its memory, which holds the gold answers (Linux only).
set_non_dumpable()

# Older openenv releases lack these parameters.
_params = inspect.signature(create_app).parameters
_kwargs = {}
if "reset_observation_cls" in _params:
    _kwargs["reset_observation_cls"] = Observation
if "state_cls" in _params:
    _kwargs["state_cls"] = DataAnalysisState

app = create_app(
    DataAnalysisEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="data_analysis_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
    **_kwargs,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
