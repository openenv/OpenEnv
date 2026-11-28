# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the chart-to-code environment."""

import os

from openenv.core.env_server import create_app

from ..models import Chart2CodeAction, Chart2CodeObservation, Chart2CodeState
from .chart2code_environment import Chart2CodeEnvironment

# The class is passed rather than an instance so each WebSocket session gets
# its own environment and its own task.
app = create_app(
    Chart2CodeEnvironment,
    Chart2CodeAction,
    Chart2CodeObservation,
    env_name="chart2code_env",
    state_cls=Chart2CodeState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
