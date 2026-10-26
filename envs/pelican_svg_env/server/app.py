# SPDX-License-Identifier: BSD-3-Clause

"""FastAPI application for the Pelican SVG environment."""

import os

from openenv.core.env_server import create_app

from ..models import PelicanSvgAction, PelicanSvgObservation, PelicanSvgState
from .pelican_svg_environment import PelicanSvgEnvironment

# The class is passed rather than an instance so each WebSocket session gets
# its own environment and its own sampled task.
app = create_app(
    PelicanSvgEnvironment,
    PelicanSvgAction,
    PelicanSvgObservation,
    env_name="pelican_svg_env",
    state_cls=PelicanSvgState,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
