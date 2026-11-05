# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Pathway Analysis Environment."""

import inspect
import logging
import os
from pathlib import Path

# Pathway lab is meant to be used at /web; OpenEnv defaults web off unless set.
if "ENABLE_WEB_INTERFACE" not in os.environ:
    os.environ["ENABLE_WEB_INTERFACE"] = "true"

# Some dependencies (e.g. gseapy) import matplotlib, which tries to write a font/cache
# directory under the user's home. In sandboxed / CI contexts this can be unwritable.
if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(__file__).resolve().parent.parent / "outputs" / ".mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

from openenv.core.env_server.http_server import create_app

from ..models import PathwayAction, PathwayObservation, PathwayState
from .gradio_ui import build_pathway_gradio_app
from .pathway_environment import PathwayEnvironment

_logger = logging.getLogger(__name__)

_sig = inspect.signature(create_app)
_kw: dict = {
    "env": PathwayEnvironment,
    "action_cls": PathwayAction,
    "observation_cls": PathwayObservation,
    "env_name": "pathway_analysis_env",
}
if "gradio_builder" in _sig.parameters:
    _kw["gradio_builder"] = build_pathway_gradio_app
else:
    _logger.warning(
        "openenv-core does not support gradio_builder; Pathway lab tab will be unavailable."
    )

app = create_app(**_kw, state_cls=PathwayState)


def main():
    """Entry point for ``uv run --project . server``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
