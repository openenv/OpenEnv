# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application entry point for the Calendar environment.

This module re-exports the existing FastAPI app from main.py and provides
the standard server entry point used by OpenEnv tooling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    REPO_ROOT = SERVER_DIR.parents[3]
except IndexError:
    REPO_ROOT = None

if REPO_ROOT is not None:
    SRC_DIR = REPO_ROOT / "src"
    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

try:
    from . import main as _main
except ImportError:
    import importlib

    _main = importlib.import_module("main")

app = _main.app


def main():
    """Run the Calendar environment server with uvicorn."""

    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8004")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
