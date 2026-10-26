# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data loading utilities for CARLA scenarios.

Adapted from SinatrasC/carla-env:
https://github.com/SinatrasC/carla-env
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

__all__ = ["load_json", "load_trolley_micro_benchmarks"]

_DATA_DIR = Path(__file__).parent


def load_json(name: str) -> Dict[str, Any]:
    path = _DATA_DIR / name
    with open(path, "r") as f:
        return json.load(f)


def load_trolley_micro_benchmarks() -> Dict[str, Any]:
    return load_json("trolley_micro_benchmarks.json")
