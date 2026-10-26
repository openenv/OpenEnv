# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple logging utilities for CARLA environment.

Adapted from SinatrasC/carla-env logging module.
"""

from __future__ import annotations

import logging
import os

_LOGGER_BASE = "openenv.carla_env"

# Package-level parent logger. Defaults to WARNING; override via env vars
# or configure_logging().
_pkg_logger = logging.getLogger(_LOGGER_BASE)
if _pkg_logger.level == logging.NOTSET:
    _pkg_logger.setLevel(logging.WARNING)


def _normalize_level(level: str | int) -> str | int:
    """Accept common level formats (e.g. ``"debug"``, ``"10"``)."""
    if isinstance(level, str):
        s = level.strip()
        if s.isdigit():
            return int(s)
        return s.upper()
    return level


def configure_logging(log_level: str | int | None = None) -> None:
    """
    Set the package parent logger level.

    Precedence: ``CARLA_ENV_LOG_LEVEL`` > *log_level* argument.
    """
    env_level = os.getenv("CARLA_ENV_LOG_LEVEL")
    if env_level:
        try:
            _pkg_logger.setLevel(_normalize_level(env_level))
            return
        except Exception:
            pass

    if log_level is not None:
        _pkg_logger.setLevel(_normalize_level(log_level))


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``openenv.carla_env``."""
    return logging.getLogger(f"{_LOGGER_BASE}.{name}")


# Apply env-var overrides at import time.
configure_logging(None)
