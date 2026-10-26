# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenApp Environment - Web application simulation environment for UI agents."""

from .models import OpenAppAction, OpenAppObservation

__all__ = ["OpenAppAction", "OpenAppObservation", "OpenAppEnv"]


def __getattr__(name: str):
    if name == "OpenAppEnv":
        from .client import OpenAppEnv

        return OpenAppEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
