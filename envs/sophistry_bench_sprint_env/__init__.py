# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sophistry-Bench Sprint Environment (OpenEnv port).

Single-step advocacy environment: reset() issues a QuALITY reading-comprehension
advocacy task, step(AdvocacyAction(text=...)) scores the argument and returns the
reward plus all eight sprint reward components in observation.metadata.
"""

from .client import SophistryBenchSprintEnv
from .models import AdvocacyAction, AdvocacyObservation

__all__ = ["SophistryBenchSprintEnv", "AdvocacyAction", "AdvocacyObservation"]
