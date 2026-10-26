# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""τ²-bench environment server."""

import litellm
from loguru import logger

# τ²-bench logs its registry on import and every message of the simulation, and
# litellm prints a notice for every call to a model it has no price for.
logger.disable("tau2")
litellm.suppress_debug_info = True

from .tau2_environment import Tau2Environment

__all__ = ["Tau2Environment"]
