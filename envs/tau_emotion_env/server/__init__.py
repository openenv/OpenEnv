# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""τ-emotion-bench environment server."""

import litellm
from loguru import logger

# τ-emotion-bench logs its data directory on import and every message of the
# simulation, and litellm prints a notice for every call to a model it has no
# price for.
logger.disable("tau_emotion")
litellm.suppress_debug_info = True

from .tau_emotion_environment import TauEmotionEnvironment

__all__ = ["TauEmotionEnvironment"]
