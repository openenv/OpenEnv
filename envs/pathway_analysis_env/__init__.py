# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pathway Analysis Environment for OpenEnv.

A toy computational-biology environment where an agent identifies the
activated signaling pathway from synthetic omics data.

Example:
    >>> from pathway_analysis_env import PathwayEnv, PathwayAction
    >>>
    >>> with PathwayEnv(base_url="http://localhost:8000") as client:
    ...     result = client.reset()
    ...     result = client.step(PathwayAction(action_type="inspect_dataset"))
    ...     result = client.step(PathwayAction(action_type="run_differential_expression"))
    ...     result = client.step(PathwayAction(action_type="run_pathway_enrichment"))
    ...     result = client.step(PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling"))
"""

from .client import PathwayEnv
from .models import PathwayAction, PathwayObservation, PathwayState

__all__ = ["PathwayEnv", "PathwayAction", "PathwayObservation", "PathwayState"]
