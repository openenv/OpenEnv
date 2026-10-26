# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QED Math Environment."""

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import QEDMathEnv
    from .models import (
        GetGradingGuidelines,
        GetProblem,
        ProblemObservation,
        ProofSubmissionObservation,
        QEDMathAction,
        QEDMathObservation,
        SubmitProof,
    )

__all__ = [
    "QEDMathAction",
    "QEDMathObservation",
    "QEDMathEnv",
    "SubmitProof",
    "GetProblem",
    "GetGradingGuidelines",
    "ProblemObservation",
    "ProofSubmissionObservation",
]


def __getattr__(name: str) -> Any:
    """Keep client and wire-model imports out of spawned math workers."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".client" if name == "QEDMathEnv" else ".models", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
