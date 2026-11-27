# SPDX-License-Identifier: BSD-3-Clause

"""openenvd: the control plane that runs each environment as kernel-isolated zones.

openenvd is PID 1 of an environment unit. It stays in the unit's parent
namespaces and creates three zones of containers in child namespaces: the agent
zone, hidden services and privileged observers. Every call out of the agent
zone crosses a relay in the control plane, and graders start only after the
agent zone is empty.
"""

from .contract import (
    ContainerSpec,
    load_manifest,
    Manifest,
    ManifestError,
    parse_manifest,
    Phase,
    validate_manifest,
    ZoneKind,
    ZoneSpec,
)
from .phases import Actor, PhaseError, PhaseMachine, Transition

__all__ = [
    "Actor",
    "ContainerSpec",
    "Manifest",
    "ManifestError",
    "Phase",
    "PhaseError",
    "PhaseMachine",
    "Transition",
    "ZoneKind",
    "ZoneSpec",
    "load_manifest",
    "parse_manifest",
    "validate_manifest",
]
