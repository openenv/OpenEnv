# SPDX-License-Identifier: BSD-3-Clause

"""The unit's phase machine.

A unit is always in exactly one [`Phase`]. Only some actors may cause each
transition: the orchestrator drives the episode, the environment can only say
it's done, and openenvd itself moves the unit when the kernel reports progress
(readiness, `populated 0`, a verdict) or a fault. The agent is not an actor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .contract import Phase


class Actor(str, Enum):
    ORCHESTRATOR = "orchestrator"
    ENVIRONMENT = "environment"
    SYSTEM = "system"


class PhaseError(RuntimeError):
    """A transition that the phase machine does not allow."""


_O, _E, _S = Actor.ORCHESTRATOR, Actor.ENVIRONMENT, Actor.SYSTEM

# (from, to) -> actors allowed to cause it. `reset` and `failed` are handled below.
_EDGES: dict[tuple[Phase, Phase], frozenset[Actor]] = {
    (Phase.PROVISIONING, Phase.READY): frozenset({_S}),
    (Phase.READY, Phase.RUNNING): frozenset({_O, _S}),
    (Phase.RUNNING, Phase.FROZEN): frozenset({_O}),
    (Phase.FROZEN, Phase.RUNNING): frozenset({_O}),
    (Phase.READY, Phase.SEALED): frozenset({_O, _E, _S}),
    (Phase.RUNNING, Phase.SEALED): frozenset({_O, _E, _S}),
    (Phase.FROZEN, Phase.SEALED): frozenset({_O, _S}),
    (Phase.SEALED, Phase.GRADING): frozenset({_S}),
    (Phase.GRADING, Phase.CLOSED): frozenset({_O, _S}),
    (Phase.SEALED, Phase.CLOSED): frozenset({_O, _S}),
}


@dataclass(frozen=True)
class Transition:
    """One recorded phase change."""

    from_phase: Phase
    to_phase: Phase
    actor: Actor
    reason: str
    at: float


@dataclass
class PhaseMachine:
    """Tracks the unit's phase and refuses transitions the contract forbids.

    Args:
        on_transition (`Callable[[Transition], None]`, *optional*):
            Called after every transition, for example to log it.
    """

    on_transition: Callable[[Transition], None] | None = None
    phase: Phase = Phase.CLOSED
    history: list[Transition] = field(default_factory=list)

    def allowed(self, to: Phase, actor: Actor) -> bool:
        if to is Phase.PROVISIONING:
            return actor is Actor.ORCHESTRATOR
        if to is Phase.FAILED:
            return actor is Actor.SYSTEM and self.phase not in (
                Phase.FAILED,
                Phase.CLOSED,
            )
        if to is Phase.CLOSED and self.phase is Phase.FAILED:
            return actor in (Actor.ORCHESTRATOR, Actor.SYSTEM)
        return actor in _EDGES.get((self.phase, to), frozenset())

    def transition(self, to: Phase, actor: Actor, reason: str = "") -> Transition:
        """Move to `to`, or raise [`PhaseError`].

        Args:
            to (`Phase`):
                The target phase.
            actor (`Actor`):
                Who is causing the transition.
            reason (`str`, *optional*):
                Recorded with the transition.

        Returns:
            [`Transition`]: The recorded transition.
        """
        if not self.allowed(to, actor):
            raise PhaseError(
                f"{actor.value} cannot move the unit from {self.phase.value} to {to.value}"
            )
        record = Transition(self.phase, to, actor, reason, time.time())
        self.phase = to
        self.history.append(record)
        if self.on_transition is not None:
            self.on_transition(record)
        return record
