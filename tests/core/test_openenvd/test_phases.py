# SPDX-License-Identifier: BSD-3-Clause

import pytest
from openenv.core.openenvd.contract import Phase
from openenv.core.openenvd.phases import Actor, PhaseError, PhaseMachine


def _to(machine, *steps):
    for phase, actor in steps:
        machine.transition(phase, actor)


def test_happy_path_and_history():
    seen = []
    m = PhaseMachine(on_transition=seen.append)
    _to(
        m,
        (Phase.PROVISIONING, Actor.ORCHESTRATOR),
        (Phase.READY, Actor.SYSTEM),
        (Phase.RUNNING, Actor.SYSTEM),
        (Phase.FROZEN, Actor.ORCHESTRATOR),
        (Phase.RUNNING, Actor.ORCHESTRATOR),
        (Phase.SEALED, Actor.ENVIRONMENT),
        (Phase.GRADING, Actor.SYSTEM),
        (Phase.CLOSED, Actor.SYSTEM),
    )
    assert [t.to_phase for t in seen][-1] is Phase.CLOSED
    assert len(m.history) == 8


@pytest.mark.parametrize(
    "setup, target, actor",
    [
        ([], Phase.PROVISIONING, Actor.ENVIRONMENT),
        ([(Phase.PROVISIONING, Actor.ORCHESTRATOR)], Phase.READY, Actor.ORCHESTRATOR),
        (
            [(Phase.PROVISIONING, Actor.ORCHESTRATOR), (Phase.READY, Actor.SYSTEM)],
            Phase.FROZEN,
            Actor.ORCHESTRATOR,
        ),
        (
            [
                (Phase.PROVISIONING, Actor.ORCHESTRATOR),
                (Phase.READY, Actor.SYSTEM),
                (Phase.RUNNING, Actor.SYSTEM),
            ],
            Phase.FROZEN,
            Actor.ENVIRONMENT,
        ),
        (
            [
                (Phase.PROVISIONING, Actor.ORCHESTRATOR),
                (Phase.READY, Actor.SYSTEM),
                (Phase.RUNNING, Actor.SYSTEM),
                (Phase.SEALED, Actor.ENVIRONMENT),
            ],
            Phase.GRADING,
            Actor.ORCHESTRATOR,
        ),
    ],
)
def test_forbidden_transitions(setup, target, actor):
    m = PhaseMachine()
    _to(m, *setup)
    with pytest.raises(PhaseError):
        m.transition(target, actor)


def test_only_the_system_fails_a_unit_and_only_the_orchestrator_resets():
    m = PhaseMachine()
    _to(m, (Phase.PROVISIONING, Actor.ORCHESTRATOR))
    with pytest.raises(PhaseError):
        m.transition(Phase.FAILED, Actor.ORCHESTRATOR)
    m.transition(Phase.FAILED, Actor.SYSTEM)
    m.transition(Phase.PROVISIONING, Actor.ORCHESTRATOR)
    assert m.phase is Phase.PROVISIONING
