import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from llm_capacity_env.server.engine import (
    eligible_starts,
    EPISODE_DAYS,
    HISTORY_DAYS,
    load_trace,
    modeled_cost,
    MODELS,
    Replay,
    validate_reservations,
)


def plan(value=0):
    return {m: [value] * 24 for m in MODELS}


def test_trace_integrity_and_license():
    data = Path(__file__).resolve().parents[2] / "envs/llm_capacity_env/server/data"
    provenance = json.loads((data / "PROVENANCE.json").read_text())
    assert (
        hashlib.sha256((data / "hourly_demand.csv").read_bytes()).hexdigest()
        == provenance["sha256"]
    )
    assert (
        "Attribution 4.0 International" in (data / "LICENSE-BurstGPT.txt").read_text()
    )
    assert len(load_trace()) == 231


def test_chronology_and_split_disjointness():
    trace = load_trace()
    target_sets = {}
    for split in ("train", "test"):
        starts = eligible_starts(trace, split)
        assert starts
        target_sets[split] = {d for s in starts for d in range(s, s + EPISODE_DAYS)}
        for s in starts:
            assert set(range(s - HISTORY_DAYS, s + EPISODE_DAYS)) <= trace.keys()
    assert target_sets["train"].isdisjoint(target_sets["test"])


def test_known_cost_and_perfect_information_lower_bound():
    demand = plan(1000)
    assert modeled_cost(plan(0), demand) == pytest.approx(3.792)
    assert modeled_cost(plan(1000), demand) == pytest.approx(0.948)
    assert modeled_cost(plan(2000), demand) == pytest.approx(1.896)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"ChatGPT": [0] * 24},
        {**plan(), "extra": [0] * 24},
        {**plan(), "GPT-4": [0] * 23},
        plan(-1),
        plan(1000001),
        plan(float("nan")),
        plan(float("inf")),
        plan(True),
        plan("1"),
    ],
)
def test_invalid_actions_do_not_advance_or_change_cost(bad):
    env = Replay()
    before = copy.deepcopy(env.reset(7))
    with pytest.raises(ValueError):
        env.reserve(bad)
    assert env.observe() == before


def test_determinism_terminal_and_non_mutating_observations():
    a, b = Replay(), Replay()
    assert a.reset(10) == b.reset(10)
    view = a.observe()
    view["history"][0]["tokens"]["GPT-4"][0] = -99
    assert a.observe() == b.observe()
    for _ in range(EPISODE_DAYS):
        x, y = a.reserve(plan(10000)), b.reserve(plan(10000))
        assert x == y
        assert -2 <= x["previous"]["reward"] <= 0
    assert a.done
    with pytest.raises(RuntimeError):
        a.reserve(plan())
    assert a.observe() == b.observe()


def test_current_and_future_demand_cannot_affect_predecision_observation():
    a, b = Replay(), Replay()
    assert a.reset(19) == b.reset(19)
    # Causal counterfactual: change all current/future truth but no revealed history.
    b.trace = copy.deepcopy(b.trace)
    for d in range(b.start, b.start + EPISODE_DAYS):
        b.trace[d] = plan(98765)
    assert a.observe() == b.observe()
    assert a.reserve(plan())["previous"] != b.reserve(plan())["previous"]


def test_all_legal_extremes_bound_reward_and_lower_bound():
    for demand in load_trace().values():
        lower = modeled_cost(demand, demand)
        for reservations in (plan(0), plan(1000000)):
            cost = modeled_cost(validate_reservations(reservations), demand)
            assert math.isfinite(cost) and lower <= cost <= 2000


def test_environment_instances_are_independent():
    a, b = Replay(), Replay()
    a.reset(0)
    before = b.reset(1)
    a.reserve(plan())
    assert b.observe() == before
