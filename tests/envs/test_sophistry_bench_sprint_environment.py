# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the sophistry-bench sprint OpenEnv environment."""

import asyncio
import inspect
import os
import sys

import pytest

# Add the project root to the path for envs imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# The env imports the published ``sophistry-bench-sprint`` package (canonical
# scoring source). It is not part of the repo's base test deps, so skip the whole
# module when it (or its ``verifiers`` dependency) is unavailable.
pytest.importorskip("sophistry_bench_sprint")

from envs.sophistry_bench_sprint_env.client import SophistryBenchSprintEnv
from envs.sophistry_bench_sprint_env.models import AdvocacyAction, AdvocacyObservation
from envs.sophistry_bench_sprint_env.server.sophistry_bench_sprint_environment import (
    SophistryBenchSprintEnvironment,
)
from openenv.core.env_server.serialization import serialize_observation
from sophistry_bench_sprint import load_environment

# All eight scored components (full vector used for the weighted reward).
_ALL_KEYS = {
    "aggregate_reward",
    "correctness_reward",
    "n_claims",
    "n_citations",
    "alternation_canary",
    "starts_with_canary",
    "length_band_canary",
    "template_echo_canary",
}
# What is surfaced on the wire by default: everything except the hidden
# ground-truth ``correctness_reward`` (opt in via expose_correctness=True).
_WIRE_KEYS = _ALL_KEYS - {"correctness_reward"}


def _env():
    # Small dataset keeps the test fast; reads the bundled QuALITY split.
    return SophistryBenchSprintEnvironment(n_items=2, passage_chars=500, seed=0)


def test_advocacy_action_carries_text():
    a = AdvocacyAction(text="<claim>x</claim>")
    assert a.text == "<claim>x</claim>"


def test_advocacy_observation_defaults():
    o = AdvocacyObservation(prompt="P", answer_to_defend="A", item_id="id1")
    assert o.prompt == "P"
    assert o.answer_to_defend == "A"
    assert o.item_id == "id1"
    assert o.reward is None  # inherits base Observation default (no action scored)
    assert o.done is False
    assert o.metadata == {}


def test_client_parses_step_result():
    # Exercise the pure parsing hooks without a live server.
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    payload = client._step_payload(AdvocacyAction(text="<claim>x</claim>"))
    assert payload["text"] == "<claim>x</claim>"

    raw = {
        "observation": {
            "prompt": "",
            "answer_to_defend": "",
            "item_id": "",
            "reward": 0.5,
            "done": True,
            "metadata": {"aggregate_reward": 0.5},
        },
        "reward": 0.5,
        "done": True,
        "info": {},
    }
    result = client._parse_result(raw)
    assert isinstance(result.observation, AdvocacyObservation)
    assert result.observation.metadata["aggregate_reward"] == 0.5
    assert result.reward == 0.5
    assert result.done is True


def test_parse_result_rejects_malformed_payload():
    # A missing/null "observation" is a protocol error: fail loudly, don't build
    # a silently-empty observation.
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    with pytest.raises(ValueError):
        client._parse_result({"reward": 0.0, "done": True})
    with pytest.raises(ValueError):
        client._parse_result({"observation": None, "reward": 0.0, "done": True})


def test_step_text_is_async():
    # Must be a coroutine function so `await env.step_text(...)` yields a
    # StepResult (not a coroutine) and `.sync()` auto-wraps it like base step().
    assert inspect.iscoroutinefunction(SophistryBenchSprintEnv.step_text)


def test_reset_returns_task_observation():
    env = _env()
    obs = env.reset(seed=0)
    assert obs.done is False
    assert obs.reward is None  # no action scored on reset (framework convention)
    assert obs.prompt  # non-empty system prompt
    assert "DEFEND THIS ANSWER" in obs.prompt
    assert obs.answer_to_defend in obs.prompt
    assert obs.item_id  # article id present


def test_reset_is_deterministic_for_fixed_seed():
    a = _env().reset(seed=3)
    b = _env().reset(seed=3)
    assert (a.item_id, a.answer_to_defend, a.prompt) == (
        b.item_id,
        b.answer_to_defend,
        b.prompt,
    )


def test_step_returns_all_components_and_is_done():
    env = _env()
    env.reset(seed=0)
    text = "".join(f"<claim>point {i}</claim>" for i in range(8))  # 8 claims, 0 cites
    obs = env.step(AdvocacyAction(text=text))
    assert obs.done is True
    # correctness_reward (hidden ground truth) is withheld by default.
    assert set(obs.metadata.keys()) == _WIRE_KEYS
    assert "correctness_reward" not in obs.metadata
    assert obs.metadata["n_claims"] == 8.0
    assert obs.metadata["n_citations"] == 0.0
    # Values come from the upstream sophistry-bench-sprint spec: cliff(8 claims)=1.0
    # and citation_grounding(claims, [], passage)=0.0 -> aggregate mean 0.5.
    assert obs.metadata["aggregate_reward"] == 0.5
    # default weights -> reward == aggregate
    assert obs.reward == 0.5


def test_correctness_reward_exposed_only_when_opted_in():
    # Default: hidden ground truth absent from the wire observation.
    default_env = _env()
    default_env.reset(seed=0)
    obs = default_env.step(AdvocacyAction(text="<claim>x</claim>"))
    assert "correctness_reward" not in obs.metadata
    assert "correctness_reward" not in obs.components

    # Opt in (trusted measurement code): full eight-component vector surfaces.
    exposed_env = SophistryBenchSprintEnvironment(
        n_items=2, passage_chars=500, seed=0, expose_correctness=True
    )
    exposed_env.reset(seed=0)
    obs = exposed_env.step(AdvocacyAction(text="<claim>x</claim>"))
    assert set(obs.metadata.keys()) == _ALL_KEYS
    assert "correctness_reward" in obs.metadata


def test_malformed_env_vars_raise_with_context(monkeypatch):
    # Bad env-var values should raise a message naming the var, not a bare ValueError.
    monkeypatch.setenv("SPRINT_WEIGHTS", "1,two,3,4,5,6,7,8")
    with pytest.raises(ValueError, match="SPRINT_WEIGHTS"):
        SophistryBenchSprintEnvironment(n_items=2, passage_chars=500, seed=0)
    monkeypatch.delenv("SPRINT_WEIGHTS", raising=False)

    monkeypatch.setenv("SPRINT_N_ITEMS", "lots")
    with pytest.raises(ValueError, match="SPRINT_N_ITEMS"):
        SophistryBenchSprintEnvironment(passage_chars=500, seed=0)


def test_wrong_length_weights_rejected():
    # A mis-sized weight vector must raise, not silently truncate via zip.
    with pytest.raises(ValueError):
        SophistryBenchSprintEnvironment(
            n_items=2, passage_chars=500, seed=0, weights=[1.0, 0.0]
        )


def test_step_before_reset_errors_gracefully():
    env = _env()
    obs = env.step(AdvocacyAction(text="<claim>x</claim>"))
    assert obs.done is True
    assert obs.reward == 0.0
    assert "error" in obs.metadata


def test_aggregate_matches_canonical_verifiers_reward():
    """Anti-drift: the OpenEnv aggregate must equal the PI Hub aggregate_reward
    for the passage selected at the same seed (dataset + formula parity)."""
    env = _env()
    env.reset(seed=0)
    # Capture the episode passage now: step() flips _has_task and ends the episode.
    passage = env.current_passage
    text = "<claim>alpha</claim><cite>beta gamma delta epsilon zeta</cite>"
    obs = env.step(AdvocacyAction(text=text))

    vf_env = load_environment(n_items=2, passage_chars=500, seed=0)
    # Dataset parity: both sides build from the same quality_to_advocacy_dataset
    # builder, so reset(seed=0) (idx 0) must select the same passage as row 0.
    canonical_passage = vf_env.dataset[0]["info"]["passage"]
    assert canonical_passage == passage

    # Newer verifiers wrap the reward Rubric in a RubricGroup, so funcs live on
    # the inner rubric; older versions expose them directly. aggregate_reward is index 0.
    rubric = vf_env.rubric
    if not getattr(rubric, "funcs", None) and getattr(rubric, "rubrics", None):
        rubric = rubric.rubrics[0]
    aggregate_fn = rubric.funcs[0]  # aggregate_reward is index 0
    # Fail loudly (not with IndexError/AttributeError) if upstream reorders funcs
    # within the pinned <0.2.0 range.
    assert aggregate_fn.__name__ == "aggregate_reward", (
        f"expected rubric.funcs[0] to be aggregate_reward, got {aggregate_fn.__name__}"
    )
    completion = [{"role": "assistant", "content": text}]
    # Formula parity: feed the canonical fn the canonical side's own passage.
    state = {"info": {"passage": canonical_passage}}
    canonical = asyncio.run(
        aggregate_fn(prompt=[], completion=completion, answer="", state=state)
    )
    assert abs(obs.metadata["aggregate_reward"] - canonical) < 1e-9


def test_metadata_survives_wire_serialization_round_trip():
    """Lock in the wire contract: the framework strips base ``metadata`` from the
    serialized observation, but the declared ``components`` field survives and the
    typed client re-populates ``metadata`` from it on the way back."""
    env = _env()
    env.reset(seed=0)
    obs = env.step(
        AdvocacyAction(text="".join(f"<claim>c{i}</claim>" for i in range(8)))
    )

    # Real server-side serialization. Returns
    # {"observation": {...}, "reward": float, "done": bool}; the obs dict
    # excludes reward/done/metadata but keeps declared subclass fields.
    payload = serialize_observation(obs)
    obs_dict = payload["observation"]
    assert "metadata" not in obs_dict  # framework strips base metadata
    assert set(obs_dict["components"].keys()) == _WIRE_KEYS

    # Reconstruct the wire payload in the shape ``_parse_result`` reads.
    wire = {
        "observation": obs_dict,
        "reward": payload["reward"],
        "done": payload["done"],
    }
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    result = client._parse_result(wire)
    assert set(result.observation.metadata.keys()) == _WIRE_KEYS
    assert result.reward == obs.reward


def test_error_survives_wire_serialization_round_trip():
    """The error path declares an ``error`` field so the step-before-reset
    message survives the framework's metadata-stripping serialization and is
    restored into ``metadata`` by the typed client on the way back."""
    env = _env()
    obs = env.step(AdvocacyAction(text="<claim>x</claim>"))  # step before reset

    payload = serialize_observation(obs)
    obs_dict = payload["observation"]
    assert "metadata" not in obs_dict  # framework strips base metadata
    assert obs_dict["error"] == "call reset() before step()"

    wire = {
        "observation": obs_dict,
        "reward": payload["reward"],
        "done": payload["done"],
    }
    client = SophistryBenchSprintEnv.__new__(SophistryBenchSprintEnv)
    result = client._parse_result(wire)
    assert result.observation.metadata["error"] == "call reset() before step()"
    assert result.reward == 0.0
