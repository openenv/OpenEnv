# SPDX-License-Identifier: BSD-3-Clause

"""The contract `examples/pelican_svg_grpo.py` relies on, checked without TRL.

The training script does exactly three things against the environment: pin the
task, submit a completion, read `result.reward`. If any of those drift, a GRPO
run either crashes hours in or, worse, trains happily against a reward that is
zero everywhere and learns nothing.

This runs on CPU with no network and no TRL installed, so it can guard the
training path in CI. It deliberately does not import trl: that would make the
whole module skip on a machine without it, which is most CI machines, and the
contract being guarded is the environment's, not the trainer's.
"""

from __future__ import annotations

import asyncio
import math
import pathlib
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("resvg_py", reason="resvg-py is not installed")

from envs.pelican_svg_env.models import PelicanSvgAction
from envs.pelican_svg_env.server.pelican_svg_environment import PelicanSvgEnvironment
from envs.pelican_svg_env.server.scoring import component_weights

# A minimal drawing that clears the gate: two level wheels, a frame between
# them, and a body above. Enough to exercise the reward the trainer sees.
DRAWABLE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">'
    '<circle cx="110" cy="230" r="52" fill="none" stroke="#222" stroke-width="6"/>'
    '<circle cx="290" cy="230" r="52" fill="none" stroke="#222" stroke-width="6"/>'
    '<line x1="110" y1="230" x2="290" y2="230" stroke="#c00" stroke-width="6"/>'
    '<line x1="110" y1="230" x2="200" y2="160" stroke="#c00" stroke-width="6"/>'
    '<line x1="290" y1="230" x2="200" y2="160" stroke="#c00" stroke-width="6"/>'
    '<ellipse cx="200" cy="120" rx="48" ry="34" fill="#fff" stroke="#333" stroke-width="3"/>'
    "</svg>"
)


@pytest.fixture
def env():
    """The environment as the training script configures it: no judge."""
    return PelicanSvgEnvironment(enable_judge=False)


def roll(env, completion: str) -> float:
    """One rollout the way the reward function does it."""
    env.reset(task_id="pelican_bicycle")
    return env.step(PelicanSvgAction(response=completion)).reward


class TestRewardContract:
    def test_reward_is_a_finite_float_in_range(self, env):
        reward = roll(env, DRAWABLE)
        assert isinstance(reward, float)
        assert math.isfinite(reward)
        assert 0.0 <= reward <= 1.0

    def test_a_drawing_earns_more_than_garbage(self, env):
        """Without this ordering GRPO has no signal to follow."""
        assert roll(env, DRAWABLE) > roll(env, "I cannot draw that.")

    def test_garbage_scores_exactly_zero(self, env):
        assert roll(env, "no svg here") == 0.0

    def test_reward_is_deterministic_without_a_judge(self, env):
        """A reward that wobbles run to run makes the curve unreadable."""
        assert len({roll(env, DRAWABLE) for _ in range(5)}) == 1

    def test_offline_reward_is_the_structural_score(self, env):
        """With no judge, structure carries the whole weight."""
        env.reset(task_id="pelican_bicycle")
        observation = env.step(PelicanSvgAction(response=DRAWABLE))
        assert observation.reward == pytest.approx(observation.structure_score)
        assert component_weights(judge_enabled=False) == (1.0, 0.0)

    def test_judge_never_runs_when_disabled(self, env):
        """Training must not make a paid API call per rollout."""
        env.reset(task_id="pelican_bicycle")
        observation = env.step(PelicanSvgAction(response=DRAWABLE))
        assert observation.judged is False
        assert observation.semantic_score == 0.0


class TestTruncationTrap:
    """The failure mode that quietly wastes a GPU run.

    `max_completion_length` too small cuts every SVG off mid-document. Every
    rollout then scores zero, GRPO sees no variance, and the run burns hours
    learning nothing. The environment has to name that case so it is visible in
    the logs rather than looking like a model that cannot draw.
    """

    def test_truncated_completion_is_named(self, env):
        env.reset(task_id="pelican_bicycle")
        observation = env.step(
            PelicanSvgAction(response=DRAWABLE[: len(DRAWABLE) // 2])
        )
        assert observation.reward == 0.0
        assert "truncated_svg" in observation.violations

    def test_absent_svg_is_a_different_code(self, env):
        env.reset(task_id="pelican_bicycle")
        observation = env.step(PelicanSvgAction(response="Sorry, I will not."))
        assert "no_svg_in_response" in observation.violations


class TestTrainingTaskIsPinned:
    def test_every_reset_serves_the_same_prompt(self, env):
        """The trainer builds one dataset row per episode from one prompt."""
        prompts = {env.reset(task_id="pelican_bicycle").prompt for _ in range(5)}
        assert len(prompts) == 1

    def test_the_pinned_task_is_the_original_prompt(self, env):
        observation = env.reset(task_id="pelican_bicycle")
        assert observation.subject == "pelican"
        assert observation.vehicle == "bicycle"
        assert observation.held_out is False


class TestStructuralProxyIsGameable:
    """Documents, in a test, the thing the training run is looking for.

    The structural layer is satisfiable without drawing an animal, which is why
    the training script probes with the judge before and after. Pinning it here
    means nobody later mistakes the proxy for the task.
    """

    def test_a_bare_vehicle_already_scores_most_of_the_structural_reward(self, env):
        bare = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">'
            '<circle cx="110" cy="230" r="52" fill="none" stroke="#222" stroke-width="6"/>'
            '<circle cx="290" cy="230" r="52" fill="none" stroke="#222" stroke-width="6"/>'
            '<line x1="110" y1="230" x2="290" y2="230" stroke="#222" stroke-width="6"/>'
            '<line x1="110" y1="230" x2="200" y2="165" stroke="#222" stroke-width="6"/>'
            '<line x1="290" y1="230" x2="200" y2="165" stroke="#222" stroke-width="6"/>'
            "</svg>"
        )
        reward = roll(env, bare)
        assert reward > 0.5, "a vehicle with no rider should still score well"
        assert reward < 1.0, "but it must not score full marks"


_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))


class _FakeStream:
    """An async iterable of chat-completion chunks."""

    def __init__(self, deltas):
        self._deltas = deltas

    def __aiter__(self):
        async def gen():
            for delta in self._deltas:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        return gen()


class _FakeClient:
    def __init__(self, deltas):
        self._deltas = deltas

    async def chat_completion(self, **kwargs):
        assert kwargs.get("stream") is True
        return _FakeStream(self._deltas)


def _delta(content=None, reasoning_content=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning_content)


def _args():
    return SimpleNamespace(no_stream=False, max_tokens=100, temperature=0.7)


class TestReasoningModelStreams:
    """A reasoning model streams its thinking apart from its answer.

    `delta.reasoning_content` carries the chain of thought while `delta.content`
    stays empty. Reading only `content` makes a model that is still thinking look
    like a model that refused, and a model that spent its whole budget reasoning
    look like a model that cannot draw.
    """

    def run(self, deltas):
        from pelican_svg_eval import generate

        return asyncio.run(
            generate(
                _FakeClient(deltas),
                asyncio.Semaphore(1),
                "fake/model",
                "draw",
                1,
                _args(),
            )
        )

    def test_answer_and_thinking_are_kept_apart(self):
        out = self.run(
            [
                _delta(reasoning_content="let me think"),
                _delta(content="<svg/>"),
            ]
        )
        assert out["reply"] == "<svg/>"
        assert out["reasoning_chars"] == len("let me think")
        assert out["error"] is None

    def test_thinking_only_is_reported_as_an_exhausted_budget(self):
        """Not as a refusal, which is what reading `content` alone suggests."""
        out = self.run([_delta(reasoning_content="thinking " * 20)])
        assert out["reply"] == ""
        assert out["error"] is not None
        assert "reasoning" in out["error"] and "max-tokens" in out["error"]

    def test_a_plain_model_with_no_thinking_still_works(self):
        out = self.run([_delta(content="<svg"), _delta(content="/>")])
        assert out["reply"] == "<svg/>"
        assert out["reasoning_chars"] == 0
        assert out["error"] is None


class TestEvalHoldsNoIdleConnection:
    """The environment connection must not span the generation phase.

    Holding an idle WebSocket across a long generation gets it closed by a proxy
    ("ConnectionClosedOK") and every generated sample is lost with it. Cheap to
    reintroduce by tidying generation back inside the `with` block, and expensive
    when it happens, so it is pinned here.
    """

    def test_scoring_opens_the_connection_after_generation(self):
        source = (_EXAMPLES / "pelican_svg_eval.py").read_text()
        assert source.index("outputs = await asyncio.gather") < source.index(
            "with PelicanSvgEnv(base_url=args.env_url) as env:"
        )


def _proxy_gap():
    """Load the detector without importing trl, which CI does not have."""
    source = (_EXAMPLES / "pelican_svg_grpo.py").read_text()
    fn = source[
        source.index("def report_proxy_gap") : source.index("def main() -> None:")
    ]
    namespace: dict = {}
    exec(fn, namespace)
    return namespace["report_proxy_gap"]


class TestProxyGapDetector:
    """Did the policy learn the task, or just the measurable part of it?

    Pinned with the real numbers from a Qwen3-1.7B run: 80 GRPO steps lifted the
    held-out structural score from 0.381 to 0.643 while the judged semantic score
    stayed at 0.014, meaning the judge still recognised nothing. An earlier
    version of the check used absolute thresholds and missed that case by under
    0.002, so it now compares the size of the two gains.
    """

    def test_fires_on_the_run_that_slipped_past_the_old_thresholds(self, capsys):
        _proxy_gap()(
            {"structure": 0.381, "semantic": 0.0023, "gate_failures": 6},
            {"structure": 0.6429, "semantic": 0.0139, "gate_failures": 2},
        )
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "reward hacking" in out

    def test_stays_quiet_when_both_layers_rise_together(self, capsys):
        _proxy_gap()(
            {"structure": 0.38, "semantic": 0.10, "gate_failures": 6},
            {"structure": 0.64, "semantic": 0.45, "gate_failures": 2},
        )
        out = capsys.readouterr().out
        assert "WARNING" not in out
        assert "moved together" in out

    def test_says_so_when_nothing_moved(self, capsys):
        _proxy_gap()(
            {"structure": 0.38, "semantic": 0.01, "gate_failures": 6},
            {"structure": 0.39, "semantic": 0.01, "gate_failures": 6},
        )
        out = capsys.readouterr().out
        assert "did not move" in out
        assert "WARNING" not in out

    def test_a_high_judged_score_is_not_flagged_even_on_a_big_structural_gain(
        self, capsys
    ):
        """Structure can legitimately outpace a judge already scoring well."""
        _proxy_gap()(
            {"structure": 0.30, "semantic": 0.60, "gate_failures": 4},
            {"structure": 0.90, "semantic": 0.62, "gate_failures": 1},
        )
        assert "WARNING" not in capsys.readouterr().out
