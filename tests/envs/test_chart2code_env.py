# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the chart-to-code environment."""

import asyncio

import pytest

pytest.importorskip("matplotlib", reason="matplotlib is not installed")

from chart2code_env.models import Chart2CodeAction  # noqa: E402
from chart2code_env.server.chart2code_environment import (  # noqa: E402
    Chart2CodeEnvironment,
)
from chart2code_env.server.specs import (  # noqa: E402
    reference_code,
    sample_spec,
    task_seed,
    TEST_SEED_OFFSET,
)

BAR_CODE = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["a", "b", "c"], [3, 5, 2], color="#1f77b4")
ax.set_title("t")
"""


def submit(env, code):
    return env.step(Chart2CodeAction(code=code, mode="submit"))


def reference(env):
    spec, _, _ = env._task
    return reference_code(spec)


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_code_scores_one(level, seed):
    env = Chart2CodeEnvironment(level=level)
    env.reset(seed=seed)
    result = submit(env, reference(env))
    assert result.done and result.gate_passed
    assert result.reward == pytest.approx(1.0)
    assert set(result.scores) == {"structure", "data", "text", "color"}


@pytest.mark.parametrize(
    "code, reason",
    [
        ("import matplotlib.pyplot as plt\nplt.figure()\n", "no data drawn"),
        ("x = 1\n", "no figure"),
        ("def broken(:\n", "SyntaxError"),
        ("while True:\n    pass\n", "ran past"),
        ("raise SystemExit(0)\n", "SystemExit"),
    ],
)
def test_code_that_draws_nothing_scores_zero(code, reason):
    env = Chart2CodeEnvironment(timeout_s=5)
    env.reset(seed=0)
    result = submit(env, code)
    assert result.reward == 0.0 and not result.gate_passed
    assert reason in result.error


@pytest.mark.parametrize(
    "paste",
    [
        "ax.imshow(img)",
        "fig.figimage(img)",
        "from matplotlib.offsetbox import AnnotationBbox, OffsetImage\n"
        "ax.add_artist(AnnotationBbox(OffsetImage(img), (0.5, 0.5)))",
    ],
)
def test_pasting_the_target_image_scores_zero(paste):
    env = Chart2CodeEnvironment()
    target = env.reset(seed=0).target_png_base64
    code = (
        "import base64, io\nimport matplotlib.pyplot as plt\n"
        f"img = plt.imread(io.BytesIO(base64.b64decode({target!r})))\n"
        "fig, ax = plt.subplots()\n" + paste + "\n"
    )
    # Even next to the honest chart, a raster image zeroes the reward.
    result = submit(
        env, reference(env) + code.replace("fig, ax = plt.subplots()\n", "")
    )
    assert result.reward == 0.0
    assert result.error == "raster image"


def test_extra_bars_score_below_the_honest_answer():
    env = Chart2CodeEnvironment(level="easy")
    env.reset(task_index=0)
    spec, _, _ = env._task
    assert spec.panels[0].kind == "bar"
    honest = reference(env)
    sprayed = honest + "ax.bar(['x1', 'x2', 'x3'], [40, 50, 60])\n"
    env.reset(task_index=0)
    honest_reward = submit(env, honest).reward
    env.reset(task_index=0)
    assert submit(env, sprayed).reward < honest_reward


def test_equivalent_code_scores_the_same():
    env = Chart2CodeEnvironment()
    variants = [
        BAR_CODE,
        BAR_CODE.replace(
            "fig, ax = plt.subplots(figsize=(6, 4))\nax.bar",
            "plt.figure(figsize=(6, 4))\nplt.bar",
        ).replace("ax.set_title", "plt.title"),
        "import matplotlib.pyplot as plt\nimport pandas as pd\n"
        "ax = pd.Series([3, 5, 2], index=['a', 'b', 'c']).plot.bar(color='#1f77b4', rot=0)\n"
        "ax.set_title('t')\n",
    ]
    rewards = []
    for code in variants:
        env.reset(task_index=0)
        rewards.append(submit(env, code).reward)
    assert rewards[0] > 0
    assert rewards == pytest.approx([rewards[0]] * len(variants))


def test_scoring_is_repeatable():
    env = Chart2CodeEnvironment(level="medium")
    rewards = []
    for _ in range(2):
        env.reset(seed=5)
        rewards.append(submit(env, BAR_CODE).reward)
    assert rewards[0] == rewards[1]


def test_render_returns_a_drawing_but_no_score_and_is_capped():
    env = Chart2CodeEnvironment(max_renders=2)
    env.reset(seed=1)
    for left in (1, 0):
        draft = env.step(Chart2CodeAction(code=BAR_CODE, mode="render"))
        assert draft.rendered_png_base64 and draft.error == ""
        assert draft.reward is None and not draft.done and draft.scores == {}
        assert draft.renders_left == left
    refused = env.step(Chart2CodeAction(code=BAR_CODE, mode="render"))
    assert refused.rendered_png_base64 is None and "no renders left" in refused.error
    assert submit(env, reference(env)).reward == pytest.approx(1.0)


def test_render_reports_the_error():
    env = Chart2CodeEnvironment()
    env.reset(seed=1)
    draft = env.step(Chart2CodeAction(code="1/0", mode="render"))
    assert "ZeroDivisionError" in draft.error and draft.rendered_png_base64 is None


def test_steps_after_the_submission_return_the_final_observation():
    env = Chart2CodeEnvironment()
    env.reset(seed=2)
    final = submit(env, BAR_CODE)
    steps = env.state.step_count
    again = submit(env, reference(env))
    assert again == final and env.state.step_count == steps
    render = env.step(Chart2CodeAction(code=BAR_CODE, mode="render"))
    assert render == final
    env.reset(seed=2)
    assert submit(env, reference(env)).reward == pytest.approx(1.0)


def test_async_step_matches_step():
    env = Chart2CodeEnvironment()
    asyncio.run(env.reset_async(seed=4))
    result = asyncio.run(env.step_async(Chart2CodeAction(code=reference(env))))
    assert result.reward == pytest.approx(1.0)


def test_reset_seed_and_split_pick_the_task():
    env = Chart2CodeEnvironment()
    first = env.reset(seed=11, episode_id="ep-1")
    again = env.reset(seed=11)
    assert first.target_png_base64 == again.target_png_base64
    assert first.task_id == "easy-train-11" and env.state.task_id == "easy-train-11"
    assert env.reset(seed=12).target_png_base64 != first.target_png_base64
    test = env.reset(task_index=3, split="test", level="hard")
    assert test.task_id == "hard-test-3" and test.level == "hard"


def test_train_and_test_seeds_never_overlap():
    assert task_seed("test", 0) == TEST_SEED_OFFSET
    with pytest.raises(ValueError):
        task_seed("train", TEST_SEED_OFFSET)
    with pytest.raises(ValueError):
        task_seed("test", 500)
    assert sample_spec("easy", "train", 0) != sample_spec("easy", "test", 0)


def test_the_code_runs_without_the_servers_environment(monkeypatch):
    monkeypatch.setenv("CHART2CODE_TEST_SECRET", "hunter2")
    env = Chart2CodeEnvironment()
    env.reset(seed=0)
    code = "import os\nassert 'CHART2CODE_TEST_SECRET' not in os.environ\n" + reference(
        env
    )
    assert submit(env, code).reward == pytest.approx(1.0)
