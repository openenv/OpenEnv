# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TitanRL recipes and reward functions.

Unlike ``test_bridge.py`` these need a TitanRL install (``torchtitan/rl`` on
``PYTHONPATH`` plus its training stack), and skip without one. They build the
configs on CPU -- no GPU, model weights, or chess server -- and pin down the
settings whose absence fails *silently* or only deep into a GPU run:

* no trainer checkpointer means no initial HF load, so the run trains random
  weights with only a warning;
* no ``global_vocab_size`` at trainer TP > 1 raises on the first loss call,
  after model load, validation, and a full batch of rollouts;
* a stale import anywhere under ``config_registry`` is reported by
  ``ConfigManager`` as "Config function ... not found", not as the ImportError;
* fp32 Adam moments at 30B OOM the trainer right after the first mid-run
  checkpoint save (step 51 of the recipe).

Run from the torchtitan repo root::

    PYTHONPATH="$PWD:/path/to/OpenEnv/examples" \\
        pytest /path/to/OpenEnv/examples/titanrl_openenv/tests/test_recipes.py
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys

import pytest

pytest.importorskip("torchtitan.rl")

# Make the example importable as the ``titanrl_openenv`` package, the way
# ``--module titanrl_openenv`` resolves it.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TEST_DIR)
_EXAMPLES_DIR = os.path.dirname(_PKG_DIR)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from titanrl_openenv import config_registry  # noqa: E402
from titanrl_openenv.rubric import OpenEnvReward, OpenEnvShapingReward  # noqa: E402
from titanrl_openenv.tasks import CHESS_POSITION_REWARD  # noqa: E402
from torchtitan.config.manager import ConfigManager  # noqa: E402
from torchtitan.models.common.config_utils import decoder_vocab_size  # noqa: E402
from torchtitan.rl.controller import Controller  # noqa: E402
from torchtitan.rl.rollout import Rollout, RolloutStatus  # noqa: E402
from torchtitan.rl.rollout.types import RolloutTurn  # noqa: E402
from torchtitan.rl.types import RolloutTurnID  # noqa: E402

RECIPES = [
    "rl_grpo_muse_glimmer_30b_openenv_chess",
    "rl_grpo_muse_glimmer_30b_openenv_chess_smoke",
    "rl_grpo_qwen3_1_7b_openenv_chess",
]


def _load(name: str) -> Controller.Config:
    # The same path ``python -m torchtitan.rl.train --module ... --config ...``
    # takes, CLI parsing included.
    return ConfigManager().parse_args(["--module", "titanrl_openenv", "--config", name])


def _serve_chess_default_max_sessions() -> int:
    # Read from source rather than imported: ``serve_chess`` imports the chess
    # environment at module level, and the training venv needs no chess packages.
    path = os.path.join(_PKG_DIR, "serve_chess.py")
    with open(path) as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DEFAULT_MAX_SESSIONS"
        ):
            # int(os.environ.get("CHESS_MAX_SESSIONS", "<default>"))
            return int(node.value.args[0].args[1].value)
    raise AssertionError("DEFAULT_MAX_SESSIONS not found in serve_chess.py")


def test_registry_exposes_only_the_recipes():
    public = sorted(
        name
        for name in vars(config_registry)
        if name.startswith("rl_") and callable(getattr(config_registry, name))
    )
    assert public == sorted(RECIPES)


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_loads_through_config_manager(name):
    config = _load(name)
    assert isinstance(config, Controller.Config)


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_loads_pretrained_weights(name):
    # Without a checkpointer the trainer skips the HF load and starts from random
    # weights -- only a warning marks it.
    checkpointer = _load(name).trainer.checkpointer
    assert checkpointer is not None
    assert checkpointer.initial_load_in_hf


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_sets_vocab_size_for_vocab_parallel_loss(name):
    config = _load(name)
    loss_fn = config.trainer.loss.loss_fn
    assert loss_fn.global_vocab_size == decoder_vocab_size(config.model)


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_disables_trainer_cuda_graphs(name):
    # Every upstream TitanRL recipe does; varlen attention cannot be graphed
    # without fixed-shape document metadata.
    assert _load(name).trainer.training.disable_cuda_graphs


@pytest.mark.parametrize("name", RECIPES)
def test_rollout_budget_matches_trainer_context(name):
    config = _load(name)
    training = config.trainer.training
    token_env = config.rollouter.worker.token_env
    assert token_env.max_rollout_tokens == training.max_context_length
    assert (
        training.num_tokens_per_microbatch_per_dp_rank % training.max_context_length
        == 0
    )
    # A rollout reaches the trainer as one sample (last prompt + last completion),
    # and the batcher silently drops one longer than the context. So every turn's
    # completion has to fit with room left for the chess prompt and observations
    # (~730 tokens to open, then ~150 per move).
    completions = token_env.max_num_turns * config.generator.sampling.max_tokens
    assert completions <= training.max_context_length - 2048


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_targets_the_chess_server(name):
    worker = _load(name).rollouter.worker
    assert worker.message_env.task == "chess"
    assert worker.message_env.base_url == "http://127.0.0.1:8000"
    # Chess rollouts end ``truncated_max_turns``; a truncation reward would
    # overwrite every one of them with the same constant.
    assert worker.rubric.truncation_reward is None
    assert [type(fn).__qualname__ for fn in worker.rubric.reward_fns] == [
        "OpenEnvReward.Config",
        "OpenEnvShapingReward.Config",
    ]


@pytest.mark.parametrize("name", RECIPES)
def test_chess_server_default_capacity_covers_the_recipe(name):
    # The async loop keeps up to ``target_offpolicy_steps + 1`` batches of groups
    # in flight; validation runs only before and after training, so it counts
    # only if larger (TitanRL sizes vLLM's concurrency the same way). An
    # undersized server does not refuse cleanly: it accepts the WebSocket and
    # closes it, and the rollouts are scored as errors.
    async_loop = _load(name).async_loop
    demand = max(
        async_loop.max_active_rollout_groups * async_loop.num_samples_per_prompt,
        async_loop.validation.num_samples,
    )
    assert _serve_chess_default_max_sessions() >= demand


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_muse_glimmer_30b_openenv_chess",
        "rl_grpo_muse_glimmer_30b_openenv_chess_smoke",
    ],
)
def test_30b_recipe_keeps_adam_moments_in_bf16(name):
    # At 30B on 95 GiB cards, fp32 moments leave too little headroom for the
    # weight push's bf16 copy when it overlaps the next forward/backward (it
    # does right after a mid-run checkpoint save): the run OOMed at step 51.
    optimizers = _load(name).trainer.optimizer.optimizers
    assert [o.moment_dtype for o in optimizers] == ["bfloat16"]


def test_smoke_recipe_skips_the_mid_run_checkpoint():
    config = _load("rl_grpo_muse_glimmer_30b_openenv_chess_smoke")
    assert config.trainer.checkpointer.interval > config.async_loop.num_training_steps


def test_cli_override_reaches_the_openenv_env_config():
    # ``worker.message_env`` is typed as the base ``MessageEnv.Config``; the CLI
    # must still narrow to ``OpenEnvMessageEnv.Config`` and accept its fields.
    config = ConfigManager().parse_args(
        [
            "--module",
            "titanrl_openenv",
            "--config",
            "rl_grpo_muse_glimmer_30b_openenv_chess",
            "--rollouter.worker.message-env.base-url",
            "http://10.0.0.7:9000",
        ]
    )
    assert config.rollouter.worker.message_env.base_url == "http://10.0.0.7:9000"


# --------------------------------------------------------------------------- #
# Reward functions, scored through TitanRL's own Rubric.
# --------------------------------------------------------------------------- #


def _turn(turn_id: int, env_rewards: dict[str, float]) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=turn_id),
        prompt_token_ids=[1],
        completion_token_ids=[2],
        completion_logprobs=[-0.1],
        env_rewards=env_rewards,
    )


def _rollout(*env_rewards: dict[str, float]) -> Rollout:
    return Rollout(
        group_id=0,
        rollout_id=0,
        turns=[_turn(i, rewards) for i, rewards in enumerate(env_rewards)],
        status=RolloutStatus.TRUNCATED_MAX_TURNS,
    )


def test_recipe_rubric_scores_a_partial_game():
    rubric = config_registry._openenv_chess_rollouter_config().worker.rubric.build()
    rollout = _rollout(
        {"openenv": 0.0, CHESS_POSITION_REWARD: 0.10},
        {"openenv": 0.0, CHESS_POSITION_REWARD: 0.30},
        # A turn cut off before its tool call carries no env rewards and must
        # not reset the position score.
        {},
    )
    (output,) = asyncio.run(rubric.score_group([rollout], env_input=None))
    assert output.reward_breakdown == {
        "OpenEnvReward": 0.0,
        "OpenEnvShapingReward": pytest.approx(0.30),
    }
    # Weights 1.0 and 0.5, normalized to sum to 1.
    assert output.reward == pytest.approx(0.30 / 3)


def test_shaping_reward_builds_its_own_class():
    # Redeclaring ``Config`` is what makes ``build()`` return the subclass; the
    # rubric rejects two reward fns with the same class name.
    shaping = OpenEnvShapingReward.Config(reward_name=CHESS_POSITION_REWARD).build()
    assert type(shaping) is OpenEnvShapingReward
    assert type(OpenEnvReward.Config().build()) is OpenEnvReward


def test_env_passes_the_game_outcome_to_shaping():
    # The wire format strips reward/done from the observation; the env must put
    # them back before shaping, or a finished game is scored by a static
    # evaluation taken from the wrong side of the board.
    from titanrl_openenv.data import OpenEnvSample
    from titanrl_openenv.openenv_bridge import BridgeTurn
    from titanrl_openenv.titanrl_env import OpenEnvMessageEnv

    worker = config_registry._openenv_chess_rollouter_config().worker
    env = worker.message_env.build(env_input=OpenEnvSample())
    won = BridgeTurn(
        text="Game over — result 1-0.",
        done=True,
        reward=1.0,
        raw_observation={
            "fen": "k7/8/8/8/8/8/8/K7 b - - 0 1",
            "metadata": {"evaluation": -900.0},
        },
    )

    async def act(_tool_calls):
        return won

    env._bridge.act_from_tool_calls = act
    call = {"name": "chess_move", "arguments": {"move": "a1a2"}}
    out = asyncio.run(
        env.step({"role": "assistant", "content": "", "tool_calls": [call]})
    )
    assert isinstance(env, OpenEnvMessageEnv)
    assert out.done
    assert out.env_rewards == {"openenv": 1.0, CHESS_POSITION_REWARD: 1.0}


@pytest.mark.parametrize(
    ("aggregate", "expected"), [("sum", -0.2), ("last", 0.0), ("mean", -0.2 / 3)]
)
def test_openenv_reward_aggregates(aggregate, expected):
    # Two illegal moves (-0.1 each), then a legal one.
    reward = OpenEnvReward.Config(aggregate=aggregate).build()
    rollout = _rollout({"openenv": -0.1}, {"openenv": -0.1}, {"openenv": 0.0})
    assert asyncio.run(reward(rollout, None)) == pytest.approx(expected)
