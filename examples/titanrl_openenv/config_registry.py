# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the TitanRL <-> OpenEnv example.

These mirror ``search_r1``'s config registry (``torchtitan/rl/examples``): each
sets the full recipe from the example module, leaving the TorchTitan core
defaults untouched. ``ConfigManager`` discovers them directly from the example
module. From the torchtitan repo root, with this example's parent directory on
``PYTHONPATH``::

    python -m torchtitan.rl.train \\
        --module titanrl_openenv \\
        --config rl_grpo_muse_glimmer_30b_openenv_chess

All three recipes train on OpenEnv's built-in chess environment
(``envs/chess_env``) — start it first with ``python
examples/titanrl_openenv/serve_chess.py`` (the stock
``python -m envs.chess_env.server.app`` serves only one session at a time, which
is not enough for a batch of concurrent rollouts) and point
``OpenEnvMessageEnv.Config.base_url`` at it (see ``README.md``). Switch
``message_env.task`` to ``"generic"`` to point the same recipe at any other
OpenEnv environment.

This module imports ``torchtitan``.
"""

from __future__ import annotations

import dataclasses

from renderers import Qwen3RendererConfig
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import (
    AdamW,
    LRSchedulersContainer,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import CompileConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_model_registry
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.model.muse_glimmer.renderer import MuseGlimmerRendererConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer

from .data import OpenEnvDataset
from .rubric import OpenEnvReward, OpenEnvShapingReward
from .tasks import CHESS_POSITION_REWARD
from .titanrl_env import OpenEnvMessageEnv

# A chess rollout is a partial game: one turn per move, each echoing a fresh
# legal-move list. Keep the trainer's context equal to the rollout budget in
# ``_openenv_chess_rollouter_config`` -- the batcher requires the microbatch token
# budget to be a multiple of ``max_context_length``, so the three move together.
# A finished rollout (last prompt + last completion) longer than this is dropped
# by the batcher, so ``max_num_turns * sampling.max_tokens`` plus the prompt and
# observations must fit it as well.
_SEQ_LEN = 12288


def _openenv_chess_rollouter_config() -> Rollouter.Config:
    """Datasets, environment, rubric, and token budget shared by every recipe.

    The defaults target OpenEnv's built-in chess environment; switch
    ``message_env.task`` to ``"generic"`` (and point ``base_url`` elsewhere) to
    train against any other OpenEnv environment.
    """
    return Rollouter.Config(
        train_dataset=OpenEnvDataset.Config(seed=42),
        validation_dataset=OpenEnvDataset.Config(seed=99, shuffle=False),
        worker=RolloutWorker.Config(
            # Two reward fns over the same ``env_rewards`` dict: the environment's
            # own outcome reward, plus the shaped position score the chess profile
            # derives from ``metadata["evaluation"]``. The outcome reward is the
            # one that matters, but within a few-move rollout it is almost always
            # 0.0 (chess pays out only on checkmate/draw, or -0.1 for an illegal
            # move), so on its own it gives GRPO no way to rank siblings and the
            # batcher aborts with "N consecutive untrainable batches". The
            # position score is dense and differs per rollout, which is what makes
            # the group trainable; it is weighted below the real reward so it
            # shapes rather than replaces it.
            rubric=Rubric.Config(
                reward_fns=[
                    OpenEnvReward.Config(weight=1.0, aggregate="sum"),
                    OpenEnvShapingReward.Config(
                        reward_name=CHESS_POSITION_REWARD,
                        # The score of the final position reached, not a sum
                        # over the game: it already accounts for everything
                        # played so far.
                        aggregate="last",
                        weight=0.5,
                    ),
                ],
                # Deliberately left unset (``search_r1`` sets 0.0). It
                # short-circuits the reward fns for *any* truncated status, and
                # ``is_truncated()`` covers ``truncated_max_turns`` as well as
                # ``truncated_length`` -- so setting it flattens every rollout in a
                # partial-game task, since none of them ever reach ``completed``.
                truncation_reward=None,
            ),
            # Served with enough session capacity for a full batch of rollouts
            # by ``python examples/titanrl_openenv/serve_chess.py``.
            message_env=OpenEnvMessageEnv.Config(
                task="chess", base_url="http://127.0.0.1:8000"
            ),
            # Chess is genuinely long-horizon: one turn per move, and every turn
            # echoes a fresh legal-move list, so the budget has to cover a partial
            # game rather than a single question/answer exchange. Muse Glimmer
            # reasons more as the game history grows (see ``sampling.max_tokens``
            # below), and a turn cut off before its tool call is a wasted turn.
            # Four turns of completion plus the prompt and observations is what
            # the trainer context holds; raising the turn count means raising
            # both.
            token_env=TokenEnv.Config(max_rollout_tokens=_SEQ_LEN, max_num_turns=4),
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
    )


def rl_grpo_muse_glimmer_30b_openenv_chess() -> Controller.Config:
    """GRPO/DAPO on OpenEnv's chess environment for Muse Glimmer 30B.

    8 GPUs: 6 trainer (FSDP=3 x TP=2) + 2 generator (TP=2). Requires a running
    OpenEnv chess server (``python examples/titanrl_openenv/serve_chess.py``)
    plus the model assets; see ``README.md``.

    Two constraints are specific to this model (both inherited from torchtitan's
    ``rl_grpo_muse_glimmer_30b_search_r1``):

    * **Generator TP <= 2.** Muse Glimmer has 2 KV heads, so attention cannot be
      tensor-split further. Scale the trainer with FSDP rather than TP.
    * **Full activation checkpointing is required.** Adam's m/v are allocated on
      the *first* ``optimizer.step()``, so per-GPU memory jumps between step 1
      and step 2. With the default ``SelectiveAC`` that jump OOMs at step 2;
      ``FullAC`` frees the headroom it needs.

    A third is specific to running it on torchtitan ``main`` at this context
    length: Adam's moments are kept in bf16 (see the ``optimizer`` comment).

    varlen attention is used for both roles so the trainer and the vLLM generator
    run one model config. The state-dict adapter handles the HF checkpoint's Q/K
    RoPE layout on load, and the renderer handles Muse Glimmer's harmony chat
    format and ATEM tool calls — which is what carries the ``chess_move`` tool.
    """
    model_config = muse_glimmer_model_registry("30B", attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Muse-Glimmer-30B",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=64),
            # ``windowed_fifo_batches`` stays at upstream's greedy default, as in
            # ``search_r1``: finished groups are trained oldest-first with no cap
            # on policy age, rather than stalling on a slow rollout. Watch
            # ``train_batch/pct_samples_over_target_age``; ``1`` restores a
            # FIFO bound of ``target_offpolicy_steps + 1`` steps.
        ),
        compile=None,
        rollouter=_openenv_chess_rollouter_config(),
        renderer=MuseGlimmerRendererConfig(),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            wandb_project="openenv-titanrl-chess",
            # Also write TensorBoard events under the dump folder, so the run is
            # plottable without a W&B account.
            enable_tensorboard=True,
        ),
        trainer=Trainer.Config(
            # Adam's moments in bf16 (params and updates stay fp32), as
            # torchtitan's own 27B dense RL recipe does. It frees ~17 GiB per
            # trainer GPU, and at 30B on 95 GiB cards that headroom is not
            # optional: the weight push to the generator holds a bf16 copy of
            # every weight shard (~8.7 GiB per GPU) and can overlap the next
            # forward/backward -- it does whenever a batch is already waiting
            # when the optimizer step returns, which a mid-run checkpoint save
            # (inside the optimizer step, minutes long) guarantees. With fp32
            # moments that overlap needs ~91 GiB and the trainer runs out of
            # memory at the first step after a mid-run checkpoint; with bf16
            # moments it peaks at ~79.
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(pattern=r".*", lr=1e-6, moment_dtype="bfloat16")
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(
                # As in every upstream RL recipe: the shared training engine
                # captures CUDA graphs unless told not to, and varlen attention
                # needs fixed-shape document metadata to be graphed.
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=_SEQ_LEN,
                max_context_length=_SEQ_LEN,
            ),
            activation_checkpoint=FullAC.Config(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=3,
                tensor_parallel_degree=2,
            ),
            # Required, not just for saving: without a checkpointer the trainer
            # skips the initial HF load and trains from random weights, with
            # only a warning to show for it.
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    # Needed whenever the trainer has TP > 1: policy statistics
                    # are computed vocab-parallel, and the first loss call raises
                    # without it.
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,  # <= 2 KV heads
            ),
            # Leave ``gpu_memory_limit`` at its 0.9 default, as torchtitan's own
            # Muse Glimmer 30B recipe does. Its Qwen3-8B recipe caps it at 0.6,
            # but at 30B the weights take ~26 GiB per card at TP=2, so 0.6 leaves
            # ~28 GiB of KV cache where 0.9 leaves ~57 GiB. A run capped at 0.6
            # queued requests in vLLM, decoded at ~130 s per turn, and hit NCCL's
            # 600 s watchdog at step 3; runs at 0.9 have not. (Its log shows no
            # KV preemption, so the exact mechanism is not established.)
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                # One turn is "think briefly, then call chess_move", but Muse
                # Glimmer's reasoning grows with the game history: in the 60-step
                # run, completions averaged ~380 tokens on move 1, ~870 on
                # move 2 and ~1,600 on moves 3-4. A turn cut off before the tool
                # call is a wasted turn -- at 384 every rollout was
                # truncated_length, at 1536 only 27% of third moves still reached
                # the tool call, and at 2560 about a quarter of third and fourth
                # moves still hit the cap. What bounds it is the context:
                # 2560 x 4 turns plus the ~730-token prompt and the observations
                # must fit _SEQ_LEN (finished rollouts averaged ~5.2k tokens,
                # longest ~9.3k).
                max_tokens=2560,
            ),
        ),
    )


def rl_grpo_muse_glimmer_30b_openenv_chess_smoke() -> Controller.Config:
    """Tiny smoke variant of the Muse Glimmer recipe.

    A handful of steps, a small batch, and no mid-run checkpoint save -- enough to
    confirm the OpenEnv server, the rollouter, the generator, and the trainer all
    talk to each other before committing GPUs to the full run.
    """
    config = rl_grpo_muse_glimmer_30b_openenv_chess()
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=5,
        num_prompts_per_train_step=4,
        num_samples_per_prompt=4,
        validation=ValidationConfig(num_samples=8),
    )
    checkpointer = config.trainer.checkpointer
    assert checkpointer is not None
    config.trainer = dataclasses.replace(
        config.trainer,
        # Past the smoke run's step count: a mid-run DCP save of the full
        # training state (~208 GB with optimizer state at 30B) is heavy I/O the
        # smoke run does not need to exercise. The final-step save still runs.
        checkpointer=dataclasses.replace(checkpointer, interval=1000),
    )
    return config


def rl_grpo_qwen3_1_7b_openenv_chess() -> Controller.Config:
    """The same OpenEnv chess recipe at small scale, for Qwen3-1.7B.

    Useful for smoke-testing the integration on a single node before committing
    GPUs to the 30B run: 4 generator (TP=4) + 1 trainer (TP=1).
    """
    model_config = qwen3_model_registry("1.7B", seq_len=_SEQ_LEN, attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=64),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_openenv_chess_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            wandb_project="openenv-titanrl-chess",
            # Also write TensorBoard events under the dump folder, so the run is
            # plottable without a W&B account.
            enable_tensorboard=True,
        ),
        trainer=Trainer.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-6)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=_SEQ_LEN,
                max_context_length=_SEQ_LEN,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=1536,
            ),
        ),
    )
