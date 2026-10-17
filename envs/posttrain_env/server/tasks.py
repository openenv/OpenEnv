# SPDX-License-Identifier: BSD-3-Clause

"""T0 post-training tasks.

Each task tells the agent what to train, which HF Jobs hardware it may use and
how much it may spend. The evaluator script runs as its own job: the agent's
checkpoint is mounted read-only at `/model`, and the job prints one
`POSTTRAIN_SCORE {...}` line that the broker parses. Evaluators pin their
dependencies in their own PEP 723 header, so scores don't drift with releases.
"""

from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).parent / "eval_scripts"


@dataclass(frozen=True)
class Task:
    """
    One post-training task.

    Attributes:
        task_id (`str`):
            Stable task identifier, passed to `reset(task_id=...)`.
        prompt (`str`):
            Instructions shown to the agent. `{nonce}` is filled in at reset.
        flavors (`tuple[str, ...]`):
            HF Jobs hardware flavors the agent may request.
        max_timeout_minutes (`int`):
            Longest timeout a single agent job may request.
        budget_usd (`float`):
            Per-rollout budget for agent jobs and dev evaluations.
        max_jobs (`int`):
            Maximum number of agent jobs per episode.
        max_dev_evals (`int`):
            Maximum number of `dev_evaluate` calls per episode.
        eval_script (`str`):
            File name of the evaluator under `server/eval_scripts/`.
        eval_flavor (`str`):
            Hardware flavor for dev and score evaluations.
        eval_timeout_minutes (`int`):
            Timeout for one evaluation job.
        dev_seed (`int`):
            Seed of the public dev split.
        max_episode_minutes (`int`):
            Wall-clock limit for the whole episode.
    """

    task_id: str
    prompt: str
    flavors: tuple[str, ...]
    max_timeout_minutes: int
    budget_usd: float
    max_jobs: int
    max_dev_evals: int
    eval_script: str
    eval_flavor: str
    eval_timeout_minutes: int
    dev_seed: int = 0
    max_episode_minutes: int = 240

    @property
    def eval_script_path(self) -> Path:
        return EVAL_DIR / self.eval_script


OUTPUTS_NOTE = (
    "Your job's working storage is mounted at `/data`. Anything you write under "
    "`/data/outputs/` is the job's artifact: `dev_evaluate` and `submit` mount "
    "it read-only at `/model`."
)

TASKS = {
    "t0_artifact": Task(
        task_id="t0_artifact",
        prompt=(
            "Plumbing check. Submit one HF Jobs script that writes the exact text "
            "`{nonce}` to `/data/outputs/answer.txt`, then call `submit` with that "
            "job's id. " + OUTPUTS_NOTE
        ),
        flavors=("cpu-basic",),
        max_timeout_minutes=10,
        budget_usd=0.05,
        max_jobs=3,
        max_dev_evals=2,
        eval_script="t0_artifact_eval.py",
        eval_flavor="cpu-basic",
        eval_timeout_minutes=5,
        max_episode_minutes=30,
    ),
    "t0_arithmetic_format": Task(
        task_id="t0_arithmetic_format",
        prompt=(
            "Fine-tune `HuggingFaceTB/SmolLM2-135M-Instruct` so that it answers "
            "two-digit addition in an exact format. The evaluator sends the chat "
            "message `What is {{a}} + {{b}}?` (0 <= a, b <= 99) through the base "
            "model's chat template with `add_generation_prompt=True`, decodes up to "
            "16 new tokens greedily, and counts an answer as correct only if the "
            "stripped reply is exactly `#### {{a+b}}`. Only your weights are used: "
            "the tokenizer, chat template, config and generation settings are the "
            "base model's, so keep its architecture and vocabulary. Save the model "
            "with `save_pretrained` (safetensors) to `/data/outputs/`. The reward is "
            "the accuracy on a hidden test split; `dev_evaluate` scores the public "
            "dev split (seed 1234). " + OUTPUTS_NOTE
        ),
        flavors=("cpu-upgrade", "t4-small", "a10g-small"),
        max_timeout_minutes=60,
        budget_usd=3.0,
        max_jobs=6,
        max_dev_evals=3,
        eval_script="t0_arithmetic_format_eval.py",
        eval_flavor="cpu-upgrade",
        eval_timeout_minutes=20,
        dev_seed=1234,
    ),
}
