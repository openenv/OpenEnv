# /// script
# dependencies = [
#     "trl[vllm]==1.16.0",
#     "trackio",
#     "openenv==0.10.0",
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
# ]
# ///
"""
GRPO on the OpenEnv Wordle environment, starting from the SFT warmup model, with thinking on.

```sh
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN grpo.py \
    --model <user>/Qwen3-1.7B-Wordle-SFT --hub-model-id <user>/Qwen3-1.7B-Wordle-SFT-GRPO
```
"""

import argparse
import re

import torch
from datasets import Dataset
from huggingface_hub import get_token
from textarena_env import TextArenaAction, TextArenaEnv
from trl import GRPOConfig, GRPOTrainer


PROMPT = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""

FEEDBACK = re.compile(r"Feedback:\n([A-Z ]+)\n([GYX ]+)")


# The environment pays a win the same at any speed and pays partial credit only at the end of a game, so a policy can
# learn to think less and spend every guess. These terms reward winning fast, give partial credit from the best guess
# whether or not the game finished, and penalize invalid moves. Each one is logged separately.
def reward_win(environments, **kwargs) -> list[float]:
    return [
        1.0 + 0.5 * (1 - env.num_guesses / 6) if env.won else 0.0
        for env in environments
    ]


def reward_partial(environments, **kwargs) -> list[float]:
    return [0.0 if env.won else 0.1 * env.best_greens for env in environments]


def reward_invalid(environments, **kwargs) -> list[float]:
    return [-0.1 * env.invalid_moves for env in environments]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--env-url", default="https://openenv-wordle.hf.space")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--trackio-space-id", default=None)
    args = parser.parse_args()
    env_url = args.env_url
    # A private Space needs your token. A public one ignores it.
    headers = {"Authorization": f"Bearer {get_token()}"} if get_token() else None

    class WordleEnv:
        def __init__(self):
            self.client = TextArenaEnv(base_url=env_url, headers=headers)

        def reset(self, **kwargs) -> str | None:
            result = self.client.reset()
            # The game returns cumulative feedback each turn, so keep the full text and return only the new part.
            self._last_full_feedback = result.observation.messages[0].content
            self.done = False
            self.won = False
            self.num_guesses = 0
            self.best_greens = 0
            self.invalid_moves = 0
            return self._last_full_feedback

        def guess(self, guess: str) -> str:
            """
            Make a guess in the Wordle environment.

            Args:
                guess: The guessed word, formatted as '[abcde]'

            Returns:
                The feedback message from the environment.
            """
            if self.done or self.num_guesses + self.invalid_moves >= 12:
                raise ValueError("Game over.")
            result = self.client.step(TextArenaAction(message=guess))
            full_feedback = result.observation.messages[0].content
            feedback = full_feedback[len(self._last_full_feedback) :]
            self._last_full_feedback = full_feedback
            if "invalid move" in feedback.lower():
                self.invalid_moves += 1
            else:
                self.num_guesses += 1
                match = FEEDBACK.search(feedback)
                if match:
                    self.best_greens = max(self.best_greens, match.group(2).count("G"))
                self.won = result.done and result.reward >= 1.0
            self.done = result.done
            return feedback

    # 4 prompts x 16 games per step. Each game is one environment session, so the Space must allow 64 at once.
    prompts_per_step, num_generations, per_device_batch_size = 4, 16, 8
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": PROMPT}]]
            * (args.max_steps * prompts_per_step)
        }
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_win, reward_partial, reward_invalid],
        train_dataset=dataset,
        environment_factory=WordleEnv,
        args=GRPOConfig(
            output_dir=args.hub_model_id.split("/")[-1],
            max_steps=args.max_steps,
            learning_rate=1e-6,
            # Constant: with a linear decay to 0 the reward was still rising when the learning rate ran out.
            lr_scheduler_type="constant",
            num_generations=num_generations,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=prompts_per_step
            * num_generations
            // per_device_batch_size,
            gradient_checkpointing=True,
            bf16=True,
            chat_template_kwargs={"enable_thinking": True},
            max_completion_length=4096,
            mask_truncated_completions=True,
            # Multi-turn rollouts: clip per-token ratios instead of masking whole sequences.
            vllm_importance_sampling_mode="token_truncate",
            # Invalid warnings use tool rounds, but not legal guesses.
            # The adapter caps submissions at 12; the environment still allows six legal guesses.
            max_tool_calling_iterations=12,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.35,
            report_to="trackio",
            trackio_space_id=args.trackio_space_id,
            project="openenv-wordle",
            run_name="grpo",
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=1,
            save_steps=25,
            save_total_limit=2,
            push_to_hub=True,
            hub_model_id=args.hub_model_id,
            hub_private_repo=False,
        ),
    )
    trainer.train()
    # Share the weights in bf16, as the SFT model.
    trainer.model.to(torch.bfloat16)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
