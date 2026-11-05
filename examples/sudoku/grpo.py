# /// script
# dependencies = [
#     "trl[vllm]==1.16.0",
#     "trackio",
#     "openenv==0.10.0",
#     "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env",
# ]
# ///
"""
GRPO on the OpenEnv Sudoku environment, starting from the SFT warmup model, with thinking on.

The Space must serve `Sudoku-v0` at the level you train on, e.g. `TEXTARENA_ENV_ID=Sudoku-v0`, `TEXTARENA_KW_CLUES=55`
and `MAX_CONCURRENT_ENVS=96` (64 games at once, plus room for the sessions being closed between games).

```sh
hf jobs uv run --flavor a100-large --timeout 18h --secrets HF_TOKEN grpo.py \
    --model <user>/Qwen3-1.7B-Sudoku-SFT --hub-model-id <user>/Qwen3-1.7B-Sudoku-SFT-GRPO \
    --env-url https://<user>-sudoku.hf.space
```
"""

import argparse

import torch
from datasets import Dataset
from huggingface_hub import get_token
from textarena_env import TextArenaAction, TextArenaEnv
from trl import GRPOConfig, GRPOTrainer


PROMPT = """You are playing Sudoku. Fill every empty cell (shown as '.') of the 9x9 grid with a digit from 1 to 9 so that \
each row, each column and each 3x3 box contains every digit exactly once.

Use the tool `place` to fill one cell at a time. Rows and columns are numbered from 1 to 9. A digit that is not the \
solution of that cell is a mistake. Two mistakes in a row end the game.

"""


def reward_solved(environments, **kwargs) -> list[float]:
    return [1.0 if env.solved else 0.0 for env in environments]


# Partial credit for the cells filled, whether the game ended on mistakes or the model stopped calling the tool.
def reward_progress(environments, **kwargs) -> list[float]:
    return [env.filled / env.empty for env in environments]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--run-name", default="grpo")
    parser.add_argument("--trackio-space-id", default=None)
    args = parser.parse_args()
    env_url = args.env_url
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else None

    class SudokuEnv:
        def __init__(self):
            self.client = TextArenaEnv(base_url=env_url, headers=headers).sync()

        def reset(self, **kwargs) -> str | None:
            self.client.close()
            self.client = TextArenaEnv(base_url=env_url, headers=headers).sync()
            result = self.client.reset()
            # The game returns the whole log each turn, so keep the full text and read only the new part.
            self._last_full = result.observation.messages[0].content
            board = self._last_full.split("Game Board:\n")[-1]
            self.empty = board.count(".")
            self.filled = 0
            self.solved = False
            self.done = False
            return board

        def place(self, row: int, col: int, number: int) -> str:
            """
            Place a number in an empty cell of the Sudoku grid.

            Args:
                row: Row number (1-9).
                col: Column number (1-9).
                number: Number to place (1-9).

            Returns:
                The updated grid, or the reason the move was rejected.
            """
            if self.done:
                raise ValueError("Game over.")
            result = self.client.step(
                TextArenaAction(message=f"[{row} {col} {number}]")
            )
            full = result.observation.messages[0].content
            feedback, self._last_full = full[len(self._last_full) :], full
            self.done = result.done
            if "invalid" in feedback.lower():
                return feedback.split("[GAME]")[-1].strip()
            self.filled += 1
            self.solved = result.done and result.reward >= 1.0
            return feedback.split("Board state: \n")[-1]

    # 4 prompts x 16 games per step, one environment session each. MAX_CONCURRENT_ENVS=96 leaves room for the sessions
    # still being closed between games.
    # One game per forward pass: with games up to 16k tokens, 8 at once ran out of memory on an A100 80GB.
    prompts_per_step, num_generations, per_device_batch_size = 4, 16, 1
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": PROMPT}]]
            * (args.max_steps * prompts_per_step)
        }
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_solved, reward_progress],
        train_dataset=dataset,
        environment_factory=SudokuEnv,
        args=GRPOConfig(
            output_dir=args.hub_model_id.split("/")[-1],
            max_steps=args.max_steps,
            learning_rate=1e-6,
            lr_scheduler_type="constant",
            num_generations=num_generations,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=prompts_per_step
            * num_generations
            // per_device_batch_size,
            gradient_checkpointing=True,
            bf16=True,
            chat_template_kwargs={"enable_thinking": True},
            max_completion_length=16384,
            mask_truncated_completions=True,
            # Multi-turn rollouts: clip per-token ratios instead of masking whole sequences.
            vllm_importance_sampling_mode="token_truncate",
            # One call per empty cell, plus the moves rejected along the way.
            max_tool_calling_iterations=40,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.35,
            report_to="trackio",
            trackio_space_id=args.trackio_space_id,
            project="openenv-sudoku",
            run_name=args.run_name,
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
