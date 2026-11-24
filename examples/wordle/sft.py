# /// script
# dependencies = [
#     "trl==1.16.0",
#     "trackio",
# ]
# ///
"""
Short SFT warmup on Wordle tool-calling games, before GRPO.

```sh
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN sft.py \
    --dataset <user>/wordle-tool-calls-sft --hub-model-id <user>/Qwen3-1.7B-Wordle-SFT
```
"""

import argparse

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--trackio-space-id", default=None)
    args = parser.parse_args()

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=load_dataset(args.dataset, split="train"),
        args=SFTConfig(
            output_dir=args.hub_model_id.split("/")[-1],
            # Train every assistant turn: reasoning, tool calls and the closing turn.
            assistant_only_loss=True,
            max_length=2048,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            warmup_steps=3,
            # Keep small weight updates in fp32 while computing in bf16.
            model_init_kwargs={"dtype": "float32"},
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            report_to="trackio",
            trackio_space_id=args.trackio_space_id,
            project="openenv-wordle",
            run_name="sft",
            push_to_hub=True,
            hub_model_id=args.hub_model_id,
            hub_private_repo=False,
        ),
    )
    trainer.train()
    # Cast only after training, to share the weights in bf16.
    trainer.model.to(torch.bfloat16)
    trainer.push_to_hub(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
