# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "trl[peft]==1.18.0",
#     "transformers==5.24.1",
#     "torchvision",
#     "huggingface_hub==2.4.0",
# ]
# ///
"""Merge the saved LoRA adapter into its base model for vLLM evaluation."""

import argparse
from contextlib import nullcontext
from tempfile import TemporaryDirectory


def main():
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="Local folder or hf://buckets/<user>/<bucket>/<prefix>.",
    )
    args = parser.parse_args()
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter, dtype=torch.bfloat16, device_map="cpu"
    )
    processing_class = AutoProcessor.from_pretrained(args.adapter)
    bucket_output = args.output.startswith("hf://buckets/")
    # Serialize locally before transferring to object storage; mounted filesystems may not preserve atomic renames.
    with TemporaryDirectory() if bucket_output else nullcontext(args.output) as output:
        model.merge_and_unload().save_pretrained(output)
        processing_class.save_pretrained(output)
        if bucket_output:
            from huggingface_hub import HfApi

            HfApi().sync_bucket(output, args.output)


if __name__ == "__main__":
    main()
