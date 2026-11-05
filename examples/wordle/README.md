# Wordle: SFT, then GRPO

Train Qwen3-1.7B to play Wordle against the [`openenv/wordle`](https://huggingface.co/spaces/openenv/wordle) environment. The [tutorial](https://huggingface.co/docs/openenv/tutorials/wordle-grpo) walks through each step, with results, hardware and costs.

| Script | What it does | Where it runs |
|---|---|---|
| `generate_sft_data.py` | A solver plays 1,000 games and writes them as tool-calling conversations | Laptop, 10 s |
| `sft.py` | SFT warmup on those games | Hugging Face Jobs, `a100-large` |
| `grpo.py` | GRPO against the environment, starting from the SFT model | Hugging Face Jobs, `a100-large` |
| `evaluate.py` | Plays a model and saves every game (`--secrets` fixes the words, so models play the same ones) | Laptop or any GPU |
| `make_gif.py` | Turns saved games into a side-by-side GIF | Laptop |
| `wordle.ipynb` | The same pipeline in a notebook | Colab |

```bash
uv run generate_sft_data.py --games 1000 --push-to-hub <user>/wordle-tool-calls-sft
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN sft.py \
    --dataset <user>/wordle-tool-calls-sft --hub-model-id <user>/Qwen3-1.7B-Wordle-SFT
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN grpo.py \
    --model <user>/Qwen3-1.7B-Wordle-SFT --hub-model-id <user>/Qwen3-1.7B-Wordle-SFT-GRPO \
    --env-url https://<user>-wordle.hf.space
uv run evaluate.py --model <user>/Qwen3-1.7B-Wordle-SFT-GRPO --thinking --max-new-tokens 512 \
    --secrets "$(cat secrets30.txt)" --out grpo.jsonl
```

GRPO plays 64 games at once, one environment session each. Duplicate the `openenv/wordle` Space and set `MAX_CONCURRENT_ENVS=64` on your copy before training.
