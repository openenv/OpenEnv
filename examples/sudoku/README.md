# Sudoku: SFT, then GRPO

Train Qwen3-1.7B to play Sudoku against the TextArena `Sudoku-v0` environment, one `place(row, col, number)` tool call per cell.

| Script | What it does | Where it runs |
|---|---|---|
| `make_puzzles.py` | Writes a fixed set of puzzles and solutions, so every model plays the same boards | Laptop, seconds |
| `generate_sft_data.py` | A solver plays 1,000 games and writes them as tool-calling conversations | Laptop, about a minute |
| `sft.py` | SFT warmup on those games | Hugging Face Jobs, `a100-large`, 27 min |
| `grpo.py` | GRPO against the environment, starting from the SFT model | Hugging Face Jobs, `a100-large`, about 15 h |
| `evaluate.py` | Plays a model on a puzzle file and saves every game | Laptop or any GPU (`l4x1`, under an hour for 30 puzzles) |
| `summarize.py` | One line per evaluation file: solved, cells filled, mistakes, how games ended | Laptop |
| `make_gif.py` | Turns saved games into a side-by-side GIF | Laptop |

`puzzles_c55.json` (26 empty cells on average) and `puzzles_c65.json` (16) are the 30 puzzles we evaluate on.

```bash
uv run generate_sft_data.py --games 1000 --push-to-hub <user>/sudoku-tool-calls-sft
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN sft.py \
    --dataset <user>/sudoku-tool-calls-sft --hub-model-id <user>/Qwen3-1.7B-Sudoku-SFT
hf jobs uv run --flavor a100-large --timeout 18h --secrets HF_TOKEN grpo.py \
    --model <user>/Qwen3-1.7B-Sudoku-SFT --hub-model-id <user>/Qwen3-1.7B-Sudoku-SFT-GRPO \
    --env-url https://<user>-sudoku.hf.space
uv run evaluate.py --model <user>/Qwen3-1.7B-Sudoku-SFT-GRPO --thinking --max-new-tokens 384 \
    --puzzles puzzles_c55.json --out grpo_c55.jsonl
uv run summarize.py grpo_c55.jsonl
```

GRPO plays 64 games at once, one environment session each. Duplicate the [`openenv/sudoku`](https://huggingface.co/spaces/openenv/sudoku) Space and set `TEXTARENA_KW_CLUES=55` (about 26 empty cells, the Space default is 60 clues, about 21) and `MAX_CONCURRENT_ENVS=96` on your copy.

## Results

Greedy, the same 30 puzzles for every model. The base model plays without thinking (256 new tokens per move), the trained models with thinking (384).

| Model | 16 empty: solved | 16 empty: cells | 26 empty: solved | 26 empty: cells |
|---|---|---|---|---|
| Qwen3-1.7B | 0/30 | 1.9% | 0/30 | 0.1% |
| + SFT | 19/30 | 79.1% | 10/30 | 55.8% |
| + SFT + GRPO (75 steps) | 26/30 | 87.5% | 22/30 | 83.2% |
