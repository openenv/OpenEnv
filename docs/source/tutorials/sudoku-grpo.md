# Train a Sudoku Agent: SFT, then GRPO

This tutorial trains [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) to play Sudoku against the OpenEnv TextArena environment, one cell per tool call. The model starts out unable to fill a single cell correctly. A short supervised warmup teaches it to read the grid and fill it, and in our run GRPO then more than doubled the puzzles it solves at the level it trained on. Every step runs on one GPU on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), and the whole pipeline costs about $42, most of it GRPO.

![Base model, SFT and SFT + GRPO playing the same puzzles](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/sudoku-progress.gif)

*Puzzles 7 to 12 of `puzzles_c55.json` (26 empty cells), where the SFT model solves 2 and SFT + GRPO 4, close to their rates over all 30.*

| Model | 16 empty cells: solved | Cells filled | 26 empty cells: solved | Cells filled |
|---|---|---|---|---|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0/30 | 1.9% | 0/30 | 0.1% |
| + SFT ([model](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT)) | 19/30 | 79.1% | 10/30 | 55.8% |
| + SFT + GRPO ([model](https://huggingface.co/openenv/Qwen3-1.7B-Sudoku-SFT-GRPO)) | 26/30 | 87.5% | 22/30 | 83.2% |

These numbers come from our run, with one training run per step, evaluated greedily on the same 30 puzzles per level ([`puzzles_c65.json`](https://github.com/openenv/OpenEnv/tree/main/examples/sudoku) and `puzzles_c55.json`). GRPO trained at 26 empty cells, which is where it helps most. At 16 empty cells the SFT model already solves most puzzles. Thirty puzzles is a small sample: GRPO fills more cells than the SFT model on 19 of the 30 puzzles at 26 empty cells and fewer on 4. Greedy decoding on a different GPU also changes a few games, which moved the SFT model by 3 puzzles between our runs, so expect the same ordering in a new run rather than the exact numbers.

See the training curves on [Trackio](https://huggingface.co/spaces/openenv/sudoku-runs). The scripts are in [`examples/sudoku`](https://github.com/openenv/OpenEnv/tree/main/examples/sudoku).

## What you need

| Step | Script | Hardware | Time | Cost |
|---|---|---|---|---|
| 1. Make the evaluation puzzles | `make_puzzles.py` | Laptop CPU | seconds | $0 |
| 2. Generate SFT data | `generate_sft_data.py` | Laptop CPU | 35 s | $0 |
| 3. SFT warmup | `sft.py` | 1x A100 80GB (`a100-large`) | 27 min | $1.10 |
| 4. GRPO | `grpo.py` | 1x A100 80GB (`a100-large`) | 15 h | $38 |
| 5. Evaluate and make the GIF | `evaluate.py`, `summarize.py`, `make_gif.py` | 1x L4 (`l4x1`) per puzzle file | under 1 h | about $2 for three models |

Times and costs are from our runs, at the Hugging Face Jobs prices ($2.50/h for `a100-large`, $0.80/h for `l4x1`), and include installing packages and loading the model. You need a Hugging Face account with a payment method for Jobs, and [uv](https://docs.astral.sh/uv/). The scripts pin their dependencies (`trl==1.16.0`, `openenv==0.10.0`), so `uv run` and `hf jobs uv run` install the versions used here.

## 1. Play the environment

The Sudoku environment runs as a Space, [`openenv/sudoku`](https://huggingface.co/spaces/openenv/sudoku). Open its playground to play by hand, or connect from Python:

```python
from textarena_env import TextArenaAction, TextArenaEnv

with TextArenaEnv(base_url="https://openenv-sudoku.hf.space").sync() as env:
    result = env.reset()
    print(result.observation.messages[0].content)
    result = env.step(TextArenaAction(message="[1 5 3]"))  # row 1, column 5, digit 3
    print(result.observation.messages[0].content)
```

Install the client with `pip install "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env"`.

A move is `[row col digit]`, and the environment answers with the whole grid:

```text
   C1 C2 C3   C4 C5 C6   C7 C8 C9
R1  4  1  9 |  7  .  . |  .  8  6
R2  6  3  5 |  8  9  2 |  .  .  4
R3  2  7  8 |  .  6  1 |  5  .  .
   - - - - - - - - - - - - - - - -
...
```

Three rules matter for training:

- **Any digit that is not the solution of its cell is rejected**, even if it breaks no rule yet. So there are no dead ends: every accepted move is correct.
- **Two rejected moves in a row end the game.** A correct move resets the count.
- **The reward is 1.0 for a solved puzzle**, and the fraction of the grid filled otherwise.

The log grows with every move: each observation repeats the whole game so far, so the scripts keep the previous text and read only the new part.

## 2. Pick a level

TextArena's `Sudoku-v0` takes a `clues` argument, the number of digits given at the start. The generator removes digits while the solution stays unique, so `clues` is a floor: 55 clues leaves about 26 empty cells, 60 (the `Sudoku-v0` default, which the `openenv/sudoku` Space serves) about 21, and 65 about 16.

The level decides whether GRPO has something to learn from. After SFT, our model solved most puzzles at 16 empty cells and a third of them at 26, filling about half the grid. Training at 26 empty cells leaves room to improve while every group still has games that go further than others. We trained at 55 clues and evaluated at 55 and 65.

The Space reads the game arguments from `TEXTARENA_KW_*` variables, as JSON. To serve the training level, [duplicate the `openenv/sudoku` Space](https://huggingface.co/spaces/openenv/sudoku?duplicate=true) and set these variables on your copy:

| Variable | Value | Why |
|---|---|---|
| `TEXTARENA_KW_CLUES` | `55` | About 26 empty cells |
| `MAX_CONCURRENT_ENVS` | `96` | GRPO plays 64 games at once, plus room for the sessions being closed between games |

`make_puzzles.py` writes the evaluation puzzles with TextArena's own generator, from fixed seeds, so every model plays the same boards:

```bash
uv run make_puzzles.py --clues 55 --n 30 --out puzzles_c55.json
uv run make_puzzles.py --clues 65 --n 30 --out puzzles_c65.json
```

## 3. Why start with SFT

Qwen3-1.7B does not play Sudoku at all. On our 30 puzzles at 16 empty cells, it solved none and filled 2% of the cells. In 20 of the 30 games its first move was on a cell that was already given. Its games end on two rejected moves in a row, or when it stops calling the tool.

A model trained earlier with GRPO alone, from the base model and with hints in the prompt, did not do much better on these puzzles: 0/30 without hints and 1/30 with them. With almost no correct moves, GRPO has very little to compare. So the recipe starts with a short SFT on games played by a solver, which teaches the format, how to read the grid and a simple strategy. GRPO then trains against the environment and fixes what the SFT model still gets wrong.

## 4. Generate the SFT data

`generate_sft_data.py` plays 1,000 games with a solver, against the same environment class the Space runs (in-process, so it takes about 35 seconds). Boards have 50, 55, 60 or 65 clues (15 to 31 empty cells). The solver only fills a cell when one digit fits, so every step of its reasoning can be checked on the grid:

```text
Cell (8, 1) is empty. Row 8 has 1, 2, 3, 4, 5, 6, 7, 9, column 1 has 1, 2, 3, 4, 5, 6, 7, 9 and its box has
1, 2, 3, 4, 5, 6, 7, 9. The only digit missing from all three is 8.
```

Each game is stored as a tool-calling conversation: the user prompt with the grid, then one assistant turn per move (the reasoning as `reasoning_content` and a call to the `place` tool), each followed by the updated grid, and a last turn saying the puzzle is solved. This is the format TRL renders during GRPO, so the SFT model plays the way RL will train it.

```bash
uv run generate_sft_data.py --games 1000 --push-to-hub <user>/sudoku-tool-calls-sft
```

The dataset used here is [`openenv/sudoku-tool-calls-sft`](https://huggingface.co/datasets/openenv/sudoku-tool-calls-sft), with no board in common with the evaluation puzzles. Games are long: every tool result carries the whole grid (about 320 tokens), so a game is 8k to 15k tokens.

## 5. SFT warmup

`sft.py` fine-tunes Qwen3-1.7B on those games for one epoch (63 steps), training only the assistant turns, with a max length of 16,384 tokens:

```bash
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN sft.py \
    --dataset <user>/sudoku-tool-calls-sft --hub-model-id <user>/Qwen3-1.7B-Sudoku-SFT
```

In our run the loss went from 0.40 to 0.017. As in the [Wordle tutorial](wordle-grpo), the trainable weights stay in fp32 with bf16 compute, and the script casts them to bf16 just before the upload.

The SFT model solves 19 of 30 puzzles at 16 empty cells and 10 of 30 at 26. Its most common failure: it misreads a row or column, concludes that the missing digit is 0, and ends the game with the closing turn it learned from the dataset ("Solved: every cell is filled.") while cells are still empty.

## 6. GRPO

`grpo.py` starts from the SFT model and trains it against your Space with TRL's `GRPOTrainer` and `environment_factory`. The environment class exposes one tool, `place(row, col, number)`, and keeps what the rewards need:

```python
def reward_solved(environments, **kwargs):
    return [1.0 if env.solved else 0.0 for env in environments]

def reward_progress(environments, **kwargs):
    return [env.filled / env.empty for env in environments]
```

The SFT model solves only a few puzzles at 26 empty cells, so with the binary reward alone many groups would score all zeros and give no signal. The fraction of cells filled separates games that stop at 5 cells from games that stop at 20, and it pays the model for not ending early.

Each step plays 4 prompts x 16 games (64 games) with thinking on, at a constant learning rate of 1e-6:

```bash
hf jobs uv run --flavor a100-large --timeout 18h --secrets HF_TOKEN grpo.py \
    --model <user>/Qwen3-1.7B-Sudoku-SFT --hub-model-id <user>/Qwen3-1.7B-Sudoku-SFT-GRPO \
    --env-url https://<user>-sudoku.hf.space
```

Two settings come from runs that failed:

- **One game per forward pass** (`per_device_train_batch_size=1`, `gradient_accumulation_steps=64`). Games reach 16k tokens, and with 8 games per pass the first step ran out of memory on the A100 80GB.
- **A step gets slower as the model gets better.** Solved games are longer: 13 moves per game at the start, 24 by the end. Steps went from about 8 minutes to about 16, and our first long run hit its 20-hour timeout at step 92 of 100. The script runs 75 steps (about 15 hours) and saves a checkpoint every 25.

![Puzzles solved and cells filled during GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/sudoku-grpo-curve.png)

In our run, puzzles solved during training (sampling at temperature 1) went from 18% to about 62%, and cells filled from 39% to about 75%. Most of the gain came in the first 60 steps. The two dips at steps 26 and 74 are the subject of the next section.

## 7. Long runs against a Space

Each game is a WebSocket session with the Space. In our run, the Space closed all its WebSockets about every 2 hours. The client reconnects on its own, but a new connection is a new session with a different board. When the drop came in the middle of the games, the model kept playing the board it had in its context while the server applied its moves to another one: moves were rejected, games ended early, and that step trained on broken games (step 26: 0 puzzles solved, against about 35% before and after). When the drop came between games, nothing happened.

`grpo.py` opens a new connection at the start of every game, so no connection lives long enough to be dropped:

```python
def reset(self, **kwargs):
    self.client.close()
    self.client = TextArenaEnv(base_url=env_url, headers=headers)
    result = self.client.reset()
    ...
```

The old session is released asynchronously, which is why `MAX_CONCURRENT_ENVS` is 96 rather than 64.

Since [#1536](https://github.com/openenv/OpenEnv/pull/1536), the client no longer reconnects silently in the middle of an episode. It raises `SessionLostError` and the next `reset()` starts a new session. A dropped game then fails instead of being played on the wrong board, and a new connection per game still keeps the drops away from training.

## 8. Evaluate and make the GIF

`evaluate.py` plays a model greedily on a puzzle file and saves every game. It runs the environment in-process and sets the board itself, so every model plays the same puzzles:

```bash
uv run evaluate.py --model <user>/Qwen3-1.7B-Sudoku-SFT-GRPO --thinking --max-new-tokens 384 \
    --puzzles puzzles_c55.json --out grpo_c55.jsonl
uv run summarize.py base_c55.jsonl sft_c55.jsonl grpo_c55.jsonl
uv run make_gif.py base_c55.jsonl sft_c55.jsonl grpo_c55.jsonl --labels "Qwen3-1.7B" "+ SFT" "+ SFT + GRPO" \
    --skip 6 --games 6 --out progress.gif
```

A trained model takes up to 9 minutes per puzzle on a laptop (Apple silicon), since solved games are long. On an `l4x1` job, 30 puzzles take under an hour:

```bash
hf jobs uv run --flavor l4x1 --timeout 4h evaluate.py --model <user>/Qwen3-1.7B-Sudoku-SFT-GRPO \
    --thinking --max-new-tokens 384 --puzzles puzzles_c55.json --out /dev/stdout
```

A game ends on two rejected moves in a row or after 40 moves, the same budget as `max_tool_calling_iterations` in GRPO. The games are printed as JSON lines in the job logs (`hf jobs logs <job_id> | grep '^{"model"' > grpo_c55.jsonl`).

The base model runs without thinking (`--max-new-tokens 256`). With thinking on and the same 384-token budget, it never finished reasoning before the limit and made no move at all.

## Going further

- The [Wordle tutorial](wordle-grpo) follows the same recipe with shorter games and a shaped reward.
- The SFT reasoning only uses single candidates. Puzzles with more empty cells need other techniques, so a richer solver is the next thing to try at harder levels.
- Sending only the changed cell instead of the whole grid would cut the game length several times, and with it the cost of GRPO.
