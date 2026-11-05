# Train a Wordle Agent: SFT, then GRPO

This tutorial trains [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) to play [Wordle](https://en.wikipedia.org/wiki/Wordle) against the OpenEnv TextArena environment. The model starts out unable to win a single game. A short supervised warmup teaches it the game, and in our run GRPO then doubled its wins and removed most of its invalid guesses. Every step runs on one GPU, and the whole pipeline costs about $6 on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs).

![Base model, SFT and SFT + GRPO playing the same 30 words](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/wordle-progress.gif)

| Model | Wins on 30 words | Guesses per win | Invalid guesses |
|---|---|---|---|
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0/30 | - | 24 |
| + SFT ([model](https://huggingface.co/openenv/Qwen3-1.7B-Wordle-SFT)) | 10/30 | 3.2 | 41 |
| + SFT + GRPO ([model](https://huggingface.co/openenv/Qwen3-1.7B-Wordle-SFT-GRPO)) | 20/30 | 4.1 | 3 |

These numbers come from our run, with one training run per step, evaluated greedily on the same 30 words ([`secrets30.txt`](https://github.com/openenv/OpenEnv/blob/main/examples/wordle/secrets30.txt)). Thirty words is a small sample and training is not deterministic, so a new run will land a few wins away. Two SFT runs with the same recipe won 10 and 13 games, for example. Expect the same ordering (base, SFT, then SFT + GRPO clearly ahead) rather than the exact numbers.

The GRPO run used an earlier version of the invalid-move penalty, which missed the invalid move that ends a game, and capped rollouts at six tool rounds, including invalid warnings. The current `grpo.py` counts terminal invalid moves and allows up to 12 tool rounds and 12 submissions, while the environment still ends games after six legal guesses. The reported results have not been regenerated with this corrected budget.

The 30 words come from the 156 nouns TextArena draws its secrets from, the same list used for the SFT games and during GRPO. So the table measures how well each model plays the words of this environment. We did not test words outside that list.

Watch all three play any of the 30 words in the [Wordle Arena](https://huggingface.co/spaces/openenv/wordle-arena) Space, and see the training curves on [Trackio](https://huggingface.co/spaces/openenv/wordle-runs). The scripts are in [`examples/wordle`](https://github.com/openenv/OpenEnv/tree/main/examples/wordle).

## What you need

| Step | Script | Hardware | Time | Cost |
|---|---|---|---|---|
| 1. Generate SFT data | `generate_sft_data.py` | Laptop CPU | 10 s | $0 |
| 2. SFT warmup | `sft.py` | 1x A100 80GB (`a100-large`) | 7 min | $0.30 |
| 3. GRPO | `grpo.py` | 1x A100 80GB (`a100-large`) | 2 h 15 min | $5.60 |
| 4. Evaluate and make the GIF | `evaluate.py`, `make_gif.py` | Laptop (Apple silicon) or any GPU | 20 min for three models | $0 |

Times and costs are from our runs, at the `a100-large` price on Hugging Face Jobs ($2.50/h), and include the time to install packages and load the model. You need a Hugging Face account with a payment method for Jobs, and [uv](https://docs.astral.sh/uv/). The scripts pin their dependencies (`trl==1.16.0`, `openenv==0.10.0`, vLLM 0.31), so `uv run` and `hf jobs uv run` install the same versions used here.

## 1. Play the environment

The Wordle environment runs as a Space, [`openenv/wordle`](https://huggingface.co/spaces/openenv/wordle). Open its playground to play a game by hand, or connect from Python:

```python
from textarena_env import TextArenaAction, TextArenaEnv

with TextArenaEnv(base_url="https://openenv-wordle.hf.space").sync() as env:
    result = env.reset()
    print(result.observation.messages[0].content)
    result = env.step(TextArenaAction(message="[crane]"))
    print(result.observation.messages[0].content)
```

Install the client with `pip install "openenv-textarena @ git+https://github.com/openenv/OpenEnv.git@v0.10.0#subdirectory=envs/textarena_env"`.

Each guess returns the feedback for every letter: `G` (right letter, right spot), `Y` (in the word, wrong spot) and `X` (not in the word). A repeated word or a word that isn't English is an invalid move. The first one is a warning, and a second one in the same turn ends the game. A win ends with `Congratulations! You guessed the word correctly!`, and a loss shows the secret word.

The environment's own reward is `1.0` for a win. A lost game scores (greens + half the yellows) / 5 for its last guess. A game that never reaches its end scores `0`.

## 2. Why start with SFT

Training Qwen3-1.7B with GRPO directly does not work well, for two reasons we saw in practice:

- **The base model almost never wins.** It always opens with `apple` and soon repeats a word, so it wins around 2-3% of games while sampling. GRPO learns from the difference between good and bad games in a group, and with almost no wins there is little to learn from. A 25-step run from the base model stayed at a 2-5% win rate.
- **The environment reward can be gamed.** A game only scores once it ends, and a win pays the same after 1 guess or 6. With Qwen3's long thinking, the easiest way to score more was to think less and use all 6 guesses. In one run, reasoning dropped from 7,000 to 2,300 tokens in 13 steps while the reward fell below where it started.

So the recipe starts with a short SFT on well-played games, which teaches the format and a reasonable strategy. GRPO then trains against the environment, with a reward that pays for winning fast, and fixes what the SFT model still gets wrong.

## 3. Generate the SFT data

`generate_sft_data.py` plays 1,000 games with a solver, against the same environment class the Space runs (in-process, so it takes seconds). Each turn the solver keeps the words that fit all the feedback so far and picks the guess that splits them best, and writes a short reasoning before each guess:

```text
ARISE gave X X X Y Y: A gray, R gray, I gray, S yellow, E yellow.
In the word but misplaced: E (not position 5), S (not position 4). Ruled out: A, I, R.
Words that fit: SLEEP, SHEEP, SMELL, STEEL, SHELF, and more.
SHEEP fits the clues and its letters split the remaining options best, so I'll guess it.
```

Each game is stored as a tool-calling conversation: the user prompt, then one assistant turn per guess (the reasoning as `reasoning_content` and a call to the `guess` tool), each followed by the environment feedback. This is the same format TRL renders during GRPO, so the SFT model plays exactly the way RL will train it.

```bash
uv run generate_sft_data.py --games 1000 --push-to-hub <user>/wordle-tool-calls-sft
```

The dataset used here is [`openenv/wordle-tool-calls-sft`](https://huggingface.co/datasets/openenv/wordle-tool-calls-sft). TextArena draws the secret from 156 common nouns, so the solver wins every game, in 2.7 guesses on average.

## 4. SFT warmup

`sft.py` fine-tunes Qwen3-1.7B on those games for one epoch (63 steps), training only the assistant turns:

```bash
hf jobs uv run --flavor a100-large --timeout 1h --secrets HF_TOKEN sft.py \
    --dataset <user>/wordle-tool-calls-sft --hub-model-id <user>/Qwen3-1.7B-Wordle-SFT
```

In our run the loss went from 4.2 to 0.09. Keep the trainable weights in fp32, with bf16 mixed-precision compute. When we loaded the weights in bf16 for training, the model came out clearly worse (final loss 0.21), because bf16 weights lose the small updates. The script sets `model_init_kwargs={"dtype": "float32"}` explicitly (also the default in TRL 1.16.0) and keeps `bf16=True` for compute. It casts the weights to bf16 after training, just before the final upload.

## 5. GRPO

`grpo.py` starts from the SFT model and trains it against the environment with TRL's `GRPOTrainer` and `environment_factory`. The environment class exposes one tool, `guess`, and keeps what the reward needs:

```python
def reward_win(environments, **kwargs):
    return [1.0 + 0.5 * (1 - env.num_guesses / 6) if env.won else 0.0 for env in environments]

def reward_partial(environments, **kwargs):
    return [0.0 if env.won else 0.1 * env.best_greens for env in environments]

def reward_invalid(environments, **kwargs):
    return [-0.1 * env.invalid_moves for env in environments]
```

- A win pays more the faster it comes (1.42 in one guess, 1.0 in six).
- A loss pays for the best guess of the game, whether or not the game reached its end.
- Each invalid move costs 0.1.

TRL logs the three terms separately, so `rewards/reward_win/mean` follows the wins (each one weighted by how fast it came).

Each step plays 4 prompts x 16 games (64 games) with thinking on, and runs 150 steps at a constant learning rate of 1e-6 (in an earlier run with a linear decay to 0, the reward was still rising when the learning rate ran out):

```bash
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN grpo.py \
    --model <user>/Qwen3-1.7B-Wordle-SFT --hub-model-id <user>/Qwen3-1.7B-Wordle-SFT-GRPO \
    --env-url https://<user>-wordle.hf.space
```

Every game is one environment session, so the environment must accept 64 at once. [Duplicate the `openenv/wordle` Space](https://huggingface.co/spaces/openenv/wordle?duplicate=true) and set the `MAX_CONCURRENT_ENVS=64` variable on your copy, rather than training against the shared Space. A private copy works too, since the script sends your token.

![Reward during GRPO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/openenv/wordle-grpo-reward.png)

In our run, the reward (averaged over 10 steps) went from about 0.37 to about 0.58, with a lot of noise from step to step. Most of the gain came from more wins and fewer invalid moves, while the reasoning stayed around 500 tokens per game and entropy stayed stable. A new run will give different numbers, but it should show the same trend.

## 6. Evaluate and make the GIF

`evaluate.py` plays a model greedily and saves every game. With `--secrets`, it runs the environment in-process and sets the secret word itself, so every model plays the same words:

```bash
uv run evaluate.py --model <user>/Qwen3-1.7B-Wordle-SFT-GRPO --thinking --max-new-tokens 512 \
    --secrets "$(cat secrets30.txt)" --out grpo.jsonl
uv run make_gif.py base.jsonl sft.jsonl grpo.jsonl --labels "Qwen3-1.7B" "+ SFT" "+ SFT + GRPO" --out progress.gif
```

The base model runs without thinking (`--max-new-tokens 256`). With thinking it writes thousands of tokens per guess, which is slow on a laptop and, in our tests, didn't help it win.

## Run it in Colab

The [notebook](https://colab.research.google.com/github/openenv/OpenEnv/blob/main/examples/wordle/wordle.ipynb) runs the same pipeline. On a free T4 it loads the published models and plays them against each other, and launches SFT and GRPO on Hugging Face Jobs. On an A100 (Colab Pro) it can train in the notebook itself.

## Going further

- The [SFT warmup tutorial](sft-warmup) shows the same idea with rollouts collected from a stronger model.
- Training against a larger model (Qwen3-4B), more games per step or a richer SFT reasoning (listing every remaining candidate) are the next things to try for a higher win rate.
