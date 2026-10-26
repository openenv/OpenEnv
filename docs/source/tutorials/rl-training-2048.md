# Play 2048 with GRPO

Train a language model to play 2048 with GRPO. The game runs in the OpenSpiel environment, and TRL's `GRPOTrainer` drives it through `environment_factory`: the model plays by calling a `move` tool, and the game score is the reward.

The recipe follows TRL's [`examples/grpo_2048`](https://github.com/huggingface/trl/tree/main/examples/grpo_2048), which implements the game in plain Python. Here the game logic lives in an OpenEnv server instead.

> [!TIP]
> Unsloth maintains its own [2048 tutorial with OpenEnv](https://github.com/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_%2820B%29_Reinforcement_Learning_2048_Game.ipynb), which trains `gpt-oss-20b` with Unsloth on the same OpenSpiel server to write a 2048 strategy instead of playing move by move.

> [!NOTE]
> **Difficulty**: Advanced | **GPU Required**: Yes

## 1. Install dependencies

```python
!pip install -Uq "trl[peft]" trackio open_spiel
!pip install -q "openenv-openspiel-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/openspiel_env"
```

`open_spiel` ships the game engine (wheels for Linux and macOS). `openenv-openspiel-env` installs the client (`OpenSpielEnv`) and the server.

## 2. Start the 2048 server

An OpenSpiel server serves one game, chosen with the `OPENSPIEL_GAME` variable when it starts. Run one locally for 2048:

```python
import os, subprocess, sys, time

import requests

server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "openspiel_env.server.app:app", "--port", "8000"],
    env={**os.environ, "OPENSPIEL_GAME": "2048"},
)

for _ in range(60):
    try:
        if requests.get("http://localhost:8000/health").ok:
            break
    except requests.ConnectionError:
        time.sleep(1)
```

The hosted [`openenv/openspiel_env`](https://huggingface.co/spaces/openenv/openspiel_env) Space serves Catch. To host 2048 instead, duplicate it, set the `OPENSPIEL_GAME=2048` variable in the Space settings, and use its `https://<user>-openspiel-env.hf.space` URL as `base_url` below.

The server accepts 8 concurrent sessions by default (`MAX_CONCURRENT_ENVS`), one per rollout in a generation batch with the settings below.

## 3. Define the environment class

`environment_factory` takes a class the trainer instantiates once per rollout. Its public methods with docstrings become tools, `reset()` returns the start of the conversation, and `get_reward()` scores the finished rollout.

OpenSpiel's 2048 observation is the board as 16 tile values (`info_state`), the step reward is the points gained by merging tiles, and the actions are `0` (up), `1` (right), `2` (down) and `3` (left).

```python
from openspiel_env import OpenSpielAction, OpenSpielEnv

PROMPT = "Play 2048 on a 4x4 board. Use the tool `move` with one of: up, down, left, right. Maximize the score."
MOVES = {"up": 0, "right": 1, "down": 2, "left": 3}


class Game2048Env:
    def __init__(self):
        self.client = OpenSpielEnv(base_url="http://localhost:8000").sync()

    def reset(self, **kwargs) -> str:
        result = self.client.reset()
        self.observation = result.observation
        self.score = 0.0
        self.done = result.done
        return f"{PROMPT}\n\n{self._render()}"

    def move(self, direction: str) -> str:
        """
        Play one move in 2048.

        Args:
            direction: One of "up", "down", "left", "right".

        Returns:
            The board and score after the move.
        """
        if self.done:
            raise ValueError("Game over.")
        action_id = MOVES.get(direction.strip().lower())
        if action_id not in self.observation.legal_actions:
            raise ValueError(f"Illegal move: {direction}.")
        result = self.client.step(OpenSpielAction(action_id=action_id, game_name="2048"))
        self.observation = result.observation
        self.score += result.reward or 0.0
        self.done = result.done
        return self._render()

    def get_reward(self) -> float:
        return self.score

    def _render(self) -> str:
        tiles = [int(v) for v in self.observation.info_state]
        rows = [" ".join(f"{v:4d}" for v in tiles[i : i + 4]) for i in range(0, 16, 4)]
        return f"score={self.score}\n" + "\n".join(rows) + f"\ndone={self.done}"
```

An error raised inside a tool is sent back to the model as the tool response, so an illegal move costs a turn without ending the game.

## 4. Train with GRPO

The environment generates the starting board and owns the reward, so the trainer needs no dataset and no reward function. `max_steps` sets the length of the run.

```python
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

trainer = GRPOTrainer(
    model="Qwen/Qwen3-4B",
    args=GRPOConfig(
        max_steps=200,
        chat_template_kwargs={"enable_thinking": False},
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        report_to="trackio",
        trackio_space_id="openenv-2048",
        max_completion_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
    ),
    environment_factory=Game2048Env,
    peft_config=LoraConfig(),
)
trainer.train()
```

Each generation batch holds `per_device_train_batch_size × gradient_accumulation_steps = 8` rollouts, all from the same starting prompt (`num_generations` defaults to 8). The trainer keeps one `Game2048Env` instance per rollout and reuses them across batches. A rollout ends when the model stops calling `move` or reaches `max_completion_length`, so longer completions let the model play longer games.

In the Trackio dashboard, `reward` (also logged as `rewards/Game2048Env/mean`) is the mean game score and `tools/call_frequency` is how often the model calls `move`. Expect the call frequency to rise first, then the score.

## 5. Save the model

```python
trainer.push_to_hub()
server.terminate()
```

## Next steps

- The [end-to-end walkthrough](end-to-end-walkthrough) covers the same `environment_factory` pattern on a single-step task, and [Wordle GRPO](wordle-grpo) on another multi-turn game.
- The [Training overview](../guides/training) lists the other frameworks that train on OpenEnv environments.
