# Rubrics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/rubrics.ipynb)

Rubrics are OpenEnv's first-class abstraction for computing rewards. They let you build multi-criteria reward functions from small reusable pieces. This tutorial walks through the API end-to-end, from a one-line rubric to a full environment that introspects its reward signal at training time.

## Why Rubrics?

Before rubrics, each environment rolled its own reward logic. Three pain points surfaced repeatedly:

1. **No standard interface**. Every environment author invented their own `compute_reward(...)` shape, so reusing a reward component across environments meant copy-pasting.
2. **Multi-criteria evaluation was ad-hoc**. "Code must compile, tests must pass, style matters a bit" becomes a tangle of nested `if`/`else` and hand-rolled weighted averages. There was no consistent way to ask *which* criterion caused a low reward.
3. **LLM judges and sandboxed checks are slow**. Without a framework-level concept of "reward component", batch evaluation couldn't parallelise the I/O-bound pieces.

The Rubric API is small: you subclass, implement `forward`, and the framework gives you composition, introspection, and parallel evaluation for free.

## Your First Rubric

A rubric is a callable with a `forward(action, observation) -> float` method.

```python
from openenv.core.rubrics import Rubric


class MessageLengthRubric(Rubric):
    """Reward 1.0 if the message is 5–20 characters long, else 0.0."""

    def forward(self, action, observation) -> float:
        length = len(action.message)
        return 1.0 if 5 <= length <= 20 else 0.0
```

That's the whole contract. Instantiate it and call it:

```python
rubric = MessageLengthRubric()
score = rubric(action, observation)   # runs forward + hooks
print(rubric.last_score)              # latest score is cached on the rubric
```

`Rubric.__call__` runs pre- and post-hooks around your `forward`, caches the result on `self.last_score`, and supports async `forward` implementations transparently. (If you've used `torch.nn.Module`, the subclass-and-implement-`forward` pattern will feel familiar — children assigned as instance attributes auto-register with the parent.)

### Optional hooks for observability

You can attach hooks without subclassing — useful for logging every component's score without polluting `forward`. Post-hooks run after `forward` completes and see the returned score; pre-hooks run before `forward` and are handy for input validation or instrumentation. When a rubric is async, hooks are awaited transparently.

```python
def log_score(rubric, action, obs, result):
    print(f"{type(rubric).__name__}: {result:.2f}")

rubric.register_forward_hook(log_score)              # fires after forward()
rubric.register_forward_pre_hook(lambda r, a, o: None)  # fires before forward()
```

### State dict

Rubrics implement `state_dict()` / `load_state_dict(state)` so their configuration (thresholds, prompt templates, etc.) can be serialised alongside model checkpoints. The default implementations return an empty dict — override them when your rubric has tunable parameters.

## Composing Rubrics

The real power shows up when you stack rubrics. `openenv.core.rubrics` ships with four containers.

### `WeightedSum` — multi-criteria averaging

Use when several independent criteria each contribute to the final score.

```python
from openenv.core.rubrics import WeightedSum


class TestsPassRubric(Rubric):
    def forward(self, action, observation) -> float:
        return observation.tests_passed / max(observation.tests_total, 1)


class StyleRubric(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0 if action.code.count("\n\n\n") == 0 else 0.6


reward = WeightedSum(
    [TestsPassRubric(), StyleRubric()],
    weights=[0.7, 0.3],
)
```

Weights must sum to `1.0`. `WeightedSum` evaluates its children with `asyncio.gather` when any of them is async, so an LLM-backed child does not block the synchronous ones.

### `Gate` — hard constraints

Use when a child score below a threshold should short-circuit the reward to zero.

```python
from openenv.core.rubrics import Gate

reward = Gate(TestsPassRubric(), threshold=0.5)  # 0.0 if fewer than half the tests pass
```

`Gate` returns `0.0` when the child score is below the threshold, and passes the child score through unchanged otherwise.

### `Sequential` — fail-fast pipeline

Use when criteria are ordered: a later criterion only matters if the earlier ones passed. Sequential returns `0.0` the moment any child returns `0.0` and does not evaluate the remaining children — great for gating expensive checks like sandboxed test runs or LLM calls.

```python
from openenv.core.rubrics import Sequential

reward = Sequential(
    Gate(CompilesRubric(), threshold=1.0),   # skip everything if it doesn't compile
    Gate(TestsPassRubric(), threshold=0.5),  # and skip style if tests are failing
    WeightedSum([TestsPassRubric(), StyleRubric()], [0.7, 0.3]),
)
```

### `RubricList` and `RubricDict` — dynamic dispatch

When the right rubric depends on the current observation (e.g. one rubric per game in a multi-game environment), wrap the options in a `RubricList` or `RubricDict` and dispatch in your parent rubric's `forward`.

```python
from openenv.core.rubrics import Rubric, RubricDict


class MultiGameRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.games = RubricDict({
            "pong": PongRubric(),
            "breakout": BreakoutRubric(),
        })

    def forward(self, action, observation) -> float:
        return self.games[observation.game_id](action, observation)
```

`RubricList` and `RubricDict` do not aggregate on their own — calling them directly raises. Their job is auto-registration (so their children show up in `named_rubrics()`) and indexed access. Reach for them when the parent rubric needs to pick a child *at runtime* based on the observation — if the set of children is fixed, plain attributes are simpler.

### Introspection: `named_rubrics()`

Assigning a child rubric as an attribute auto-registers it with the parent. Training code can then walk the tree:

```python
composite = WeightedSum(
    [Gate(CompilesRubric(), 1.0), TestsPassRubric(), StyleRubric()],
    [0.2, 0.5, 0.3],
)

for name, child in composite.named_rubrics():
    print(f"{name:30s} last_score={child.last_score}")
```

After running the composite once, every component's most recent score is cached on `last_score` — no manual bookkeeping.

## LLM-as-judge: `LLMJudge`

When a criterion is too subjective for a handwritten heuristic ("is this argument persuasive?", "is this explanation clear?"), use an LLM as the judge. `LLMJudge` wraps an `LLMClient` with a prompt template and a score extractor.

Any OpenAI-compatible endpoint works: hosted OpenAI / Anthropic, or open-weight models served through vLLM, Ollama, Hugging Face Inference Providers, etc. Pick a client and hand it to `LLMJudge`:

```python
import os

from openenv.core.llm_client import OpenAIClient, create_llm_client
from openenv.core.rubrics import LLMJudge

# Option 1 — hosted OpenAI (the factory also supports "anthropic").
client = create_llm_client(
    "openai",
    model="gpt-4.1-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Option 2 — open-weight model served via a local OpenAI-compatible endpoint
# (vLLM, Ollama, Hugging Face Inference Providers, …). Point OpenAIClient
# at the base URL and the model id the server exposes. `api_key` is optional
# and defaults to "not-needed" for local endpoints.
client = OpenAIClient(
    endpoint="http://localhost",
    port=8000,
    model="Qwen/Qwen3-1.7B",
)

clarity_judge = LLMJudge(
    client=client,
    prompt_template=(
        "Rate the clarity of this explanation on a 0-10 scale. "
        "Reply with the number only.\n\n"
        "Explanation:\n{action}\n"
    ),
    score_pattern=r"(\d+(?:\.\d+)?)",
    normalize=True,   # clamps extracted score to [0, 1]
)
```

`LLMJudge.forward` is async. When you put it inside `WeightedSum` or `Sequential`, the container awaits it transparently. A few caveats worth stating up front:

- **Cost and latency** scale with the number of episodes and the number of rubric calls per step. `Sequential` + `Gate` earlier in the pipeline is the usual answer.
- **Determinism** is not free. Cache scores when you can, and consider temperature 0 for repeatable eval runs.
- **API keys** belong in environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …), not in code that ships to the Hub.

## Delayed Rewards: `TrajectoryRubric`

Some signals only materialise at the end of an episode — chess win/loss, unit-test suite success, a goal reached after many steps. `TrajectoryRubric` accumulates `(action, observation)` pairs internally and only invokes your scoring logic on the terminal observation.

```python
from openenv.core.rubrics import TrajectoryRubric


class WinLossRubric(TrajectoryRubric):
    def score_trajectory(self, trajectory) -> float:
        _, final_obs = trajectory[-1]
        return final_obs.reward   # +1 win, -1 loss, 0 draw

    def compute_step_rewards(self):
        # Credit assignment: distribute the final score across steps however you like.
        final = self.score_trajectory(self._trajectory)
        return [final] * len(self._trajectory)
```

`forward(action, obs)` returns `intermediate_reward` (default `0.0`) until `observation.done` is `True`, then calls `score_trajectory`. After the episode ends, call `rubric.compute_step_rewards()` to get one reward per step — same length as the trajectory. This is the hook for credit assignment: training code feeds these per-step rewards back into advantage estimation, return-to-go, or whatever your optimizer expects. `ExponentialDiscountingTrajectoryRubric` precomputes `gamma^(T-1-t) * final_score` for you; override `compute_step_rewards` in your subclass if you want a different strategy (all-to-last, equal split, task-specific shaping).

> [!CAUTION]
> If `observation.done` never becomes `True`, `score_trajectory` is never called and the trajectory grows unbounded in memory. Make sure `step` flips `done` on every terminal transition, and call `self._reset_rubric()` in `Environment.reset` so trajectories do not leak across episodes.

For the common exponentially-discounted case, subclass `ExponentialDiscountingTrajectoryRubric` instead and only implement `score_trajectory`:

```python
from openenv.core.rubrics import ExponentialDiscountingTrajectoryRubric


class ChessOutcomeRubric(ExponentialDiscountingTrajectoryRubric):
    def score_trajectory(self, trajectory) -> float:
        _, final_obs = trajectory[-1]
        return final_obs.reward    # already +1 / 0 / -1 from the engine
```

This is exactly the pattern the built-in `envs/chess_env/` uses — see `envs/chess_env/server/rubrics.py` for the complete real-world example.

> [!CAUTION]
> The `TrajectoryRubric` keeps the trajectory in CPU memory. If your observation carries GPU tensors (images, embeddings), detach and move them to CPU before returning from `step()` — otherwise the trajectory holds onto GPU memory across the whole episode.

## Wiring a Rubric into an `Environment`

Rubrics are **server-side**. Each environment declares its rubric in `__init__`, and `step` runs it via the `_apply_rubric` helper. The base `Environment` class accepts the rubric through its constructor and stores it as `self.rubric`.

Here is a complete minimal environment that composes a `Sequential` gate-then-`WeightedSum` pipeline and exposes the reward through its observation:

```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum


class CodeAction(Action):
    code: str


class CodeObservation(Observation):
    compiles: bool = False
    tests_passed: int = 0
    tests_total: int = 0


class CodeState(State):
    attempts: int = 0


class CompilesRubric(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0 if observation.compiles else 0.0


class TestsPassRubric(Rubric):
    def forward(self, action, observation) -> float:
        if observation.tests_total == 0:
            return 0.0
        return observation.tests_passed / observation.tests_total


class StyleRubric(Rubric):
    def forward(self, action, observation) -> float:
        return 1.0 if action.code.count("\n\n\n") == 0 else 0.6


def build_code_rubric() -> Rubric:
    return Sequential(
        Gate(CompilesRubric(), threshold=1.0),  # gate everything on compilation
        WeightedSum(
            [
                TestsPassRubric(),
                StyleRubric(),
            ],
            weights=[0.7, 0.3],
        ),
    )


class CodeEnvironment(Environment[CodeAction, CodeObservation, CodeState]):
    def __init__(self):
        super().__init__(rubric=build_code_rubric())
        self._state = CodeState()

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeObservation:
        self._reset_rubric()               # clear any trajectory / cached last_score
        self._state = CodeState()
        return CodeObservation()

    def step(self, action: CodeAction, timeout_s=None, **kwargs) -> CodeObservation:
        self._state.attempts += 1
        obs = self._run_code(action)       # your domain-specific execution
        obs.reward = self._apply_rubric(action, obs)
        return obs

    @property
    def state(self) -> CodeState:
        return self._state

    def _run_code(self, action: CodeAction) -> CodeObservation:
        # Placeholder for whatever your environment actually does.
        compiles = "def " in action.code
        return CodeObservation(
            compiles=compiles,
            tests_passed=3 if compiles else 0,
            tests_total=3,
        )
```

The three pieces the base class expects from you:

1. **Pass the rubric to `super().__init__(rubric=...)`** so `self.rubric` is set.
2. **Call `self._reset_rubric()` from `reset`** so trajectory state does not leak between episodes.
3. **Call `self._apply_rubric(action, obs)` from `step`** and attach the result to `obs.reward`. There is also `_apply_rubric_async` for `step_async`.

> [!NOTE]
> Some environments already compute `obs.reward` from game mechanics or a handcrafted multi-component signal (see `envs/chess_env/` and `envs/carla_env/`). In that case, call `self._apply_rubric(action, obs)` without assigning its return value — the rubric still accumulates the trajectory for `compute_step_rewards()` and still exposes per-component scores via `named_rubrics()`, but `obs.reward` stays authoritative.

### Inspecting rewards from training code

Because children are auto-registered, the training loop can walk the rubric tree and log component-level diagnostics without the environment exposing a custom API:

```python
env = CodeEnvironment()
obs = env.reset()
obs = env.step(CodeAction(code="def solution(): return 42"))

for name, component in env.rubric.named_rubrics():
    print(f"{name:30s} last_score={component.last_score:.2f}")
```

That snippet works for *any* OpenEnv environment that sets `self.rubric`, regardless of whether the rubric is a single scalar or a deeply nested composition.

### Where the reward ends up during training

Training frameworks consume the reward through the same channel as any other OpenEnv observation field: `step()` returns an `Observation` whose `reward` is the rubric's output, and the client delivers it via `result.reward`.

With [TRL](https://huggingface.co/docs/trl/main/en/openenv), the recommended path is `GRPOTrainer`'s `environment_factory`: you define a thin wrapper class with tool methods that call the OpenEnv client, store `self.reward = result.observation.reward` after each step, and a plain reward function reads it off the `environments` parameter. The [TRL OpenEnv integration guide](https://huggingface.co/docs/trl/main/en/openenv) has the full recipe, and TRL's [`examples/`](https://github.com/huggingface/trl/tree/main/examples) folder ships ready-to-run OpenEnv scripts (`grpo_wordle`, `grpo_catch`, `grpo_sudoku`, ...). The same observation shape works with [torchforge](https://github.com/pytorch-labs/torchforge) and other OpenEnv-compatible training stacks.

`named_rubrics()` is orthogonal: use it to **log per-component scores** (to Weights & Biases, TensorBoard, trackio, …) while training, without changing the reward the optimiser sees.

## Using Rubrics for Evaluation

A rubric is just a callable — nothing forces you to run it inside a training loop. Drop it into a for-loop over a static dataset and you have a multi-criteria scoring function for offline eval:

```python
rubric = build_code_rubric()

scores = []
for action, obs in eval_dataset:
    scores.append(rubric(action, obs))

print(f"mean reward: {sum(scores) / len(scores):.3f}")
for name, component in rubric.named_rubrics():
    print(f"  {name:30s} last_score={component.last_score:.3f}")
```

The same rubric object used to compute training rewards doubles as the eval metric — one source of truth for "what is a good response". Per-component `last_score` gives you a per-criterion breakdown for free (useful for regression dashboards and failure analysis). When a component like `LLMJudge` is async, wrap the loop with `asyncio.run(...)` and `await rubric(action, obs)` so the judge calls can overlap.

## Next Steps

- **Real-world trajectory example** — walk through `envs/chess_env/server/rubrics.py` and `chess_environment.py` to see `ExponentialDiscountingTrajectoryRubric` wired into a game environment.
- **Design details** — [RFC 004](https://github.com/huggingface/OpenEnv/blob/main/rfcs/004-rubrics.md) covers the rationale for the composable API and the "rewards inside the environment" invariant.
- **Reward design basics** — the [Rewards](../guides/rewards.md) guide covers sparse-vs-dense signals and common pitfalls that still apply on top of any rubric composition.
- **Training loop integration** — see [Training with OpenEnv](../guides/training.md) and the [TRL OpenEnv integration guide](https://huggingface.co/docs/trl/main/en/openenv) for the recommended `environment_factory` pattern.
