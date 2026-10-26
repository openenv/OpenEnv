# Rewards

Good reward signals are the single biggest lever on RL training outcomes — and the hardest thing to get right. This page covers the design principles that apply in OpenEnv and points to the [Rubrics tutorial](../tutorials/rubrics.md) for the concrete implementation API.

## Rewards Live Inside the Environment

The OpenEnv contract is that reward computation stays on the server side, inside `Environment.step`. The `openenv.core.rubrics` module exists to make that computation composable — think `torch.nn.Module` but for reward functions. An environment declares `self.rubric` in its constructor and the base class helper `self._apply_rubric(action, observation)` runs it every step.

See the [Rubrics tutorial](../tutorials/rubrics.md) for the end-to-end walkthrough, including `WeightedSum`, `Gate`, `Sequential`, `LLMJudge`, and the trajectory rubrics used for delayed rewards like chess outcomes.

## Design Principles

The principles below are independent of the rubric mechanics — they apply whether the reward is a single scalar or a composition of a dozen components.

### 1. Start simple

Begin with a sparse success/failure signal. Only shape the reward after you've confirmed the agent can reach the positive signal at all.

```python
class WinLossRubric(Rubric):
    def forward(self, action, observation) -> float:
        if not observation.done:
            return 0.0
        return 1.0 if observation.success else -1.0
```

### 2. Shape carefully

Dense intermediate rewards speed up learning but invite reward hacking. Prefer adding shaping as a small-weighted *component* of a `WeightedSum`, so the dominant signal is still the real outcome.

```python
reward = WeightedSum(
    [WinLossRubric(), ProgressRubric()],
    weights=[0.8, 0.2],
)
```

### 3. Consider density

- **Sparse** rewards (signal only on terminal transitions) are cleaner but slower to learn from.
- **Dense** rewards (signal every step) are faster but can push the agent into local optima that optimise the proxy instead of the goal.

For long-horizon tasks with a delayed outcome, reach for `TrajectoryRubric` and its built-in per-step credit assignment instead of hand-crafting a dense proxy.

## Worked Examples

### Chess — sparse, trajectory-based

```python
class ChessOutcomeRubric(ExponentialDiscountingTrajectoryRubric):
    def score_trajectory(self, trajectory) -> float:
        _, final_obs = trajectory[-1]
        return final_obs.reward   # +1 / 0 / -1 from the engine
```

Rewards accumulate silently during the game, then the final step produces the outcome and `compute_step_rewards()` distributes it back with exponential discounting. This is exactly what `envs/chess_env/` ships with.

### Coding — gated composition

```python
reward = Sequential(
    Gate(CompilesRubric(), threshold=1.0),              # must compile
    Gate(TestsPassRubric(), threshold=0.5),             # must pass at least half the tests
    WeightedSum([TestsPassRubric(), StyleRubric()], [0.7, 0.3]),
)
```

`Sequential` short-circuits to `0.0` the moment any child fails its gate, so expensive style / LLM judge calls never run on broken submissions.

### Text tasks — mixed signal

```python
reward = WeightedSum(
    [WinLossRubric(), PerTurnProgressRubric()],
    weights=[0.7, 0.3],
)
```

Per-turn progress helps the agent explore, but the terminal outcome still dominates the final score.

## Common Pitfalls

1. **Reward hacking.** The agent optimises the proxy instead of the goal. Prefer shaping as a small-weighted component rather than a large intermediate bonus.
2. **Sparse rewards that never fire.** If the agent cannot reach the success signal in practice, training stalls. Measure success rate on a random policy before relying on a sparse reward.
3. **Conflicting signals.** Two criteria pulling in opposite directions produce a flat optimisation landscape. If you catch yourself subtracting rewards to "cancel out" a bad behaviour, consider a `Gate` instead — make the bad case a hard zero.
4. **Component score drift.** Without component-level logging, you won't know which criterion dropped the total. Rubric introspection (`env.rubric.named_rubrics()` → `last_score`) gives you this for free.

## Next Steps

- [Rubrics tutorial](../tutorials/rubrics.md) — full API walkthrough with composable examples.
- [RFC 004](https://github.com/huggingface/OpenEnv/blob/main/rfcs/004-rubrics.md) — design rationale.
- [Training with OpenEnv](training.md) — consume the reward signal in a training loop.
- [Concepts](concepts.md) — where the rubric plugs in.
