# RFC: Task Sampling and In-Training Evaluation

**Status**: In Review
**Created**: 2026-09-26
**Updated**: 2026-11-23
**Revised**: 2026-10-12 — removed a CLI limitation fixed by #1237, updated the RFC 008 reference to #1321, and corrected four small inaccuracies; 2026-10-16 — added results on three more seeds, replaced the token-weighted formula, scoped the first evaluation to environments whose turns the trainer drives, listed the gaps in the collect pipeline, offered the curriculum as an example, linked the trial's code and logs, and stated how the token weights treat unseen keys and many keys; 2026-10-22 — after review, removed weighting by cost and the `cost` argument from the curriculum, stated the two hooks a sampler needs in a trainer and their limits in TRL, added `metrics()` to the protocol, left shares across sources to the trainer, reported how far the signal formula over-predicts, cited prior work on rollout allocation and filtering, narrowed the reference experiment to uniform sampling against the curriculum on five new seeds, answered open question 6 and proposed a direction for open question 5; 2026-10-28 — removed the prime-rl wrapper sketch and the TRL dataset path, as the approving review suggested, changed how samplers treat a rollout the environment could not score (a reward of None), a rule RFC 014's `posttrain_env` relies on, so that it counts in groups, episodes and cost but in nothing estimated from rewards, added open question 10, and stated how evaluation counts each case: such a rollout, and one lost to a failure of the infrastructure, leaves `success_rate`, `mean_reward` and `pass@k` and is reported separately, a failure of the policy gets the environment's own score or counts as not solved, `mean_reward` is read only together with `error_rate`, and checkpoint comparison lists the tasks it leaves out; 2026-11-15 — reported the reference experiment (the curriculum ahead of uniform sampling on all five seeds, by 8.2 points per generated token and 7.2 per episode on average) and linked its code, logs, reading rules and analysis, replaced the example's untrained scores and overlap counts with those of the reference runs, stated that the endpoint must serve the weights of the step being evaluated and how evaluation counts an episode the harness ends after a step that returned a reward, aligned `EvalSchedule`, `SequentialSampler` and `id_key` with the implementation, and updated huggingface/trl#7609 (merged), #1156 (closed) and Harbor's `reset`; 2026-11-23 — aligned the interfaces with the code built for every rollout step: `EnvEvalHarness` takes the policy as a model step and writes the records of `openenv collect`, its reproducibility check and the line between infrastructure and policy failures are stated, `TaskAPIClient` takes its options by keyword, the samplers' signal rule is a tolerance, `openenv eval` keeps only collect's endpoint flags, the collect changes of rollout step 5 are named, a text answer's reading rule is stated, and tasks are read with `iter_tasks`
**Authors**: @Matheus-F-Scatolin, @tiagoperrupato
**RFC ID**: 013

## Summary

This RFC adds two trainer-side capabilities on top of the existing Task API: **task sampling**, which decides which task each episode runs, and **in-training evaluation**, which measures a policy checkpoint on a held-out split in a reproducible way. Together they turn "train on an environment" into a closed loop: evaluate a baseline, train, and evaluate the same fixed test set at regular steps, with the same environment and the same tool interface throughout.

It introduces a small `TaskSampler` protocol with two implementations in core (`SequentialSampler`, `UniformSampler`), a generic Task API client helper, an `EnvEvalHarness` that implements the existing `EvalHarness` base class with the same pieces as the `openenv collect` rollout pipeline, and an `openenv eval` CLI command. A third sampler, `CurriculumSampler`, is offered as an example; whether it belongs in core is a question for the maintainers. The environment boundary does not change: environments keep computing rewards, and samplers only consume them.

The protocol and the evaluation harness are the core of the proposal and are useful with any sampler. The curriculum is one pluggable sampler, and we measured it on a laptop (see [Validation Plan](#validation-plan)). In a reference experiment on five new seeds, with the test and the reading rules written down before it ran, a curriculum that targets GRPO signal reached higher test accuracy than uniform sampling on every seed: on average by 8.2 points at the same number of generated tokens and by 7.2 at the same number of environment episodes (95% intervals +1.9 to +14.6 and +1.0 to +13.3), with one 1B model and one task family. Before it, a single-seed pilot and a trial on three more seeds had found the curriculum ahead per episode on all four seeds, and per generated token on the three new ones. The pilot's seed had pointed the other way per token, because its curriculum run happened to produce much longer answers; on the other seeds, answer length did not follow the method. Dividing the curriculum's weights by the token cost of each tier, as the first version of this RFC proposed, made it worse, and a corrected form matched the curriculum that leaves cost out per generated token, the unit it was meant to improve, while being slightly ahead of it per episode. So the `CurriculumSampler` weights by signal only; the samplers proposed here count the cost the trainer reports and report signal per episode and per unit of that cost.

## Motivation

### Problem Statement

**1. Task selection has no shared mechanism.** The Task API (`list_splits`, `list_tasks`, `num_tasks`, `get_task`, `get_task_range`, and `reset(split=..., index=...)`) lets a dataset-backed environment publish its tasks, but it is discovery only. Nothing in OpenEnv decides which task an episode should use. Harbor and the wrappers generated by `openenv import` implement it; no hand-written environment under `envs/` does, including the dataset-backed `finqa_env` and the procedurally generated `reasoning_gym_env`. The core clients deliberately ship no Task API methods, because task specs are environment-specific (Task API guide), so each environment client writes its own HTTP helpers; `HarborEnv` has its own, and the guide shows the same pattern for a LaTeX OCR client.

Today, selecting tasks well is manual. The Harbor harness takes an explicit `indices` list, and its code explains why: a task that every generation solves and one that none solves both give `reward_std` 0, and "the band that splits has to be chosen per model" (`envs/harbor_env/harness.py`). This RFC turns that manual choice into a sampler that measures the band as training runs.

**2. Small models waste most of their GRPO rollouts.** GRPO computes each completion's advantage relative to the other completions in its group. When all `G` completions for a prompt receive the same reward, every advantage is zero and the group contributes no gradient. For a binary reward with success probability `p`, the chance that a group of size `G` carries any signal is

```
P(signal) = 1 - p^G - (1 - p)^G
```

which is maximal at `p = 0.5` and collapses as `p` approaches 0 or 1. In our pilot, 80% of GRPO groups carried no gradient under uniform sampling (Llama-3.2-1B, `G = 8`). DAPO's dynamic sampling filters these groups after generation; a sampler that targets tasks near `p = 0.5` generates fewer of them in the first place. What a wasted group costs depends on the environment: in a single-turn reasoning task it is mostly generated tokens, while in a sandboxed or tool-using environment it is mostly the episode itself (container time, tool calls, API calls). A trainer has to see that cost next to the signal, so every sampler here reports both.

**3. There is no shared way to evaluate a checkpoint on an environment.** `openenv.core.evals` provides the abstract `EvalHarness` and an Inspect AI wrapper. There is no shared way to run a checkpoint over an environment's `test` split with fixed seeds, aggregate metrics, and compare two checkpoints: `examples/claude_code_harness_eval/run_eval.py` and the SFT warm-up tutorial (`evaluate_model`) each write their own loop. Reproducibility is also fragile: environments have accepted `seed` in `reset()` and discarded it (`textarena_env` in #1102, and `finqa_env` until #1252 fixed it), so two evaluation runs of the same checkpoint can see different tasks.

RFC 000 schedules evals for phase three (0.6 to 0.9), starting with "simple evals that can be scored by a script". This RFC proposes that starting point, built so that training and evaluation share one interface (design principle 1: minimize lifecycle deltas).

### Goals

1. A trainer can choose tasks for each episode through one small protocol, independent of the training framework.
2. A reproducible evaluation of any policy reachable through an OpenAI-compatible endpoint, returning the existing `EvalResult` type. The first version covers environments that implement the Task API and whose episodes the trainer drives turn by turn; environments whose server runs a whole agent in one call, such as Harbor, are open question 8.
3. A way to run that evaluation every N training steps and compare checkpoints task by task.
4. A curriculum sampler, offered as an example, that raises the number of GRPO groups with non-zero advantage per episode, with no environment changes beyond publishing tasks. Every sampler proposed here reports those groups per episode and per unit of the cost the trainer reports (for example generated tokens).
5. A reference experiment that runs on a single laptop-class machine and reports results per episode and per generated token.

### Non-goals

- Owning GRPO or any trainer implementation (consistent with the non-goals of the draft recipes RFC, #924).
- LLM-judge or multi-agent evals; this RFC covers evals scored by the environment's own reward.
- Changing `reset`, `step`, or `state` signatures, or the wire types.
- Exposing task selection to the agent. Samplers run in orchestration code only.
- Setting shares across environments or data sources, or filtering groups after generation. Both belong to the trainer and its rollout scheduler; a sampler works inside one source's share (see Trainer integration).
- Claiming that a curriculum beats uniform sampling in general. The samplers make the comparison easy to run; the answer depends on the environment and the cost unit.

## Design

### Architecture Overview

```
          trainer process (orchestration side)                  environment container
 ┌─────────────────────────────────────────────────────┐       ┌───────────────────────┐
 │                                                     │  HTTP │ Task API routes       │
 │  TaskAPIClient ──── list_splits / get_task_range ───┼──────▶│ (metadata only)       │
 │        │                                            │       │                       │
 │        ▼                                            │       │                       │
 │  TaskSampler ── sample(n) ──▶ task specs            │       │                       │
 │        ▲                          │                 │  WS   │                       │
 │        │                          ▼                 │       │ reset(split, index,   │
 │   update(task, rewards,    session factory +        ├──────▶│       seed)           │
 │          costs)            harness adapter          │       │ step(action)          │
 │        └──── rewards, costs ◀─────┘                 │◀──────┤ reward (computed      │
 │                                                     │       │   inside the env)     │
 │  EnvEvalHarness = SequentialSampler + rollouts      │       └───────────────────────┘
 │                   + metric aggregation → EvalResult │
 └─────────────────────────────────────────────────────┘
```

Training uses a `UniformSampler` or `CurriculumSampler` over the `train` split. Evaluation uses a `SequentialSampler` over the `test` split, with fixed seeds. Both feed the same rollout path, so the only difference between a training episode and an evaluation episode is which task is chosen and whether a gradient step follows. The sampler hands each task to the session factory, which turns it into what the environment needs: `reset(split=..., index=..., seed=...)` for an environment that follows the Task API guide.

### Core Abstractions

#### 1. Task spec conventions

Task specs stay untyped (`Any`), as in the Task API guide. This RFC reserves a few optional keys so samplers and evaluators can work across environments:

| Key | Type | Meaning |
|-----|------|---------|
| `id` | `str` | Stable task identifier. Already the convention. `CurriculumSampler`, the only sampler that reads it, takes the key's name as a parameter (`id_key`, default `"id"`), because Harbor's task specs use `task_id`. |
| `split` | `str` | Split name. Already the convention. |
| `index` | `int` | Position in the split. Already the convention. |
| `difficulty` | `int` or `float` | Optional. Higher means harder. Tiers (`0..K`) or a continuous score. |
| `group` | `str` | Optional. A family of related tasks (for example a reasoning_gym dataset name). |

An environment that publishes neither `difficulty` nor `group` still works with every sampler; the curriculum then keys its statistics by task `id`.

Tiers work best when they are close together. The curriculum can only concentrate on a tier whose success rate is near 0.5; in the pilot, once the middle tier reached about 95%, the next tier was still below 40% and no tier sat in between. The Task API guide should recommend enough tiers that neighbouring tiers differ by a modest step in difficulty. Two more points for the guide: train and test seeds that never collide do not make the questions disjoint, because a tier with fewer possible questions than train tasks repeats its test questions in train; and a test set should take no questions from such a tier, or report it apart. These keys and points are an addition to the guide, not a requirement: the Task API checklist and existing environments are unaffected.

#### 2. `TaskAPIClient`

A generic helper in `openenv.core` that calls the standard Task API routes for any environment, deriving the HTTP base URL from a `ws://` client URL:

```python
class TaskAPIClient:
    def __init__(self, base_url: str, *, env_name: str,
                 headers: dict[str, str] | None = None): ...
    @classmethod
    def from_env_client(cls, env: EnvClient | SyncEnvClient, *,
                        env_name: str | None = None,
                        headers: dict[str, str] | None = None) -> "TaskAPIClient": ...
    def list_splits(self) -> list[dict[str, Any]]: ...
    def list_tasks(self, split: str) -> list[Any]: ...
    def num_tasks(self, split: str) -> int: ...
    def get_task(self, split: str, index: int) -> Any: ...
    def get_task_range(self, split: str, start: int | None = None,
                       stop: int | None = None) -> list[Any]: ...
    def iter_tasks(self, split: str, page_size: int = 256) -> Iterator[Any]: ...
```

It returns task specs as untyped JSON, exactly as the routes serve them, and maps `501` to `NotImplementedError` and `400` to `IndexError`, mirroring the server. Since an environment may cap `get_task_range`, a caller that needs a whole split reads it with `iter_tasks`. `iter_tasks` pages with `get_task_range` and relies on `num_tasks` for the true total, since `list_tasks` may return a bounded preview. `headers` are sent with every request, as `EnvClient` and `MCPToolClient` send theirs since #1403 (for example a token for a private Space). Environment clients can keep their own typed helpers (for example `HarborEnv.get_task` returning a `HarborTaskRef`) and build them on this client. See D6 for why this belongs in core although core clients currently leave Task API methods out.

#### 3. `TaskSampler`

```python
class TaskSampler(Protocol):
    def sample(self, n: int) -> list[Any]:
        """Return task specs for the next n prompts (GRPO groups): n of them,
        or, from a finite sampler at its end, the ones left, then none."""

    def update(self, task: Any, rewards: Sequence[float | None],
               costs: Sequence[float] | None = None) -> None:
        """Report the rewards of every rollout of one task (one GRPO group).

        A reward of None marks a rollout the environment could not score
        because of a failure of its own, such as a grader that failed.
        Samplers leave it out of everything they estimate from rewards, since
        a missing reward is not zero, and still count its episode and its
        cost. costs, if given, has one entry per rollout, scored or not, in
        the trainer's cost unit (for example the number of generated tokens).
        """

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def metrics(self) -> dict[str, float]:
        """Return numbers for the trainer's log; the samplers proposed here return the counters below."""
```

The rewards passed to `update` are the environment's own, one per rollout, not a total that the trainer forms with reward terms of its own. A failure of the policy that reaches the environment, such as a wrong or malformed answer in a valid tool call or hitting a limit the environment enforces, gets the environment's own score, usually 0, and that score is what reaches `update`: the trainer never fills in a reward. A harness used for training therefore hands the policy's answer to the environment; for a plain-text reply in a single-turn environment, that is the model step of rollout step 5. An episode the harness ends before the environment ends it, such as on a tool call it cannot parse or at its own turn limit in a multi-turn environment, has no environment score even when an earlier step returned a reward, and neither does an episode lost to a failure of the infrastructure; open question 10 asks what a trainer reports for them. `None` is only for a rollout the environment could not score because of a failure of its own, such as a grader that failed, never for an answer it rejects. A trainer calls `update` for every group it ran to the end, scored or not, including the groups it then leaves out of the gradient because they carry no signal (see Trainer integration).

`state_dict` and `load_state_dict` let a trainer checkpoint and resume the sampler together with the model, which a long run on one machine needs. (The pilot lost its intermediate checkpoints because they were not saved separately; resuming requires the model, the optimizer and the sampler state together.)

The samplers proposed here keep the same counters and return them from `metrics()`, so their training logs report efficiency the same way:

- `groups` and `episodes` (every group and every rollout reported, scored or not), `signal_groups` (groups whose scored rewards have a population standard deviation of at least `1e-6`, so that a difference from rounding does not count; a group with fewer than two scored rewards has no signal), and `cost` (the sum of reported costs)
- derived: `signal_groups_per_episode`, `signal_groups_per_cost`, and `zero_signal_fraction`

A trainer reports costs for every group or for none: these samplers raise `ValueError` on a group without costs after one with costs, and the reverse, and with none they leave `cost` and `signal_groups_per_cost` out of `metrics()`. Another `TaskSampler` returns from `metrics()` whatever it tracks, as numbers for the log. The zero-signal fraction on its own is misleading: in the pilot the curriculum had the lower zero-signal fraction and the worse result per token. Logs should report signal groups per episode and per unit of cost.

The protocol has the shape of prime-rl's [`TaskSampler`](https://github.com/PrimeIntellect-ai/prime-rl/blob/855a2b815b804857835c635b3fd2b1ae5400f96c/src/prime_rl/orchestrator/curriculum/samplers/base.py) (`__next__`, `observe`, `state_dict`, `load_state_dict`, `metrics`), with two differences: `sample(n)` draws the tasks of a whole step at once (a trainer that opens one group at a time calls `sample(1)`), and `update` takes rewards and costs instead of that framework's episode objects, so core depends on no training framework.

Two implementations ship in core:

- **`SequentialSampler(tasks, *, seeds=None)`**: deterministic enumeration of a split. With `seeds`, it returns `(task, seed)` pairs, each task with every seed before the next task. At the end of the split, `sample(n)` returns the items left, possibly fewer than `n`, and then an empty list. Used by evaluation. `update` only updates the counters.
- **`UniformSampler(tasks, *, rng_seed=0)`**: shuffled sampling without replacement per epoch. The baseline for training.

A third, **`CurriculumSampler(tasks, *, group_size, key="difficulty", ...)`**, described below, is offered as an example in `examples/`, and moves to core if the maintainers prefer (open question 9).

The tasks a sampler draws, for example `sampler.sample(num_episodes)`, are a valid `tasks` iterable for the existing `CollectRunner`, so `openenv collect` and the evaluation harness gain task selection without changes to how tasks are passed. Selection does not work end to end yet, though, for four reasons. Inside `CollectRunner`: it passes no seed to the session, and it writes no record for an episode that fails. Outside it: the `reasoning_gym_env` session factory receives the task and does not use it, and the `openenv collect` command never passes tasks. Rollout steps 3 and 5 close these gaps. An adaptive sampler also needs each episode's reward, which `CollectRunner` does not report back to its `tasks` iterable today; trainers call `update` from their own loop (see the example), and open question 7 asks whether `CollectRunner` should take a per-episode callback.

#### 4. `CurriculumSampler`

The sampler keeps, for each key `k` (a difficulty tier, a group, or a task id), decayed success counts and an estimate `p̂_k`:

```
s_k ← γ · s_k + successes
f_k ← γ · f_k + failures
p̂_k = (s_k + 1) / (s_k + f_k + 2)          # Beta(1, 1) prior
```

A rollout counts as a success when its reward is at least `success_threshold` (default `1.0` for binary rewards; open question 5 asks about graded rewards), and a rollout the environment could not score (a reward of None) counts as neither. The decay `γ` (default `0.9`), applied once per update that has at least one scored reward, lets the estimate follow a policy that is improving.

Each key is sampled with probability proportional to the chance that a group there carries signal:

```
w_k = 1 - p̂_k^G - (1 - p̂_k)^G + ε
```

where `G` is the trainer's group size and `ε` (default `0.05`) keeps every key reachable so the sampler can notice when a hard tier becomes learnable. Unseen keys start at `p̂ = 0.5`, the maximum signal, so every tier is tried early. Within a key, tasks are drawn uniformly without replacement.

The weight ranks keys; it is not a calibrated probability. It treats the `G` rollouts of a group as independent draws with one success rate, but the questions of one tier differ in difficulty, so the outcomes inside a group are correlated and groups in which every rollout fails are far more common than the formula assumes. In the pilot, at the success rates of the uniform run's last quarter, it predicted about twice as many groups with signal as a regression on the log's per-step tier counts estimates for the two hardest tiers (0.98 and 0.92 against 0.46 and 0.44). In the trial the curriculum's own estimates did the same on tiers 2 to 4: 0.96 and 0.89 against 0.52 and 0.46 on the two hardest, in the last quarter of each run, with a correlation of about 0.5 between the outcomes of one group (pooled over the three seeds, so part of it may come from differences between seeds and from learning within the window). The factor is similar on those tiers, so it moves the sampling probabilities little. On a solved key the weight never reaches `ε`: with the default decay and `G = 8`, `p̂` approaches 81/82, so the key keeps a signal term of about 0.09, and in the trial the curriculum gave each of tiers 0 and 1, both solved from the start, about 7% of its episodes.

The `CurriculumSampler` counts and reports costs but does not weight by them; trainers use them for budgets, concurrency and rollout allocation. The first version of this RFC divided the weights by the token cost of each key, and on our trial that did not help (Alternatives Considered). Costs come from the trainer, which knows them: the number of tokens each completion generated (the length of its generated ids, never a count made by tokenizing the text again), the token records that RFC 006's harness already carries, or a spend that the environment reports with each episode, as issue #1382 proposes (`metadata["service_spend"]`). An episode the environment could not score still costs something: `posttrain_env` (#1500), the prototype of RFC 014 (#1508), ends an episode with a reward of None when its score job fails, times out or is cancelled, and still reports its spend, so the samplers count that episode and its cost. The sampler never sees rollouts, so it takes no cost function.

These weights are a heuristic, not an optimum: they move probability toward keys with more signal without abandoning the others. [Knapsack RL](https://arxiv.org/abs/2509.25849) uses the same expression, with a prompt's number of rollouts in place of `G`, to share a fixed rollout budget among the prompts of a batch. It multiplies that expression by an information-gain term that favours harder prompts, and warns that selecting prompts for a high share of groups with signal can concentrate on easy ones; in our trial the curriculum gave tiers 2 to 4 85% to 87% of its episodes. On the ten finer tiers of the reference experiment, it gave tiers 4 to 9, the ones not solved from the start, about 88% of its episodes, and tiers 0 to 3 about 3% each.

#### 5. `EnvEvalHarness`

An implementation of the existing `EvalHarness` base class:

```python
class EnvEvalHarness(EvalHarness):
    def __init__(self, session_factory: ResourceSessionFactory,
                 harness_adapter: HarnessAdapter,
                 tasks: Iterable[Mapping[str, Any]], *,
                 model_step: ModelStep,
                 output_dir: str | Path | None = None): ...

    def run(self, harness_version, library_versions, dataset, eval_parameters) -> dict:
        ...
```

`tasks` is the list of task specs to evaluate, for example `list(TaskAPIClient.iter_tasks("test"))`; the harness does not need the client itself. Each task spec is a mapping with a unique string `id`, which `per_task` and checkpoint comparison key on. The policy is a model step, as `CollectRunner.run` takes it: `build_model_step(OpenAIClient(...))` for an OpenAI-compatible endpoint, or a step that generates in process, for example on the weights a trainer holds. `eval_parameters` accepts:

| Key | Default | Meaning |
|-----|---------|---------|
| `limit` | `None` | Evaluate only the first `limit` tasks (for cheap in-training evals). |
| `seeds` | `[0]` | Seeds crossed with each task. |
| `samples_per_task` | `1` | `n` completions per task, for pass@k. |
| `temperature` | `0.0` | Sampling temperature; must be > 0 when `samples_per_task > 1`. |
| `success_threshold` | `1.0` | Reward at or above which an episode counts as solved. |
| `max_tokens`, `max_turns` | `200`, `9` | Passed to the model step, and the harness's turn limit. |
| `model`, `llm_endpoint` | | Recorded in the result's config; the model step reaches the policy. |

It walks `tasks` in the order a `SequentialSampler` gives, each task with every seed, starts each episode through the session factory with its task and seed, runs it through the harness adapter as `CollectRunner` does, and reads who ended it and its reward as `openenv collect` does. With `output_dir`, each run writes a new folder `run-NNNN` with the files and records of `openenv collect`, through the existing `RolloutSerializer`: `results.jsonl` for the episodes with a reward, `errors.jsonl` for the others, each record with its outcome, seed and generated ids, and `metadata.json`. It returns scores:

- `success_rate` with its standard error, `mean_reward`, `mean_turns`, `mean_completion_tokens`, and `num_tasks` and `num_episodes`
- `pass@k` for each `k ≤ samples_per_task`, with the unbiased estimator `1 - C(n - c, k) / C(n, k)`
- `error_rate`: episodes the harness ended because of the policy before the environment ended them, whatever reward an earlier step returned, for example at the harness's own turn limit in a multi-turn environment. A failure the policy's output can cause counts here even when it surfaces as an exception, such as a tool call the harness cannot parse or that names no tool of the session, or a request past the context length; when the cause is unclear, it counts here. They count as not solved in `success_rate` and `pass@k`, so a policy cannot raise its score by failing to answer, and `error_rate` shows how many of the failures were not wrong answers. `mean_reward` averages only the scores of episodes the environment ended, so it leaves these episodes out and is read only together with `error_rate`: a policy that stops answering the questions it would fail raises `mean_reward`
- `unscored_rate`: episodes the environment ended with a reward of None, which it does only for a failure of its own, such as a failed score job
- `infra_error_rate`: episodes lost to a failure of the infrastructure that the policy's output cannot cause, in the harness, the model endpoint or the environment's server, such as a crash or an unreachable server. In practice an exception counts here when it is a lost connection to the environment (a `ConnectionError`, such as `SessionLostError`, or a closed WebSocket), when the model client reports a failure to connect, a timeout, an authentication, permission or routing error, a rate limit or a server error, when the session cannot be created, or when `verify` forwards a reward that differs from the environment's. Any other exception, such as an environment step that times out or the same failures raised by a client the harness does not know, counts in `error_rate` by the rule for unclear causes
- `by_difficulty` and `by_group` breakdowns when task specs carry those keys
- `per_task`: task id → outcome (solved, not solved, or left out when every one of its episodes was unscored or an infrastructure failure), which is what checkpoint comparison needs, and `reproducible`, the result of the checks of D4

Unscored episodes and infrastructure failures are left out of `success_rate`, `mean_reward` and `pass@k` and reported separately. In `pass@k`, `n` and `c` count the task's remaining samples, over all its seeds (a failure of the policy stays in `n`, not solved), the average runs over the tasks with at least `k` of them, and each `pass@k` reports how many tasks it averages; a score with nothing to average is None. Leaving an episode out is sound only if whether it is lost does not depend on the policy, which this RFC asks of environments and harnesses but cannot check, so `unscored_rate` and `infra_error_rate` are always reported next to `success_rate`.

An episode the harness ends after a step that returned a reward, such as at its own turn limit, keeps in its collect record the last reward a step returned, as `openenv collect` does today, though the environment never scored the episode as a whole (section 3); rollout step 5 adds a mark that the harness ended it, the record's `outcome`. Evaluation writes such an episode the same way and reads the mark, not that reward, and counts the episode in `error_rate`. What a trainer passes to `update` for such an episode is still open question 10.

`mean_completion_tokens` needs the generated token count of each episode. The pipeline has it and drops it today: the model client does not keep the provider's usage numbers, and the episode record does not copy the generated ids that the harness collects. Rollout step 5 keeps them.

In a single-turn text environment such as `reasoning_gym_env`, the harness expects the answer as a tool call: a plain-text answer ends the episode without a reward, and it counts as not solved (`error_rate`). The harness adapter for such an environment needs a model step that turns the text into that call (rollout step 5); with it, the environment scores a text answer like any other, in evaluation and in training. Which part of the text is the answer is the step's rule: `reasoning_gym_env`'s step reads the last `<answer>` block, which every dataset can ask for, while a task with numeric answers may also read the last number, as the reference experiment's trainer did. A small model that does not write the tags is scored very differently by the two: on the first 10 test questions of the example of rollout step 6, the untrained Llama-3.2-1B wrote them in 2 replies, and the tags-only rule solved none of the 10 where the numeric rule solved 5. Transcripts made with such a step hold a tool call the model did not write, so they are not supervised fine-tuning data.

A helper compares two results:

```python
diff = compare_eval_results(before, after)
diff.fixed        # task ids that went from fail to pass
diff.regressed    # task ids that went from pass to fail
diff.left_out     # task ids left out in either result, in neither list above
```

#### 6. `openenv eval`

A CLI command with `openenv collect`'s flags for the environment, an OpenAI-compatible endpoint and the model, plus `--split`, `--system-prompt` and the `eval_parameters` above; collect's `--provider`, `--llm-port` and `--dataset-config` are left out. It reads the tasks through `TaskAPIClient`, writes the episodes to `run-0000/` in `--output-dir` and the `EvalResult` to `result.json` beside it, and `--compare <other_result.json>` prints the fixed, regressed and left-out tasks. The endpoint must answer with a tool call: a plain-text reply counts in `error_rate`.

#### 7. Trainer integration

Core provides a framework-neutral schedule object; framework adapters live in examples, following the recipes RFC:

```python
class EvalSchedule:
    def __init__(self, harness: EvalHarness, config: EvalConfig, *,
                 every_n_steps: int, on_result: Callable[[int, EvalResult], None]): ...
    def run(self, step: int) -> EvalResult: ...
    def maybe_run(self, step: int) -> EvalResult | None: ...
```

It takes any `EvalHarness`, not only `EnvEvalHarness`. A plain Python loop, an MLX loop, or a TRL callback calls `maybe_run(step)` after each optimizer step; it evaluates when `step` is a multiple of `every_n_steps`, step 0 included, and `every_n_steps=0` turns it off. `run(step)` evaluates at once, for example after the last step. The policy under evaluation is served through an OpenAI-compatible endpoint (vLLM, `mlx_lm.server`, or any other server), so the harness does not depend on how the trainer holds the weights. That endpoint must serve the weights of the step being evaluated: a trainer that sends new weights to its server only before its next generation, as TRL's GRPO trainer does in vLLM server mode, checks the interval itself and, on a step it evaluates, syncs the weights first and then calls `run(step)`. `on_result` receives the step, and the trainer should log the sampler's `metrics()` next to it, so learning curves can be drawn against episodes and against cost, not only against steps or wall-clock time.

A sampler needs two hooks in a trainer: the trainer draws its prompts from `sample(n)`, and it calls `update` on every group it ran to the end, scored or not, before anything drops groups. Trainers increasingly drop the groups whose rewards are all equal: DAPO's dynamic sampling, the zero-advantage gate of prime-rl, which updates its sampler before the gate ([`Curriculum.on_result`](https://github.com/PrimeIntellect-ai/prime-rl/blob/855a2b815b804857835c635b3fd2b1ae5400f96c/src/prime_rl/orchestrator/curriculum/base.py)), and TRL's asynchronous GRPO trainer since [huggingface/trl#7609](https://github.com/huggingface/trl/pull/7609), merged on 2026-10-28. A sampler updated only with the groups that survive such a filter never sees that a key is solved or out of reach. The trial's trainer calls `update` on every group, then leaves the groups without signal out of the gradient.

Asynchronous trainers limit a sampler in two ways. Updates arrive late and out of order, and the decay weighs groups in the order they arrive, so keys whose rollouts finish first are refreshed sooner; large runs report this bias toward short rollouts, mostly after a start or a restart ([MiMo-V2.6](https://arxiv.org/abs/2610.11959), section 5.5), and [MiniMax-M2](https://arxiv.org/abs/2605.26494) bounds it with a windowed FIFO over the generation queue. And a seed reproduces a run only in a synchronous trainer such as the trial's: in an asynchronous one the draw order depends on completion times.

Shares across environments or data sources are also the trainer's. Large runs set a share per source and keep it in the rollout scheduler, giving slow sources more concurrent rollouts (MiMo-V2.6) or a larger rollout quota ([LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725)), and prime-rl draws the source by a fixed ratio before asking that source's sampler for a task. A sampler works inside one source's share, keyed by `difficulty` or `group` within that source; the trainer already knows which environment each task belongs to.

### Key Design Decisions

#### D1. Sampling lives on the trainer side, not in the environment

- **Chosen approach**: Samplers run in orchestration code and hand each task to the session factory, which turns it into what the environment needs: `reset(split=..., index=...)` for an environment that follows the Task API guide. Environments only publish task metadata.
- **Rationale**: This keeps the invariants intact. Rewards are still computed inside the environment; the sampler only reads them. The agent has no path to influence which task it gets, because selection goes through the session factory and `reset()`, which stay on the infrastructure side of the dual API boundary. The environment server stays free of cross-episode state, which also keeps "one env = one trajectory" and stacking of environment instances simple. It is also where task selection already happens today: the Harbor harness picks `indices` on the trainer side.
- **Trade-offs**: Each trainer process holds its own statistics. A distributed trainer with many rollout workers must aggregate them; `state_dict` gives a merge point, but a shared sampler service is out of scope for this RFC.

#### D2. The curriculum targets GRPO signal, and reports cost

- **Chosen approach**: Weight keys by the probability that a group of size `G` has non-zero reward variance. Count cost in a unit the trainer chooses and report signal per unit of cost, without weighting by it.
- **Rationale**: The probability of signal is the quantity that decides whether a GRPO group produces a gradient, and it needs one parameter the trainer already knows (`G`). On the pilot's seed, the signal-only curriculum produced 39% more gradient-bearing groups per episode than uniform sampling but 42% fewer per generated token, and the first version of this RFC read that as a reason to divide by cost. Three more seeds changed the reading: there the signal-only curriculum got more groups with signal per generated token than uniform sampling on every seed, and higher test success per token, while dividing by token cost as first proposed lowered both, and a corrected division matched the signal-only weights per token without beating them (Alternatives Considered). Learning-progress curricula (weighting by the change in success rate) remain a reasonable alternative and can be added as another `TaskSampler`, as can a sampler that accounts for cost.
- **Trade-offs**: Gradient-bearing groups are a proxy for learning, not the objective. In the pilot, uniform sampling trained mostly on an easier tier and still improved the two hardest tiers, from 7% and 0% to 46% and 24%. The evaluation harness, not the sampler's counters, is what decides whether a sampler helps. The estimates also need repeated observations per key: for large datasets without `difficulty` or `group`, per-task statistics are sparse and the sampler behaves close to uniform until tasks repeat. Publishing closely spaced difficulty tiers is the intended remedy.

#### D3. Evaluation reuses the collect pipeline's pieces

- **Chosen approach**: `EnvEvalHarness` is built from the same pieces as `CollectRunner` (the session factory, the harness adapter, `EpisodeRecord` and `RolloutSerializer`), plus the order of a `SequentialSampler` and aggregation.
- **Rationale**: Training rollouts, collected datasets, and evaluation episodes then go through one code path and one serialization format (design principle 1). Evaluation episodes can be pushed to the Hub with the existing `push_to_hf_hub` for inspection.
- **Trade-offs**: `CollectRunner` itself passes no seed to the session and writes nothing for a failed episode, so until rollout step 5 adds both, the harness drives the pieces itself instead of calling `CollectRunner`. Evaluation also inherits the limits of the collect pipeline. The `openenv collect` CLI resolves only `openspiel:<game>` and `reasoning_gym:<dataset>` environment specs today, so `openenv eval` would start with the same environment coverage and grow with `collect`; the Python `EnvEvalHarness` accepts any `ResourceSessionFactory` from the start. And the pipeline drives every turn of an episode, while a Harbor server runs a whole agent in one tool call, `run_rollout`, which takes the task as its arguments (Harbor's `reset` selects no task), so the first version does not cover Harbor (open question 8).

#### D4. Evaluation checks reproducibility instead of assuming it

- **Chosen approach**: Before a run, `EnvEvalHarness` makes two checks with fresh sessions. It resets two sessions with the first task and seed and compares their initial messages; then it resets a session with the second task and checks that its initial messages differ from the first. If either check fails, it warns and records `reproducible: false` in the result.
- **Rationale**: An eval that silently sees different tasks across checkpoints produces curves that cannot be trusted. Environments that ignore `seed` have existed (#1102, #1250; #1647 found 14 more), and RFC 008's `runtime.seed_control` and `runtime.episode_determinism` checks (pending in #1321) address the same concern at validation time. The second check catches an environment that ignores the task selection, which the first cannot see: the server passes `reset()` only the arguments its signature names, so an environment whose `reset()` does not name `split` and `index` returns the same default task twice and looks reproducible. `reasoning_gym_env` behaves this way today.
- **Trade-offs**: Three extra resets per run. The checks compare the initial messages, which a session builds from the observation; a session that copied the task spec into its messages would pass the second check for an environment that ignores the task. They cover the start of an episode only, not later nondeterminism in the environment.

#### D5. Task metadata stays on HTTP for now

- **Chosen approach**: `TaskAPIClient` uses the existing HTTP Task API routes. This RFC adds no routes and no wire types.
- **Rationale**: The Task API guide documents these routes as HTTP-only, with no WebSocket message types, so that the `/ws` session protocol stays focused on `reset`, `step`, `state` and `close`. The routes serve metadata, never the step loop.
- **Trade-offs**: This sits in tension with the communication-patterns invariant, which lists WebSocket for all environment communication, "Gym-like API + metadata", and notes that HTTP is being deprecated (#252); #1156, which proposed deprecating the HTTP `/reset`, `/step` and `/state` routes, was closed as not planned on 2026-10-27. This RFC does not resolve that tension; it builds on the Task API as documented. Because every sampler and the evaluation harness go through `TaskAPIClient`, moving the Task API to WebSocket later changes only that class. See open question 1.

#### D6. A generic Task API client in core

- **Chosen approach**: Add `TaskAPIClient` to `openenv.core`, separate from `EnvClient`, returning untyped task specs.
- **Rationale**: The Task API guide leaves Task API methods out of the core clients because task specs are environment-specific. `TaskAPIClient` keeps specs untyped and adds no methods to `EnvClient`, so that reasoning still holds; what it removes is the duplicated plumbing that is not environment-specific (route names, HTTP base derivation from a `ws://` URL, headers, paging, error mapping). Samplers and the evaluation harness need to list tasks for any environment, which is impossible if each client names its helpers differently.
- **Trade-offs**: One more public class in core. Typed task specs remain each environment client's job.

## Examples

### Server side: publishing difficulty tiers

A procedurally generated environment exposes tiers as a finite split. For `reasoning_gym_env`, a task maps to a dataset name, a difficulty configuration, and a seed. The tiers below come from reasoning-gym's own curriculum for `chain_sum`, whose levels set the number of terms (2 to 12) and of digits (1 to 10):

```python
from reasoning_gym.arithmetic.chain_sum import ChainSumCurriculum
from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode

# (level of num_terms, level of num_digits): 2x1, 3x1, 3x2, 4x2, ... 7x5 terms x digits
LEVELS = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4)]
TASKS_PER_TIER = {"train": 2000, "test": 100}
SPLIT_SEED_OFFSET = {"train": 0, "test": 500_000_000}   # train and test seeds never collide

def tier_config(tier: int):
    curriculum = ChainSumCurriculum()
    terms, digits = LEVELS[tier]
    curriculum.set_attr_level("num_terms", terms)
    curriculum.set_attr_level("num_digits", digits)
    return curriculum.generate_configuration(
        context=DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND))

class ReasoningGymEnvironment(Environment):
    def list_splits(self) -> list[str]:
        return ["train", "test"]

    def num_tasks(self, split: str) -> int:
        return len(LEVELS) * TASKS_PER_TIER[split]

    def get_task(self, split: str, index: int) -> dict:
        if not 0 <= index < self.num_tasks(split):
            raise IndexError(index)
        tier = index % len(LEVELS)    # tiers interleave: the first N tasks cover every tier
        return {"id": f"chain_sum-{split}-{index}", "split": split, "index": index,
                "group": "chain_sum", "difficulty": tier}

    def reset(self, split: str | None = None, index: int | None = None,
              seed: int | None = None, episode_id: str | None = None, **kwargs):
        if index is None:
            # No selection: keep today's seed-driven behaviour, as the Task API checklist asks.
            return self._reset_from_kwargs(seed=seed, episode_id=episode_id, **kwargs)
        split = split or "train"
        task = self.get_task(split, index)
        # The seed picks a variant of the task, so tasks crossed with seeds differ.
        question_seed = (seed or 0) * 1_000_000_000 + SPLIT_SEED_OFFSET[split] + index
        return self._reset_dataset("chain_sum", tier_config(task["difficulty"]),
                                   seed=question_seed)
```

`_reset_from_kwargs` and `_reset_dataset` stand for the environment's existing reset paths.

The library has no level with negative numbers, so these tiers have none, unlike the pilot's two hardest tiers. On them, the untrained Llama-3.2-1B-Instruct scored 100, 100, 100, 100, 68, 51, 17, 9, 3 and 4% (100 test questions per tier, greedy, the step-0 evaluation of the reference experiment): tiers 4 to 9 are the ones a curriculum can use for that model. The two easiest tiers have few possible questions: tier 0 has only 75 distinct test questions, and 100 and 47 of the 100 test questions of tiers 0 and 1 also appear among the 2,000 train tasks; the guidance of section 1 applies. The pilot numbered its tasks inside each tier and derived each seed from the tier and the position, so the same index names different questions there and in this example.

### Client side: a plain training loop

```python
client = TaskAPIClient.from_env_client(env)
tasks = list(client.iter_tasks("train"))
sampler = CurriculumSampler(tasks, group_size=8, key="difficulty")

evaluator = EnvEvalHarness(
    session_factory, harness_adapter, list(client.iter_tasks("test")),
    model_step=build_model_step(OpenAIClient(POLICY_URL, None, model="policy")),
)
schedule = EvalSchedule(
    evaluator,
    EvalConfig(harness_name="EnvEvalHarness", harness_version="0.1",
               library_versions={}, dataset="reasoning_gym:chain_sum",
               eval_parameters={"llm_endpoint": POLICY_URL,
                                "model": "policy", "temperature": 0.0}),
    every_n_steps=50,
    on_result=lambda step, r: logger.log({"step": step, **sampler.metrics(), **r.scores}),
)

schedule.maybe_run(step=0)                          # baseline before any training
for step in range(1, num_steps + 1):
    batch = sampler.sample(n=prompts_per_step)
    groups = [rollout_group(task, G=8) for task in batch]   # trainer-specific
    for task, group in zip(batch, groups):
        sampler.update(task, [ep.reward for ep in group],
                       costs=[len(ep.completion_ids) for ep in group])
    trainer_update(groups)              # trainer-specific; may skip groups without signal
    logger.log({"step": step, **sampler.metrics()})
    schedule.maybe_run(step)
```

### CLI: comparing checkpoints

The environment spec is positional, as in `openenv collect`:

```bash
openenv eval reasoning_gym:chain_sum --base-url http://localhost:8001 --split test \
  --llm-endpoint http://localhost:8080/v1 --model base \
  --output-dir runs/eval-step0000

openenv eval reasoning_gym:chain_sum --base-url http://localhost:8001 --split test \
  --llm-endpoint http://localhost:8080/v1 --model ckpt-1000 \
  --output-dir runs/eval-step1000 --compare runs/eval-step0000/result.json
```

## Validation Plan

The claims in this RFC are measured, not assumed. Every run below fits one laptop-class machine: a pilot and a trial (Preliminary Results), then the reference experiment that this RFC proposed (Reference experiment).

### Preliminary Results

We ran the samplers before proposing the API: first a pilot on one seed, then a trial on three more seeds, with predictions and the rules for reading the results written down before it ran. The code is a standalone trainer, not the proposed API: a GRPO loop in MLX with the samplers of section 4 and of "Weighting by cost" in Alternatives Considered. The pilot's code, its per-step logs and a figure of both learning curves are at [Matheus-F-Scatolin/OpenEnv@rfc-013-pilot](https://github.com/Matheus-F-Scatolin/OpenEnv/tree/rfc-013-pilot/pilot); the figure is also in the PR description. The trial's code, its 28 run logs, the predictions and a script that regenerates every table and figure from those logs are at [Matheus-F-Scatolin/OpenEnv@rfc-013-trial](https://github.com/Matheus-F-Scatolin/OpenEnv/tree/rfc-013-trial).

Setup, the same for both:

- **Model**: Llama-3.2-1B-Instruct, LoRA rank 16 on every linear layer (11.3M trainable parameters).
- **Environment**: `reasoning_gym_env` vendored from commit `9e0fd9b`, run in-process, with the pilot's five `chain_sum` tiers: 2×1, 3×2, 4×3, 5×4 and 6×5 terms × digits, negative numbers in the last two. 2,000 train tasks and 100 test tasks per tier, with disjoint seeds. The reward is whatever the environment's `step()` returns.
- **Training**: 8 prompts × 8 completions per step (64 episodes), temperature 1.0, at most 384 tokens per completion, on-policy GRPO with one update per batch, no KL term, token-level loss normalization. The sampler is updated with every group; groups with no reward variance are then left out of the gradient and not replaced.
- **Evaluation**: greedy decoding on all 500 test tasks. The standard error of one evaluation is about 2 points.
- **Reproducibility**: the model's chat template writes today's date into every prompt, so the date is fixed to the pilot's day; with it fixed, the trial's trainer reproduced the pilot's seed-0 logs step for step, for the first 370 steps of uniform sampling and 158 of the curriculum.
- **Hardware**: a MacBook Pro (M5 Pro, 24 GB).

#### The pilot (seed 0)

150 training minutes per sampler, one seed, an evaluation every 100 steps. At the same number of environment episodes, the curriculum was ahead at 9 of 10 evaluation points:

| Training episodes (steps) | Uniform | Curriculum | Difference |
|---|---|---|---|
| 6,400 (100) | 50.6% | 58.0% | +7.4 |
| 19,200 (300) | 62.2% | 65.2% | +3.0 |
| 32,000 (500) | 65.4% | 68.4% | +3.0 |
| 44,800 (700) | 62.8% | 63.6% | +0.8 |
| 51,200 (800) | 62.4% | 69.0% | +6.6 |
| 64,000 (1,000) | 67.4% | 71.4% | +4.0 |

Over the whole run:

| | Uniform | Curriculum |
|---|---|---|
| Steps in 150 minutes | 2,196 | 1,081 |
| Tokens per completion | 44 | 106 |
| Groups with no gradient | 79.5% | 71.5% |
| Gradient-bearing groups per 1,000 episodes | 25.7 | **35.6** |
| Gradient-bearing groups per million generated tokens | **578** | 337 |
| Test success, start → end | 52.0% → 73.2% | 52.0% → 72.4% |
| Test success at ~6 M generated tokens | **73.6%** | 67.8% |
| Tiers 3 and 4, start → end | 7%, 0% → 46%, 24% | 7%, 0% → 43%, 22% |

Per episode, the curriculum learned faster: it reached 72.4% after 69,184 episodes, which uniform sampling reached only after about 100,000. Per generated token, uniform sampling was ahead from about 3.3 M tokens on. The curriculum's completions were 2.4 times longer than uniform's, but choosing harder tiers explains a factor of only about 1.2 of that: priced at the uniform run's own completion lengths per tier, the curriculum's mix of tiers costs 54 tokens per completion, against 44. The rest came from the two trained models moving apart: on the same 500 test questions, the untrained model answered with 73 tokens on average, the uniform-trained model with 45 and the curriculum-trained model with 86, both read at 5 million generated tokens.

#### Three more seeds

Seeds 1, 2 and 3, each run stopped at 5 million generated tokens, with an evaluation every 100 steps and every 250,000 tokens. Four samplers: uniform; the curriculum of section 4; the token-weighted curriculum as this RFC first proposed it; and a corrected token-weighted form. Both token-weighted forms are described under "Weighting by cost" in Alternatives Considered. Test success is the mean of the evaluations at 4.0 to 5.0 million generated tokens ("per token") or at steps 100 to 700 ("per episode"). Difference to uniform sampling, in points:

| | Seed 0 (pilot) | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| Curriculum, per token | −7.7 | +5.5 | +18.3 | +14.4 |
| Curriculum, per episode | +2.3 | +4.3 | +9.7 | +6.5 |
| Token-weighted, first form, per token | | +1.2 | −1.6 | −1.7 |
| Token-weighted, corrected, per token | | +5.6 | +15.2 | +16.1 |
| Token-weighted, corrected, per episode | | +10.3 | +10.1 | +9.2 |

Uniform sampling reached 66.0%, 55.6% and 59.9% per token on seeds 1 to 3 (72.2% on seed 0); the curriculum 71.6%, 73.9% and 74.4% (64.5%). Per episode, seed 2 is read at steps 100 to 700 for each pair of runs that reached step 700. The rule we wrote before the run read every run of a seed only up to the step all of them reached: one run of seed 2 stopped at step 223, and that reading gives +3.9 for the curriculum, with the same conclusion.

At 5 million generated tokens:

| | Uniform | Curriculum | Token-weighted, first form | Token-weighted, corrected |
|---|---|---|---|---|
| Groups with signal per million tokens, seeds 1 to 3 | 285, 287, 706 | 526, 730, 835 | 492, 119, 750 | 345, 476, 850 |
| Share of episodes on tier 0, solved from the start | 20% | 7%, 7%, 7% | 17%, 14%, 22% | 8%, 7%, 5% |
| Tokens per training completion, seeds 1 to 3 | 80, 79, 33 | 73, 59, 49 | 64, 351, 45 | 111, 80, 47 |

What we take from both:

1. **The mechanism works as designed.** The curriculum concentrated on the tiers nearest `p = 0.5` and raised the groups with signal per episode on every seed.
2. **The pilot's per-token result belonged to its seed.** The curriculum's completions were longer in the pilot; in a 30-step check on four new seeds they were longer by 20 tokens or more in only one, and on seeds 1 and 2 of the trial the curriculum's completions were the shorter ones. Across the four seeds the curriculum was ahead per episode four times and per token three times.
3. **Dividing by token cost did not help.** The comparisons with the signal-only curriculum were not among the questions we wrote down before the trial; we read them with the thresholds we wrote there. The first form sent 14% to 22% of its episodes to tier 0, which the model solved from the start, against about 7% for the signal-only curriculum, and learned less on every new seed: −4.3, −19.9 and −16.2 points per token. The corrected form fixed the allocation and was neither better nor worse on the 500 test questions: +0.1, −3.1 and +1.6 points per token, and +6.1, +0.4 and +2.7 per episode, below the +4 that "better" needs. On tiers 2 to 4 alone it was ahead per episode by +9.6, +0.7 and +4.3 (+4.9 on average), which passes that bar when each pair of runs is read to step 700, but not in the window we wrote before the trial, which reads seed 2 to step 200, where the corrected form is 0.7 points behind. Per token, the unit that weighting by token cost exists to improve, it did not help, and it got fewer groups with signal per generated token than the signal-only curriculum on two of three seeds. This version of the RFC therefore leaves cost out of the weights.
4. **A reward that scores a cut answer lets a run degenerate.** In one of the twelve runs (the first token-weighted form, seed 2), the policy learned to repeat its tagged answer until the length limit. From about step 50 on, every completion was cut at 384 tokens, and test success stayed at 54% (52% untrained), so the run spent its 5 million tokens by step 223. The environment scores the answer a cut completion contains, so nothing discouraged the repetition. This is the environment's reward, not the sampler, and the run stays in every number above.
5. **The tiers were too coarse.** After tier 2 reached about 95%, no tier was near 50%, which limited what any curriculum could do.

Caveats: three new seeds per sampler are a pattern, not a significance test; ahead on three seeds of three happens one time in eight by chance alone. One model, one task family, one budget. The trainer is synchronous, so the trial says nothing about asynchronous trainers, and the rewards of `chain_sum` are almost binary (about 0.1% of training answers got partial credit), so it says nothing about graded rewards. All 100 test questions of tier 0 also appear in train, because that tier has about 200 possible questions; the model solves it from the start, so the comparisons do not change, and on tiers 2 to 4 alone they are larger. In the pilot the two arms ran one after the other on the same machine, and the curriculum arm was slowed for its last 40 minutes by an underpowered charger; this affects the comparison by wall-clock time but not the per-episode or per-token comparisons, which do not depend on machine speed.

### Reference experiment

Uniform sampling against the curriculum of section 4, with the same group size, on five seeds not used before (5 to 9). The test, the reading windows, how the tiers whose test questions also appear in train are read, and what happens to a run that degenerates were written down and pushed to the authors' private repository before the first run, and every run logs a commit that contains them. The script that applies them was committed before the runs but pushed only after the results, and has since gained readings added after the results, marked below. Rules and script are published with the logs. The code, the ten run logs, those rules (`notes/expectations.md`, section 5) and the report and figures that the script regenerates from the logs (`notes/reference-results/`) are at [Matheus-F-Scatolin/OpenEnv@rfc-013-trial](https://github.com/Matheus-F-Scatolin/OpenEnv/tree/rfc-013-trial).

- **Environment**: `reasoning_gym_env` with the Task API and the ten tiers of the example above, from commit `7186e70e` of the fork, run in-process; the trainer reads its tasks with `get_task_range`. 2,000 train and 100 test tasks per tier.
- **Training**: as in Preliminary Results (the model, LoRA, 8 prompts × 8 completions, temperature 1.0, at most 384 tokens, the sampler updated with every group and groups without signal left out of the gradient, the fixed prompt date), each run stopped at 5 million generated tokens. Before the runs, the trainer reproduced the pilot's stored 20-step logs of seed 0 exactly, for both samplers, on the pilot's tiers.
- **Evaluation**: greedy decoding on all 1,000 test questions, at step 0, every 100 steps, every 250,000 generated tokens and at the last step.
- **Reading, as written before the runs**: 100 and 47 of the 100 test questions of tiers 0 and 1 also appear in train, so the two main measures use the 800 test questions of tiers 2 to 9, and the rest is reported apart. Per token, a run's score is the mean of its evaluations at 4.00, 4.25, ..., 5.00 million generated tokens; per episode, the mean of those at steps 100, 200, ..., 700. Each measure gets a two-sided paired t-test on the five per-seed differences (df 4), with no correction for the two tests, and reads "better" when that test is significant at 5% with a positive mean, "worse" when it is significant with a negative mean. The proposal is supported when, for both measures, the curriculum is not "worse" and its mean difference is −2 points or more; the report also says whether the lower end of the 95% interval is −2 points or more ("shown by the interval"). A run whose training answers are at least 90% cut at the length limit over 50 steps would stay in every number, with a reading without its seed added.
- **Hardware**: the MacBook Pro of Preliminary Results. The ten runs ran back to back in about a day: 1.5 to 2.1 hours of training each, and 2.1 to 2.7 hours from one run's start to the next, evaluations and model loading included. A CUDA machine with vLLM runs the same experiment faster.

Test success on tiers 2 to 9, curriculum minus uniform sampling, in points:

| | Seed 5 | Seed 6 | Seed 7 | Seed 8 | Seed 9 | Mean | 95% interval | t (df 4) | p |
|---|---|---|---|---|---|---|---|---|---|
| Per generated token (4.00 to 5.00 M) | +3.7 | +15.2 | +10.4 | +2.7 | +9.2 | **+8.2** | +1.9 to +14.6 | 3.59 | 0.023 |
| Per episode (steps 100 to 700) | +12.4 | +3.2 | +5.3 | +2.4 | +12.6 | **+7.2** | +1.0 to +13.3 | 3.22 | 0.032 |

Both measures read "better", and the lower end of both intervals is above −2 points, so the proposal is supported, shown by the interval: the curriculum was at least as good as uniform sampling per episode and per token, and it was ahead on every seed in both. Mean test success was 77.7% under uniform sampling and 85.9% under the curriculum per token, and 59.6% and 66.7% per episode; one evaluation on the 800 questions has a standard error over its questions of at most 1.8 points. We had predicted +5 points per token and +4 per episode, ahead on four seeds of five; the effect was larger. The two tests are not corrected for each other, as written before the runs; with a Bonferroni correction, a reading we added after the results, only the per-token test would stay significant (p = 0.023 against 0.025; per episode, 0.032). On all 1,000 test questions, with no test, the differences are +6.5 points per token and +5.7 per episode. The sampler protocol and the evaluation harness did not depend on this outcome.

The differences come from tiers 4 to 9. Tiers 2 and 3 were at 100% at step 0 in every run and mostly stayed near it, and per token the curriculum was ahead by 2.6, 5.0, 15.0, 14.3, 15.6 and 13.0 points on tiers 4 to 9, on average over the seeds. It gave those six tiers about 88% of its episodes, against 60% under uniform sampling, and got more groups with signal on every seed: 54 to 57 per 1,000 episodes against 29 to 36, and 620 to 1,063 per million generated tokens against 294 to 648. Its training answers were not consistently shorter or longer: longer on three seeds and shorter on two (51 to 87 tokens on average, against 55 to 100). Under 0.4% of training answers were cut at the length limit in any run, no run degenerated, and every run reached step 700.

What the result does not settle:

- **Updates per token.** At the same token budget the curriculum got 885 to 1,525 optimizer updates per run and uniform sampling 674 to 1,318: 11% more in total (a mean of 1,140 against 1,025), but fewer on seeds 5 and 8, the two seeds with the curriculum's smallest per-token leads. How many updates a token budget buys depends on answer length and on how many steps have no group with signal, and the two differ between the arms.
- **Falls between evaluations.** We read them after the results, with thresholds chosen then; this changes no number above. On tiers 2 to 9, uniform sampling had five falls of 5 points or more between consecutive evaluations, the curriculum one. One (uniform, seed 5, step 658) was a short burst of answers cut at the length limit, gone at the next evaluation and read by neither measure. In the others, none of the lost answers was cut, and each writes a wrong sum, as nearly every wrong answer in these runs does. Three of the uniform falls (seed 6 at steps 400 and 860, seed 7 at step 784) and the curriculum's one land on evaluations that the measures read. The three falls of 10 points or more are all uniform (seed 5 at step 658, seed 6 at steps 400 and 860); replacing each by the mean of its neighbouring evaluations, as a description with no test, gives +7.8 points per token and +6.7 per episode, with the curriculum still ahead on every seed.
- **Scope.** One 1B model with LoRA on MLX, one task family whose rewards are almost binary, one budget, a synchronous trainer, five seeds. Tiers 0 and 1 are left out of the main measures and tiers 2 and 3 are solved from the start, so the result is about the six tiers the model had to learn.

## Rollout Plan

Each step is a separate PR:

1. This RFC.
2. Reserved task spec keys in the Task API guide, including guidance on tier spacing and on train and test overlap, and `TaskAPIClient` (with `headers`) with tests against a stub server.
3. Task API support in `reasoning_gym_env`, with difficulty tiers from reasoning-gym's curriculum levels and a session factory that turns a task into `reset(split, index, seed)`; then `finqa_env`, whose `reset(seed)` fix (#1252) is merged.
4. `SequentialSampler` and `UniformSampler` in core, and `CurriculumSampler` as an example (or in core, if the maintainers prefer), with unit tests on synthetic reward and cost streams (for example, a simulated policy whose success rate per tier improves over time and whose completion length grows with difficulty).
5. `EnvEvalHarness`, `compare_eval_results`, and `openenv eval`. In the collect pipeline: pass the seed to the session (`CollectRunner(seeds=...)`), write a record for a failed episode (`errors.jsonl`, with the record's `outcome` and `error`), keep apart in the episode record a reward of None with which the environment ended the episode, a failure of the policy in the harness and a failure of the infrastructure (the record's reward is a `float` today, because `_resolve_env_reward`, which `build_harness_rollout_func` also uses, raises when the environment gave none), keep the generated token count of each episode (the provider's usage numbers in the model client, the generated ids in the episode record's `completion_ids`), let `openenv collect` pass tasks (`--tasks`), and a model step that turns a text answer into a single-turn environment's tool call.
6. `EvalSchedule`, with a TRL `TrainerCallback` that calls it, and an example in `examples/` with the reference experiment's settings and reading rule (a plain loop and an MLX loop for Apple Silicon); it stops after a number of steps rather than on a token budget, so its numbers are not the experiment's. A TRL example that draws its prompts from a sampler is left for later.

## Alternatives Considered

- **Curriculum inside the environment.** The server would track success rates and pick the next task itself. Rejected: it makes the environment stateful across episodes, mixes orchestration into the environment boundary, and breaks when many trainer workers share or stack environment instances.
- **Only filtering zero-signal groups after generation (DAPO-style).** This is complementary, not a replacement: the filter still pays for generating the groups it drops, and where they are not replaced, the step trains on fewer groups. In our trial, which does not replace them, uniform sampling made no update at all on 19% to 21% of its steps. DAPO resamples until the batch is full, prime-rl keeps dispatching new tasks while its zero-advantage gate, when configured, rejects groups, and TRL's asynchronous GRPO trainer drops them in its rollout worker ([huggingface/trl#7609](https://github.com/huggingface/trl/pull/7609)). A trainer can use both, as long as the sampler sees each group before the filter.
- **Weighting by cost.** The first version of this RFC divided the curriculum's weights by the decayed mean token cost of each key, `ĉ_k`, as `(signal + ε) / ĉ_k`, and used it in its example. On three more seeds it learned less than the signal-only weights per token (−4.3, −19.9 and −16.2 points; the run on seed 2 degenerated, see result 4). A solved key keeps a signal term of about 0.09 under the fixed prior, its answers are the shortest, and `ε` divided by a small cost is large, so tier 0, solved from the start, got 14% to 22% of the episodes. A corrected form, `w_k = signal_k / ĉ_k` with every key kept at a share of at least `ε` outside the division and a prior that decays with the data, gave that tier 5% to 8% of the episodes, matched the signal-only weights per token (+0.1, −3.1 and +1.6 points) and was slightly ahead per episode (+6.1, +0.4 and +2.7). It also had fewer groups with signal per generated token on two of three seeds, and longer training answers on two (111 and 80 tokens against 73 and 59): the policy changes its answer length, so dividing by a decayed mean of past lengths does not control what the next answers cost. Both forms moved episodes away from the longest, hardest tiers: tiers 3 and 4 got 25% to 34% of the episodes under the first form and 47% to 56% under the corrected one, against 52% to 60% under the signal-only weights and 40% under uniform sampling. No lab report we read divides a task's sampling weight by its cost; large RL runs keep a share per source and give slow sources more rollouts instead (MiMo-V2.6, LongCat-Flash-Thinking-2601). Outside RL for language models, [Graves et al. (2017)](https://arxiv.org/abs/1704.03003) divide a task's learning progress by the time it takes to process. Two 2026 preprints that account for cost in GRPO divide a usefulness term by the square root of the cost, to share rollouts among the prompts of a batch ([DUET](https://arxiv.org/abs/2605.08441)) or to sample the minibatches of an update ([Cost-Aware Learning](https://arxiv.org/abs/2604.28020)), not to choose tasks. A sampler that accounts for cost can be added later as another `TaskSampler`.
- **reasoning-gym's own curricula.** The library defines difficulty levels for each dataset, and its curriculum experiment lets the training code raise or lower a dataset's level. That schedule lives inside one library and one dataset at a time; the `CurriculumSampler` works on the tasks any environment publishes through the Task API, and keeps the environment unaware of the schedule (D1). The example above uses the library's levels to define the tiers, so the two fit together.
- **Relying on the Inspect AI harness for evaluation.** Inspect AI remains supported, but it runs its own task and solver definitions. An evaluation that goes through the same environment, tools and rollout path as training is what design principle 1 asks for. The two can meet at the results: an `EvalResult` could later be exported in Inspect's log format, which this RFC does not attempt.

## Open Questions

1. Should the Task API gain WebSocket message types, given the WebSocket invariant, or remain an HTTP metadata surface?
2. Should `difficulty` be standardized further (for example, normalized to `[0, 1]`), or remain environment-defined and only compared within one environment?
3. Where should shared sampler state live for distributed trainers with many rollout workers?
4. Should `openenv eval` results be pushable to the Hub in a schema that the environments datasets RFC (#795) can reference?
5. What is the right default for `success_threshold` on environments whose rewards are graded rather than binary? One direction, suggested in review and not yet tested: estimate each key's signal directly, as the decayed share, among its groups with a scored reward, of those whose scored rewards vary. That is the quantity a zero-variance filter keeps, it is calibrated by construction (section 4), and it needs no `success_threshold`, only a tolerance on the spread of the rewards. With continuous rewards, where almost every group varies a little, that tolerance would have to follow the scale of the rewards, or the weight would have to grow with the size of the variance. Our task can test it only on binary rewards: about 0.1% of its training answers got partial credit. There it would mainly lower the weight of solved keys: in the trial's last quarter, 0.2% to 0.4% of the curriculum's groups on tiers 0 and 1 had signal, against the formula's 0.10. We have not run it.
6. Answered in review: should environments declare their cost? Environments declare cost meters, such as #1382's `service_spend`, for reporting, and trainers use them for budgets, concurrency and rollout allocation. The `CurriculumSampler` counts and reports them and does not weight by them.
7. Should `CollectRunner` accept a per-episode callback (task, `EpisodeRecord`) so adaptive samplers can be driven by `openenv collect` and not only by a trainer loop?
8. Should the first version of the evaluation cover environments whose server runs a whole agent per call, such as Harbor's `run_rollout`, which takes its task as arguments instead of through `reset`?
9. Should `CurriculumSampler` ship in core, next to the two simple samplers, or stay an example? A review suggested keeping it as an example, which is how this RFC offers it.
10. What should a trainer pass to `update` for an episode the harness ends before the environment ends it, such as at the harness's own turn limit in a multi-turn environment, or for one lost to a failure of the infrastructure (the harness, the model endpoint or the environment's server)? Neither has an environment score, a 0 from the trainer would be a reward computed outside the environment, and `None` is for the environment's own failures. For the first, one option is a harness whose limits are never lower than the environment's, so that the environment enforces them and scores the episode; an environment can declare its episode timeout in `openenv.yaml` (`resources.episode_timeout_s`), which a harness could read.

## Related RFCs and Work

- RFC 000: phase three (evals) of the roadmap.
- RFC 004 (Rubrics): the source of the rewards that samplers consume.
- RFC 005 (Agentic harnesses) and the collect pipeline: the rollout pieces reused by evaluation.
- RFC 006 (Agentic RL harness interception): how an agent's model calls are captured; its token records are one source of per-rollout cost.
- RFC 008 (Environment auto-validation) and #1321: repeatability checks at validation time; D4 applies the same idea at evaluation time.
- RFC 012: how Harbor's agents are captured, relevant to open question 8.
- Draft recipes RFC (#924): its open question 2 asks which environment is simple enough for smoke validation. `reasoning_gym_env` answers the "simple and cheap" half, but it is single-turn and does not exercise harness-mediated tool use.
- Environments datasets RFC (#795): possible home for published evaluation results.
- Task API guide (`docs/source/guides/task-api.md`) and the Harbor harness (`envs/harbor_env/harness.py`): the current Task API contract and the manual task selection this RFC generalizes.
- Training guide (`docs/source/guides/training.md`): training stays in the user's framework, and core depends on no trainer; the samplers and `EvalSchedule` follow that.
- Issue #1382: environments that report a per-episode spend, which a trainer can count as cost and use for budgets and concurrency.
- RFC 014 (#1508) and its prototype `posttrain_env` (#1500): an episode whose score job fails ends with a reward of None and still reports its spend, which the samplers count as cost.
- [Knapsack RL](https://arxiv.org/abs/2509.25849): the signal term of section 4, used to share a rollout budget among the prompts of a batch.
- prime-rl's [`TaskSampler`](https://github.com/PrimeIntellect-ai/prime-rl/blob/855a2b815b804857835c635b3fd2b1ae5400f96c/src/prime_rl/orchestrator/curriculum/samplers/base.py) and `Curriculum`: the same two hooks, with the sampler updated before the zero-advantage gate.
- [huggingface/trl#7609](https://github.com/huggingface/trl/pull/7609) (merged on 2026-10-28): a zero-variance filter in TRL's asynchronous GRPO trainer.
- [MiMo-V2.6](https://arxiv.org/abs/2610.11959), [LongCat-Flash-Thinking-2601](https://arxiv.org/abs/2601.16725), [MiniMax-M2](https://arxiv.org/abs/2605.26494) and [INTELLECT-3](https://arxiv.org/abs/2512.16144): shares per source, rollout scheduling and difficulty filtering in large RL runs.
- [DUET](https://arxiv.org/abs/2605.08441) and [Cost-Aware Learning](https://arxiv.org/abs/2604.28020): choices in GRPO that divide by the square root of the cost, for the rollouts of the prompts in a batch and for the minibatches of an update.
- #1250 and its fix #1252 (by one of the authors), merged on 2026-10-13: the `finqa_env` seed issue named in rollout step 3.
