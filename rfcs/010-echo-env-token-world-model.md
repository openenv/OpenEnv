# RFC 010: Env-token World Modeling (ECHO) — trajectory token-role masks + an optimizer world-loss seam

**Status**: Draft
**Created**: 2026-06-14
**Authors**: @thegovind
**RFC ID**: 010

> Numbers 006–009 are reserved for the self-improving-gym RFC family
> (curriculum/adversarial designer, harness optimization, mock foundry,
> environments-from-traces). This RFC is a small, cross-cutting amendment to
> **RFC 007** (the optimizer seam) and **RFC 009** (the trajectory schema).

## Summary

During agent RL we normally **mask out** the environment's response tokens and
train only on the agent's action tokens. **ECHO** (*"Terminal Agents Learn
World Models for Free"*, arXiv 2605.24517, `microsoft/echo-rl` on SkyRL; and
Prime Intellect's *"True Agents Model the World"*) adds a small cross-entropy
loss that makes the policy **predict the environment's observation tokens** too:

```
L_ECHO = L_GRPO(action tokens) + λ · L_env(observation tokens)
```

The policy already conditions on those observation tokens and already computes
their logits in the same forward pass, so the world-modeling signal is **~free**
— no extra rollouts, no teacher, no separate world model, just a different mask
over logits you already have. Reported results: **~2.3× faster** RL,
TerminalBench-2.0 pass@1 roughly **doubles**, recovers **50–104%** of expert-SFT
gain with no teacher, and even **verifier-free** (reward off) improves held-out
tasks. The authors expect it to generalize to any agent setting where the world
responds in tokens (browser, multi-tool, long-horizon coding, assistants).

OpenEnv's `step()` observation stream is exactly that supervisable target. This
RFC adds the two small things a trainer needs to switch the objective on.

## Motivation — the two gaps

ECHO is "a few lines on top of any GRPO trainer," but OpenEnv cannot express it
today:

1. **No per-token role masks.** OpenEnv's rollout/trajectory artifacts
   (`core/harness/collect.py` `EpisodeRecord`, the planned RFC 009 `Trajectory`
   import schema, `TrajectoryRubric`) are **message-level** (`messages` +
   `tool_trace`). A trainer cannot tell, *per token*, which tokens were the
   agent's **actions** (the GRPO target) vs the environment's **observations**
   (the world-model target) — nor the finer **real env output** vs **harness
   warning** boundary ECHO uses to avoid training on boilerplate. Without these
   spans the env loss cannot be applied.
2. **No world-loss config on the optimizer seam.** The RFC 007 `Optimizer` /
   `Trainer` protocol (and the `grpo_training_loop` example) is action-token
   only. There is nowhere to set `λ`, the loss target, or the rollout filters.

## Proposal

### Part A — Trajectory token-role masks (extends RFC 009 + the rollout format)

A trajectory/rollout SHOULD be representable as a token sequence with aligned
per-token **role** labels:

| role | meaning | loss |
|---|---|---|
| `context` | system prompt / scaffolding / shell prompt | none |
| `action` | agent/assistant tokens | RL / GRPO target |
| `env_output` | real tool/world output | world-model (ECHO) target |
| `warning` | harness boilerplate (echoed command, `[stdout]` tags, wrappers) | excluded from the env loss by default |

Concretely this is three boolean masks aligned to the token ids
(`action_mask`, `obs_mask`, `warning_mask`) — the same disjoint masks ECHO's
reference code carries (`completion_masks` / `completion_observation_masks` /
`completion_warning_masks`). They can be produced either by emitting token-role
spans at rollout time, or by re-deriving them from the existing role-tagged
`messages` + the chat template. The RFC 009 `Trajectory` schema and
`collect.py`'s record SHOULD carry enough structure to recover them.

A reference realization is in
[`examples/echo_world_model/trajectory.py`](../examples/echo_world_model/trajectory.py).

### Part B — World-loss config on the `Optimizer` seam (extends RFC 007)

The `Optimizer`/`Trainer` seam SHOULD expose:

- `world_model_coeff` (**λ**) — `0.0` = vanilla GRPO; small positive = ECHO.
- `world_loss_target` — `env_only` (default; real output, exclude warnings) | `all`.
- rollout filters — e.g. min valid-tool-call %, parse-clean %, correct % — gate
  which rollouts contribute the env term.

The objective is exactly:

```python
L = L_GRPO(action tokens) + world_model_coeff * CE(observation tokens)
```

See [`examples/echo_world_model/echo_loss.py`](../examples/echo_world_model/echo_loss.py)
for the reference loss (with `world_model_coeff=0` ⇒ vanilla GRPO and
`use_rl=False` ⇒ verifier-free env-only as pinned identities).

**Implementation note (no extra forward pass).** The env term is a
length-normalized cross-entropy on the observation tokens — i.e. SFT on the
tool-response tokens. SFT is RL with a *constant positive advantage*, so a
trainer can implement ECHO without a second loss function: set a per-token
advantage vector that is the GRPO group-relative advantage on `action` tokens
and a constant `λ` on `env_output` tokens. Normalize the RL and SFT token
contributions independently. (This is how the [Tinker](../examples/echo_world_model/backends/tinker.md)
and [Foundry Fine-Tuning](../examples/echo_world_model/backends/foundry-finetuning.md) adapters express it;
[SkyRL](../examples/echo_world_model/backends/skyrl.md) is the reference backend.)

## Why it is worth a protocol change

- **Dense supervision for sparse/collapsing reward.** GRPO gets ~no signal when
  a small model's pass-rate is low or post-PMF variance collapses; the env term
  turns every turn — *including failed ones* — into supervision.
- **Partly substitutes for a teacher.** Much of what expert SFT buys is an
  interaction prior the environment supplies for free; "the terminal is the
  teacher."
- **Verifier-free bootstrap.** A customer with an environment (or replayable
  traces) but **no rubric yet** can still self-improve — then GRPO layers on
  once a grader (RFC 004 / 006) exists.
- **Near-universal & near-free.** Any OpenEnv observation stream is a valid
  target, at no extra rollout cost.

## Backward compatibility & caveats

- Purely additive. `world_model_coeff = 0.0` is exactly today's behavior; the
  masks are optional metadata existing trainers can ignore.
- **Keep λ small.** Prime Intellect observed collapse at `0.05` for GLM-4.5-Air
  (stable at `0.005`); echo-rl's published Qwen3-8B config uses `0.05`. Expose
  it as config and sweep.
- Works best when tool outputs are **predictable-without-memorization and
  complex**; pure-retrieval outputs can fail to generalize — hence `env_only`
  and the rollout filters.

## Reference implementation

Two runnable, CPU-only references ship with this RFC:

- [`examples/echo_world_model/`](../examples/echo_world_model/): a toy terminal env that
  **trains**, including the verifier-free result on a small model and the role-mask schema,
  with adapters to SkyRL (reference), Tinker, and Foundry Fine-Tuning for the full GPU run.
- [`examples/echo_on_agent_world_model/`](../examples/echo_on_agent_world_model/): the same
  role masks on a **real upstream env** (`agent_world_model_env`), showing AWM observations
  map onto ECHO roles almost one to one.

## References

- ECHO — *"Terminal Agents Learn World Models for Free"*, arXiv `2605.24517`;
  code `github.com/microsoft/echo-rl` (on `NovaSky-AI/SkyRL`).
- Prime Intellect — *"True Agents Model the World"* (2026-06-05) — ECHO + PaW;
  SFT-on-tool-tokens = RL with a constant positive advantage.
- Relationship: extends **RFC 007** (optimizer seam) and **RFC 009** (trajectory
  schema); complements **RFC 004** rubrics (reward when a grader exists).
