# AWM expert-in-the-loop: run notes

This note captures the current experimental state for the AWM expert-in-the-loop
recipe. It is intentionally conservative. The Foundry Fine-Tuning backend ran
successfully, but the current setup does not yet show an expert advantage.

## What ran

- Environment: `agent_world_model_env`
- Split: `splits/workflow_automation.json`
- Train tasks: 63
- Held-out validation tasks: 33
- Verifier: deterministic `code` verifier
- Trivial no-op passes: filtered out before split creation
- Base model: Qwen/Qwen3.5-4B
- Frontier advisor: GPT-5.5-compatible Azure OpenAI deployment
- Backend: Tinker-style `forward_backward` + `optim_step` on Foundry Fine-Tuning

Two full train runs completed:

| Condition | Train steps | Wall-clock |
|---|---:|---:|
| Expert available | 31 | 36.7 min |
| No expert | 31 | 51.0 min |

## Held-out evaluation

Both final checkpoints were evaluated expert-free on the same 33 validation tasks.

| Condition | Complete rate | Mean verifier reward | Mean shaped total |
|---|---:|---:|---:|
| Expert available | 1/33 | 0.0303 | 0.1333 |
| No expert | 1/33 | 0.0303 | 0.1409 |

Both checkpoints completed the same validation task:

- `workflow_automation_1`, task `1`

The paired delta on verifier success is `0.0`. This means the current experiment is
a tie on real task success. It should not be presented as an expert improvement.

## Why the first reading was misleading

The training logs include `env/all/reward/total`, which is a shaped rollout reward.
It includes small tool-use shaping terms, so it partly rewards taking more useful
turns. The no-expert run took more turns per episode, so it earned a higher shaped
training reward even though held-out verifier success was tied.

The expert condition also had extra protocol complexity:

- The prompt included a third virtual tool, `ask_expert`.
- The Qwen renderer produced multiple tool-call shapes.
- Some `ask_expert` calls were malformed or sparse.
- Expert advice was not consistently used before task verification.

## Next experiments before making claims

1. Keep the prompt fixed between conditions and only add or remove the expert tool.
2. Normalize the tool-call protocol so direct environment tools, `call_tool`, and
   `ask_expert` parse consistently.
3. Evaluate several seeds on the held-out split.
4. Increase group size above 2 so GRPO has more useful within-group contrast.
5. Report verifier completion as the headline metric and shaped reward only as a
   training diagnostic.

Until those are done, the honest statement is:

> The Foundry Fine-Tuning backend run and held-out eval succeeded. This configuration
> does not yet demonstrate an expert-in-the-loop improvement.
